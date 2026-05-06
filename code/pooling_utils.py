"""
Centralised pooling-strategy resolver for layer-wise embedding extraction.

Three pooling strategies are supported:
    - "cls"        : take hidden_states[:, 0, :]            (BERT [CLS] token)
    - "mean"       : attention-mask weighted mean over tokens
    - "last_token" : last non-padding token                  (decoder LMs)

Resolution order in get_pooling_strategy():
    1. Manual override in MODEL_POOLING_OVERRIDES.
    2. Auto-detect from the model's `1_Pooling/config.json`
       (sentence-transformers convention).
    3. Fallback to architecture-based default
       (decoder -> last_token, otherwise -> mean).

This module centralises what used to be a hard-coded
`pooling = "last_token" if is_decoder else "mean"` in three different
analysis scripts, so that BERT-family models that were trained with
CLS-pooling (BGE, GTE, Snowflake, mxbai, ...) are evaluated with the
same pooling that produced their published MTEB scores.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Pooling functions
# ----------------------------------------------------------------------

def cls_pool(hs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Take the first token ([CLS]) of the sequence.

    `mask` is unused but kept for signature compatibility.
    """
    return hs[:, 0, :]


def mean_pooling(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-weighted mean over the sequence dimension."""
    return (hs * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).float()


def last_token_pool(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Take the last non-padding token (assumes left-padding for decoders)."""
    seq_len = mask.sum(dim=1, keepdim=True) - 1
    idx = seq_len.unsqueeze(-1).expand(-1, -1, hs.size(-1))
    return hs.gather(1, idx.long()).squeeze(1)


def apply_pooling(strategy: str, hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Dispatch to the requested pooling function."""
    if strategy == "cls":
        return cls_pool(hs, mask)
    if strategy == "last_token":
        return last_token_pool(hs, mask)
    return mean_pooling(hs, mask)


# ----------------------------------------------------------------------
# Manual overrides for models whose 1_Pooling/config.json is missing
# or whose architecture-based default would be wrong.
# ----------------------------------------------------------------------

MODEL_POOLING_OVERRIDES: dict[str, str] = {
    # ---------------------------------------------------------------------- BGE family: trained with CLS pooling ----------------------------------------------------------------------
    "BAAI/bge-base-en-v1.5":          "cls",
    "BAAI/bge-large-en-v1.5":         "cls",
    "BAAI/bge-small-en-v1.5":         "cls",
    "BAAI/bge-m3":                    "cls",

    # ---------------------------------------------------------------------- GTE family (BERT-based variants): trained with CLS ----------------------------------------------------------------------
    "thenlper/gte-base":              "cls",
    "thenlper/gte-large":             "cls",
    "thenlper/gte-small":             "cls",
    "Alibaba-NLP/gte-base-en-v1.5":   "cls",
    "Alibaba-NLP/gte-large-en-v1.5":  "cls",

    # ---------------------------------------------------------------------- Snowflake Arctic Embed (BERT-based): CLS ----------------------------------------------------------------------
    "Snowflake/snowflake-arctic-embed-l":     "cls",
    "Snowflake/snowflake-arctic-embed-m":     "cls",
    "Snowflake/snowflake-arctic-embed-s":     "cls",
    "Snowflake/snowflake-arctic-embed-xs":    "cls",
    "Snowflake/snowflake-arctic-embed-m-v2.0":"cls",
    "Snowflake/snowflake-arctic-embed-l-v2.0":"cls",

    # ---------------------------------------------------------------------- mxbai (BERT-based): CLS ----------------------------------------------------------------------
    "mixedbread-ai/mxbai-embed-large-v1":     "cls",

    # ---------------------------------------------------------------------- GritLM (decoder Mistral, but trained with bidirectional+mean) ----------------------------------------------------------------------
    # GritLM's "embedding mode" defaults to mean pooling AND bidirectional
    # attention.  We override pooling to mean (matches the official setup),
    # but AutoModel.from_pretrained loads it as a standard causal Mistral,
    # so attention here is causal -- NOT identical to the official
    # bidirectional embedding mode.  This is acceptable for a layer-wise
    # geometric probe (we compare hidden state trajectories, not deployed
    # embedding outputs), but should be noted as a caveat in the paper.
    # See docs/probe_vs_deployment_caveats.md for details.
    "GritLM/GritLM-7B":               "mean",
}


# ----------------------------------------------------------------------
# Auto-detection from sentence-transformers' 1_Pooling/config.json
# ----------------------------------------------------------------------

def _detect_from_st_config(model_name: str) -> Optional[str]:
    """Try to read sentence-transformers' `1_Pooling/config.json`.

    Returns one of {"cls", "mean", "last_token"} or None if not detectable.
    """
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:
        return None

    try:
        cfg_path = hf_hub_download(
            repo_id=model_name,
            filename="1_Pooling/config.json",
            repo_type="model",
        )
    except (EntryNotFoundError, RepositoryNotFoundError, OSError, Exception):
        return None

    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    # sentence-transformers Pooling config keys
    if cfg.get("pooling_mode_cls_token"):
        return "cls"
    if cfg.get("pooling_mode_lasttoken"):
        return "last_token"
    if cfg.get("pooling_mode_mean_tokens"):
        return "mean"
    if cfg.get("pooling_mode_max_tokens"):
        # fall back to mean for max-pooled models (rare in MTEB)
        return "mean"
    return None


# ----------------------------------------------------------------------
# Public resolver
# ----------------------------------------------------------------------

def get_pooling_strategy(
    model_name: str,
    is_decoder: bool,
    use_st_autodetect: bool = True,
    verbose: bool = True,
) -> str:
    """Resolve which pooling strategy to use for *this* model.

    Resolution order:
        1. MODEL_POOLING_OVERRIDES (case-insensitive lookup on full repo id).
        2. (optional) sentence-transformers `1_Pooling/config.json`.
        3. Fallback: decoder -> "last_token", else -> "mean".

    Set `use_st_autodetect=False` to skip the network call (useful when
    HF Hub is unreachable; only the override table + arch fallback are used).
    """
    # 1. manual override (exact match, case-insensitive on the lower key)
    for key, val in MODEL_POOLING_OVERRIDES.items():
        if key.lower() == model_name.lower():
            if verbose:
                print(f"  [pool] OVERRIDE: {model_name} -> {val}")
            return val

    # 2. sentence-transformers config
    if use_st_autodetect:
        detected = _detect_from_st_config(model_name)
        if detected is not None:
            if verbose:
                print(f"  [pool] 1_Pooling/config.json: {model_name} -> {detected}")
            return detected

    # 3. architecture-based default
    fallback = "last_token" if is_decoder else "mean"
    if verbose:
        print(f"  [pool] arch-default: {model_name} (is_decoder={is_decoder}) -> {fallback}")
    return fallback


__all__ = [
    "cls_pool",
    "mean_pooling",
    "last_token_pool",
    "apply_pooling",
    "MODEL_POOLING_OVERRIDES",
    "get_pooling_strategy",
]
