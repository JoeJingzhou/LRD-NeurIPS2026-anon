#!/usr/bin/env python3
"""
GFMI All-Layers Profile.

For each (dataset, model), extract embeddings from ALL hidden layers,
compute GFMI(l_i, last) for every layer i, and produce:
  - A GFMI-AUC profile curve across layers
  - Correlation analysis with MTEB scores (raw + partial controlling dim)
  - JSON results + PNG visualisation

Reuses model loading / multi-pass extraction from extract_all_layers_id.py,
with a fully vectorized graph-filtration MI implementation.
"""

import argparse
import gc
import inspect
import gzip
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

MAX_N = 5000
K_DEFAULT = 30
N_SCALES = 20
LAYERS_PER_PASS = 8

MODELS = [
    # ── Original models ──
    ("sentence-transformers/all-MiniLM-L6-v2", "all_MiniLM_L6_v2"),
    ("intfloat/e5-large-v2", "e5_large_v2"),
    ("NovaSearch/stella_en_1.5B_v5", "stella_en_1.5B_v5"),
    ("BAAI/bge-multilingual-gemma2", "bge_multilingual_gemma2"),
    ("Alibaba-NLP/gte-Qwen2-7B-instruct", "gte_Qwen2_7B_instruct"),
    ("Linq-AI-Research/Linq-Embed-Mistral", "Linq_Embed_Mistral"),
    ("Salesforce/SFR-Embedding-Mistral", "SFR_Embedding_Mistral"),
    ("zeta-alpha-ai/Zeta-Alpha-E5-Mistral", "Zeta_Alpha_E5_Mistral"),
    # ── New 4096-dim models ──
    ("intfloat/e5-mistral-7b-instruct", "e5_mistral_7b_instruct"),
    ("Salesforce/SFR-Embedding-2_R", "SFR_Embedding_2_R"),
    ("GritLM/GritLM-7B", "GritLM_7B"),
    ("BAAI/bge-en-icl", "bge_en_icl"),
    ("Qwen/Qwen3-Embedding-8B", "Qwen3_Embedding_8B"),
    # ── New 1024-dim models ──
    ("BAAI/bge-large-en-v1.5", "bge_large_en_v1_5"),
    ("mixedbread-ai/mxbai-embed-large-v1", "mxbai_embed_large_v1"),
    ("Snowflake/snowflake-arctic-embed-l", "snowflake_arctic_embed_l"),
    ("Alibaba-NLP/gte-large-en-v1.5", "gte_large_en_v1_5"),
    ("Qwen/Qwen3-Embedding-0.6B", "Qwen3_Embedding_0_6B"),
    # ── healthy embed extensions used by formal30 backfill ──
    ("nomic-ai/nomic-embed-text-v1.5", "nomic_embed_text_v1_5"),
    ("BAAI/bge-base-en-v1.5", "bge_base_en_v1_5"),
    ("intfloat/e5-base-v2", "e5_base_v2"),
    ("sentence-transformers/all-mpnet-base-v2", "all_mpnet_base_v2"),
    ("jinaai/jina-embeddings-v3", "jina_embeddings_v3"),
    ("Snowflake/snowflake-arctic-embed-m-v2.0", "snowflake_arctic_embed_m_v2"),
    ("thenlper/gte-large", "gte_large"),
    ("thenlper/gte-base", "gte_base"),
]

DATASET_CONFIGS = {
    # ── Already completed ──
    "mteb_banking77": dict(
        hf="mteb/banking77", config=None, split="train",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_arguana": dict(
        hf="mteb/arguana", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_MassiveIntentClassification_en": dict(
        hf="mteb/MassiveIntentClassification", config="en", split="train",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_scidocs": dict(
        hf="mteb/scidocs", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    # ── Retrieval (new) ──
    "mteb_NFCorpus": dict(
        hf="mteb/nfcorpus", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_FiQA2018": dict(
        hf="mteb/fiqa", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_SciFact": dict(
        hf="mteb/scifact", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    # ── Classification (new) ──
    "mteb_ToxicConversationsClassification": dict(
        hf="mteb/toxic_conversations_50k", config=None, split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_ImdbClassification": dict(
        hf="mteb/imdb", config=None, split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_EmotionClassification": dict(
        hf="mteb/emotion", config=None, split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    # ── wave 2 (sync with nrs_profile) ──
    "mteb_AmazonCounterfactualClassification": dict(
        hf="mteb/amazon_counterfactual", config="en", split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_AmazonPolarityClassification": dict(
        hf="mteb/amazon_polarity", config=None, split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_TweetSentimentExtractionClassification": dict(
        hf="mteb/tweet_sentiment_extraction", config=None, split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_QuoraRetrieval": dict(
        hf="mteb/quora", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_TRECCOVID": dict(
        hf="mteb/trec-covid", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    # ── wave 3 (sync with nrs_profile) ──
    "mteb_MTOPIntentClassification": dict(
        hf="mteb/MTOPIntentClassification", config="en", split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_DBpediaClassification": dict(
        hf="fancyzhx/dbpedia_14", config=None, split="test",
        text_col="content", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_TweetTopicSingleClassification": dict(
        hf="mteb/TweetTopicSingleClassification", config=None, split="test_2021",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_HotpotQA": dict(
        hf="mteb/hotpotqa", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_NQ": dict(
        hf="mteb/nq", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_CQADupstackRetrieval": dict(
        hf="mteb/cqadupstack-android", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_Touche2020": dict(
        hf="mteb/touche2020", config="corpus", split="corpus",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    # ── formal30 extension (sync with frenet_curvature) ──
    "mteb_MTOPDomainClassification_en": dict(
        hf="mteb/MTOPDomainClassification", config="en", split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_MassiveScenarioClassification_en": dict(
        hf="mteb/amazon_massive_scenario", config="en", split="test",
        text_col="text", label_col="label", task="CLS", load_mode="standard",
    ),
    "mteb_AskUbuntuDupQuestions": dict(
        hf="mteb/AskUbuntuDupQuestions", config="corpus", split="test",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_STSBenchmark": dict(
        hf="mteb/stsbenchmark-sts", config=None, split="test",
        task="STS", load_mode="sts_pairs", col_a="sentence1", col_b="sentence2",
    ),
    "mteb_SICK_R": dict(
        hf="mteb/sickr-sts", config=None, split="test",
        task="STS", load_mode="sts_pairs", col_a="sentence1", col_b="sentence2",
    ),
    "mteb_MindSmallReranking": dict(
        hf="mteb/MindSmallReranking", config="corpus", split="test",
        text_col="text", label_col=None, task="RET", load_mode="standard",
    ),
    "mteb_SprintDuplicateQuestions": dict(
        hf="mteb/sprintduplicatequestions-pairclassification", config=None, split="test",
        task="PAIR", load_mode="pair_cls", col_a="sent1", col_b="sent2",
    ),
    "mteb_TwitterURLCorpus": dict(
        hf="mteb/twitterurlcorpus-pairclassification", config=None, split="test",
        task="PAIR", load_mode="pair_cls", col_a="sent1", col_b="sent2",
    ),
}

MTEB_SCORES = {
    "mteb_banking77": {
        "bge_multilingual_gemma2": 92.53, "gte_Qwen2_7B_instruct": 87.57,
        "Linq_Embed_Mistral": 87.88, "SFR_Embedding_Mistral": 88.81,
        "Zeta_Alpha_E5_Mistral": 83.11, "all_MiniLM_L6_v2": 80.04,
        "stella_en_1.5B_v5": 89.79, "e5_large_v2": 84.55,
        "e5_mistral_7b_instruct": 88.23, "SFR_Embedding_2_R": 90.02,
        "GritLM_7B": 88.47, "bge_en_icl": 91.49, "Qwen3_Embedding_8B": 87.27,
        "bge_large_en_v1_5": 87.79, "mxbai_embed_large_v1": 87.82,
        "snowflake_arctic_embed_l": 80.06, "gte_large_en_v1_5": 87.33,
        "Qwen3_Embedding_0_6B": 81.01,
    },
    "mteb_arguana": {
        "bge_multilingual_gemma2": 77.37, "gte_Qwen2_7B_instruct": 54.56,
        "Linq_Embed_Mistral": 59.65, "SFR_Embedding_Mistral": 67.17,
        "Zeta_Alpha_E5_Mistral": 65.82, "all_MiniLM_L6_v2": 50.17,
        "stella_en_1.5B_v5": 57.06, "e5_large_v2": 45.43,
        "e5_mistral_7b_instruct": 61.88, "SFR_Embedding_2_R": 62.34,
        "GritLM_7B": 63.24, "bge_en_icl": 83.08, "Qwen3_Embedding_8B": 77.00,
        "bge_large_en_v1_5": 63.54, "mxbai_embed_large_v1": 66.02,
        "snowflake_arctic_embed_l": 59.09, "gte_large_en_v1_5": 72.11,
        "Qwen3_Embedding_0_6B": 71.00,
    },
    "mteb_MassiveIntentClassification_en": {
        "bge_multilingual_gemma2": 82.05, "gte_Qwen2_7B_instruct": 85.43,
        "Linq_Embed_Mistral": 76.42, "SFR_Embedding_Mistral": 75.88,
        "Zeta_Alpha_E5_Mistral": 77.31, "all_MiniLM_L6_v2": 66.94,
        "stella_en_1.5B_v5": 84.51, "e5_large_v2": 68.14,
        "e5_mistral_7b_instruct": 71.14, "SFR_Embedding_2_R": 85.97,
        "GritLM_7B": 80.78, "bge_en_icl": 82.93, "Qwen3_Embedding_8B": 75.57,
        "bge_large_en_v1_5": 77.56, "mxbai_embed_large_v1": 76.24,
        "snowflake_arctic_embed_l": 65.79, "gte_large_en_v1_5": 78.94,
        "Qwen3_Embedding_0_6B": 73.76,
    },
    "mteb_scidocs": {
        "bge_multilingual_gemma2": 26.93, "gte_Qwen2_7B_instruct": 23.48,
        "Linq_Embed_Mistral": 21.93, "SFR_Embedding_Mistral": 19.91,
        "Zeta_Alpha_E5_Mistral": 20.86, "all_MiniLM_L6_v2": 21.64,
        "stella_en_1.5B_v5": 26.77, "e5_large_v2": 20.50,
        "e5_mistral_7b_instruct": 16.30, "SFR_Embedding_2_R": 24.87,
        "GritLM_7B": 24.41, "bge_en_icl": 25.26, "Qwen3_Embedding_8B": 32.74,
        "bge_large_en_v1_5": 22.64, "mxbai_embed_large_v1": 23.32,
        "snowflake_arctic_embed_l": 21.36, "gte_large_en_v1_5": 26.35,
        "Qwen3_Embedding_0_6B": 24.41,
    },
    # ── Retrieval (new) ──
    "mteb_NFCorpus": {
        "gte_Qwen2_7B_instruct": 40.60, "bge_multilingual_gemma2": 38.11,
        "Linq_Embed_Mistral": 42.03, "SFR_Embedding_Mistral": 41.88,
        "e5_large_v2": 37.13, "all_MiniLM_L6_v2": 31.59,
        "Zeta_Alpha_E5_Mistral": 40.46, "stella_en_1.5B_v5": 42.00,
        "e5_mistral_7b_instruct": 38.62, "SFR_Embedding_2_R": 41.34,
        "GritLM_7B": 40.89, "bge_en_icl": 41.85, "Qwen3_Embedding_8B": 41.00,
        "bge_large_en_v1_5": 38.13, "mxbai_embed_large_v1": 38.64,
        "snowflake_arctic_embed_l": 37.65, "gte_large_en_v1_5": 36.95,
        "Qwen3_Embedding_0_6B": 37.00,
    },
    "mteb_FiQA2018": {
        "gte_Qwen2_7B_instruct": 62.03, "bge_multilingual_gemma2": 60.04,
        "Linq_Embed_Mistral": 61.21, "SFR_Embedding_Mistral": 60.40,
        "e5_large_v2": 41.14, "all_MiniLM_L6_v2": 36.87,
        "Zeta_Alpha_E5_Mistral": 58.78, "stella_en_1.5B_v5": 60.48,
        "e5_mistral_7b_instruct": 56.59, "SFR_Embedding_2_R": 61.77,
        "GritLM_7B": 59.95, "bge_en_icl": 59.67, "Qwen3_Embedding_8B": 65.00,
        "bge_large_en_v1_5": 45.02, "mxbai_embed_large_v1": 45.27,
        "snowflake_arctic_embed_l": 44.71, "gte_large_en_v1_5": 63.23,
        "Qwen3_Embedding_0_6B": 47.00,
    },
    "mteb_SciFact": {
        "gte_Qwen2_7B_instruct": 79.06, "bge_multilingual_gemma2": 72.05,
        "Linq_Embed_Mistral": 78.32, "SFR_Embedding_Mistral": 77.66,
        "e5_large_v2": 72.24, "all_MiniLM_L6_v2": 64.51,
        "Zeta_Alpha_E5_Mistral": 77.38, "stella_en_1.5B_v5": 80.09,
        "e5_mistral_7b_instruct": 76.41, "SFR_Embedding_2_R": 85.91,
        "GritLM_7B": 79.17, "bge_en_icl": 79.09, "Qwen3_Embedding_8B": 78.00,
        "bge_large_en_v1_5": 74.61, "mxbai_embed_large_v1": 74.73,
        "snowflake_arctic_embed_l": 73.82, "gte_large_en_v1_5": 82.43,
        "Qwen3_Embedding_0_6B": 70.00,
    },
    # ── Classification (new) ──
    "mteb_ToxicConversationsClassification": {
        "gte_Qwen2_7B_instruct": 85.74, "bge_multilingual_gemma2": 87.34,
        "Linq_Embed_Mistral": 71.29, "SFR_Embedding_Mistral": 69.33,
        "e5_large_v2": 63.29, "all_MiniLM_L6_v2": 62.09,
        "Zeta_Alpha_E5_Mistral": 75.88, "stella_en_1.5B_v5": 88.76,
        "e5_mistral_7b_instruct": 69.59, "SFR_Embedding_2_R": 91.14,
        "GritLM_7B": 70.80, "bge_en_icl": 93.17, "Qwen3_Embedding_8B": 91.65,
        "bge_large_en_v1_5": 70.91, "mxbai_embed_large_v1": 71.48,
        "snowflake_arctic_embed_l": 64.71, "gte_large_en_v1_5": 82.61,
        "Qwen3_Embedding_0_6B": 82.13,
    },
    "mteb_ImdbClassification": {
        "gte_Qwen2_7B_instruct": 96.75, "bge_multilingual_gemma2": 96.66,
        "Linq_Embed_Mistral": 94.78, "SFR_Embedding_Mistral": 94.79,
        "e5_large_v2": 91.69, "all_MiniLM_L6_v2": 61.76,
        "Zeta_Alpha_E5_Mistral": 95.50, "stella_en_1.5B_v5": 96.66,
        "e5_mistral_7b_instruct": 94.78, "SFR_Embedding_2_R": 96.80,
        "GritLM_7B": 95.00, "bge_en_icl": 96.91, "Qwen3_Embedding_8B": 97.37,
        "bge_large_en_v1_5": 92.85, "mxbai_embed_large_v1": 92.83,
        "snowflake_arctic_embed_l": 72.88, "gte_large_en_v1_5": 92.10,
        "Qwen3_Embedding_0_6B": 95.44,
    },
    "mteb_EmotionClassification": {
        "gte_Qwen2_7B_instruct": 79.46, "bge_multilingual_gemma2": 92.98,
        "Linq_Embed_Mistral": 51.82, "SFR_Embedding_Mistral": 50.24,
        "e5_large_v2": 49.45, "all_MiniLM_L6_v2": 40.83,
        "Zeta_Alpha_E5_Mistral": 57.72, "stella_en_1.5B_v5": 84.30,
        "e5_mistral_7b_instruct": 49.77, "SFR_Embedding_2_R": 93.37,
        "GritLM_7B": 52.81, "bge_en_icl": 93.36, "Qwen3_Embedding_8B": 66.29,
        "bge_large_en_v1_5": 51.52, "mxbai_embed_large_v1": 50.88,
        "snowflake_arctic_embed_l": 46.46, "gte_large_en_v1_5": 46.77,
        "Qwen3_Embedding_0_6B": 63.25,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Model loading / embedding extraction (from extract_all_layers_id.py)
# ──────────────────────────────────────────────────────────────────────

def mean_pooling(hidden: Tensor, mask: Tensor) -> Tensor:
    """GFMI variant: explicit zero-mask before sum, plus eps-clamped denom."""
    masked = hidden.masked_fill(~mask[..., None].bool(), 0.0)
    denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1e-9)
    return masked.sum(dim=1) / denom


def last_token_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    """GFMI variant: gather via integer index (assumes left-padding)."""
    seq_lens = mask.sum(dim=1) - 1
    bs = hidden.shape[0]
    return hidden[torch.arange(bs, device=hidden.device), seq_lens]


def cls_pool(hidden: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Take the [CLS] (first) token; mask unused, kept for signature parity."""
    return hidden[:, 0, :]


from pooling_utils import get_pooling_strategy  # noqa: E402


def _align_hidden_and_mask(hidden: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Some trust_remote_code models (e.g. Snowflake Arctic) return hidden sequences
    whose length disagrees with the tokenizer's attention_mask; truncate both to
    the common length before pooling.
    """
    if hidden.dim() != 3 or mask.dim() != 2:
        return hidden, mask
    sh, sm = hidden.size(1), mask.size(1)
    if sh == sm:
        return hidden, mask
    n = min(sh, sm)
    return hidden[:, :n].contiguous(), mask[:, :n].contiguous()


def _reshape_hidden_to_mask_layout(hidden: Tensor, mask: Tensor) -> Tensor:
    """
    Some encoder checkpoints flatten (batch, seq) into a single sequence axis and
    return hidden states shaped like (1, B*S, H). If the total token count matches
    the tokenizer mask, reshape back to the standard (B, S, H) layout before pooling.
    """
    if hidden.dim() != 3 or mask.dim() != 2:
        return hidden
    bsz, seqlen = mask.shape
    if hidden.shape[0] == bsz and hidden.shape[1] == seqlen:
        return hidden
    if hidden.shape[0] * hidden.shape[1] != bsz * seqlen:
        return hidden
    return hidden.contiguous().reshape(bsz, seqlen, hidden.shape[-1])


def _load_model(model_name, device, force_bidir=False):
    import transformers.utils.import_utils as _tu
    import transformers.modeling_utils as _mu
    _tu.check_torch_load_is_safe = lambda: None
    _mu.check_torch_load_is_safe = lambda: None

    # LLM2Vec models need special adapter-based loading
    from bidir_models import is_llm2vec_model, _load_llm2vec_as_bidir, \
        _patch_nemotron_py39, _get_nested_config_value, \
        is_nvembed_model, _load_nvembed
    if is_llm2vec_model(model_name):
        return _load_llm2vec_as_bidir(model_name, device)
    if is_nvembed_model(model_name):
        return _load_nvembed(model_name, device)

    if "nemotron" in model_name.lower():
        _patch_nemotron_py39()

    from transformers import AutoModel, AutoTokenizer, AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_layers = _get_nested_config_value(config, "num_hidden_layers")
    hidden_size = _get_nested_config_value(config, "hidden_size")
    if n_layers is None:
        raise ValueError(f"Cannot determine num_hidden_layers for {model_name}")
    arch = getattr(config, "model_type", "unknown")

    _DECODER_TYPES = (
        "qwen2", "qwen3", "mistral", "llama", "gemma", "gemma2",
        "gpt2", "falcon", "phi", "phi3", "phimoe", "starcoder2",
    )
    is_decoder = arch in _DECODER_TYPES and not force_bidir

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if (arch in _DECODER_TYPES) else torch.float32,
    }
    if arch in _DECODER_TYPES:
        load_kwargs["attn_implementation"] = "eager"
    model = AutoModel.from_pretrained(model_name, **load_kwargs).to(device)
    try:
        model.resize_token_embeddings(len(tokenizer))
    except NotImplementedError:
        pass
    model.eval()

    info = dict(n_layers=n_layers, hidden_size=hidden_size,
                arch=arch, is_decoder=is_decoder)
    return model, tokenizer, info


def _filter_forward_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pass only the kwargs that ``forward`` actually accepts, to avoid encoders
    complaining about use_cache or custom heads complaining about extra kwargs."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params):
        return kwargs
    names = {p.name for p in params if p.name != "self"}
    return {k: v for k, v in kwargs.items() if k in names}


def _forward_with_hidden_states(model, enc: Dict[str, torch.Tensor], is_decoder: bool):
    """
    Call the backbone so that the returned object exposes ``hidden_states``
    (a tuple of the embedding output plus each layer).
    - Decoders typically require ``use_cache=False``; some encoders do not accept
      ``use_cache``.
    - For some trust_remote_code wrappers, ``output_hidden_states`` is only
      available on a submodule.
    """
    base_kw: Dict[str, Any] = {**enc, "output_hidden_states": True, "return_dict": True}
    if is_decoder:
        base_kw["use_cache"] = False

    def try_module(m, kw: Dict[str, Any]):
        fk = _filter_forward_kwargs(m.forward, kw)
        return m(**fk)

    candidates: List[Any] = [model]
    for name in (
        "model",
        "auto_model",
        "base_model",
        "embedding_model",
        "bert",
        "nomic_bert",
        "roberta",
        "distilbert",
        "transformer",
        "text_encoder",
        "backbone",
        "text_model",
    ):
        sub = getattr(model, name, None)
        if sub is not None and sub not in candidates:
            candidates.append(sub)

    errors: List[str] = []
    for m in candidates:
        for variant in (
            base_kw,
            {**enc, "output_hidden_states": True, "return_dict": True},
            {**enc, "output_hidden_states": True, "return_dict": True, "use_cache": False},
        ):
            try:
                out = try_module(m, variant)
            except Exception as e:
                errors.append(f"{m.__class__.__name__}: {e}")
                continue
            hs = getattr(out, "hidden_states", None)
            if hs is not None:
                return out
    raise RuntimeError(
        "Could not obtain per-layer hidden_states from this checkpoint "
        "(forward does not support it or submodule probing failed). "
        f"Tried {len(candidates)} modules. Last errors: {errors[-3:]}"
    )


def _hidden_states_batch_is_valid(out: Any, batch_size: int) -> bool:
    """Validate that returned hidden_states preserve the input batch dimension."""
    hs = getattr(out, "hidden_states", None)
    if hs is None or len(hs) == 0:
        return False
    last = hs[-1]
    if not isinstance(last, torch.Tensor):
        return False
    return last.dim() == 3 and last.shape[0] == batch_size


def _setup_hooks_if_needed(model, tokenizer, device, is_decoder):
    """Test if model returns hidden_states; if not, set up forward hooks."""
    import re
    hooked_states: Dict[int, Tensor] = {}
    hooks = []
    use_hooks = False

    try:
        # Use a multi-item batch here: some trust_remote_code models incorrectly
        # return hidden_states with batch size hardcoded to 1.
        test_enc = tokenizer(["test", "another test"], max_length=32, padding=True,
                             truncation=True, return_tensors="pt")
        test_enc = {k: v.to(device) for k, v in test_enc.items()}
        with torch.inference_mode():
            test_out = _forward_with_hidden_states(model, test_enc, is_decoder)
        if not _hidden_states_batch_is_valid(test_out, batch_size=test_enc["input_ids"].shape[0]):
            use_hooks = True
        del test_out, test_enc
    except Exception:
        use_hooks = True

    if use_hooks:
        print("  [INFO] Using hook-based hidden state extraction (GFMI)")
        embeddings_module = None
        for name, mod in model.named_modules():
            if ("embed" in name and not any(x in name for x in
                    ["word", "position", "token_type", "LayerNorm", "dropout", "self_attn", "mlp"])) \
               or (name == "embed_tokens"):
                embeddings_module = mod
                break
        _layer_re = re.compile(r'^(?:.*\.)?(layers?|h|block)\.\d+$')
        encoder_layers = []
        for name, mod in model.named_modules():
            if _layer_re.match(name) and hasattr(mod, "forward"):
                encoder_layers.append((name, mod))

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hooked_states[layer_idx] = output[0].detach()
                else:
                    hooked_states[layer_idx] = output.detach()
            return hook_fn

        if embeddings_module:
            hooks.append(embeddings_module.register_forward_hook(make_hook(0)))
        for idx, (name, mod) in enumerate(encoder_layers):
            hooks.append(mod.register_forward_hook(make_hook(idx + 1)))

    return use_hooks, hooked_states, hooks


def _run_one_pass(model, tokenizer, texts, target_layers: Set[int],
                  batch_size, max_length, device, pooling, normalize, is_decoder,
                  _hook_ctx=None):
    layer_chunks: Dict[int, List[np.ndarray]] = {i: [] for i in target_layers}

    use_hooks = False
    hooked_states: Dict[int, Tensor] = {}
    hooks = []
    if _hook_ctx is not None:
        use_hooks, hooked_states, hooks = _hook_ctx

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size),
                          desc=f"  fwd layers {min(target_layers)}-{max(target_layers)}",
                          leave=False):
            batch = texts[start:start + batch_size]
            if is_decoder:
                tokenizer.padding_side = "left"
            batch = [t if t.strip() else " " for t in batch]
            enc = tokenizer(
                batch, max_length=max_length, padding=True,
                truncation=True, return_tensors="pt",
            )
            if enc["input_ids"].numel() == 0:
                continue
            enc = {k: v.to(device) for k, v in enc.items()}
            hooked_states.clear()

            if use_hooks:
                fwd_kwargs: Dict[str, Any] = {**enc}
                if is_decoder:
                    fwd_kwargs["use_cache"] = False
                outputs = model(**fwd_kwargs)
            else:
                outputs = _forward_with_hidden_states(model, enc, is_decoder)

            mask = enc["attention_mask"]
            for i in target_layers:
                if use_hooks:
                    if i not in hooked_states:
                        continue
                    hs = hooked_states[i].float()
                else:
                    hs = outputs.hidden_states[i].float()
                hs = _reshape_hidden_to_mask_layout(hs, mask)
                hs, mask_aligned = _align_hidden_and_mask(hs, mask)
                if pooling == "last_token":
                    pooled = last_token_pool(hs, mask_aligned)
                elif pooling == "cls":
                    pooled = cls_pool(hs, mask_aligned)
                else:
                    pooled = mean_pooling(hs, mask_aligned)
                if normalize:
                    pooled = F.normalize(pooled, p=2, dim=1)
                layer_chunks[i].append(pooled.cpu().numpy())
            del outputs, enc
            hooked_states.clear()
            if start % (batch_size * 20) == 0:
                gc.collect(); torch.cuda.empty_cache()

    result = {}
    for i in target_layers:
        result[i] = np.vstack(layer_chunks[i]).astype(np.float32)
    return result


# ──────────────────────────────────────────────────────────────────────
# Vectorized GFMI
# ──────────────────────────────────────────────────────────────────────

def _sanitize(Z: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf rows with zero vectors + small noise."""
    bad = ~np.isfinite(Z).all(axis=1)
    if bad.any():
        n_bad = int(bad.sum())
        print(f"    WARNING: {n_bad} NaN/Inf rows → replaced with noise", flush=True)
        rng = np.random.RandomState(0)
        Z[bad] = rng.randn(n_bad, Z.shape[1]).astype(Z.dtype) * 1e-6
    return Z


def knn_graph_data(Z: np.ndarray, k: int = 30):
    """Compute kNN and return distances, indices, flat distances."""
    Z = _sanitize(Z)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute")
    nn.fit(Z)
    dists, idxs = nn.kneighbors(Z)
    return dists[:, 1:], idxs[:, 1:]


def _cc_labels_at_scales(dists, idxs, percentiles_of_flat, n, k):
    """
    Precompute connected-component labels at each percentile scale.
    Returns list of label arrays, one per scale.
    Uses vectorized graph construction.
    """
    flat_dists = dists.ravel()
    thresholds = np.percentile(flat_dists, percentiles_of_flat)

    n_eff, k_eff = dists.shape
    rows = np.repeat(np.arange(n_eff), k_eff)
    cols = idxs.ravel()

    labels_per_scale = []
    for thresh in thresholds:
        mask = flat_dists <= thresh
        g = csr_matrix(
            (np.ones(mask.sum(), dtype=np.int8),
             (rows[mask], cols[mask])),
            shape=(n_eff, n_eff),
        )
        g = g + g.T
        _, labels = connected_components(g, directed=False)
        labels_per_scale.append(labels)

    return labels_per_scale, thresholds


def gfmi_auc(labels_layer: list, labels_last: list, percentiles: np.ndarray):
    """Compute MI at each scale and return AUC + per-scale MI."""
    mi_values = np.array([
        mutual_info_score(la, lb)
        for la, lb in zip(labels_layer, labels_last)
    ])
    auc = float(np.trapz(mi_values, percentiles))
    peak_mi = float(mi_values.max())
    peak_idx = int(mi_values.argmax())
    peak_pct = float(percentiles[peak_idx])
    return dict(auc=auc, peak_mi=peak_mi, peak_pct=peak_pct, mi_curve=mi_values.tolist())


# ──────────────────────────────────────────────────────────────────────
# Correlation helpers
# ──────────────────────────────────────────────────────────────────────

def residualize(x, z):
    z = np.asarray(z, dtype=float)
    x = np.asarray(x, dtype=float)
    zc = z - z.mean()
    d = (zc ** 2).sum()
    if d < 1e-12:
        return x - x.mean()
    b = (x * zc).sum() / d
    return x - b * z - (x.mean() - b * z.mean())


def sig(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.1: return "*"
    return ""


# ──────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────

def load_texts(ds_key: str, max_n: int) -> List[str]:
    from datasets import load_dataset
    cfg = DATASET_CONFIGS[ds_key]
    mode = cfg.get("load_mode", "standard")
    print(f"Loading dataset: {cfg['hf']}  config={cfg.get('config')}  "
          f"split={cfg['split']}  mode={mode}")

    try:
        if cfg.get("config"):
            ds = load_dataset(cfg["hf"], cfg["config"], split=cfg["split"])
        else:
            ds = load_dataset(cfg["hf"], split=cfg["split"])
    except ValueError as e:
        # Some MMTEB pair-classification datasets store list-valued features
        # that older `datasets` versions cannot deserialize from cached metadata.
        if mode != "pair_cls" or "Feature type 'List' not found" not in str(e):
            raise
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=cfg["hf"],
            repo_type="dataset",
            filename=f"{cfg['split']}.json.gz",
        )
        with gzip.open(path, "rt", encoding="utf-8") as f:
            ds = json.load(f)

    if mode == "standard":
        texts = [str(x) for x in ds[cfg["text_col"]]]

    elif mode == "sts_pairs":
        col_a, col_b = cfg["col_a"], cfg["col_b"]
        seen = set()
        texts = []
        for row in ds:
            for s in (str(row[col_a]), str(row[col_b])):
                if s not in seen:
                    seen.add(s)
                    texts.append(s)

    elif mode == "reranking":
        seen = set()
        texts = []
        for row in ds:
            q = str(row["query"])
            if q not in seen:
                seen.add(q)
                texts.append(q)
            for lst_key in ("positive", "negative"):
                if lst_key in row and row[lst_key]:
                    for s in row[lst_key]:
                        s = str(s)
                        if s not in seen:
                            seen.add(s)
                            texts.append(s)

    elif mode == "pair_cls":
        col_a, col_b = cfg["col_a"], cfg["col_b"]
        seen = set()
        texts = []
        if isinstance(ds, dict):
            a_seq = ds[col_a]
            b_seq = ds[col_b]
            for s in list(a_seq) + list(b_seq):
                s = str(s)
                if s not in seen:
                    seen.add(s)
                    texts.append(s)
        else:
            for row in ds:
                a_list = row[col_a] if isinstance(row[col_a], list) else [row[col_a]]
                b_list = row[col_b] if isinstance(row[col_b], list) else [row[col_b]]
                for s in a_list + b_list:
                    s = str(s)
                    if s not in seen:
                        seen.add(s)
                        texts.append(s)

    elif mode == "clustering":
        seen = set()
        texts = []
        scol = cfg.get("sentences_col", "sentences")
        for row in ds:
            for s in row[scol]:
                s = str(s)
                if s not in seen:
                    seen.add(s)
                    texts.append(s)
    else:
        raise ValueError(f"Unknown load_mode: {mode}")

    if len(texts) > max_n:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(texts), max_n, replace=False)
        idx.sort()
        texts = [texts[i] for i in idx]
    print(f"  {len(texts)} unique texts loaded (max_n={max_n})")
    return texts


# ──────────────────────────────────────────────────────────────────────
# Main pipeline for one (dataset, model)
# ──────────────────────────────────────────────────────────────────────

def compute_gfmi_profile(
    texts: List[str],
    hf_model: str,
    device: str = "cuda",
    batch_size_small: int = 64,
    batch_size_large: int = 4,
    max_length: int = 512,
    k: int = K_DEFAULT,
    n_scales: int = N_SCALES,
    layers_per_pass: int = LAYERS_PER_PASS,
) -> dict:
    """
    Extract all-layer embeddings, compute GFMI(l_i, last) for every i.
    Returns dict with profile data.
    """
    force_bidir = globals().get("_FORCE_BIDIR", False)
    model, tokenizer, info = _load_model(hf_model, device, force_bidir=force_bidir)
    n_layers = info["n_layers"]
    is_decoder = info["is_decoder"]
    pooling = get_pooling_strategy(hf_model, is_decoder)
    normalize = True
    batch_size = batch_size_large if is_decoder else batch_size_small
    if info["hidden_size"] >= 4096 and not is_decoder:
        batch_size = min(batch_size, batch_size_large)

    print(f"  arch={info['arch']}  layers={n_layers}  "
          f"hidden={info['hidden_size']}  pooling={pooling}  bs={batch_size}")

    percentiles = np.linspace(5, 95, n_scales)
    n = len(texts)

    # ── Hook setup (for models that don't return hidden_states) ──
    hook_ctx = _setup_hooks_if_needed(model, tokenizer, device, is_decoder)

    # ── Step 1: extract last layer, build kNN + precompute CC labels ──
    last_idx = n_layers  # hidden_states[n_layers] == last hidden state
    print(f"  [1/3] Extracting last layer (idx={last_idx}) ...")
    last_data = _run_one_pass(
        model, tokenizer, texts, {last_idx},
        batch_size, max_length, device, pooling, normalize, is_decoder,
        _hook_ctx=hook_ctx,
    )
    Z_last = last_data[last_idx]
    del last_data; gc.collect()

    print(f"  [2/3] Building last-layer kNN (k={k}) ...")
    dists_last, idxs_last = knn_graph_data(Z_last, k=k)
    labels_last_per_scale, _ = _cc_labels_at_scales(
        dists_last, idxs_last, percentiles, n, k)
    del dists_last, idxs_last
    gc.collect()

    # ── Step 2: extract other layers in groups, compute GFMI ──
    other_layers = list(range(n_layers))  # 0 .. n_layers-1
    groups = [other_layers[i:i + layers_per_pass]
              for i in range(0, len(other_layers), layers_per_pass)]

    profile = {}  # layer_idx -> gfmi result dict

    print(f"  [3/3] Processing {len(other_layers)} layers in {len(groups)} pass(es) ...")
    for gi, group in enumerate(groups):
        t0 = time.time()
        print(f"    Pass {gi+1}/{len(groups)}: layers {group[0]}-{group[-1]}", flush=True)
        layer_data = _run_one_pass(
            model, tokenizer, texts, set(group),
            batch_size, max_length, device, pooling, normalize, is_decoder,
            _hook_ctx=hook_ctx,
        )

        for li in group:
            Z_li = layer_data[li]
            dists_li, idxs_li = knn_graph_data(Z_li, k=k)
            labels_li_per_scale, _ = _cc_labels_at_scales(
                dists_li, idxs_li, percentiles, n, k)
            del dists_li, idxs_li, Z_li

            result = gfmi_auc(labels_li_per_scale, labels_last_per_scale, percentiles)
            profile[li] = result
            del labels_li_per_scale

            print(f"      layer {li:3d}  AUC={result['auc']:8.2f}  "
                  f"peak_MI={result['peak_mi']:.4f}  peak@{result['peak_pct']:.0f}%",
                  flush=True)

        del layer_data
        gc.collect(); torch.cuda.empty_cache()
        print(f"    Pass {gi+1} done in {time.time()-t0:.0f}s", flush=True)

    # last layer vs itself
    self_mi = gfmi_auc(labels_last_per_scale, labels_last_per_scale, percentiles)
    profile[last_idx] = self_mi
    print(f"      layer {last_idx:3d} (self)  AUC={self_mi['auc']:8.2f}", flush=True)

    del labels_last_per_scale, Z_last
    for h in hook_ctx[2]:
        h.remove()
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()

    return dict(
        hf_model=hf_model,
        n_layers=n_layers,
        hidden_size=info["hidden_size"],
        arch=info["arch"],
        n_samples=n,
        k=k,
        n_scales=n_scales,
        profile={int(k): v for k, v in sorted(profile.items())},
    )


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_gfmi_profiles(all_results: dict, ds_key: str, save_dir: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, (short_name, data) in enumerate(all_results.items()):
        prof = data["profile"]
        layers = sorted(int(k) for k in prof.keys())
        aucs = [prof[str(l)]["auc"] if str(l) in prof else prof[l]["auc"]
                for l in layers]
        n_layers = data["n_layers"]
        rel = [l / n_layers for l in layers]
        ax.plot(rel, aucs,
                label=f"{short_name} ({n_layers}L, d={data['hidden_size']})",
                color=colors[idx], marker="o", markersize=3, linewidth=1.5)

    ax.set_xlabel("Relative layer depth (0=embedding, 1=last)")
    ax.set_ylabel("GFMI-AUC (l_i vs last)")
    ax.set_title(f"GFMI Profile — {ds_key}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"gfmi_profile_{ds_key}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {path}")


# ──────────────────────────────────────────────────────────────────────
# Correlation analysis
# ──────────────────────────────────────────────────────────────────────

def analyse_correlations(all_results: dict, ds_key: str):
    """
    For selected relative depths (0%, 25%, 50%, 75%, 95%, 100%),
    compute GFMI-AUC correlation with MTEB.
    """
    mteb = MTEB_SCORES.get(ds_key, {})
    if not mteb:
        print(f"  No MTEB scores for {ds_key}")
        return

    rel_targets = [0.0, 0.25, 0.50, 0.75, 0.95]

    models_ordered = []
    mteb_vals = []
    dims = []
    auc_at_rel: Dict[float, list] = {r: [] for r in rel_targets}

    for short_name, data in all_results.items():
        if short_name not in mteb:
            continue
        prof = data["profile"]
        n_layers = data["n_layers"]

        models_ordered.append(short_name)
        mteb_vals.append(mteb[short_name])
        dims.append(data["hidden_size"])

        for rt in rel_targets:
            target_layer = int(round(rt * n_layers))
            target_layer = min(target_layer, n_layers)
            key = str(target_layer) if str(target_layer) in prof else target_layer
            if key in prof:
                auc_at_rel[rt].append(prof[key]["auc"])
            else:
                auc_at_rel[rt].append(float("nan"))

    if len(models_ordered) < 4:
        print(f"  Too few models ({len(models_ordered)}) for correlation")
        return

    mteb_arr = np.array(mteb_vals)
    log_dims = np.log(np.array(dims, dtype=float))

    print(f"\n  Correlation with MTEB ({ds_key}, n={len(models_ordered)}):")
    print(f"  {'RelDepth':>10s}  {'Spearman':>10s} {'p':>7s}   "
          f"{'Pearson':>10s} {'p':>7s}   {'Partial_r':>10s} {'p':>7s}")

    for rt in rel_targets:
        vals = np.array(auc_at_rel[rt])
        if np.any(np.isnan(vals)):
            print(f"  {rt:10.2f}  NaN")
            continue
        rs, ps = spearmanr(vals, mteb_arr)
        rp, pp = pearsonr(vals, mteb_arr)
        vr = residualize(vals, log_dims)
        mr = residualize(mteb_arr, log_dims)
        pr, prp = pearsonr(vr, mr)
        print(f"  {rt:10.2f}  {rs:+.4f}  {ps:.4f}{sig(ps):3s}  "
              f"{rp:+.4f}  {pp:.4f}{sig(pp):3s}  "
              f"{pr:+.4f}  {prp:.4f}{sig(prp):3s}")

    # log(dim) baseline
    rs, ps = spearmanr(log_dims, mteb_arr)
    rp, pp = pearsonr(log_dims, mteb_arr)
    print(f"  {'log(dim)':>10s}  {rs:+.4f}  {ps:.4f}{sig(ps):3s}  "
          f"{rp:+.4f}  {pp:.4f}{sig(pp):3s}  {'---':>10s}")

    # Also check AUC slope (last - first) as a shape feature
    auc_first = np.array(auc_at_rel[0.0])
    auc_last = np.array(auc_at_rel[0.95])
    if not (np.any(np.isnan(auc_first)) or np.any(np.isnan(auc_last))):
        slope = auc_last - auc_first
        rs, ps = spearmanr(slope, mteb_arr)
        rp, pp = pearsonr(slope, mteb_arr)
        vr = residualize(slope, log_dims)
        mr = residualize(mteb_arr, log_dims)
        pr, prp = pearsonr(vr, mr)
        print(f"  {'AUC_slope':>10s}  {rs:+.4f}  {ps:.4f}{sig(ps):3s}  "
              f"{rp:+.4f}  {pp:.4f}{sig(pp):3s}  "
              f"{pr:+.4f}  {prp:.4f}{sig(prp):3s}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, default="banking77,arguana",
                        help="Comma-separated dataset key fragments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--scales", type=int, default=N_SCALES)
    parser.add_argument("--max_n", type=int, default=MAX_N)
    parser.add_argument("--layers_per_pass", type=int, default=LAYERS_PER_PASS)
    parser.add_argument("--output_dir", default="./outputs/gfmi_profile")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model short-name fragments (e.g. MiniLM,e5)")
    parser.add_argument(
        "--model_source", type=str, default="embed",
        choices=["embed", "llm", "bidir"],
        help="embed = embedding models; llm = pure LLM; bidir = bidirectional LLM embeddings",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    global _FORCE_BIDIR
    _FORCE_BIDIR = (args.model_source == "bidir")

    if args.model_source == "llm":
        from llm_models import LLM_MODELS, llm_benchmark_scores_for
        _model_pool = LLM_MODELS
        global MTEB_SCORES
        MTEB_SCORES = llm_benchmark_scores_for(DATASET_CONFIGS.keys())
    elif args.model_source == "bidir":
        from bidir_models import BIDIR_MODELS
        _model_pool = BIDIR_MODELS
        MTEB_SCORES = {}
    else:
        _model_pool = MODELS

    tags = [t.strip() for t in args.ds.split(",") if t.strip()]
    ds_list = [d for d in sorted(DATASET_CONFIGS.keys())
               if any(tag in d for tag in tags)]

    model_list = _model_pool
    if args.models:
        mtags = [t.strip() for t in args.models.split(",") if t.strip()]
        model_list = [(hf, sn) for hf, sn in _model_pool
                      if any(mt in sn for mt in mtags)]

    print(f"GFMI All-Layers Profile")
    print(f"  Datasets: {ds_list}")
    print(f"  Models: {[sn for _, sn in model_list]}")
    print(f"  k={args.k}, scales={args.scales}, max_n={args.max_n}, "
          f"layers_per_pass={args.layers_per_pass}\n")

    for ds_key in ds_list:
        print(f"\n{'#'*80}")
        print(f"#  Dataset: {ds_key}  [{DATASET_CONFIGS[ds_key]['task']}]")
        print(f"{'#'*80}\n")

        texts = load_texts(ds_key, args.max_n)
        ds_dir = os.path.join(args.output_dir, ds_key)
        os.makedirs(ds_dir, exist_ok=True)

        all_results = {}

        # Load existing results to allow incremental runs
        combined_path = os.path.join(ds_dir, "all_results.json")
        if os.path.exists(combined_path):
            with open(combined_path) as f:
                all_results = json.load(f)
            print(f"  Loaded {len(all_results)} existing results from {combined_path}")

        for hf_model, short_name in model_list:
            if short_name in all_results:
                print(f"\n  SKIP {short_name} (already computed)")
                continue

            print(f"\n{'='*70}")
            print(f"  Model: {short_name}  ({hf_model})")
            print(f"{'='*70}")

            t0 = time.time()
            try:
                result = compute_gfmi_profile(
                    texts=texts,
                    hf_model=hf_model,
                    device=args.device,
                    k=args.k,
                    n_scales=args.scales,
                    layers_per_pass=args.layers_per_pass,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                continue

            elapsed = time.time() - t0
            print(f"  Completed in {elapsed:.0f}s")

            all_results[short_name] = result

            with open(combined_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"  Saved: {combined_path}")

        # ── Plot + Correlations ──
        if len(all_results) >= 2:
            plot_gfmi_profiles(all_results, ds_key, ds_dir)
            analyse_correlations(all_results, ds_key)


if __name__ == "__main__":
    main()
