#!/usr/bin/env python3
"""
Evaluate a pruned embedder with MTEB by skipping selected transformer blocks
at inference time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from pooling_utils import apply_pooling, get_pooling_strategy
from frenet_curvature import MODELS as EMBED_MODELS
from llm_models import LLM_MODELS
from bidir_models import BIDIR_MODELS


def _mteb_task_name(dataset_key: str) -> str:
    return dataset_key[len("mteb_") :] if dataset_key.startswith("mteb_") else dataset_key


def _model_map() -> Dict[str, str]:
    pairs = list(EMBED_MODELS) + list(LLM_MODELS) + list(BIDIR_MODELS)
    return {short: hf for hf, short in pairs}


def _get_layer_modules(model) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    if hasattr(model, "model") and hasattr(model.model, "encoder") and hasattr(model.model.encoder, "layer"):
        return model.model.encoder.layer
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot locate transformer layers for model type {type(model)}")


def _skip_returns_tuple(model) -> bool:
    """Match the contract of the layer stack actually being skipped.

    BERT/RoBERTa-style encoder stacks unpack `layer_outputs[0]`, so the skip
    path must return a tuple. Decoder-style stacks (including Mistral-backed
    embedding wrappers such as LLM2Vec) pass the hidden states directly to the
    next block, so the skip path must return a tensor instead.
    """
    layers = _get_layer_modules(model)
    if len(layers) == 0:
        return False
    layer0 = layers[0]
    module_path = type(layer0).__module__.lower()
    class_name = type(layer0).__name__.lower()
    if "bert" in module_path or "roberta" in module_path or "modernbert" in module_path:
        return True
    if "encoderlayer" in class_name and "mistral" not in module_path:
        return True
    return False


def _install_skip_layers(model, pruned_layers: Iterable[int]) -> None:
    layers = _get_layer_modules(model)
    pruned = set(int(x) for x in pruned_layers)
    return_tuple = _skip_returns_tuple(model)
    for idx in pruned:
        layer = layers[idx]

        def _skip_forward(hidden_states, *args, **kwargs):
            if return_tuple:
                return (hidden_states,)
            return hidden_states

        layer.forward = _skip_forward  # type: ignore[assignment]


class PrunedHFEncoder:
    def __init__(
        self,
        hf_model: str,
        short_name: str,
        pruned_layers: Sequence[int],
        device: str,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        from frenet_curvature import _load_model
        from mteb.model_meta import ModelMeta

        self.device = device
        self.hf_model = hf_model
        self.short_name = short_name
        self.pruned_layers = [int(x) for x in pruned_layers]
        self.batch_size = batch_size
        self.max_length = max_length

        self.model, self.tokenizer, self.info = _load_model(hf_model, device)
        _install_skip_layers(self.model, self.pruned_layers)

        self.is_decoder = bool(self.info["is_decoder"])
        self.pooling = get_pooling_strategy(hf_model, self.is_decoder, verbose=False)

        self.mteb_model_meta = ModelMeta(
            name=f"local/{short_name}_pruned",
            revision=None,
            release_date=None,
            languages=["eng-Latn"],
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=None,
            embed_dim=int(self.info["hidden_size"]),
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            reference=None,
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
        )

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type=None,
        **kwargs: Any,
    ) -> np.ndarray:
        if self.is_decoder:
            self.tokenizer.padding_side = "left"
        out_batches: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(sentences), self.batch_size):
                batch = [s if str(s).strip() else " " for s in sentences[start : start + self.batch_size]]
                enc = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                fwd_kwargs = {}
                if not _skip_returns_tuple(self.model):
                    fwd_kwargs["use_cache"] = False
                outputs = self.model(**enc, **fwd_kwargs)
                hs = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
                pooled = apply_pooling(self.pooling, hs.float(), enc["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                out_batches.append(pooled.cpu().numpy().astype(np.float32))
                del outputs, enc, hs, pooled
        return np.vstack(out_batches) if out_batches else np.zeros((0, int(self.info["hidden_size"])), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a pruned model on one MTEB task")
    ap.add_argument("--plan-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--output-dir", default="./outputs/pruning_results")
    ap.add_argument("--verbosity", type=int, default=1)
    args = ap.parse_args()

    plan = json.load(open(args.plan_json, "r"))
    model_short = plan["model"]
    dataset_key = plan["dataset"]
    hf_model = _model_map()[model_short]

    # MTEB is only available in the nf-mi / gritlm environments. Import here
    # so score-building scripts remain runnable in a plain system python.
    import mteb

    task_name = _mteb_task_name(dataset_key)
    encoder = PrunedHFEncoder(
        hf_model=hf_model,
        short_name=model_short,
        pruned_layers=plan["pruned_layers"],
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    output_dir = Path(args.output_dir) / dataset_key / model_short / Path(args.plan_json).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = mteb.MTEB(tasks=[task_name])
    evaluation.run(
        encoder,
        output_folder=str(output_dir),
        verbosity=args.verbosity,
        overwrite_results=True,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
