#!/usr/bin/env python3
"""
Build per-layer pruning guidance scores from existing Frenet / NRS / GFMI
outputs.

This script does not rerun geometry. It reuses saved JSON outputs and converts
them into layer-level guidance profiles for three independent pruning rules:

    frenet_l = -z(speed_l) - z(curvature_l)
    nrs_l    =  z(J_l)
    gfmi_l   =  z(gfmi_l)

It also keeps the earlier composite score for debugging:

    composite_l = frenet_l + nrs_l + gfmi_l

The resulting JSON is consumed by `select_pruned_layers.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(".")
FRENET_ROOTS = [
    ROOT / "outputs/frenet_v2_95var",
    ROOT / "outputs/llm_frenet_v2_95var",
    ROOT / "outputs/bidir_frenet_v2_95var",
]
NRS_ROOTS = [
    ROOT / "outputs/nrs_profile",
    ROOT / "outputs/llm_nrs_profile",
    ROOT / "outputs/bidir_nrs_profile",
]
GFMI_ROOTS = [
    ROOT / "outputs/gfmi_profile",
    ROOT / "outputs/llm_gfmi_profile",
    ROOT / "outputs/bidir_gfmi_profile",
]


def _load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _zscore(values: List[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    if std < 1e-12:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _ensure_len(seq: List[float], target_len: int, fill: float = 0.0) -> List[float]:
    seq = list(seq)
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [fill] * (target_len - len(seq))


def _edge_to_layer_profile(edge_values: List[float], n_layers: int) -> List[float]:
    """
    Convert transition-level values of length (n_layers - 1) to a layer-level
    profile of length n_layers by averaging adjacent transitions.
    """
    if n_layers <= 0:
        return []
    if not edge_values:
        return [0.0] * n_layers
    if len(edge_values) == 1:
        return [edge_values[0]] * n_layers

    out = []
    for i in range(n_layers):
        vals = []
        if i - 1 >= 0 and i - 1 < len(edge_values):
            vals.append(edge_values[i - 1])
        if i < len(edge_values):
            vals.append(edge_values[i])
        if vals:
            out.append(sum(vals) / len(vals))
        else:
            out.append(edge_values[-1])
    return out


def _center_to_layer_profile(center_values: List[float], n_layers: int) -> List[float]:
    """
    Convert center-of-triple values (typically curvature) of length
    (n_layers - 2) to a layer-level profile of length n_layers.

    Each center value at index j is treated as belonging primarily to layer
    j + 1, with light spillover to adjacent layers via local averaging.
    """
    if n_layers <= 0:
        return []
    if not center_values:
        return [0.0] * n_layers
    if n_layers <= 2:
        mean_val = sum(center_values) / len(center_values)
        return [mean_val] * n_layers

    out = []
    for layer in range(n_layers):
        vals = []
        for j, v in enumerate(center_values):
            center = j + 1
            if abs(center - layer) <= 1:
                vals.append(v)
        if vals:
            out.append(sum(vals) / len(vals))
        else:
            if layer == 0:
                out.append(center_values[0])
            elif layer == n_layers - 1:
                out.append(center_values[-1])
            else:
                nearest = min(len(center_values) - 1, max(0, layer - 1))
                out.append(center_values[nearest])
    return out


def _find_per_model_json(roots: Iterable[Path], dataset: str, model: str) -> Optional[Path]:
    for root in roots:
        candidate = root / dataset / f"{model}.json"
        if candidate.exists():
            return candidate
    return None


def _find_gfmi_entry(roots: Iterable[Path], dataset: str, model: str) -> Tuple[dict, Path]:
    for root in roots:
        ds_dir = root / dataset
        model_json = ds_dir / f"{model}.json"
        if model_json.exists():
            return _load_json(model_json), model_json
        all_results = ds_dir / "all_results.json"
        if all_results.exists():
            obj = _load_json(all_results)
            if model in obj:
                return obj[model], all_results
    raise FileNotFoundError(f"GFMI result not found for dataset={dataset}, model={model}")


def _extract_gfmi_layer_auc(entry: dict, n_layers: int) -> List[float]:
    profile = entry.get("profile")
    if isinstance(profile, dict):
        vals = []
        for idx in range(n_layers):
            layer_entry = profile.get(str(idx))
            if layer_entry is None:
                vals.append(0.0)
            else:
                vals.append(float(layer_entry.get("auc", 0.0)))
        return vals
    if isinstance(profile, list):
        vals = [float(x.get("auc", 0.0)) if isinstance(x, dict) else 0.0 for x in profile]
        return _ensure_len(vals, n_layers, fill=0.0)
    if "auc" in entry:
        return [float(entry["auc"])] * n_layers
    return [0.0] * n_layers


def build_scores(dataset: str, model: str) -> dict:
    fr_path = _find_per_model_json(FRENET_ROOTS, dataset, model)
    nr_path = _find_per_model_json(NRS_ROOTS, dataset, model)
    if fr_path is None:
        raise FileNotFoundError(f"Frenet result not found for dataset={dataset}, model={model}")
    if nr_path is None:
        raise FileNotFoundError(f"NRS result not found for dataset={dataset}, model={model}")

    fr = _load_json(fr_path)
    nr = _load_json(nr_path)
    gf, gf_path = _find_gfmi_entry(GFMI_ROOTS, dataset, model)

    layers = fr.get("layers") or nr.get("layers")
    if not layers:
        raise ValueError(f"Cannot infer layer list for dataset={dataset}, model={model}")
    layer_ids = [int(x) for x in layers]
    n_layers = len(layer_ids)

    speed_layer = _edge_to_layer_profile([float(x) for x in fr.get("speed", [])], n_layers)
    curvature_layer = _center_to_layer_profile([float(x) for x in fr.get("curvature", [])], n_layers)
    nrs_layer = _edge_to_layer_profile([float(x) for x in nr.get("jaccard_profile", [])], n_layers)
    gfmi_layer = _extract_gfmi_layer_auc(gf, n_layers)

    z_speed = _zscore(speed_layer)
    z_curvature = _zscore(curvature_layer)
    z_nrs = _zscore(nrs_layer)
    z_gfmi = _zscore(gfmi_layer)

    layers_out = []
    for i, layer in enumerate(layer_ids):
        frenet_score = -z_speed[i] - z_curvature[i]
        nrs_score = z_nrs[i]
        gfmi_score = z_gfmi[i]
        composite_score = frenet_score + nrs_score + gfmi_score
        layers_out.append(
            {
                "layer": layer,
                "speed": speed_layer[i],
                "curvature": curvature_layer[i],
                "nrs": nrs_layer[i],
                "gfmi": gfmi_layer[i],
                "z_speed": z_speed[i],
                "z_curvature": z_curvature[i],
                "z_nrs": z_nrs[i],
                "z_gfmi": z_gfmi[i],
                "frenet_score": frenet_score,
                "nrs_score": nrs_score,
                "gfmi_score": gfmi_score,
                "composite_score": composite_score,
            }
        )

    return {
        "model": model,
        "dataset": dataset,
        "n_layers": n_layers,
        "layer_ids": layer_ids,
        "sources": {
            "frenet": str(fr_path),
            "nrs": str(nr_path),
            "gfmi": str(gf_path),
            "nrs_signal": "jaccard_profile",
        },
        "layers": layers_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build pruning redundancy scores from existing geometry outputs")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/pruning_scores"))
    args = ap.parse_args()

    result = build_scores(args.dataset, args.model)
    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
