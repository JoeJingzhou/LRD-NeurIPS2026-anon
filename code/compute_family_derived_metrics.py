#!/usr/bin/env python3
"""Aggregate clean31 layer-wise measurements by architecture family.

The script reads the formal30 raw Frenet, NRS, and GFMI outputs and writes
family-level summaries used by Section 4.3 and the appendix.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(".")
OUT_DIR = ROOT / "outputs" / "paper_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORMAL_TASKS = [
    p.name
    for p in sorted((ROOT / "outputs" / "frenet_v2_95var").glob("mteb_*"))
    if p.name != "mteb_TwentyNewsgroupsClustering"
]

ENCODER_EMBEDDERS = [
    "all_MiniLM_L6_v2",
    "all_mpnet_base_v2",
    "bge_base_en_v1_5",
    "bge_large_en_v1_5",
    "e5_base_v2",
    "e5_large_v2",
    "gte_base",
    "gte_large",
    "jina_embeddings_v3",
    "mxbai_embed_large_v1",
    "nomic_embed_text_v1_5",
    "snowflake_arctic_embed_l",
    "snowflake_arctic_embed_m_v2",
]

DECODER_EMBEDDERS = [
    "GritLM_7B",
    "Linq_Embed_Mistral",
    "Qwen3_Embedding_0_6B",
    "Qwen3_Embedding_8B",
    "SFR_Embedding_2_R",
    "SFR_Embedding_Mistral",
    "Zeta_Alpha_E5_Mistral",
    "bge_en_icl",
    "bge_multilingual_gemma2",
    "e5_mistral_7b_instruct",
    "gte_Qwen2_7B_instruct",
    "NV_Embed_v2",
]

BASE_LLMS = [
    "Llama_3_2_1B",
    "Mistral_7B_v0_1",
    "falcon_7b",
    "gemma_2_2b",
    "gpt2",
    "gpt2_xl",
]

MODEL_FAMILY: Dict[str, str] = {
    **{m: "Encoder embedder" for m in ENCODER_EMBEDDERS},
    **{m: "Decoder embedder" for m in DECODER_EMBEDDERS},
    **{m: "Base LLM" for m in BASE_LLMS},
}

BASES = {
    "frenet": {
        "Encoder embedder": "frenet_v2_95var",
        "Decoder embedder": "frenet_v2_95var",
        "Base LLM": "llm_frenet_v2_95var",
    },
    "nrs": {
        "Encoder embedder": "nrs_profile",
        "Decoder embedder": "nrs_profile",
        "Base LLM": "llm_nrs_profile",
    },
    "gfmi": {
        "Encoder embedder": "gfmi_profile",
        "Decoder embedder": "gfmi_profile",
        "Base LLM": "llm_gfmi_profile",
    },
}

# NV-Embed-v2 is stored in the bidirectional output directories.
BIDIR_BASES = {
    "frenet": "bidir_frenet_v2_95var",
    "nrs": "bidir_nrs_profile",
    "gfmi": "bidir_gfmi_profile",
}

GRID = [i / 100.0 for i in range(101)]


def finite(x) -> Optional[float]:
    if x is None:
        return None
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_result(metric: str, dataset: str, model: str):
    family = MODEL_FAMILY[model]
    base = BIDIR_BASES[metric] if model == "NV_Embed_v2" else BASES[metric][family]
    ds_dir = ROOT / "outputs" / base / dataset
    direct = ds_dir / f"{model}.json"
    if direct.exists():
        return read_json(direct)
    merged = ds_dir / "all_results.json"
    data = read_json(merged)
    if isinstance(data, dict):
        return data.get(model)
    return None


def normalized_peak_depth(vals: Sequence[float]) -> Optional[float]:
    clean = [finite(v) for v in vals]
    clean = [v for v in clean if v is not None]
    if not clean:
        return None
    idx = max(range(len(clean)), key=lambda i: clean[i])
    denom = max(1, len(clean) - 1)
    return idx / denom


def thirds(vals: Sequence[float]) -> Tuple[List[float], List[float]]:
    n = len(vals)
    if n == 0:
        return [], []
    width = max(1, n // 3)
    return list(vals[:width]), list(vals[-width:])


def interp(vals: Sequence[float], grid: Sequence[float] = GRID) -> List[float]:
    clean = [finite(v) for v in vals]
    clean = [v for v in clean if v is not None]
    if not clean:
        return [float("nan") for _ in grid]
    if len(clean) == 1:
        return [clean[0] for _ in grid]
    out = []
    n = len(clean)
    for g in grid:
        pos = g * (n - 1)
        lo = int(math.floor(pos))
        hi = min(n - 1, lo + 1)
        w = pos - lo
        out.append(clean[lo] * (1.0 - w) + clean[hi] * w)
    return out


def ranks(vals: Sequence[float]) -> List[float]:
    pairs = sorted((v, i) for i, v in enumerate(vals))
    out = [0.0] * len(vals)
    j = 0
    while j < len(pairs):
        k = j + 1
        while k < len(pairs) and pairs[k][0] == pairs[j][0]:
            k += 1
        rank = (j + k - 1) / 2.0 + 1.0
        for _, idx in pairs[j:k]:
            out[idx] = rank
        j = k
    return out


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = math.sqrt(sum(x * x for x in dx))
    sy = math.sqrt(sum(y * y for y in dy))
    if sx == 0 or sy == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (sx * sy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(finite(x), finite(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    x, y = zip(*pairs)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return pearson(ranks(x), ranks(y))


def gfmi_auc_sequence(obj) -> List[float]:
    profile = (obj or {}).get("profile") or {}
    rows = []
    for key, row in profile.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        val = finite(row.get("auc"))
        if val is not None:
            rows.append((idx, val))
    return [v for _, v in sorted(rows)]


def gfmi_peak_mi_sequence(obj) -> List[float]:
    profile = (obj or {}).get("profile") or {}
    rows = []
    for key, row in profile.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        val = finite(row.get("peak_mi"))
        if val is not None:
            rows.append((idx, val))
    return [v for _, v in sorted(rows)]


def layer_range(vals: Iterable[int]) -> str:
    xs = sorted(set(int(v) for v in vals if v))
    if not xs:
        return ""
    return str(xs[0]) if xs[0] == xs[-1] else f"{xs[0]}-{xs[-1]}"


def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or not math.isfinite(x):
        return ""
    return f"{x:.{nd}f}"


def main() -> None:
    rows = []
    profiles = []
    aligned_values = {"s_J": ([], []), "s_G": ([], []), "J_G": ([], [])}

    for model, family in sorted(MODEL_FAMILY.items()):
        for dataset in FORMAL_TASKS:
            fr = load_result("frenet", dataset, model)
            nr = load_result("nrs", dataset, model)
            gf = load_result("gfmi", dataset, model)
            if not (fr and nr and gf):
                continue

            speed = [v for v in (finite(x) for x in fr.get("speed", [])) if v is not None]
            curvature = [
                v for v in (finite(x) for x in fr.get("curvature", [])) if v is not None
            ]
            jaccard = [
                v for v in (finite(x) for x in nr.get("jaccard_profile", [])) if v is not None
            ]
            gfmi_auc = gfmi_auc_sequence(gf)
            gfmi_peak = gfmi_peak_mi_sequence(gf)

            if not (speed and jaccard and gfmi_auc):
                continue

            early_j, late_j = thirds(jaccard)
            j_trend = mean(late_j) - mean(early_j) if early_j and late_j else float("nan")
            speed_peak = normalized_peak_depth(speed)
            curvature_peak = normalized_peak_depth(curvature)
            gfmi_peak_depth = normalized_peak_depth(gfmi_auc)

            rows.append(
                {
                    "family": family,
                    "model": model,
                    "dataset": dataset,
                    "n_layers": int(fr.get("n_layers") or len(speed)),
                    "speed_peak_depth": speed_peak,
                    "curvature_peak_depth": curvature_peak,
                    "nrs_delta_j": j_trend,
                    "gfmi_peak_depth": gfmi_peak_depth,
                    "mean_speed": mean(speed),
                    "mean_jaccard": mean(jaccard),
                    "mean_gfmi_auc": mean(gfmi_auc),
                    "mean_peak_mi": mean(gfmi_peak) if gfmi_peak else float("nan"),
                }
            )

            for metric, vals in [
                ("speed", speed),
                ("jaccard", jaccard),
                ("gfmi_auc", gfmi_auc),
            ]:
                for depth, value in zip(GRID, interp(vals)):
                    profiles.append(
                        {
                            "family": family,
                            "model": model,
                            "dataset": dataset,
                            "metric": metric,
                            "depth": depth,
                            "value": value,
                        }
                    )

            m = min(len(speed), len(jaccard), len(gfmi_auc))
            if m >= 3:
                s = speed[:m]
                j = jaccard[:m]
                g = gfmi_auc[:m]
                aligned_values["s_J"][0].extend(s)
                aligned_values["s_J"][1].extend(j)
                aligned_values["s_G"][0].extend(s)
                aligned_values["s_G"][1].extend(g)
                aligned_values["J_G"][0].extend(j)
                aligned_values["J_G"][1].extend(g)

    family_rows = []
    for family in ["Encoder embedder", "Decoder embedder", "Base LLM"]:
        fam = [r for r in rows if r["family"] == family]
        if not fam:
            continue
        layers = [r["n_layers"] for r in fam]
        delta = [r["nrs_delta_j"] for r in fam if math.isfinite(r["nrs_delta_j"])]
        family_rows.append(
            {
                "family": family,
                "n_model_task": len(fam),
                "n_models": len(set(r["model"] for r in fam)),
                "layer_range": layer_range(layers),
                "speed_peak_depth_mean": mean(
                    [r["speed_peak_depth"] for r in fam if r["speed_peak_depth"] is not None]
                ),
                "speed_peak_depth_median": median(
                    [r["speed_peak_depth"] for r in fam if r["speed_peak_depth"] is not None]
                ),
                "curvature_peak_depth_mean": mean(
                    [
                        r["curvature_peak_depth"]
                        for r in fam
                        if r["curvature_peak_depth"] is not None
                    ]
                ),
                "gfmi_peak_depth_mean": mean(
                    [r["gfmi_peak_depth"] for r in fam if r["gfmi_peak_depth"] is not None]
                ),
                "gfmi_peak_depth_median": median(
                    [r["gfmi_peak_depth"] for r in fam if r["gfmi_peak_depth"] is not None]
                ),
                "nrs_delta_j_mean": mean(delta),
                "nrs_delta_j_median": median(delta),
                "nrs_trend_sign": "Increasing" if mean(delta) > 0 else "Decreasing",
                "mean_speed": mean([r["mean_speed"] for r in fam]),
                "mean_jaccard": mean([r["mean_jaccard"] for r in fam]),
                "mean_gfmi_auc": mean([r["mean_gfmi_auc"] for r in fam]),
                "mean_peak_mi": mean([r["mean_peak_mi"] for r in fam]),
            }
        )

    # Average interpolated profiles by family and metric.
    grouped: Dict[Tuple[str, str, float], List[float]] = {}
    for r in profiles:
        grouped.setdefault((r["family"], r["metric"], r["depth"]), []).append(r["value"])
    profile_rows = []
    for (family, metric, depth), vals in sorted(grouped.items()):
        clean = [v for v in vals if math.isfinite(v)]
        profile_rows.append(
            {
                "family": family,
                "metric": metric,
                "depth": depth,
                "mean_value": mean(clean),
                "n": len(clean),
            }
        )

    pairwise_rows = []
    for name, (xs, ys) in aligned_values.items():
        rho = spearman(xs, ys)
        pairwise_rows.append(
            {
                "scope": "all_layer_points",
                "pair": name,
                "rho": rho,
                "abs_rho": abs(rho) if rho is not None else None,
                "n": len(xs),
            }
        )

    for family in ["Encoder embedder", "Decoder embedder", "Base LLM"]:
        fam_rows = [r for r in rows if r["family"] == family]
        for a, b, name in [
            ("mean_speed", "mean_jaccard", "mean_speed_mean_jaccard"),
            ("mean_speed", "mean_gfmi_auc", "mean_speed_mean_gfmi_auc"),
            ("mean_jaccard", "mean_gfmi_auc", "mean_jaccard_mean_gfmi_auc"),
        ]:
            xs = [r[a] for r in fam_rows]
            ys = [r[b] for r in fam_rows]
            rho = spearman(xs, ys)
            pairwise_rows.append(
                {
                    "scope": family,
                    "pair": name,
                    "rho": rho,
                    "abs_rho": abs(rho) if rho is not None else None,
                    "n": len(xs),
                }
            )

    # Family-profile similarity: Spearman correlation between mean profile curves.
    by_profile = {
        (r["family"], r["metric"]): [r["mean_value"] for r in profile_rows if r["family"] == r0 and r["metric"] == m]
        for r0 in ["Encoder embedder", "Decoder embedder", "Base LLM"]
        for m in ["speed", "jaccard", "gfmi_auc"]
        for r in [{"family": r0, "metric": m}]
    }
    families = ["Encoder embedder", "Decoder embedder", "Base LLM"]
    for metric in ["speed", "jaccard", "gfmi_auc"]:
        for i, fam_a in enumerate(families):
            for fam_b in families[i + 1 :]:
                xs = by_profile.get((fam_a, metric), [])
                ys = by_profile.get((fam_b, metric), [])
                rho = spearman(xs, ys)
                pairwise_rows.append(
                    {
                        "scope": f"{fam_a} vs {fam_b}",
                        "pair": f"family_profile_{metric}",
                        "rho": rho,
                        "abs_rho": abs(rho) if rho is not None else None,
                        "n": len(xs),
                    }
                )

    detail_path = OUT_DIR / "formal30_clean31_family_model_task_metrics.csv"
    with detail_path.open("w", newline="") as f:
        fields = [
            "family",
            "model",
            "dataset",
            "n_layers",
            "speed_peak_depth",
            "curvature_peak_depth",
            "nrs_delta_j",
            "gfmi_peak_depth",
            "mean_speed",
            "mean_jaccard",
            "mean_gfmi_auc",
            "mean_peak_mi",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    family_path = OUT_DIR / "formal30_clean31_family_profiles.csv"
    with family_path.open("w", newline="") as f:
        fields = [
            "family",
            "n_model_task",
            "n_models",
            "layer_range",
            "speed_peak_depth_mean",
            "speed_peak_depth_median",
            "curvature_peak_depth_mean",
            "gfmi_peak_depth_mean",
            "gfmi_peak_depth_median",
            "nrs_delta_j_mean",
            "nrs_delta_j_median",
            "nrs_trend_sign",
            "mean_speed",
            "mean_jaccard",
            "mean_gfmi_auc",
            "mean_peak_mi",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(family_rows)

    profile_path = OUT_DIR / "formal30_clean31_family_mean_profiles.csv"
    with profile_path.open("w", newline="") as f:
        fields = ["family", "metric", "depth", "mean_value", "n"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(profile_rows)

    rho_path = OUT_DIR / "formal30_clean31_pairwise_measurement_rho.csv"
    with rho_path.open("w", newline="") as f:
        fields = ["scope", "pair", "rho", "abs_rho", "n"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    print("family summaries")
    for r in family_rows:
        print(
            "{family}: n_models={n_models}, layers={layer_range}, "
            "speed_peak={speed_peak_depth_mean:.3f}, dJ={nrs_delta_j_mean:.3f}, "
            "gfmi_peak={gfmi_peak_depth_mean:.3f}, gfmi_auc={mean_gfmi_auc:.1f}".format(
                **r
            )
        )
    print("pairwise")
    for r in pairwise_rows:
        print(
            "{scope} {pair} rho={rho} n={n}".format(
                scope=r["scope"], pair=r["pair"], rho=fmt(r["rho"]), n=r["n"]
            )
        )


if __name__ == "__main__":
    main()
