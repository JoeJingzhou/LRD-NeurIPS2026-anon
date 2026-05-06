#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from mteb import load_results
from scipy.stats import spearmanr


ROOT = Path(".")
OUT_DIR = ROOT / "outputs" / "paper_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "script"))
import frenet_curvature as fc  # noqa: E402
from llm_models import _MMLU  # noqa: E402


FORMAL_TASKS = [
    p.name
    for p in sorted((ROOT / "outputs" / "frenet_v2_95var").glob("mteb_*"))
    if p.name != "mteb_TwentyNewsgroupsClustering"
]

TASK_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "mteb_AmazonCounterfactualClassification": ("AmazonCounterfactualClassification", None),
    "mteb_AmazonPolarityClassification": ("AmazonPolarityClassification", None),
    "mteb_AskUbuntuDupQuestions": ("AskUbuntuDupQuestions", None),
    "mteb_CQADupstackRetrieval": ("CQADupstackRetrieval", None),
    "mteb_DBpediaClassification": ("DBpediaClassification", None),
    "mteb_EmotionClassification": ("EmotionClassification", None),
    "mteb_FiQA2018": ("FiQA2018", None),
    "mteb_HotpotQA": ("HotpotQA", None),
    "mteb_ImdbClassification": ("ImdbClassification", None),
    "mteb_MTOPDomainClassification_en": ("MTOPDomainClassification", "en"),
    "mteb_MTOPIntentClassification": ("MTOPIntentClassification", "en"),
    "mteb_MassiveIntentClassification_en": ("MassiveIntentClassification", "en"),
    "mteb_MassiveScenarioClassification_en": ("MassiveScenarioClassification", "en"),
    "mteb_MindSmallReranking": ("MindSmallReranking", None),
    "mteb_NFCorpus": ("NFCorpus", None),
    "mteb_NQ": ("NQ", None),
    "mteb_QuoraRetrieval": ("QuoraRetrieval", None),
    "mteb_SICK_R": ("SICK-R", None),
    "mteb_STSBenchmark": ("STSBenchmark", None),
    "mteb_SciFact": ("SciFact", None),
    "mteb_SprintDuplicateQuestions": ("SprintDuplicateQuestions", None),
    "mteb_TRECCOVID": ("TRECCOVID", None),
    "mteb_Touche2020": ("Touche2020", None),
    "mteb_ToxicConversationsClassification": ("ToxicConversationsClassification", None),
    "mteb_TweetSentimentExtractionClassification": (
        "TweetSentimentExtractionClassification",
        None,
    ),
    "mteb_TweetTopicSingleClassification": ("TweetTopicSingleClassification", None),
    "mteb_TwitterURLCorpus": ("TwitterURLCorpus", None),
    "mteb_arguana": ("ArguAna", None),
    "mteb_banking77": ("Banking77Classification", None),
    "mteb_scidocs": ("SCIDOCS", None),
}

EMBED_MODELS = [
    "GritLM_7B",
    "Linq_Embed_Mistral",
    "Qwen3_Embedding_0_6B",
    "Qwen3_Embedding_8B",
    "SFR_Embedding_2_R",
    "SFR_Embedding_Mistral",
    "Zeta_Alpha_E5_Mistral",
    "all_MiniLM_L6_v2",
    "all_mpnet_base_v2",
    "bge_base_en_v1_5",
    "bge_en_icl",
    "bge_large_en_v1_5",
    "bge_multilingual_gemma2",
    "e5_base_v2",
    "e5_large_v2",
    "e5_mistral_7b_instruct",
    "gte_Qwen2_7B_instruct",
    "gte_base",
    "gte_large",
    "jina_embeddings_v3",
    "mxbai_embed_large_v1",
    "nomic_embed_text_v1_5",
    "snowflake_arctic_embed_l",
    "snowflake_arctic_embed_m_v2",
]
MTEB_MODELS = EMBED_MODELS + ["NV_Embed_v2"]
LLM_MODELS = [
    "Llama_3_2_1B",
    "Mistral_7B_v0_1",
    "falcon_7b",
    "gemma_2_2b",
    "gpt2",
    "gpt2_xl",
]

HF_NAMES = {
    "GritLM_7B": "GritLM/GritLM-7B",
    "Linq_Embed_Mistral": "Linq-AI-Research/Linq-Embed-Mistral",
    "Qwen3_Embedding_0_6B": "Qwen/Qwen3-Embedding-0.6B",
    "Qwen3_Embedding_8B": "Qwen/Qwen3-Embedding-8B",
    "SFR_Embedding_2_R": "Salesforce/SFR-Embedding-2_R",
    "SFR_Embedding_Mistral": "Salesforce/SFR-Embedding-Mistral",
    "Zeta_Alpha_E5_Mistral": "zeta-alpha-ai/Zeta-Alpha-E5-Mistral",
    "all_MiniLM_L6_v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all_mpnet_base_v2": "sentence-transformers/all-mpnet-base-v2",
    "bge_base_en_v1_5": "BAAI/bge-base-en-v1.5",
    "bge_en_icl": "BAAI/bge-en-icl",
    "bge_large_en_v1_5": "BAAI/bge-large-en-v1.5",
    "bge_multilingual_gemma2": "BAAI/bge-multilingual-gemma2",
    "e5_base_v2": "intfloat/e5-base-v2",
    "e5_large_v2": "intfloat/e5-large-v2",
    "e5_mistral_7b_instruct": "intfloat/e5-mistral-7b-instruct",
    "gte_Qwen2_7B_instruct": "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "gte_base": "thenlper/gte-base",
    "gte_large": "thenlper/gte-large",
    "jina_embeddings_v3": "jinaai/jina-embeddings-v3",
    "mxbai_embed_large_v1": "mixedbread-ai/mxbai-embed-large-v1",
    "nomic_embed_text_v1_5": "nomic-ai/nomic-embed-text-v1.5",
    "snowflake_arctic_embed_l": "Snowflake/snowflake-arctic-embed-l",
    "snowflake_arctic_embed_m_v2": "Snowflake/snowflake-arctic-embed-m-v2.0",
    "NV_Embed_v2": "nvidia/NV-Embed-v2",
}

METRIC_BASES = {
    "frenet": {
        "embed": "frenet_v2_95var",
        "llm": "llm_frenet_v2_95var",
        "bidir": "bidir_frenet_v2_95var",
    },
    "nrs": {
        "embed": "nrs_profile",
        "llm": "llm_nrs_profile",
        "bidir": "bidir_nrs_profile",
    },
    "gfmi": {
        "embed": "gfmi_profile",
        "llm": "llm_gfmi_profile",
        "bidir": "bidir_gfmi_profile",
    },
}


def finite_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def mean(xs: Iterable[float]) -> float:
    vals = [x for x in (finite_float(v) for v in xs) if x is not None]
    return float(np.mean(vals)) if vals else float("nan")


def thirds(arr: Iterable[float]) -> Tuple[List[float], List[float], List[float]]:
    vals = list(arr)
    n = len(vals)
    if n == 0:
        return [], [], []
    width = max(1, n // 3)
    mid_end = n - width
    return vals[:width], vals[width:mid_end], vals[-width:]


def slope(arr: Iterable[float]) -> float:
    vals = np.asarray(list(arr), dtype=float)
    if vals.size < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, vals.size)
    return float(np.polyfit(x, vals, 1)[0])


def public_main_score(task_result, subset: Optional[str]) -> Optional[float]:
    scores = task_result.scores
    split = "test" if "test" in scores else ("validation" if "validation" in scores else next(iter(scores)))
    rows = scores[split]
    if subset:
        subset_rows = [r for r in rows if r.get("hf_subset") == subset]
        if subset_rows:
            rows = subset_rows
    elif len(rows) > 1:
        preferred = [r for r in rows if r.get("hf_subset") in ("en", "default")]
        if preferred:
            rows = preferred
    vals = [finite_float(r.get("main_score")) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    val = float(np.mean(vals))
    return val * 100.0 if val <= 1.5 else val


def source_for_model(model: str) -> str:
    if model in EMBED_MODELS:
        return "embed"
    if model == "NV_Embed_v2":
        return "bidir"
    return "llm"


def load_result(base: str, dataset: str, model: str):
    ds_dir = ROOT / "outputs" / base / dataset
    path = ds_dir / f"{model}.json"
    if path.exists():
        return json.loads(path.read_text())
    merged = ds_dir / "all_results.json"
    if merged.exists():
        data = json.loads(merged.read_text())
        return data.get(model)
    return None


def frenet_features(obj) -> Dict[str, float]:
    if not obj:
        return {}
    out: Dict[str, float] = {}
    for key in [
        "mean_curvature",
        "std_curvature",
        "max_curvature",
        "peak_curvature_depth",
        "mean_speed",
        "std_speed",
        "max_speed",
        "speed_cv",
        "total_arc_length",
        "chord_distance",
        "straightness",
        "norm_arc_length",
        "norm_mean_curvature",
        "early_speed",
        "late_speed",
        "speed_early_late_ratio",
    ]:
        val = finite_float(obj.get(key))
        if val is not None:
            out[key] = val
    speed = [finite_float(v) for v in obj.get("speed", [])]
    speed = [v for v in speed if v is not None]
    curv = [finite_float(v) for v in obj.get("curvature", [])]
    curv = [v for v in curv if v is not None]
    if speed:
        early, _, late = thirds(speed)
        out["speed_first_over_last"] = speed[0] / (speed[-1] + 1e-12)
        out["speed_late_minus_early"] = mean(late) - mean(early)
        out["speed_slope"] = slope(speed)
    if curv:
        early, _, late = thirds(curv)
        out["early_curvature"] = mean(early)
        out["late_curvature"] = mean(late)
        out["curvature_late_minus_early"] = mean(late) - mean(early)
    return out


def nrs_features(obj) -> Dict[str, float]:
    if not obj:
        return {}
    out: Dict[str, float] = {}
    for key in [
        "mean_jaccard",
        "std_jaccard",
        "min_jaccard",
        "max_jaccard",
        "early_jaccard",
        "mid_jaccard",
        "late_jaccard",
        "jaccard_auc",
        "jaccard_trend",
        "jaccard_first_last",
    ]:
        val = finite_float(obj.get(key))
        if val is not None:
            out[key] = val
    return out


def gfmi_features(obj) -> Dict[str, float]:
    if not obj:
        return {}
    profile = obj.get("profile") or {}
    if not profile:
        return {}
    aucs = []
    peaks = []
    for _, row in sorted(((int(k), v) for k, v in profile.items()), key=lambda kv: kv[0]):
        auc = finite_float(row.get("auc"))
        peak = finite_float(row.get("peak_mi"))
        if auc is not None:
            aucs.append(auc)
        if peak is not None:
            peaks.append(peak)
    if not aucs:
        return {}
    early, _, late = thirds(aucs)
    out = {
        "mean_auc": mean(aucs),
        "late_auc": mean(late),
        "delta_late_early": mean(late) - mean(early),
        "slope": slope(aucs),
        "first_auc": aucs[0],
        "last_auc": aucs[-1],
    }
    if peaks:
        out["mean_peak_mi"] = mean(peaks)
    return out


FEATURE_FN = {
    "frenet": frenet_features,
    "nrs": nrs_features,
    "gfmi": gfmi_features,
}


def load_mteb_scores() -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, str]]]:
    benchmark = load_results(
        download_latest=False,
        models=[HF_NAMES[m] for m in MTEB_MODELS],
        tasks=sorted({task for task, _ in TASK_MAP.values()}),
        validate_and_filter=False,
        require_model_meta=False,
        only_main_score=True,
    )
    name_to_short = {v: k for k, v in HF_NAMES.items()}
    public: Dict[str, Dict[str, float]] = {ds: {} for ds in FORMAL_TASKS}
    for model_result in benchmark.model_results:
        short = name_to_short.get(str(model_result.model_name))
        if short is None:
            continue
        by_task = {str(t.task_name): t for t in model_result.task_results}
        for ds, (task, subset) in TASK_MAP.items():
            task_result = by_task.get(task)
            if task_result is None:
                continue
            score = public_main_score(task_result, subset)
            if score is not None:
                public[ds][short] = max(public[ds].get(short, -math.inf), score)

    scores: Dict[str, Dict[str, float]] = {ds: {} for ds in FORMAL_TASKS}
    sources: Dict[str, Dict[str, str]] = {ds: {} for ds in FORMAL_TASKS}
    for ds in FORMAL_TASKS:
        local = fc.MTEB_SCORES.get(ds, {})
        for model in MTEB_MODELS:
            if model in public[ds]:
                scores[ds][model] = public[ds][model]
                sources[ds][model] = "public-mteb"
                continue
            val = finite_float(local.get(model))
            if val is not None:
                scores[ds][model] = val
                sources[ds][model] = "local-fallback"
    return scores, sources


def main() -> None:
    if set(FORMAL_TASKS) != set(TASK_MAP):
        raise RuntimeError(f"Task map mismatch: {set(FORMAL_TASKS) ^ set(TASK_MAP)}")

    task_type = {
        ds: fc.DATASET_CONFIGS.get(ds, {}).get("task")
        for ds in FORMAL_TASKS
    }

    mteb_scores, score_sources = load_mteb_scores()
    performance = {
        "MTEB-25-clean": mteb_scores,
        "MMLU-6": {ds: {m: float(_MMLU[m]) for m in LLM_MODELS} for ds in FORMAL_TASKS},
    }

    metric_values: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
        "MTEB-25-clean": {},
        "MMLU-6": {},
    }
    metric_missing = []
    for track, models in [("MTEB-25-clean", MTEB_MODELS), ("MMLU-6", LLM_MODELS)]:
        for ds in FORMAL_TASKS:
            for model in models:
                source = source_for_model(model)
                for family, bases in METRIC_BASES.items():
                    obj = load_result(bases[source], ds, model)
                    features = FEATURE_FN[family](obj)
                    if not features:
                        metric_missing.append((track, ds, model, family))
                    for name, value in features.items():
                        if math.isfinite(value):
                            metric = f"{family}:{name}"
                            metric_values[track].setdefault(metric, {}).setdefault(ds, {})[model] = value

    per_dataset = []
    summary = []
    for track, metric_map in metric_values.items():
        for metric, ds_map in sorted(metric_map.items()):
            family = metric.split(":", 1)[0]
            rows = []
            for ds in FORMAL_TASKS:
                models = sorted(set(ds_map.get(ds, {})) & set(performance[track].get(ds, {})))
                x, y = [], []
                for model in models:
                    xv = finite_float(ds_map[ds][model])
                    yv = finite_float(performance[track][ds][model])
                    if xv is not None and yv is not None:
                        x.append(xv)
                        y.append(yv)
                if len(x) < 4 or len(set(x)) < 2 or len(set(y)) < 2:
                    continue
                rho, p = spearmanr(x, y)
                if math.isnan(rho) or math.isnan(p):
                    continue
                row = {
                    "track": track,
                    "family": family,
                    "metric": metric,
                    "dataset": ds,
                    "task_type": task_type.get(ds),
                    "rho": float(rho),
                    "p": float(p),
                    "n": len(x),
                }
                per_dataset.append(row)
                rows.append(row)
            if not rows:
                continue
            rhos = np.asarray([r["rho"] for r in rows], dtype=float)
            ps = np.asarray([r["p"] for r in rows], dtype=float)
            ns = np.asarray([r["n"] for r in rows], dtype=float)
            summary.append(
                {
                    "track": track,
                    "family": family,
                    "metric": metric,
                    "n_datasets": len(rows),
                    "n_min": int(np.min(ns)),
                    "n_med": float(np.median(ns)),
                    "n_max": int(np.max(ns)),
                    "median_rho": float(np.median(rhos)),
                    "mean_rho": float(np.mean(rhos)),
                    "median_abs_rho": float(np.median(np.abs(rhos))),
                    "median_p": float(np.median(ps)),
                    "sig_p05": int(np.sum(ps < 0.05)),
                    "sig_p01": int(np.sum(ps < 0.01)),
                    "pos": int(np.sum(rhos > 0)),
                    "neg": int(np.sum(rhos < 0)),
                }
            )

    summary.sort(key=lambda r: (r["track"], r["family"], -r["sig_p05"], -abs(r["median_rho"])))

    coverage = {
        "formal_tasks": len(FORMAL_TASKS),
        "mteb_models": len(MTEB_MODELS),
        "llm_models": len(LLM_MODELS),
        "reporting_models_total": len(MTEB_MODELS) + len(LLM_MODELS),
        "mteb_score_cells": sum(len(v) for v in mteb_scores.values()),
        "mteb_possible_cells": len(FORMAL_TASKS) * len(MTEB_MODELS),
        "missing_mteb_scores": {
            ds: [m for m in MTEB_MODELS if m not in mteb_scores[ds]]
            for ds in FORMAL_TASKS
        },
        "score_source_counts": {
            "public-mteb": sum(
                1
                for ds in FORMAL_TASKS
                for m in MTEB_MODELS
                if score_sources[ds].get(m) == "public-mteb"
            ),
            "local-fallback": sum(
                1
                for ds in FORMAL_TASKS
                for m in MTEB_MODELS
                if score_sources[ds].get(m) == "local-fallback"
            ),
        },
        "metric_missing_count": len(metric_missing),
        "metric_missing_first50": metric_missing[:50],
    }

    payload = {
        "coverage": coverage,
        "summary": summary,
        "per_dataset": per_dataset,
    }
    (OUT_DIR / "formal30_clean31_metric_downstream_correlations.json").write_text(
        json.dumps(payload, indent=2)
    )

    with (OUT_DIR / "formal30_clean31_metric_downstream_summary.csv").open("w", newline="") as f:
        fields = [
            "track",
            "family",
            "metric",
            "n_datasets",
            "n_min",
            "n_med",
            "n_max",
            "median_rho",
            "mean_rho",
            "median_abs_rho",
            "median_p",
            "sig_p05",
            "sig_p01",
            "pos",
            "neg",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    with (OUT_DIR / "formal30_clean31_metric_downstream_per_dataset.csv").open("w", newline="") as f:
        fields = ["track", "family", "metric", "dataset", "task_type", "rho", "p", "n"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_dataset)

    print(json.dumps(coverage, indent=2))
    for track in ["MTEB-25-clean", "MMLU-6"]:
        print(f"\n{track}")
        for family in ["frenet", "nrs", "gfmi"]:
            print(family)
            rows = [r for r in summary if r["track"] == track and r["family"] == family]
            for row in rows[:8]:
                print(
                    "  {metric} med={median_rho:.3f} sig={sig_p05}/{n_datasets} "
                    "pmed={median_p:.3g} nmed={n_med:.1f} pos/neg={pos}/{neg}".format(**row)
                )


if __name__ == "__main__":
    main()
