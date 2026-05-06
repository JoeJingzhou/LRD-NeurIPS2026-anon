#!/usr/bin/env python3
"""
Select pruned layers from precomputed pruning scores.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List


ROOT = Path(".")


def _load(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _candidate_layers(layer_ids: List[int], protect_edges: int) -> List[int]:
    if protect_edges <= 0:
        return list(layer_ids)
    if len(layer_ids) <= 2 * protect_edges:
        return []
    return list(layer_ids[protect_edges:-protect_edges])


def _pick_budget(candidate: List[int], frac: float) -> int:
    if not candidate:
        return 0
    return max(1, int(math.ceil(len(candidate) * frac)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Select layers to prune from pruning scores")
    ap.add_argument("--score-json", required=True)
    ap.add_argument("--rule", required=True, choices=["frenet", "nrs", "gfmi", "composite", "random", "lastk"])
    ap.add_argument("--budget", required=True, type=float, help="Fraction of candidate layers to prune, e.g. 0.1")
    ap.add_argument("--protect-edges", type=int, default=1, help="Protect this many layers at both ends")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default=str(ROOT / "outputs/pruning_plans"))
    args = ap.parse_args()

    score_path = Path(args.score_json)
    scores = _load(score_path)
    layer_ids = [int(x) for x in scores["layer_ids"]]
    candidates = _candidate_layers(layer_ids, args.protect_edges)
    n_prune = _pick_budget(candidates, args.budget)

    if args.rule in {"frenet", "nrs", "gfmi", "composite"}:
        score_key = {
            "frenet": "frenet_score",
            "nrs": "nrs_score",
            "gfmi": "gfmi_score",
            "composite": "composite_score",
        }[args.rule]
        score_map = {int(x["layer"]): float(x[score_key]) for x in scores["layers"]}
        ranked = sorted(candidates, key=lambda l: (score_map[l], l), reverse=True)
        pruned = ranked[:n_prune]
    elif args.rule == "lastk":
        pruned = sorted(candidates)[-n_prune:]
    else:
        rng = random.Random(args.seed)
        pruned = sorted(rng.sample(candidates, n_prune)) if n_prune > 0 else []

    out = {
        "model": scores["model"],
        "dataset": scores["dataset"],
        "rule": args.rule,
        "budget": args.budget,
        "seed": args.seed,
        "protect_edges": args.protect_edges,
        "candidate_layers": candidates,
        "protected_layers": [x for x in layer_ids if x not in candidates],
        "pruned_layers": pruned,
        "score_json": str(score_path),
    }

    out_dir = Path(args.output_dir) / scores["dataset"] / scores["model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.rule}_{int(round(args.budget * 100)):02d}"
    if args.rule == "random":
        suffix += f"_seed{args.seed}"
    out_path = out_dir / f"{suffix}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(out_path)


if __name__ == "__main__":
    main()
