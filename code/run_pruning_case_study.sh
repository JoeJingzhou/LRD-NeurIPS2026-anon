#!/usr/bin/env bash
set -euo pipefail

ROOT="."
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCORE_DIR="${ROOT}/outputs/pruning_scores"
PLAN_DIR="${ROOT}/outputs/pruning_plans"
PROTECT_EDGES="${PROTECT_EDGES:-3}"
BUDGETS_CSV="${BUDGETS_CSV:-0.10}"
IFS=',' read -r -a BUDGETS <<< "${BUDGETS_CSV}"

MODELS=(
  "LLM2Vec_Mistral_7B"
  "NV_Embed_v2"
)

DATASETS=(
  "mteb_NQ"
  "mteb_AmazonPolarityClassification"
  "mteb_STSBenchmark"
)

RULES=(frenet nrs gfmi lastk)
RANDOM_SEEDS=(0 1 2)

for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "== build score :: ${ds} :: ${model}"
    "${PYTHON_BIN}" "${ROOT}/script/build_pruning_scores.py" \
      --dataset "${ds}" \
      --model "${model}" \
      --output-dir "${SCORE_DIR}"

    score_json="${SCORE_DIR}/${ds}/${model}.json"

    for budget in "${BUDGETS[@]}"; do
      for rule in "${RULES[@]}"; do
        echo "== select :: ${rule} :: ${budget} :: ${ds} :: ${model}"
        "${PYTHON_BIN}" "${ROOT}/script/select_pruned_layers.py" \
          --score-json "${score_json}" \
          --rule "${rule}" \
          --budget "${budget}" \
          --protect-edges "${PROTECT_EDGES}" \
          --output-dir "${PLAN_DIR}"
      done

      for seed in "${RANDOM_SEEDS[@]}"; do
        echo "== select :: random(seed=${seed}) :: ${budget} :: ${ds} :: ${model}"
        "${PYTHON_BIN}" "${ROOT}/script/select_pruned_layers.py" \
          --score-json "${score_json}" \
          --rule random \
          --budget "${budget}" \
          --protect-edges "${PROTECT_EDGES}" \
          --seed "${seed}" \
          --output-dir "${PLAN_DIR}"
      done
    done
  done
done

echo "Pilot pruning plans written under ${PLAN_DIR}"
