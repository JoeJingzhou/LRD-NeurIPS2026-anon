# Layer-wise Representation Dynamics (LRD)

Anonymous code release for the NeurIPS 2026 submission *"Layer-wise
Representation Dynamics"*.

This repository contains the measurement, model-selection, and
layer-pruning pipeline used in the paper. All identifying paths have
been replaced; the placeholder `<USER>` may appear in comments.

## Contents

```
code/
  pooling_utils.py                          Encoder/decoder pooling resolver
  llm_models.py                             Base LLM panel + MMLU scores
  bidir_models.py                           Bidirectional embedder helpers

  frenet_curvature.py                       Frenet (Grassmann speed + Menger curvature)
  nrs_profile.py                            NRS (Jaccard retention)
  gfmi_all_layers.py                        GFMI (per-percentile MI + AUC)

  compute_family_derived_metrics.py         Section 4 family-level summaries
  compute_clean31_selection_correlations.py Section 5 model selection
  build_pruning_scores.py                   Section 6 per-layer pruning scores
  select_pruned_layers.py                   Section 6 layer selection
  eval_pruned_embedder.py                   Section 6 inference-time block skipping
  run_pruning_case_study.sh                 Section 6 driver script
```

Each of `frenet_curvature.py`, `nrs_profile.py`, and `gfmi_all_layers.py`
is self-contained: it loads the model, extracts per-layer hidden states,
and computes its primitive in a single command.

## Requirements

Python 3.9+ with the packages in `requirements.txt`. A single GPU is
required for forward passes; the rest of the pipeline runs on CPU.

```
pip install -r requirements.txt
```

## Quick Start

The scripts assume that the repository root is the current working
directory. Each script writes outputs under `./outputs/<subdir>/`.

### 1. Compute the three layer-wise primitives

```
python code/frenet_curvature.py  --models e5_large_v2 --datasets mteb_banking77
python code/nrs_profile.py        --models e5_large_v2 --datasets mteb_banking77
python code/gfmi_all_layers.py    --models e5_large_v2 --datasets mteb_banking77
```

### 2. Section 4 / Section 5: aggregate to model-level scores

```
python code/compute_family_derived_metrics.py
python code/compute_clean31_selection_correlations.py
```

### 3. Section 6: layer pruning

```
bash code/run_pruning_case_study.sh
```

## Conventions

The implementation follows the conventions reported in Appendix A of
the paper:

- **Frenet.** Subspace rank `r` is the smallest rank whose cumulative
  explained variance of the final-layer representation reaches 95%, and
  the same `r` is used for all layers. Speed is forward-indexed:
  `s_l = d_Gr(Q_l, Q_{l+1})`. Menger curvature is assigned to the
  middle layer of each consecutive triple. Degenerate triples receive
  zero curvature.
- **NRS.** Jaccard retention is computed on `min(500, N)` random
  anchors sampled once per (model, task) pair (seed 42) with
  `k_NRS = 20`. Adjacent layers `l` and `l+1` are compared.
- **GFMI.** The filtration parameter is a per-layer percentile.
  Candidate edges come from a cosine `k_GFMI = 30` graph that is
  symmetrized before connected components are computed. Per-percentile
  mutual information is integrated by trapezoidal rule on the grid
  `linspace(5, 95, 20)`, with no further normalization.
- **Pruning.** The first three and last three layers of every model are
  protected; remaining layers are ranked by the within-profile z-scored
  rule and removed in decreasing order of score up to the budget.

## Data

All experiments use publicly available datasets and model checkpoints
from the Hugging Face Hub. No new data is released.

## License

MIT.
