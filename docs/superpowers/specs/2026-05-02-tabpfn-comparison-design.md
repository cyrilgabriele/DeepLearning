# TabPFN Comparison: Design Spec

**Date:** 2026-05-02
**Author:** Gian (with Claude assistance)
**Branch:** `tabpfn`
**Scope:** Add TabPFN as a contemporary tabular-foundation-model baseline to the paper comparison. Two evaluation runs (full-feature, 20-feature), permutation-importance ranking, attention-based attribution for applicant 55728, and small targeted edits to `local_files/main (1).tex`. Measured comparison; no new section, no new figure.
**Methodology:** Same train/test split, same threshold calibration, same QWK metric, same outer-test evaluation as the existing KAN/XGBoost/MLP rows. Only difference: TabPFN is a pretrained foundation model, so it is used at default hyperparameters with no per-task tuning.

---

## Background

The paper currently compares ChebyKAN, FourierKAN, and BSplineKAN against XGBoost and an MLP. The professor (who knows the model) asked for a TabPFN comparison so the paper acknowledges current SOTA in tabular foundation models. The intent is *measured but not overdone*: get a QWK number for TabPFN in both regimes, document its interpretability, and incorporate this as a few targeted edits in §1, §2.2, §3.2.1, §3.2.2, §4, plus one row in Table 1 and two rows in Table 2. No new section, no figure, no full Optuna sweep.

The TabPFN paper (Hollmann et al., 2025) introduces a transformer pretrained on a synthetic prior over tabular tasks. Inference is performed by feeding the entire training set as in-context examples; the model produces predictions for test rows in a single forward pass with no gradient updates. TabPFN-v2 natively supports up to ~10k training samples; for larger datasets the official `tabpfn_extensions.AutoTabPFN` wrapper performs a post-hoc subsample-ensemble.

The comparison serves two narrative goals:
1. Strengthen the accuracy benchmark side of Tables 1 and 2 with a SOTA tabular foundation model.
2. Reinforce the paper's "post-hoc vs. model-native" interpretability axis: TabPFN, like XGBoost, sits firmly on the post-hoc side, leaving the sparse ChebyKAN as the only configuration that combines competitive 20-feature accuracy with model-native closed-form structure.

---

## Goals

- Add a single TabPFN row to Table 1 (single-objective Optuna QWK reference table).
- Add two TabPFN rows to Table 2 (one in the full-feature regime, one in the 20-feature regime).
- Run TabPFN at default hyperparameters using `AutoTabPFN` to accommodate Prudential's ~59k training rows.
- Apply the same QWK-based ordinal threshold calibration used by MLP and KANs.
- Derive TabPFN's own top-20 features from permutation importance (parallel to "each model uses its own top-20" in line 106 of the paper).
- Compute top-5 feature-overlap between TabPFN, XGBoost-SHAP, and ChebyKAN-native rankings.
- Extract attention weights for applicant 55728 over the training set, save top-N most-attended training applicants, and surface aggregate label statistics among them.
- Make small, targeted edits to `local_files/main (1).tex` (Tables 1 and 2 rows, plus 1–3 sentences in §1, §2.2, §3.2.1, §3.2.2, §4, §5).

## Non-goals

- New paper section, new figure, new subsection.
- Optuna or any other per-task tuning of TabPFN.
- Re-running KAN, XGBoost, or MLP. Their existing artifacts are reused as-is.
- Re-running Cyril's Stage A/B/C chain or the recently completed 20-feature rerun. TabPFN is layered on top of the current state of the paper.
- Sparse-regime λ Pareto for TabPFN (not applicable; TabPFN has no native sparsity penalty).
- Producing a TabPFN PDP plot or a TabPFN counterpart to Figure 1.
- Cross-model preprocessing alignment beyond the existing per-recipe convention. TabPFN gets its own recipe (`tabpfn_paper`).

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
                │ §1 Dependency + wrapper setup                │
                │  - add tabpfn, tabpfn_extensions to deps     │
                │  - implement TabPFNAutoRegressor             │
                │  - register; new tabpfn_paper recipe         │
                └──────────────────┬───────────────────────────┘
                                   │
                ┌──────────────────▼───────────────────────────┐
                │ §2 Run A — full-feature (128)                │
                │  - 3 seeds                                   │
                │  - outer-test QWK + 95% t-CI                 │
                │  - permutation importance → top-20 list      │
                └──────────────────┬───────────────────────────┘
                                   │
                ┌──────────────────▼───────────────────────────┐
                │ §3 Run B — 20-feature                        │
                │  - TabPFN's own permutation top-20           │
                │  - 1 seed, outer-test QWK + bootstrap CI     │
                └──────────────────┬───────────────────────────┘
                                   │
                ┌──────────────────▼───────────────────────────┐
                │ §4 Interpretability artifacts                │
                │  - feature_overlap.json (top-5 vs XGB/Cheby) │
                │  - applicant_55728_attention.csv (top-N)     │
                └──────────────────┬───────────────────────────┘
                                   │
                ┌──────────────────▼───────────────────────────┐
                │ §5 Paper edits                               │
                │  - Table 1 row, Table 2 ×2 rows              │
                │  - §1, §2.2, §3.2.1, §3.2.2, §4, §5 edits    │
                └──────────────────────────────────────────────┘
```

Sequential. Estimated wall-clock: **~2–3 h** end to end including paper edits and self-review.

---

## §1 Dependency + wrapper setup

### Dependencies

Add to `pyproject.toml`:

```toml
tabpfn = ">=2.0.0"
tabpfn-extensions = ">=0.1.0"
```

Then `uv sync`. TabPFN-v2 supports CPU and CUDA; on Apple Silicon it currently runs on CPU. Inference for one outer-test pass with `AutoTabPFN` over Prudential is approximately 5–15 minutes on a 32 GB MacBook Pro. Three seeds therefore fit easily within the 2–3 h overall budget.

### Wrapper

`src/models/tabpfn.py` — new file. Defines `TabPFNAutoRegressor` implementing the existing `BaseModel` interface used elsewhere in `src/models/`. Internally wraps `tabpfn_extensions.AutoTabPFN(TabPFNRegressor(...))`.

Responsibilities:

- `fit(X_train, y_train)` — pass through to the wrapped `AutoTabPFN`. `y_train` is the continuous Response 1–8 (regression mode).
- `predict(X)` — return continuous risk scores (regression output).
- `predict_classes(X, thresholds)` — return discrete classes 1–8 using the same threshold calibration helper used by `tabkan.py` and `mlp.py`. The wrapper does not do its own threshold fitting; calibration happens in the existing pipeline downstream.
- `save(path)` / `load(path)` — joblib-pickle the underlying `AutoTabPFN` estimator. Manifest JSON written alongside, matching the existing `*.manifest.json` convention.
- `get_internal_estimator()` — return the underlying `AutoTabPFN` (so the attention-extraction script in §4 can reach the inner transformer).

### Registry

Add a `tabpfn_auto` entry to `src/models/registry.py`. Flavor name: `tabpfn_auto`. Construction kwargs: `n_estimators` (default 8), `max_time` (seconds; default 300), `device` (default `"auto"`).

### Preprocessing recipe

`src/preprocessing/recipes/tabpfn_paper.py` — new file. Behavior identical to `xgboost_paper`: numeric encoding plus simple NaN imputation; **no scaling** (TabPFN does its own internal normalization). Categorical features are integer-encoded as in the XGBoost recipe.

The recipe is keyed `tabpfn_paper`. KAN and XGBoost recipes are unchanged.

---

## §2 Run A — full-feature

### Config

`configs/experiment_stages/stage_c_explanation_package/tabpfn/all_features/train/tabpfn_full.yaml`

```yaml
trainer:
  experiment_name: stage-c-tabpfn-full
  seeds: [42, 1337, 2024]
preprocessing:
  recipe: tabpfn_paper
model:
  flavor: tabpfn_auto
  params:
    n_estimators: 8
    max_time: 300
    device: auto
evaluate:
  outer_test: true
  threshold_calibration:
    metric: qwk
```

### Command

```bash
uv run python main.py --stage train \
  --config configs/experiment_stages/stage_c_explanation_package/tabpfn/all_features/train/tabpfn_full.yaml
```

The existing trainer loops over the seeds list and produces one checkpoint per seed:

- `checkpoints/stage-c-tabpfn-full/model-<ts>-seed42.joblib` (and 1337, 2024).
- `artifacts/stage-c-tabpfn-full/run-summary-<ts>.json` with per-seed validation and outer-test metrics.

### Validation and outer-test evaluation

For each seed, the trainer produces continuous predictions on both the validation split and the outer-test split. Threshold calibration runs on validation; thresholds are then applied to both splits to produce discrete class predictions. QWK is computed on each.

- **Validation QWK** (mean over seeds) populates the Table 1 row, matching the column semantics of the other rows in that table (validation QWK from a single-objective study).
- **Outer-test QWK** (mean and 95% t-interval over the three seed-level QWKs) populates the Table 2 full-feature row, matching line 110's protocol ("Full-feature intervals are 95% t-intervals across three seeds").

Single Run-A pass produces both numbers; no separate eval pass is needed.

### Permutation importance

After all three seeds finish, run `scripts/tabpfn_permutation_importance.py`:

- Load each seed's checkpoint, get the canonical (mean-of-three-seeds) outer-test prediction by averaging the three continuous predictions (then threshold-calibrate via the protocol-standard helper).
- For permutation importance, use the *highest-validation-QWK seed*'s estimator (single estimator, deterministic, fast). `sklearn.inspection.permutation_importance` with `n_repeats=10`, custom QWK scorer (apply thresholds, compute QWK).
- Sort features by mean importance descending.
- Write `outputs/interpretability/tabpfn_paper/stage-c-tabpfn-full/data/tabpfn_feature_ranking.csv` with columns `feature, importance, importance_std`.

The top-20 of this ranking becomes the input to Run B.

---

## §3 Run B — 20-feature

### Top-20 list

`configs/experiment_stages/stage_c_explanation_package/feature_lists/tabpfn_top20_features.json` — a top-level array of 20 feature-name strings, taken from the head of `tabpfn_feature_ranking.csv`. Generated by the same `scripts/tabpfn_permutation_importance.py` script as a side output.

### Config

`configs/experiment_stages/stage_c_explanation_package/tabpfn/20_features/train/tabpfn_top20.yaml`

```yaml
trainer:
  experiment_name: stage-c-tabpfn-top20
  seeds: [42]
preprocessing:
  recipe: tabpfn_paper
  selected_features_path: configs/experiment_stages/stage_c_explanation_package/feature_lists/tabpfn_top20_features.json
model:
  flavor: tabpfn_auto
  params:
    n_estimators: 8
    max_time: 300
    device: auto
evaluate:
  outer_test: true
  threshold_calibration:
    metric: qwk
  bootstrap_ci:
    n_resamples: 1000
    seed: 42
```

### Command

```bash
uv run python main.py --stage train \
  --config configs/experiment_stages/stage_c_explanation_package/tabpfn/20_features/train/tabpfn_top20.yaml
```

### Outer-test evaluation

The paper's sparse regime uses a 95% bootstrap CI over the outer-test set (line 110). Run B uses the same bootstrap protocol: 1000 resamples of the outer-test set, QWK on each resample, percentile CI from the resampled distribution. Single seed for the point estimate.

This is added to `scripts/bootstrap_qwk_table1.py` by appending a new entry to its `MODELS` list (`stage-c-tabpfn-top20`, single-seed checkpoint, `tabpfn_paper` recipe, `tabpfn_top20.yaml` config). The existing CI computation pipeline handles the rest.

---

## §4 Interpretability artifacts

### §4.1 Top-5 feature overlap

`scripts/tabpfn_top5_overlap.py` — new one-shot script.

Inputs:

- TabPFN top-5: `outputs/interpretability/tabpfn_paper/stage-c-tabpfn-full/data/tabpfn_feature_ranking.csv` (head 5).
- XGBoost top-5: `outputs/interpretability/xgboost_paper/stage-c-xgb-best/data/shap_xgb_values.parquet` → ranked by mean(|SHAP|) per feature, head 5.
- ChebyKAN top-5: `outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_feature_ranking.csv` (head 5).

Output: `outputs/interpretability/tabpfn_paper/feature_overlap.json`

```json
{
  "tabpfn_top5": [...],
  "xgb_top5": [...],
  "chebykan_top5": [...],
  "tabpfn_xgb_intersection_count": <int>,
  "tabpfn_chebykan_intersection_count": <int>,
  "tabpfn_xgb_intersection": [...],
  "tabpfn_chebykan_intersection": [...]
}
```

The two intersection counts are the values that fill `N` and `M` in the §3.2.1 paper edit.

### §4.2 Attention attribution for applicant 55728

`scripts/tabpfn_attention_55728.py` — new one-shot script.

Approach:

1. Load the highest-validation-QWK Run-A seed's `AutoTabPFN` estimator.
2. Identify the underlying ensemble member with the highest validation QWK (within `AutoTabPFN`'s internal members) and extract its inner `TabPFNRegressor`'s transformer module.
3. Register a forward hook on the **last** transformer block's attention layer to capture attention weights during prediction.
4. Build the inference batch: full training set plus applicant 55728 as the test query.
5. Run a forward pass; the hook captures attention weights of shape `(num_heads, n_train + 1, n_train + 1)`.
6. Reduce: take the row corresponding to applicant 55728's position as query, restrict columns to the n_train training positions, average over heads.
7. Sort training applicants by attention weight descending; take the top 20.
8. Write `outputs/interpretability/tabpfn_paper/applicant_55728_attention.csv` with columns: `train_applicant_id, attention_weight, true_response, BMI, Ins_Age, Medical_History_15, Product_Info_4` (BMI + the four most informative features for cross-reference; full feature dump available in companion JSON).
9. Write `outputs/interpretability/tabpfn_paper/applicant_55728_attention_summary.json`:

```json
{
  "applicant_id": 55728,
  "predicted_class": <int>,
  "top_n_attended_count": 20,
  "mean_true_response_among_top_n": <float>,
  "median_true_response_among_top_n": <float>,
  "std_true_response_among_top_n": <float>,
  "fraction_top_n_matching_predicted_class": <float>
}
```

The summary fields populate the §3.2.1 paper edit ("mean true Response among them: $\bar{y} = X.X$, against the predicted class $3$").

**Risk:** The attention hook depends on the internal layer naming of the version of `tabpfn` installed by `uv sync`. If the hook fails (a) because the internal API has changed, (b) because `AutoTabPFN`'s ensemble structure does not expose individual member transformers, or (c) for any other reason, **the failure is timeboxed to 1 hour**. After 1 hour:

- The paper falls back to a *mention only* form of the §3.2.1 attention sentence ("TabPFN supports attention-based attribution over training applicants in principle, but extracting it from the AutoTabPFN ensembling wrapper is non-trivial and is left to future work").
- The `applicant_55728_attention.csv` artifact is not produced.
- All other deliverables proceed unchanged.

This is a documented degradation, not a silent fallback. The decision is logged to `runs/2026-05-02-tabpfn/04_attention.log`.

---

## §5 Paper edits to `local_files/main (1).tex`

All edits are small, targeted, and additive. No existing paragraph is rewritten end-to-end.

### §5.1 §1 — one sentence (after line 46)

> "Beyond architectural innovations, recent work introduces tabular foundation models that are pretrained on synthetic priors and applied to new tasks via in-context learning rather than per-task gradient updates; TabPFN \cite{Hollmann2025} is the principal example and serves here as a contemporary no-tuning baseline."

References update: add `\cite{Hollmann2025}` entry to `local_files/references.bib` (or the equivalent .bib file used by the paper). Actual bib entry text:

```bibtex
@article{Hollmann2025,
  title   = {Accurate predictions on small data with a tabular foundation model},
  author  = {Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and others},
  journal = {Nature},
  year    = {2025}
}
```

### §5.2 §2.2 — two sentences (end of paragraph at line 66)

> "We additionally include TabPFN, a transformer-based tabular foundation model, as a no-tuning reference baseline. TabPFN is run in regression mode with the AutoTabPFN ensembling wrapper to accommodate the dataset size, and the same QWK-based threshold calibration is applied as for the MLP and KAN models."

### §5.3 Table 1 — one row (line 96)

Insert between the XGBoost row and the MLP row:

```latex
TabPFN    & --   & 0.XXXX \\
```

The QWK value is the **mean of the three Run-A seed-level validation QWKs**, matching the column semantics of the other Table 1 rows (validation QWK from a single-objective study). The `Trials` column is `--` because TabPFN is used as a pretrained model with no per-task tuning. The Table 1 caption is extended with one sentence: "For TabPFN, no Optuna study is performed; the reported value is the mean validation QWK across three seeds at default hyperparameters." This avoids the LaTeX-table-footnote complication while remaining honestly disclosed.

### §5.4 Table 2 — two rows (lines 119–126)

Full-feature regime (insert between XGBoost and ChebyKAN rows):

```latex
TabPFN, default & 0.XXX [0.XXX, 0.XXX] & in-context & SHAP / attention \\
```

Sparse regime (insert between XGBoost-tuned and ChebyKAN-sparse rows):

```latex
TabPFN, default & 0.XXX [0.XXX, 0.XXX] & in-context & SHAP / attention \\
```

The "Size" column entry is the literal string `in-context` (TabPFN has no tree count or edge count; the model is a fixed pretrained transformer that consumes the training set in-context).

### §5.5 §3.2.1 — three sentences (insert before line 154)

> "TabPFN reaches an outer-test QWK of $0.\text{XXX}$ in the full-feature setting, comparable to but not surpassing XGBoost. Its top-5 permutation-importance features overlap by $N$ with XGBoost-SHAP and by $M$ with the ChebyKAN native ranking, indicating that the foundation model recovers the same core underwriting signal. TabPFN interpretability is similar in kind to XGBoost: both rely on post-hoc methods (SHAP, permutation importance) and neither admits a model-native closed-form representation; TabPFN additionally supports attention-based attribution over training applicants, but this is example-level rather than function-level interpretability and remains on the post-hoc side of our comparison."

`X.XXX`, `N`, `M` are filled from Run A QWK and `feature_overlap.json`.

### §5.6 §3.2.1 — one sentence (after the Figure 1 description, around line 173)

> "For the same applicant (55728), TabPFN attention identifies the $N$ training applicants whose feature patterns most influence this prediction (mean true Response among them: $\bar{y} = X.X$, against the predicted class $3$), providing case-based justification but no functional decomposition; the ChebyKAN closed form, by contrast, attributes the prediction to specific polynomial responses of named features, which is the property exploited in Figure 1."

`N`, `\bar{y}` are filled from `applicant_55728_attention_summary.json`. If the §4.2 attention hook fails (timeboxed fallback), this sentence is replaced by the fallback wording given in §4.2.

### §5.7 §3.2.2 — one sentence (within sparse-regime paragraph, around line 162)

> "In the 20-feature regime, TabPFN reaches outer-test QWK $0.\text{XXX}$ on its own permutation-importance top-20, $\Delta$ from XGBoost on its top-20."

`0.XXX` is from Run B; `Δ` is the signed numeric difference from the XGBoost top-20 row of Table 2 (already in the paper at QWK 0.613).

### §5.8 §4 (Discussion) — one sentence (around line 182)

> "Even when extending the comparison to a SOTA tabular foundation model, the relative position of the sparse KAN configurations is preserved: TabPFN provides accuracy in the same broad regime as XGBoost without offering model-native interpretability, leaving the sparse ChebyKAN as the only configuration in the comparison that combines competitive 20-feature accuracy with closed-form structure."

### §5.9 §5 limitations — one sentence (around line 184)

> "TabPFN was used at default hyperparameters as a no-tuning reference; per-task fine-tuning could change the predictive ranking but lies outside the scope of the present comparison."

### §5.10 No edits to other figures or tables

`fig1_interpretability.pdf`, the closed-forms table, and bootstrap CI scripts for non-TabPFN rows are unchanged. The bootstrap CI script `scripts/bootstrap_qwk_table1.py` gains one new `MODELS` entry for the 20-feature TabPFN row; existing entries are preserved.

---

## Run-log layout

```
runs/2026-05-02-tabpfn/
├── master.log                      # high-level step boundaries + status
├── 00_setup.log                    # dep install, recipe registration smoke test
├── 01_run_a_full/
│   ├── seed42.log
│   ├── seed1337.log
│   └── seed2024.log
├── 02_permutation_importance.log
├── 03_run_b_top20/
│   └── seed42.log
├── 04_attention.log                # success or timeboxed-fallback decision
├── 05_overlap.log
└── 06_paper_edits/
    └── summary.md                  # which numeric placeholders were filled and from where
```

Each step's command is printed to `master.log` before execution, and to its own log file. On failure, the failing log file is the source of truth.

---

## Failure handling

- **§1 setup failure** (TabPFN dependency install or wrapper instantiation broken): block the entire spec, surface to user, no fallback. The remaining sections cannot proceed without a working wrapper.
- **§2 Run A failure for one seed**: continue with the other seeds; report mean and t-CI over the surviving seeds and note in the run summary. Two-of-three is acceptable; one-of-three triggers a manual check.
- **§3 Run B failure**: the 20-feature row of Table 2 is left blank with a note in the run summary; full-feature paper edits proceed.
- **§4.1 overlap script failure**: `feature_overlap.json` is not produced; §3.2.1 second-sentence overlap wording is replaced by "the foundation model's top features broadly overlap with the XGBoost-SHAP and ChebyKAN native rankings"; numeric `N`, `M` are dropped.
- **§4.2 attention hook failure**: 1-hour timebox, then fall back to mention-only wording per §4.2 above.
- **§5 paper edits**: each subsection in §5 is independently applicable. A failed-data subsection (e.g., overlap script failed) leaves its placeholder unfilled but does not block the others.

No fallback to "approximate" or "expected" numbers anywhere. If a number cannot be produced, the corresponding paper sentence is rewritten in mention-only form rather than fabricating a value.

---

## Estimated wall-clock

| Step | Time |
|---|---|
| §1 setup (deps + wrapper + recipe + registry) | 30 min |
| §2 Run A (3 seeds × ~10 min) | 30–45 min |
| §2 permutation importance (10 repeats) | 15 min |
| §3 Run B (1 seed) | 10 min |
| §4.1 top-5 overlap | 5 min |
| §4.2 attention attribution | 30 min (or 1 h timebox + fallback) |
| §5 paper edits + bibliography | 20 min |
| **Total** | **~2.5 h end to end** |

Hardware: Apple Silicon, 32 GB, CPU inference for TabPFN. Sequential.

---

## Out of scope

- Optuna or other tuning of TabPFN.
- Full SHAP analysis on TabPFN (KernelSHAP would take many hours; permutation importance is the chosen substitute).
- A new figure parallel to Figure 1 for TabPFN attention.
- Updating any non-TabPFN row in Table 1 or Table 2.
- Modifying KAN recipes, KAN configs, or KAN scripts.
- Extending the comparison to TabPFN fine-tuning, TabPFN with custom priors, or other foundation models.

---

## Open items

None. All design decisions are locked.
