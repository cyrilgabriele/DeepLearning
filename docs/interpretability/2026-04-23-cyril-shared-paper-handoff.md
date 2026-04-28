# Cyril Handoff — Large TabKAN Two-Flavor Comparison + XGBoost SHAP Baseline

## Quick Context Load

After you run the XGBoost commands, use this short read order to build
intuition before writing any comparison.

### 1. Start with the XGBoost baseline

Look in:

- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/figures/shap_xgb_beeswarm.pdf`
- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/figures/shap_xgb_dependence_*.pdf`
- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/data/shap_xgb_values.parquet`

What to learn:

- SHAP gives the baseline global ranking of important features.
- The dependence plots show the tree-based, post-hoc shape intuition for the
  top features.

### 2. Then inspect each KAN flavor's global ranking

Look in:

- `outputs/interpretability/kan_paper/stage-c-chebykan-best/figures/chebykan_feature_ranking.pdf`
- `outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_coefficient_importance.csv`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/figures/fourierkan_feature_ranking.pdf`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/data/fourierkan_coefficient_importance.csv`

What to learn:

- KAN importance is model-native and comes from first-layer coefficient
  magnitudes, not SHAP.
- First compare top-10 / top-15 feature overlap with SHAP.

### 3. Then inspect how the KANs express those features

Look in:

- `outputs/interpretability/kan_paper/stage-c-chebykan-best/figures/chebykan_activations.pdf`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/figures/fourierkan_activations.pdf`
- `outputs/interpretability/kan_paper/stage-c-chebykan-best/figures/chebykan_partial_dependence.pdf`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/figures/fourierkan_partial_dependence.pdf`

What to learn:

- `*_activations.pdf` shows the learned 1-D edge functions for top features.
- `*_partial_dependence.pdf` shows the model-level feature effect.
- Use the same feature names you saw near the top of the SHAP beeswarm.

### 4. Check whether the KAN explanations are trustworthy

Look in:

- `outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_symbolic_fits.csv`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/data/fourierkan_symbolic_fits.csv`
- `outputs/interpretability/kan_paper/stage-c-chebykan-best/reports/chebykan_r2_report.json`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/reports/fourierkan_r2_report.json`
- `outputs/interpretability/kan_paper/stage-c-chebykan-best/reports/chebykan_exact_closed_form.json`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/reports/fourierkan_exact_closed_form.json`

What to learn:

- `*_symbolic_fits.csv` tells you the per-edge recovered formula family.
- `*_r2_report.json` tells you whether the symbolic recovery is effectively
  exact at edge level.
- `*_exact_closed_form.json` tells you the full network is still not end-to-end
  exact because LayerNorm is present.

### 5. Use the feature-retention curves as the cleanest comparison

Look in:

- `outputs/interpretability/kan_paper/stage-c-chebykan-best/figures/feature_validation_curves.pdf`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/figures/feature_validation_curves.pdf`
- `outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_feature_validation_curves.json`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/data/fourierkan_feature_validation_curves.json`

What to learn:

- These curves show how much predictive signal survives when each model keeps
  only its top-ranked features.
- This is the most meaningful bridge between "ranking" and "usable
  interpretability."

### 6. Compare the three results in one pass

Use this checklist:

- Do SHAP, ChebyKAN, and FourierKAN surface the same top features?
- When they agree on a feature, do the KAN shapes look smoother or easier to
  read than the SHAP dependence plot?
- Does ChebyKAN or FourierKAN keep performance better under top-k feature
  restriction?
- Are the KAN edge recoveries near-perfect (`R² ≈ 1.0`)?
- Keep the final caveat explicit: KAN edge explanations are basis-native, but
  the full dense networks are not end-to-end exact closed forms because of
  LayerNorm.

Scope: only the two "large" TabKAN baselines in this repo:

- dense ChebyKAN: `stage-c-chebykan-best`
- dense FourierKAN: `stage-c-fourierkan-best`
- XGBoost + SHAP baseline: `stage-c-xgboost-best`

Reference methodology:
[TabKAN: Advancing Tabular Data Analysis using Kolmogorov-Arnold Network](papers/2504.06559v3.pdf)
(Ali Eslamian, Alireza Afzal Aghaei, Qiang Cheng; arXiv:2504.06559v3).
For this handoff, follow the paper's post-hoc interpretability framing for
the large models: paper-native coefficient importance, learned activation /
edge inspection, and feature-subset validation (the workflow aligned with the
paper's Section 5.7 and Figures 6-7).

Out of scope for this handoff:

- GLM
- sparse / top-k / no-LayerNorm hero models
- shared Table 1 coordination
- bootstrap CIs across dense and sparse rows

Locked reporting convention: if a single performance number is cited for
either large model, use **outer-test QWK** from the run summary / manifest,
not inner-val sweep numbers.

## Repo mapping

- In this repo, "large" means the dense stage-C best baselines, not the sparse
  or top-k retrains.
- dense ChebyKAN maps to
  [stage-c-chebykan-best](../../checkpoints/stage-c-chebykan-best) with
  hidden widths `[128, 64]`, degree `6`, LayerNorm enabled, and
  `recipe: kan_paper`.
- dense FourierKAN maps to
  [stage-c-fourierkan-best](../../configs/experiment_stages/stage_c_explanation_package/fourierkan_best.yaml)
  with hidden widths `[64, 256, 64]`, `grid_size: 8`, LayerNorm enabled, and
  `recipe: kan_paper`.

## Current audit

- Cyril has already run the three KAN commands for this narrowed scope:
  re-interpret dense Cheby from checkpoint, train dense Fourier, and interpret
  dense Fourier.
- The Fourier native symbolic fix is already present in staged code:
  [kan_symbolic.py](../../src/interpretability/kan_symbolic.py) and
  [formula_composition.py](../../src/interpretability/formula_composition.py).
- Dense Cheby training/eval already exists on the correct split/seed via
  [model-20260421-094010.manifest.json](../../checkpoints/stage-c-chebykan-best/model-20260421-094010.manifest.json)
  and
  [run-summary-20260421-094010.json](../../artifacts/stage-c-chebykan-best/run-summary-20260421-094010.json).
- Do **not** trust
  [chebykan_best.yaml](../../configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml)
  as the dense Cheby config right now: it now contains `sparsity_lambda: 0.1`,
  while the dense baseline checkpoint above is the intended `0.0` model.
- Existing dense-Cheby interpret outputs under
  [stage-c-chebykan-best-top20](../../outputs/interpretability/kan_paper/stage-c-chebykan-best-top20)
  are stale for this purpose after the native-fit change:
  [chebykan_symbolic_fits.csv](../../outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/data/chebykan_symbolic_fits.csv)
  has no `fit_mode` column and the report is still approximate, not
  basis-native.
- Dense Cheby refreshed outputs now exist under
  [stage-c-chebykan-best](../../outputs/interpretability/kan_paper/stage-c-chebykan-best).
- Dense Fourier now has a checkpoint at
  [model-20260423-184547.pt](../../checkpoints/stage-c-fourierkan-best/model-20260423-184547.pt).
- The current dense-Fourier interpretability directory exists, but at the
  moment it only contains the pruned module and pruning summary. Treat the
  Fourier train / interpret commands as already run, but the full paper-native
  artifact set still needs completion or audit before writing the comparison.
- XGBoost has a config
  [xgboost_best.yaml](../../configs/experiment_stages/stage_c_explanation_package/xgboost_best.yaml)
  but there is still no `stage-c-xgboost-best` checkpoint or interpretability
  output in this tree.
- The existing interpret pipeline already emits the paper-native artifacts
  needed for this narrowed scope:
  coefficient rankings, symbolic fits, per-edge `R²`, feature-validation
  curves, activation grids, and KAN network diagrams.
- XGBoost + SHAP is back in scope only as the baseline used to compare against
  the interpretability of the two large TabKAN flavors. GLM remains out of
  scope.

## PDP audit

Treat the current dense-Cheby PDP figure as a weak / partially misleading
supporting artifact until the plotting logic is fixed:

- [chebykan_partial_dependence.pdf](../../outputs/interpretability/kan_paper/stage-c-chebykan-best/figures/chebykan_partial_dependence.pdf)
  should not be used as clean evidence of model-level feature effects in its
  current form.
- The current PDP code in
  [partial_dependence.py](../../src/interpretability/partial_dependence.py)
  builds a `np.linspace(lo, hi, 100)` grid for every feature using the 1st and
  99th percentiles. That is acceptable for continuous variables, but it is
  wrong for discrete coded variables.
- In the current dense-Cheby top-20 PDP artifact, 14 of the 20 panels are
  binary or categorical. Those panels are being evaluated on impossible
  in-between values rather than only on observed states.
- Example failures from the current eval split:
  - `Medical_History_23` only takes `{1, 3}`, but the PDP line is drawn across
    the whole interval `1 → 3`.
  - `Medical_History_5` has counts `1:11789, 2:88`, so the 99th percentile is
    `1` and the panel collapses to one x-value.
  - `Medical_History_30` has counts `1:1, 2:11377, 3:499`, so the 1st
    percentile is `2` and category `1` disappears from the plot.
  - `Medical_History_18` has counts `1:11233, 2:641, 3:3`, so the 99th
    percentile is `2` and category `3` disappears from the plot.
- The x-axis semantics are also misleading for `recipe: kan_paper`.
  [preprocess_kan_paper.py](../../src/preprocessing/preprocess_kan_paper.py)
  does not externally scale inputs to `[-1,1]`; it keeps mostly raw/code-valued
  features and only imputes / adds missing indicators. The ChebyKAN and
  FourierKAN layers then apply `tanh(...)` internally. So labels such as
  `Encoded [-1,1]` in the current PDP figure are not faithful to the stored
  `X_eval` artifact.
- Working interpretation rule for this handoff:
  prefer `*_activations.pdf` plus feature-retention validation as the main
  KAN-side evidence. Treat the current PDPs as secondary illustrations only,
  not as the primary shape summary.
- If the PDP figure is revisited later, discrete features should be evaluated
  on observed states only, percentile clipping should not remove valid
  categories, and the axis should explicitly distinguish external preprocessing
  scale from the layer-internal `tanh` normalization.

## Ordered tasks

### 1. Re-run dense Cheby interpretability on the existing dense checkpoint

Status: already run by Cyril.

Use the existing checkpoint directly, not the current YAML:

```bash
uv run python main.py --stage interpret \
  --checkpoint checkpoints/stage-c-chebykan-best/model-20260421-094010.pt
```

Acceptance:
- `outputs/interpretability/kan_paper/stage-c-chebykan-best/` exists.
- `data/chebykan_symbolic_fits.csv` contains a `fit_mode` column and the active rows are native (`chebykan_native`).
- `reports/chebykan_r2_report.json` and/or the CSV show mean per-edge `R² ≈ 1.0`.
- `data/chebykan_coefficient_importance.csv` exists.
- `figures/chebykan_activations.pdf` exists.
- `data/chebykan_feature_validation_curves.json` and `figures/feature_validation_curves.pdf` exist.
- If a scalar performance number is cited for dense Cheby, take it from
  [run-summary-20260421-094010.json](../../artifacts/stage-c-chebykan-best/run-summary-20260421-094010.json),
  not from inner-val sweep numbers.

### 2. Train and interpret dense Fourier

Status: both commands already run by Cyril; current outputs still need audit
because the interpretability directory is incomplete.

```bash
uv run python main.py --stage train \
  --config configs/experiment_stages/stage_c_explanation_package/fourierkan_best.yaml

uv run python main.py --stage interpret \
  --config configs/experiment_stages/stage_c_explanation_package/fourierkan_best.yaml
```

Acceptance:
- `checkpoints/stage-c-fourierkan-best/` exists.
- The run summary confirms `seed: 42` and `recipe: kan_paper`.
- `outputs/interpretability/kan_paper/stage-c-fourierkan-best/` exists.
- `data/fourierkan_symbolic_fits.csv` contains `fit_mode=fourierkan_native`.
- Mean per-edge `R² ≈ 1.0`.
- `data/fourierkan_coefficient_importance.csv` exists.
- `figures/fourierkan_activations.pdf` exists.
- `data/fourierkan_feature_validation_curves.json` and `figures/feature_validation_curves.pdf` exist.
- `reports/fourierkan_exact_closed_form.json` correctly reports `exact_available=false` because LayerNorm is present.

### 3. Train and interpret the XGBoost baseline used for SHAP

```bash
uv run python main.py --stage train \
  --config configs/experiment_stages/stage_c_explanation_package/xgboost_best.yaml

uv run python main.py --stage interpret \
  --config configs/experiment_stages/stage_c_explanation_package/xgboost_best.yaml
```

Acceptance:
- `checkpoints/stage-c-xgboost-best/` exists.
- The run summary confirms `seed: 42` and `recipe: xgboost_paper`.
- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/` exists.
- `data/shap_xgb_values.parquet` exists.
- `figures/shap_xgb_beeswarm.pdf` exists.

### 4. Produce the comparison note

Goal: compare the interpretability of the two large TabKAN flavors against the
XGBoost + SHAP baseline, while using the TabKAN paper's post-hoc
interpretability framing for the KAN side.

Inputs:

- [shap_xgb_values.parquet](../../outputs/interpretability/xgboost_paper/stage-c-xgboost-best/data/shap_xgb_values.parquet)
- [chebykan_coefficient_importance.csv](../../outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_coefficient_importance.csv)
- [fourierkan_coefficient_importance.csv](../../outputs/interpretability/kan_paper/stage-c-fourierkan-best/data/fourierkan_coefficient_importance.csv)
- [chebykan_symbolic_fits.csv](../../outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_symbolic_fits.csv)
- [fourierkan_symbolic_fits.csv](../../outputs/interpretability/kan_paper/stage-c-fourierkan-best/data/fourierkan_symbolic_fits.csv)
- `figures/shap_xgb_beeswarm.pdf`
- `figures/chebykan_activations.pdf` and `figures/fourierkan_activations.pdf`
- `data/chebykan_feature_validation_curves.json` and
  `data/fourierkan_feature_validation_curves.json`
- the three run summaries / manifests for outer-test QWK

Method:

- Treat XGBoost + SHAP as the external baseline.
- Compare SHAP feature ranking against the paper-native first-layer coefficient
  rankings from ChebyKAN and FourierKAN.
- Use the per-run feature-validation curves as the paper-aligned evidence for
  whether the ranking preserves predictive signal when retaining only the top-k
  features.
- Use the activation grids and symbolic-fit tables to compare how the two large
  models express their dominant features.
- Keep GLM out of this note.

Acceptance:

- One small artifact exists that the paper text can quote directly.
- It reports, at minimum:
  - the XGBoost + SHAP baseline alongside dense ChebyKAN and dense FourierKAN
  - outer-test QWK for dense ChebyKAN and dense FourierKAN
  - the SHAP-based top features used as the baseline ranking
  - total active edge counts for the two dense models
  - mean per-edge `R²` for both native symbolic recoveries
  - a comparison of top-ranked SHAP features against the two KAN coefficient
    rankings
  - a short statement that end-to-end exact closed form is unavailable for both
    large models because LayerNorm is present
- It cites the local TabKAN paper PDF above as the methodological reference.

## Practical notes

- For dense Cheby, prefer
  `--checkpoint checkpoints/stage-c-chebykan-best/model-20260421-094010.pt`
  over the current YAML to avoid accidentally using the wrong
  `sparsity_lambda`.
- When a downstream interpretability script requires a Cheby config path, use
  [chebykan_best_top20.yaml](../../configs/experiment_stages/stage_c_explanation_package/chebykan_best_top20.yaml)
  as the architecture proxy for the `0.0` dense checkpoint. The filename is
  stale, but the model parameters match the dense baseline.
- The existing dense-Cheby interpretability directory under
  `stage-c-chebykan-best-top20` is useful only as a reminder of expected output
  filenames, not as paper-ready evidence.
- `feature_validation_curves.pdf` is already emitted by the interpret pipeline
  inside each model's output directory; reuse that artifact instead of routing
  through the SHAP comparison scripts.
- Do not claim the current dense models are end-to-end exact symbolic formulas.
  The edge-level recovery is exact / basis-native, but the full network still
  contains LayerNorm.
