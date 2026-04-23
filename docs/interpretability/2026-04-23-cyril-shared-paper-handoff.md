# Cyril Handoff — Large TabKAN Two-Flavor Comparison

Scope: only the two "large" TabKAN baselines in this repo:

- dense ChebyKAN: `stage-c-chebykan-best`
- dense FourierKAN: `stage-c-fourierkan-best`

Reference methodology:
[TabKAN: Advancing Tabular Data Analysis using Kolmogorov-Arnold Network](papers/2504.06559v3.pdf)
(Ali Eslamian, Alireza Afzal Aghaei, Qiang Cheng; arXiv:2504.06559v3).
For this handoff, follow the paper's post-hoc interpretability framing for
the large models: paper-native coefficient importance, learned activation /
edge inspection, and feature-subset validation (the workflow aligned with the
paper's Section 5.7 and Figures 6-7).

Out of scope for this handoff:

- XGBoost / SHAP
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
- Dense Fourier has a config
  [fourierkan_best.yaml](../../configs/experiment_stages/stage_c_explanation_package/fourierkan_best.yaml)
  but no checkpoint, no eval artifacts, and no interpretability outputs in
  this tree yet.
- The existing interpret pipeline already emits the paper-native artifacts
  needed for this narrowed scope:
  coefficient rankings, symbolic fits, per-edge `R²`, feature-validation
  curves, activation grids, and KAN network diagrams.
- [comparison_per_risk.py](../../src/interpretability/comparison_per_risk.py),
  [comparison_side_by_side.py](../../src/interpretability/comparison_side_by_side.py),
  and the SHAP / GLM comparison path are not required for this KAN-only
  handoff.

## Ordered tasks

### 1. Re-run dense Cheby interpretability on the existing dense checkpoint

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

### 3. Produce the KAN-only comparison note

Goal: compare the two large TabKAN flavors using the TabKAN paper's
post-hoc interpretability outputs, not SHAP / GLM baselines.

Inputs:

- [chebykan_coefficient_importance.csv](../../outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_coefficient_importance.csv)
- [fourierkan_coefficient_importance.csv](../../outputs/interpretability/kan_paper/stage-c-fourierkan-best/data/fourierkan_coefficient_importance.csv)
- [chebykan_symbolic_fits.csv](../../outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_symbolic_fits.csv)
- [fourierkan_symbolic_fits.csv](../../outputs/interpretability/kan_paper/stage-c-fourierkan-best/data/fourierkan_symbolic_fits.csv)
- `figures/chebykan_activations.pdf` and `figures/fourierkan_activations.pdf`
- `data/chebykan_feature_validation_curves.json` and
  `data/fourierkan_feature_validation_curves.json`
- the two run summaries / manifests for outer-test QWK

Method:

- Compare the paper-native first-layer coefficient rankings between ChebyKAN
  and FourierKAN directly.
- Use the per-run feature-validation curves as the paper-aligned evidence for
  whether the ranking preserves predictive signal when retaining only the top-k
  features.
- Use the activation grids and symbolic-fit tables to compare how the two large
  models express their dominant features.
- Do **not** bring XGBoost, SHAP, or GLM back into this note unless the scope is
  reopened explicitly.

Acceptance:

- One small artifact exists that the paper text can quote directly.
- It reports, at minimum:
  - outer-test QWK for dense ChebyKAN and dense FourierKAN
  - total active edge counts for the two dense models
  - mean per-edge `R²` for both native symbolic recoveries
  - a comparison of top-ranked features from the two coefficient rankings
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
