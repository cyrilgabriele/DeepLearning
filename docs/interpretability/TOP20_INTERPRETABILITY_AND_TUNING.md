# 20-Feature Models: Interpretability & Tuning

How each model's top-20 feature list was derived and how the resulting 20-feature
models were tuned. Each row in Table 2 of the paper corresponds to one of the
three models below, and every step is meant to be reproducible from the
checked-in configs.

---

## Guiding principle

Each model's top-20 list is derived from **the same implementation that ends up
being reported**. No mixing across model classes (e.g. SHAP-from-XGBClassifier
applied to XGBRegressor) and no mixing across architectures (e.g. Pareto-sweep
features applied to a different Optuna-tuned KAN). This avoids the silent
methodology drift a reviewer would (rightly) push back on.

Source: `docs/CHANGELOG.md` entries from 2026-04-24 / 2026-04-25.

---

## Step 1 — Derive each model's top-20 feature list

| Model | Importance signal | Trained-on config | Output feature list |
|---|---|---|---|
| XGBoost | mean(\|SHAP\|) via `TreeExplainer` on the full-140 `xgb-best` checkpoint (XGBRegressor + threshold calibration) | `configs/experiment_stages/stage_c_explanation_package/xgb_best.yaml` | `configs/experiment_stages/stage_c_explanation_package/feature_lists/xgb_tuned_top20_features.json` |
| ChebyKAN | first-layer Chebyshev-coefficient magnitudes on the tuned full-140 sparse run | `configs/experiment_stages/stage_c_explanation_package/chebykan_tuned_sparse_fullft.yaml` | `configs/experiment_stages/stage_c_explanation_package/feature_lists/chebykan_tuned_top20_features.json` |
| FourierKAN | first-layer Fourier-coefficient magnitudes on the tuned full-140 run | `configs/experiment_stages/stage_c_explanation_package/fourierkan_tuned_sparse_fullft.yaml` | `configs/experiment_stages/stage_c_explanation_package/feature_lists/fourierkan_tuned_top20_features.json` |

Why this signal: SHAP is the standard for tree models; for KANs the per-edge
coefficient magnitude is the model's *native* importance — no surrogate, no
sampling. Each list is then materialised as JSON and consumed by the
preprocessor via `preprocessing.selected_features_path` in the configs below.

Computation entry points:
- XGBoost SHAP: `src/interpretability/shap_xgboost.py`
- KAN coefficient ranking & pruning: `src/interpretability/kan_pruning.py`,
  feature ranking written by `src/interpretability/pipeline.py`

---

## Step 2 — Tune the 20-feature models (Optuna, 50 trials, TPE)

For each model we ran a separate per-budget Optuna sweep restricted to its own
top-20 list. Search spaces were equalised across the two KAN flavors so neither
gets an asymmetric tuning advantage.

| Model | Tune config | Sweep storage |
|---|---|---|
| XGBoost | `configs/experiment_stages/stage_c_explanation_package/xgb_top20_tune.yaml` | `sweeps/stage-c-xgb-top20-tune.db` |
| ChebyKAN | `configs/experiment_stages/stage_c_explanation_package/chebykan_top20_tune.yaml` | `sweeps/stage-c-chebykan-top20-tune.db` |
| FourierKAN | `configs/experiment_stages/stage_c_explanation_package/fourierkan_top20_tune.yaml` | `sweeps/stage-c-fourierkan-top20-tune.db` |

Sweep driver: `src/tune/sweep.py`.

What the search spaces cover:
- **XGBoost**: `n_estimators`, `max_depth`, `min_child_weight`, `learning_rate`,
  `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `gamma`.
  `gamma` was added explicitly so XGBoost is not crippled relative to its
  full-140 baseline (see CHANGELOG 2026-04-24: a real bug previously dropped
  `gamma` silently).
- **ChebyKAN / FourierKAN**: `depth`, `width`, `degree` (Cheby) or `grid_size`
  (Fourier), `max_epochs`, `lr`, `weight_decay`, `batch_size`. Sparsity λ is
  fixed per-flavor at the value chosen from the prior sparsity scan.

Objective: outer-fold QWK (threshold-calibrated, same evaluator used in §3.1).

---

## Step 3 — Retrain the winners on the 20-feature subset

After tuning, the best trial of each sweep is materialised as a stand-alone
training config and retrained once on the full training data. These are the
exact runs whose numbers appear in Table 2.

| Model | Final retrain config |
|---|---|
| XGBoost | `configs/experiment_stages/stage_c_explanation_package/xgb_top20.yaml` |
| ChebyKAN (sparse hero) | `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml` (legacy filename — wired to the new feature list and tuned hyperparameters) |
| FourierKAN | `configs/experiment_stages/stage_c_explanation_package/fourierkan_pareto_top20_noln.yaml` |

Why the LayerNorm-off (`use_layernorm: false`) variants are used for the KANs:
the symbolic-recovery story in §3.2 only holds when the network is a clean
composition of KAN edges. LayerNorm injects per-layer affine statistics that
break the closed-form composition; turning it off is what lets us claim a
single closed-form expression for the sparse ChebyKAN.

Sparsity-λ values fixed at this stage: ChebyKAN λ = 0.0108, FourierKAN
λ = 0.0249. Both came from the earlier per-flavor Pareto sparsity scans
(`chebykan_pareto_sparsity.yaml`, `fourierkan_pareto_sparsity.yaml`) — see
`local_files/interpretability_for_paper/INTERPRETABILITY_RESULTS_SUMMARY.md`
for the full Pareto tables.

---

## Step 4 — Interpretability stage on the retrained KANs

Run on the retrained 20-feature KANs to produce the symbolic + waterfall
artifacts. Pipeline entry point: `src/interpretability/pipeline.py`.

```bash
uv run python main.py \
  --stage interpret \
  --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml \
  --pruning-threshold 0.001 \
  --qwk-tolerance 0.01 \
  --candidate-library scipy \
  --max-features 20
```

Key sub-steps and the file each lives in:

| Step | File | What it does |
|---|---|---|
| L1 / variance pruning | `src/interpretability/kan_pruning.py` | Zeroes edges with output variance below `--pruning-threshold`; auto-tightens if QWK drop exceeds `--qwk-tolerance`. |
| Symbolic edge fitting | `src/interpretability/kan_symbolic.py` | Fits each surviving 1D edge against an 11-candidate scipy library (linear, quadratic, cubic, abs, sqrt, log, exp, sin, cos, sin2x, constant); reports per-edge R². |
| Closed-form composition | `src/interpretability/closed_form_surrogate.py`, `src/interpretability/formula_composition.py` | Composes the surviving symbolic edges into a single end-to-end expression. Used **only** for the LayerNorm-off configs above (composition is exact there). |
| Integrated-gradient waterfall (Fig 3) | `scripts/build_figure3_waterfall.py` | IG decomposition of one applicant (55728) against the lowest-class-1 reference (44235) — sums to `f(applicant) − f(reference)` exactly by construction. |
| Layer-0 edge atlas + PDPs | `src/interpretability/partial_dependence.py`, `src/interpretability/kan_network_diagram.py` | Per-feature learned-curve plots used in the appendix. |

Outputs are written under
`outputs/interpretability/kan_paper/<experiment_name>/` with sub-folders
`data/`, `reports/`, `figures/`, `models/`.

---

## Where the 95% intervals come from

The `[lo, hi]` values next to each QWK in Table 2 are a **non-parametric
bootstrap on outer-test predictions**, produced by
`scripts/bootstrap_qwk_table1.py`.

Recipe:

1. **Predict once on the held-out outer-test set** — `predict_on_outer_test()`
   at `scripts/bootstrap_qwk_table1.py:147`. Loads
   `outputs/eval/<recipe>/<experiment>/{X_eval,y_eval}.parquet`, runs the saved
   checkpoint forward, then applies the calibrated ordinal thresholds from the
   run's manifest (`_apply_thresholds`, line 79). Output: one `(y_true, y_pred)`
   pair per model, `n = 11 877`.
2. **Bootstrap the QWK** — `bootstrap_qwk()` at line 177:
   - `n_boot = 1000` index resamples with replacement (seeded, reproducible).
   - Per resample: `cohen_kappa_score(y_true[idx], y_pred[idx], weights="quadratic")`.
   - 95% CI = 2.5% / 97.5% quantiles of the 1000 QWKs.
   - Point estimate = QWK on the un-resampled pair (not the bootstrap mean).
3. **Persisted output** — `outputs/reports/table1_bootstrap_qwk.json`. Table 2
   reads its CI strings directly from this file.

Caveats:

- The bootstrap captures sampling variability of the **held-out evaluation
  set**, not model-training variability or Optuna selection variance.
- Training-stochasticity is reported separately via multi-seed runs (3 seeds
  each, e.g. dense ChebyKAN `0.595 ± 0.027`; see CHANGELOG 2026-04-24).
- Thresholds applied before scoring come from the same ordinal calibration used
  during training (`manifest["ordinal_calibration"]["thresholds"]`), so the CI
  is on threshold-calibrated integer predictions, matching §3.1's QWK
  definition.

---

## Quick reproduction checklist

1. Train the three full-140 source models (`xgb_best.yaml`,
   `chebykan_tuned_sparse_fullft.yaml`, `fourierkan_tuned_sparse_fullft.yaml`).
2. Extract top-20 lists from each (SHAP for XGBoost, coefficient magnitude for
   the KANs) → JSON files under `feature_lists/`.
3. Run the three Optuna sweeps (`*_top20_tune.yaml`, 50 trials each).
4. Retrain the winners (`xgb_top20.yaml`, `chebykan_pareto_q0583_top20.yaml`,
   `fourierkan_pareto_top20_noln.yaml`).
5. Run `--stage interpret` on each retrained KAN to get the symbolic +
   waterfall artifacts.
6. Bootstrap CIs over outer-test predictions:
   `scripts/bootstrap_qwk_table1.py` → `outputs/reports/table1_bootstrap_qwk.json`.
