# Human Action Needed: Interpretability Pipeline -> Paper

This document lists everything that requires **human judgment, GPU time, or manual intervention** to get from the current interpretability pipeline to a finished paper. Items are ordered by priority.

**Pipeline status after merge (2026-04-09):** Gian's interpretability analysis modules + Cyril's multi-stage infrastructure (retrain/select/config restructuring) are now unified on `interpretability-refactoring-gian`. Config lives at `src/config/`, data at `src/preprocessing/`, YAML configs under `configs/experiment_stages/`.

---

## 1. Retrain Models with Sparsity Regularization (CRITICAL)

**Why:** The current checkpoints were trained with `sparsity_lambda: 0.0`. Post-hoc pruning on a dense model is not the same as pruning a model trained with sparsity induction. A reviewer will ask: "did you train with the sparsity regularization described in Liu et al. (2024)?"

**What to do:**

Use the new **retrain stage** from the merged pipeline. This retrains candidates across multiple seeds and produces a manifest for the selection stage.

1. First, ensure you have a candidate manifest from a tune run (or create one manually).
2. Retrain with sparsity:
```bash
uv run python main.py --stage retrain \
    --candidate-manifest artifacts/tune/chebykan/candidates.json \
    --seeds 42 123 456 \
    --selection-name sparsity-experiment
```
3. Repeat for FourierKAN.
4. If QWK drops significantly, try `sparsity_lambda: 0.0005` or `0.0001`.

**What to record:**
- Final QWK (mean +/- std across seeds) for each sparsity lambda
- Sparsity ratio after pruning at threshold=0.01
- Whether the sparse-trained model prunes more cleanly (more "clean" edges, fewer "flagged")

**Expected time:** ~2-4 hours GPU per model.

---

## 2. Run Selection Stage on Retrained Models (CRITICAL)

**Why:** The selection stage picks best-performance and best-interpretable candidates per KAN family. This is needed before the interpretability pipeline can run.

**What to do:**
```bash
uv run python main.py --stage select \
    --retrain-manifest artifacts/retrain/chebykan/sparsity-experiment/manifest.json \
    --qwk-tolerance 0.01
```

This produces a selection manifest at `artifacts/selection/chebykan_selection.json` with:
- `best_performance_candidate` (highest mean QWK)
- `best_interpretable_candidate` (fewest edges within QWK tolerance)

Repeat for FourierKAN.

---

## 3. Run Full Interpretability Pipeline on Selected Models (HIGH)

**Why:** The interpretability modules (formula composition, KAN diagram, R^2 histogram, Pareto curve) are implemented and tested but have not been run on actual sparse-trained models.

**What to do:**
After steps 1-2, run interpretation on the selected checkpoints:
```bash
uv run python main.py --stage interpret \
    --config configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_interpretable.yaml \
    --pruning-threshold 0.01

uv run python main.py --stage interpret \
    --config configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_interpretable.yaml \
    --pruning-threshold 0.01
```

Then review generated artifacts in `outputs/interpretability/<recipe>/<experiment>/`:
- `figures/{flavor}_kan_diagram.pdf` -- Does the KAN diagram look publication-ready?
- `figures/{flavor}_r2_distribution.pdf` -- What's the quality tier distribution?
- `reports/{flavor}_symbolic_formulas.md` -- Are the composed formulas reasonable?
- `reports/{flavor}_symbolic_formulas.json` -- What's the end-to-end R^2?

**Human judgment needed:**
- Do the composed symbolic formulas make domain sense for insurance risk prediction?
- Are the learned feature functions consistent with prior knowledge (e.g., BMI, age)?
- Which features dominate -- is this expected?

---

## 4. Sparsity Lambda Ablation Sweep (HIGH)

**Why:** The paper should report how `sparsity_lambda` affects the sparsity-performance tradeoff.

**What to do:**
1. Train ChebyKAN with `sparsity_lambda` in {0, 0.0001, 0.0005, 0.001, 0.005, 0.01}
2. For each, record:
   - Final validation QWK (mean +/- std across seeds)
   - Edge density (fraction of edges with L1 > 0.01)
   - Mean R^2 of symbolic fits
   - Number of clean/acceptable/flagged edges
3. Plot: QWK vs. sparsity_lambda (with edge density on secondary y-axis)

**Expected time:** ~6-12 hours GPU total (6 runs x multiple seeds).

**Paper figure:** "Effect of sparsity regularization on model performance and interpretability"

---

## 5. Run Feature Subset Validation (HIGH)

**Why:** This is the TabKAN paper's primary empirical validation of feature importance (Section 5.7, Figures 6-7). The code is implemented (`feature_validation.py`) but is not yet called from the main pipeline -- it must be invoked separately.

**What to do:**
1. Load all 4 trained models (GLM, XGBoost, ChebyKAN, FourierKAN)
2. Extract their native feature importance rankings
3. Run the validation curve computation:
```python
from src.interpretability.feature_validation import run as run_validation

# Build rankings dict and model_predict dict for all 4 models
# (see final_comparison.py for patterns on how to load each model)
run_validation(rankings, model_predict, X_eval, y_eval, output_dir)
```

**Human judgment needed:**
- Compare KAN-native ranking vs. SHAP ranking -- which maintains QWK better at 50% features?
- This is a key finding: "KAN coefficient-based feature importance outperforms/matches SHAP"

---

## 6. Run Pruning Pareto Curve (MEDIUM)

**Why:** Shows the sparsity-performance tradeoff and helps identify the optimal pruning threshold.

**What to do:**
```python
from src.interpretability.quality_figures import compute_pruning_pareto, plot_pruning_pareto

pareto_data = {}
for flavor in ["chebykan", "fourierkan"]:
    pareto_data[flavor] = compute_pruning_pareto(
        module, config, flavor,
        eval_features_path, eval_labels_path,
        thresholds=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
    )
plot_pruning_pareto(pareto_data, output_dir)
```

**Human judgment needed:**
- Where is the "knee" of the curve? (Point where further pruning causes sharp QWK drop)
- Is theta=0.01 a good default, or should a different threshold be recommended?

---

## 7. PySR Runs for Complex Edges (LOW)

**Why:** Some edges have R^2 < 0.90 with the scipy library, meaning the fixed candidate set can't describe them. PySR (evolutionary symbolic regression) could find novel formulas.

**What to do:**
1. Install Julia and PySR: `pip install pysr && python -c "import pysr; pysr.install()"`
2. Re-run symbolic regression with PySR enabled:
```bash
uv run python -m src.interpretability.kan_symbolic \
    --pruned-checkpoint outputs/interpretability/<recipe>/<experiment>/models/chebykan_pruned_module.pt \
    --pruning-summary outputs/interpretability/<recipe>/<experiment>/reports/chebykan_pruning_summary.json \
    --config configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_interpretable.yaml \
    --eval-features outputs/eval/<recipe>/<experiment>/X_eval.parquet \
    --flavor chebykan \
    --use-pysr
```
3. Compare: how many edges improve from flagged/acceptable to clean with PySR?

**Expected time:** ~30-60 minutes per model (PySR is CPU-intensive).

---

## 8. Degree/Grid-Size Ablation (LOW)

**Why:** The paper should justify the choice of ChebyKAN degree=3 and FourierKAN grid_size=4.

**What to do:**
- Train ChebyKAN with degree in {2, 3, 4, 5}
- Train FourierKAN with grid_size in {2, 4, 6, 8}
- For each, record QWK and symbolic fit quality

**Expected time:** ~8-16 hours GPU total.

**Paper contribution:** "Higher-degree polynomials did not improve symbolic recovery" (or "did improve -- here's the evidence")

---

## 9. Paper Writing (HUMAN ONLY)

The pipeline produces data and figures. The narrative requires domain expertise and scientific judgment.

### 9.1 Methodology Section
- Describe the full pipeline: tune -> retrain (multi-seed) -> select -> interpret (prune -> symbolic regression -> lock-in -> compose)
- Reference Liu et al. (2024) and TabKAN (2025) as methodological sources
- Explain the BIC-penalized candidate selection
- Document the three quality tiers (clean >= 0.99, acceptable >= 0.90, flagged < 0.90)
- Describe the multi-seed retraining and selection procedure

### 9.2 Results Section
- Report QWK (mean +/- std) for all 4 models
- Report sparsity ratios and symbolic fit quality
- Present the feature validation curves (TabKAN Figures 6-7 equivalent)
- Present the pruning Pareto curves
- Show the KAN network diagrams with edge functions
- Show composed symbolic formulas and discuss their domain meaning
- Compare feature importance rankings across models (Kendall tau)

### 9.3 Discussion Points
- Do KAN-native importance rankings agree with domain knowledge?
- Do the composed symbolic formulas offer actionable insights for insurance risk?
- Is the interpretability gain worth the added complexity vs. XGBoost+SHAP?
- What fraction of the model remains "non-symbolic" (flagged edges)?
- How does sparsity regularization affect the interpretability-performance tradeoff?

---

## Figures Checklist for Paper

All figures are automatically generated by the pipeline:

| Figure | File | Status |
|--------|------|--------|
| KAN network diagram with edge functions | `{flavor}_kan_diagram.pdf` | Implemented |
| Before/after pruning comparison | `{flavor}_before_after_pruning.pdf` | Implemented |
| R^2 distribution histogram + tier pie | `{flavor}_r2_distribution.pdf` | Implemented |
| Pruning Pareto curve | `pruning_pareto_curve.pdf` | Implemented |
| Feature validation curves | `feature_validation_curves.pdf` | Implemented |
| Feature ranking comparison | `feature_ranking_comparison.pdf` | Implemented |
| Composed symbolic formulas | `{flavor}_symbolic_formulas.md` | Implemented |
| Activation function grid (top-10) | `{flavor}_activations.pdf` | Implemented |
| Feature importance bar chart | `{flavor}_feature_ranking.pdf` | Implemented |
| QWK retention curves | `qwk_feature_retention.pdf` | Implemented |
| Comparison matrix table | `final_comparison_matrix.pdf` | Implemented |
| Per-risk feature importance | `per_risk_level_comparison.pdf` | Implemented |
| Side-by-side model comparison | `side_by_side_{flavor}.pdf` | Implemented |
| Pruned network graph | `{flavor}_pruned_network.pdf` | Implemented |

---

## Summary: Code vs. Human Action

| Component | Code Status | Human Action |
|-----------|-------------|--------------|
| Multi-seed retraining pipeline | Done (merged from Cyril) | Run with sparsity lambda |
| Model selection (best-perf / best-interp) | Done (merged from Cyril) | Execute on retrained models |
| Edge pruning with QWK protection | Done | None |
| Symbolic regression (scipy + PySR) | Done | Run PySR (optional) |
| Symbolic lock-in | Done | None |
| Formula composition (SymPy) | Done | Review formulas for domain sense |
| Feature importance (paper-native) | Done | None |
| Cross-architecture normalization | Done | None |
| R^2 distribution figure | Done | None |
| KAN network diagram | Done | None |
| Pruning Pareto curve | Done | Run on actual models |
| Feature validation curves | Done | Run on actual models |
| Sparsity regularization | Config ready | **Retrain models** |
| Sparsity ablation | Not needed in code | **Run sweep, record results** |
| Degree/grid-size ablation | Not needed in code | **Run sweep, record results** |
| Non-uniform hidden widths | Done (resolved_hidden_widths) | None |
| Experiment stage configs (A/B/C) | Done (merged from Cyril) | Fill in materialized configs |
| Paper writing | N/A | **Write methods, results, discussion** |

---

## Known Gaps in Current Code

These are not blockers but should be addressed if time permits:

1. **Feature validation not in main pipeline** -- `feature_validation.py` must be invoked separately; consider integrating into the interpret stage or adding a `--validate-features` flag.
2. **SHAP figures are PNG, not PDF** -- `shap_xgboost.py` outputs PNG; should switch to PDF for publication consistency.
3. **Final comparison not in main pipeline** -- `final_comparison.py` must be invoked separately after all 4 models are interpreted.
4. **Formula composition coverage metric** -- Excludes flagged edges (R^2 < 0.90) from composition. If many edges are flagged, the composed formula underrepresents model complexity. Document this limitation in the paper.
5. **Per-risk KAN importance is global** -- `comparison_per_risk.py` shows KAN importance identically across all 8 risk panels because KAN coefficients are not conditioned on output class. Should be clearly labeled.
