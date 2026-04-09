# Human Action Needed: Interpretability Pipeline → Paper

This document lists everything that requires **human judgment, GPU time, or manual intervention** to get from the current interpretability pipeline to a finished paper. Items are ordered by priority.

---

## 1. Retrain Models with Sparsity Regularization (CRITICAL)

**Why:** The configs now have `sparsity_lambda: 0.001`, but the models in `checkpoints/` were trained with `sparsity_lambda: 0.0`. The current pruning operates on dense models that were never encouraged to become sparse during training. This fundamentally changes the scientific claim — post-hoc pruning on a dense model ≠ pruning an inherently sparse model.

**What to do:**
1. Retrain ChebyKAN with `sparsity_lambda: 0.001` (config already set)
2. Retrain FourierKAN with `sparsity_lambda: 0.001` (config already set)
3. Compare QWK of sparse-trained vs. dense-trained models
4. If QWK drops significantly, try `sparsity_lambda: 0.0005` or `0.0001`

**Expected time:** ~2-4 hours GPU time per model (depending on hardware and k-fold setup)

**What to record:**
- Final QWK for each sparsity lambda
- Sparsity ratio after pruning at threshold=0.01
- Whether the sparse-trained model prunes more cleanly (fewer "acceptable" edges, more "clean")

**Why you can't skip this:** A reviewer will ask "did you train with the sparsity regularization described in Liu et al. (2024)?" The answer must be yes.

---

## 2. Sparsity Lambda Ablation Sweep (HIGH)

**Why:** The paper should report how `sparsity_lambda` affects the sparsity-performance tradeoff.

**What to do:**
1. Train ChebyKAN with `sparsity_lambda` ∈ {0, 0.0001, 0.0005, 0.001, 0.005, 0.01}
2. For each, record:
   - Final validation QWK
   - Edge density (fraction of edges with L1 > 0.01)
   - Mean R² of symbolic fits
   - Number of clean/acceptable/flagged edges
3. Plot: QWK vs. sparsity_lambda (with edge density on secondary y-axis)

**Expected time:** ~6-12 hours GPU total (6 runs)

**Paper figure:** "Effect of sparsity regularization on model performance and interpretability"

---

## 3. Run the Full Interpretability Pipeline End-to-End (HIGH)

**Why:** The new modules (formula composition, R² histogram, KAN diagram, Pareto curve) are implemented and tested, but have not been run on actual trained models.

**What to do:**
After retraining with sparsity (item 1):
```bash
uv run python main.py --stage interpret \
    --config configs/model/chebykan_experiment.yaml \
    --pruning-threshold 0.01

uv run python main.py --stage interpret \
    --config configs/model/fourierkan_experiment.yaml \
    --pruning-threshold 0.01
```

Then review the generated artifacts in `outputs/interpretability/`:
- `figures/{flavor}_kan_diagram.pdf` — Does the KAN diagram look publication-ready?
- `figures/{flavor}_r2_distribution.pdf` — What's the quality tier distribution?
- `reports/{flavor}_symbolic_formulas.md` — Are the composed formulas reasonable?
- `reports/{flavor}_symbolic_formulas.json` — What's the end-to-end R²?

**Human judgment needed:**
- Do the composed symbolic formulas make domain sense for insurance risk prediction?
- Are the learned feature functions consistent with prior knowledge (e.g., BMI, age)?
- Which features dominate — is this expected?

---

## 4. Run Feature Subset Validation (HIGH)

**Why:** This is the TabKAN paper's primary empirical validation of feature importance. The code is implemented (`feature_validation.py`) but needs to be run with actual model checkpoints.

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
- Compare KAN-native ranking vs. SHAP ranking — which maintains QWK better at 50% features?
- This is a key finding for the paper: "KAN coefficient-based feature importance outperforms/matches SHAP"

---

## 5. Run Pruning Pareto Curve (MEDIUM)

**Why:** Shows the sparsity-performance tradeoff and helps identify the optimal pruning threshold.

**What to do:**
```python
from src.interpretability.quality_figures import compute_pruning_pareto, plot_pruning_pareto

# For each KAN flavor, compute Pareto data
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
- Is θ=0.01 a good default, or should a different threshold be recommended?

---

## 6. PySR Runs for Complex Edges (LOW)

**Why:** Some edges have R² < 0.90 with the scipy library, meaning the fixed candidate set can't describe them. PySR (evolutionary symbolic regression) could find novel formulas.

**What to do:**
1. Install Julia and PySR: `pip install pysr && python -c "import pysr; pysr.install()"`
2. Re-run symbolic regression with PySR enabled:
```bash
uv run python -m src.interpretability.kan_symbolic \
    --pruned-checkpoint outputs/models/chebykan_pruned_module.pt \
    --pruning-summary outputs/reports/chebykan_pruning_summary.json \
    --config configs/model/chebykan_experiment.yaml \
    --eval-features outputs/data/X_eval.parquet \
    --flavor chebykan \
    --use-pysr
```
3. Compare: how many edges improve from flagged/acceptable to clean with PySR?

**Expected time:** ~30-60 minutes per model (PySR is CPU-intensive)

**Why this is low priority:** The scipy library captures most common functional forms. PySR is incremental improvement. But it strengthens the "we tried everything" claim.

---

## 7. Degree/Grid-Size Ablation (LOW)

**Why:** The paper should justify the choice of ChebyKAN degree=3 and FourierKAN grid_size=4.

**What to do:**
- Train ChebyKAN with degree ∈ {2, 3, 4, 5}
- Train FourierKAN with grid_size ∈ {2, 4, 6, 8}
- For each, record QWK and symbolic fit quality

**Expected time:** ~8-16 hours GPU total

**Paper contribution:** "Higher-degree polynomials did not improve symbolic recovery" (or "did improve — here's the evidence")

---

## 8. Paper Writing (HUMAN ONLY)

The following paper sections require human authorship. The pipeline produces the data and figures, but the narrative requires domain expertise and scientific judgment.

### 8.1 Interpretability Methods Section
- Describe the 5-step pipeline: train with sparsity → prune → symbolic regression → lock-in → compose
- Reference Liu et al. (2024) and TabKAN (2025) as methodological sources
- Explain the BIC-penalized candidate selection
- Document the three quality tiers and their thresholds

### 8.2 Results Section
- Report QWK for all 4 models
- Report sparsity ratios and symbolic fit quality
- Present the feature validation curves
- Present the pruning Pareto curves
- Show the KAN network diagrams with edge functions
- Show composed symbolic formulas and discuss their domain meaning
- Compare feature importance rankings across models (Kendall τ)

### 8.3 Discussion Points to Address
- Do KAN-native importance rankings agree with domain knowledge?
- Do the composed symbolic formulas offer actionable insights for insurance risk?
- Is the interpretability gain worth the added complexity vs. XGBoost+SHAP?
- What fraction of the model remains "non-symbolic" (flagged edges)?
- How does sparsity regularization affect the interpretability-performance tradeoff?

### 8.4 Figures Checklist for Paper
All figures are now automatically generated by the pipeline:

| Figure | File | Status |
|--------|------|--------|
| KAN network diagram with edge functions | `{flavor}_kan_diagram.pdf` | **NEW** — implemented |
| Before/after pruning comparison | `{flavor}_before_after_pruning.pdf` | **NEW** — implemented |
| R² distribution histogram + tier pie | `{flavor}_r2_distribution.pdf` | **NEW** — implemented |
| Pruning Pareto curve | `pruning_pareto_curve.pdf` | **NEW** — implemented |
| Feature validation curves | `feature_validation_curves.pdf` | **NEW** — implemented |
| Feature ranking comparison | `feature_ranking_comparison.pdf` | **NEW** — implemented |
| Composed symbolic formulas | `{flavor}_symbolic_formulas.md` | **NEW** — implemented |
| Activation function grid (top-10) | `{flavor}_activations.pdf` | Existing |
| Feature importance bar chart | `{flavor}_feature_ranking.pdf` | Existing |
| QWK retention curves | `qwk_feature_retention.pdf` | Existing |
| Comparison matrix table | `final_comparison_matrix.pdf` | Existing |
| Per-risk feature importance | per-risk panels | Existing |
| Side-by-side model comparison | side-by-side panels | Existing |
| Pruned network graph | `{flavor}_pruned_network.pdf` | Existing |

---

## Summary: What's Code-Complete vs. What Needs Humans

| Component | Code Status | Human Action |
|-----------|-------------|--------------|
| Edge pruning with QWK protection | Done | None |
| Symbolic regression (scipy + PySR) | Done | Run PySR (optional) |
| Symbolic lock-in | Done | None |
| Formula composition (SymPy) | **NEW** — Done | Review formulas for domain sense |
| Feature importance (paper-native) | Done | None |
| Cross-architecture normalization | Done | None |
| R² distribution figure | **NEW** — Done | None |
| KAN network diagram | **NEW** — Done | None |
| Pruning Pareto curve | **NEW** — Done | Run on actual models |
| Feature validation curves | **NEW** — Done | Run on actual models |
| Sparsity regularization | Config ready | **Retrain models** |
| Sparsity ablation | Not needed in code | **Run sweep, record results** |
| Degree/grid-size ablation | Not needed in code | **Run sweep, record results** |
| Paper writing | N/A | **Write methods, results, discussion** |
