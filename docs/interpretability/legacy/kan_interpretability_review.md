# KAN Interpretability Review

This note documents how the current `src/interpretability/` pipeline implements (or diverges from) the interpretability workflow described in the two reference papers (arXiv:2404.19756v5 and arXiv:2504.06559v3).

Last updated: 2026-04-09 (after all 3.1–3.9 fixes and new module additions).

---

## 1. Overview of the Implemented Pipeline

| Stage | What the papers recommend | What the codebase does |
| --- | --- | --- |
| **Sparsification** | Liu et al. (2404.19756): L¹ + entropy penalties during training so the sparse graph emerges before pruning; L¹ alone is explicitly stated as insufficient. TabKAN (2504.06559): frequency-weighted ℓ₂ penalty on Chebyshev/Fourier coefficients to suppress high-order terms. | `tabkan.py:77-126` implements L¹ + entropy regularisation (parameters `sparsity_lambda`, `l1_weight`, `entropy_weight`) and wires it into `training_step()`. Configs set `sparsity_lambda: 0.001`, `l1_weight: 1.0`, `entropy_weight: 1.0`. Models need retraining with these values. |
| **Feature importance** | TabKAN (Section 5.7): compute absolute values of Chebyshev/Fourier coefficients (Eq. 8 and Eq. 14) as the model-native importance ranking. The paper shows separate bar charts per architecture; cross-model scale comparison is never attempted. | `utils/kan_coefficients.py` correctly implements `sum(abs(coefficients))` over outputs and basis terms per input feature, matching Eq. 8 / Eq. 14. Importance scores are normalized by `n_basis_terms × n_hidden_outputs` for cross-architecture comparability. |
| **Function inspection** | Plot every active 1D edge function on the normalized input domain to reveal monotonicity, periodicity, saturation. ChebyKAN: x-axis is tanh-normalized [-1,1] (Eq. 4). FourierKAN: x-axis is standardized input. | `kan_symbolic.py` samples surviving edges and renders top-10 feature grids. `sample_feature_function` correctly applies `base_weight` to raw `x` (not `tanh(x)`). `kan_network_diagram.py` produces the signature KAN visualization with mini activation function plots on edges. |
| **Symbolification** | After simplification, replace functions with human-readable formulas and document quality (R² tiers). | Symbolic fits recorded with mean/median R² and tier flags (`≥0.99` clean, `0.90–0.99` acceptable, `<0.90` flagged). `r2_pipeline.py` generalises the quality report. Clean edges are locked in via `lock_in_symbolic_edges()`, which projects the symbolic function onto the basis and replaces layer coefficients. |
| **Formula composition** | Compose edge functions through layers to extract a closed-form end-to-end expression. This is the primary differentiator of KANs over GAMs. | `formula_composition.py` traverses the pruned graph and composes per-edge symbolic fits into SymPy expressions. Computes end-to-end R² between composed formula and model output. Multi-layer coefficient analysis supported via `coefficient_importance_all_layers()`. |

---

## 2. Alignment Highlights

1. **Activation-based pruning metric** — The pruning stage measures the L¹ magnitude of learned activation functions, matching Liu et al. Eq. 2.17–2.18 (`kan_pruning.py:29-139`).
2. **Symbolic recovery tooling** — Surviving edges are exported to CSVs, fitted with formulae, and assigned quality tiers. Matches the symbolification workflow (`kan_symbolic.py`, `r2_pipeline.py`).
3. **Symbolic lock-in** — Clean edges (R² ≥ 0.99) are locked in by projecting the symbolic function onto the Chebyshev/Fourier basis via least-squares, replacing layer coefficients and zeroing `base_weight`. Produces a symbolified checkpoint.
4. **End-to-end formula composition** — `formula_composition.py` composes per-edge SymPy expressions through the network graph, producing closed-form formulas per output node.
5. **Original-scale visualisations** — Figures invert preprocessing to show KAN outputs on raw feature scales (`comparison_side_by_side.py`, `feature_risk_influence.py`).
6. **Regularisation infrastructure** — `tabkan.py:77-126` correctly implements the full L¹ + entropy formula from Eq. 2.20 of the original KAN paper. Configs set `sparsity_lambda: 0.001`.
7. **Paper-native coefficient importance** — `utils/kan_coefficients.py` correctly derives feature importance from `sum(abs(cheby_coeffs))` and `sum(abs(fourier_a) + abs(fourier_b))` per TabKAN Section 5.7, normalized by `n_basis_terms × n_hidden_outputs` for cross-architecture comparability.
8. **Multi-layer analysis** — `coefficient_importance_all_layers()` covers all KAN layers, labelling hidden-layer inputs as `h0`, `h1`, etc. `draw_pruned_network_graph()` visualises the compositional structure.
9. **KAN network diagram** — `kan_network_diagram.py` produces the canonical KAN figure (Liu et al. Figure 2.4): each edge displays a mini-plot of its learned 1D function, with opacity scaled by `tanh(3 × normalized_L1)`.
10. **Feature validation** — `feature_validation.py` implements the TabKAN §5.7 validation: QWK vs. top-k features retained, comparing KAN-native ranking against SHAP.

---

## 3. Resolved Divergences (April 2026)

All issues identified in the original review have been resolved. This section documents what was found and how each was fixed.

### 3.1 Regularisation was disabled in configs

**Problem:** `sparsity_lambda` was `0.0` in both experiment configs, meaning training ran without sparsity pressure. Post-hoc pruning on a dense model does not reflect learned sparsity.

**Resolution:** Both `configs/model/chebykan_experiment.yaml` and `configs/model/fourierkan_experiment.yaml` now set `sparsity_lambda: 0.001`, `l1_weight: 1.0`, `entropy_weight: 1.0`. Models need retraining with these values (see `docs/interpretability/human_action_needed.md`).

---

### 3.2 Layer-0 tunnel vision

**Problem:** `get_first_kan_layer()` always returned only the input-adjacent layer. All downstream analysis was restricted to layer 0, treating KAN as equivalent to a GAM.

**Resolution (commit `45959c9`):**
- Added `get_all_kan_layers()` and `coefficient_importance_all_layers()` to `kan_coefficients.py`. Layer 0 uses feature names; subsequent layers label inputs as `h0`, `h1`, etc.
- Added `draw_pruned_network_graph()` to `final_comparison.py`: a 3-column diagram (inputs → hidden → outputs) with edge width proportional to L1 magnitude.
- Added `formula_composition.py`: SymPy-based composition of per-edge formulas through the full network graph, producing end-to-end closed-form expressions.
- Tests: `test_kan_coefficients_multilayer.py` covers `get_all_kan_layers`, `coefficient_importance_all_layers`, and hidden-layer labelling.

---

### 3.3 Cross-model importance bars on incomparable scales

**Problem:** ChebyKAN (4 basis terms per edge) and FourierKAN (8 basis terms per edge) produced raw importance sums on different scales. FourierKAN scores were ~2× larger purely as an artifact.

**Resolution (commit `08f2a7a`):**
- `coefficient_importance_from_layer()` now exports `n_basis_terms` and `n_hidden_outputs` in the returned DataFrame.
- `_kan_importance_global()` in `comparison_per_risk.py` divides raw scores by `n_basis_terms × n_hidden_outputs`, making ChebyKAN and FourierKAN importance values comparable per-parameter.
- Test: `test_importance_normalization.py` verifies that two architectures with identical coefficient magnitudes but different basis counts produce equal normalized scores.

---

### 3.4 `base_weight` applied to wrong input scale

**Problem:** The forward pass applies `base_weight` to raw `x`, but `_sample_chebykan_edge`, `_sample_fourierkan_edge`, and `sample_feature_function` applied it to `x_norm = tanh(x)`. This systematically flattened the linear component in the tails.

**Resolution (commit `e908c5f`):**
- `kan_symbolic.py`: changed `base_w * x_norm` → `base_w * x` in both `_sample_chebykan_edge` (line 49) and `_sample_fourierkan_edge` (line 71).
- `kan_coefficients.py`: changed `base_weight * x_norm` → `base_weight * x` in `sample_feature_function`.
- Test: `test_base_weight_fix.py` constructs a layer with `cheby_coeffs=0` and `base_weight=1`, verifying the output equals raw `x` (not `tanh(x)`) at the tails.

---

### 3.5 Kendall τ annotations misleading when KAN ranking is constant

**Problem:** Per-risk panels computed `τ_GLM↔Cheby` and `τ_SHAP↔Cheby` inside the loop. Because KAN importance is global (identical across all 8 panels), τ variation across panels was driven entirely by the changing feature subset, not by risk-level structure.

**Resolution (commit `7b8c822`):** Removed the per-panel KAN Kendall-τ annotations. KAN importance is reported once globally in the summary table, not repeated per risk level.

---

### 3.6 No pruned-network graph visualisation

**Problem:** After pruning, only scalar counts were recorded. The compositional graph was never drawn.

**Resolution (commit `45959c9` + new `kan_network_diagram.py`):**
- `draw_pruned_network_graph()` in `final_comparison.py`: 3-column diagram with edge width proportional to L1 magnitude.
- `kan_network_diagram.py`: the signature KAN figure with mini activation function plots on each edge, opacity = `tanh(3 × normalized_L1)`, optional formula text on locked edges. Also includes `draw_before_after_pruning()` for side-by-side dense vs. sparse comparison.

---

### 3.7 Broken test and missing coverage

**Problem:** `test_comparison_per_risk.py` imported the deleted `_kan_importance_from_variance`. `kan_coefficients.py` had zero test coverage.

**Resolution (commit `0c92ece`):** Replaced the broken import test with a live test covering `_kan_importance_global` using `coefficient_importance_from_layer`. Added `test_kan_coefficients_multilayer.py` for multi-layer analysis coverage.

---

### 3.8 Symbolic lock-in step was missing

**Problem:** Symbolic fitting existed (step 1), but the fitted formula was never written back into the model (step 2). The model remained a black-box with post-hoc curve labels.

**Resolution (commit `d168eb9`):**
- Added `lock_in_symbolic_edges()` to `kan_symbolic.py`. For each clean edge (R² ≥ 0.99):
  1. Re-samples the learned activation to get `(x_norm, y_learned)`.
  2. Re-fits the scipy formula to recover exact parameter values.
  3. Evaluates the symbolic function to get `y_symbolic`.
  4. Projects `y_symbolic` onto the Chebyshev/Fourier basis via least-squares.
  5. Replaces the edge's coefficients; zeros out `base_weight`.
- Added `_project_onto_chebyshev()` and `_project_onto_fourier()` helper functions.
- Produces a symbolified checkpoint (`{flavor}_symbolified_module.pt`) and lock-in log CSV.
- Test: `test_kan_symbolic_extended.py::test_lock_in_replaces_coefficients_for_clean_edges` verifies coefficient replacement and base_weight zeroing.

---

### 3.9 PySR fallback threshold too conservative

**Problem:** PySR was only attempted when scipy R² < 0.90. Edges with acceptable but not clean scipy fits (0.90–0.95) were never tried with PySR.

**Resolution (commit `c6e0515`):** Lowered `pysr_fallback_threshold` from 0.90 to 0.95. PySR is now attempted on edges where scipy finds an acceptable fit but not a clean one. The scipy-first approach is documented as a computational shortcut, not as equivalent to the paper's PySR-primary method.

---

## 4. New Modules (April 2026)

| Module | Purpose | Paper Reference |
|--------|---------|-----------------|
| `formula_composition.py` | Compose per-edge symbolic fits into end-to-end closed-form SymPy expressions per output node. Computes end-to-end R². | Liu et al. (2024) §2.5 |
| `feature_validation.py` | QWK vs. top-k features retained for each model's native ranking. Cross-model ranking comparison figure with consensus highlighting. | TabKAN (2025) §5.7, Figures 6-7 |
| `kan_network_diagram.py` | KAN network diagram with mini activation function plots on edges, opacity = `tanh(3 × L1)`. Before/after pruning side-by-side. | Liu et al. (2024) Figure 2.4 |
| `quality_figures.py` | R² distribution histogram with tier boundaries and pie chart. Pruning Pareto curve (QWK vs. sparsity for multiple thresholds). | General paper figures |

All modules have unit tests in `tests/interpretability/` (69 tests, all passing).

---

## 5. Remaining Items Requiring Human Action

All code-level issues are resolved. Items requiring GPU time, human judgment, or paper authorship are documented in `docs/interpretability/human_action_needed.md`. Summary:

| Priority | Item |
|----------|------|
| Critical | Retrain models with `sparsity_lambda: 0.001` |
| High | Run full pipeline end-to-end on retrained models |
| High | Run feature subset validation with all 4 model checkpoints |
| High | Sparsity lambda ablation sweep |
| Medium | Run pruning Pareto curve on actual models |
| Low | PySR runs for complex edges (requires Julia) |
| Low | Degree/grid-size ablation |
| Human | Paper writing: methods, results, discussion |
