# KAN Interpretability Review

This note documents how the current `src/interpretability/` pipeline implements (or diverges from) the interpretability workflow described in the two reference papers (arXiv:2404.19756v5 and arXiv:2504.06559v3).

Last updated: 2026-03-29 (after commit c66108f).

---

## 1. Overview of the Implemented Pipeline

| Stage | What the papers recommend | What the codebase does |
| --- | --- | --- |
| **Sparsification** | Liu et al. (2404.19756): L¹ + entropy penalties during training so the sparse graph emerges before pruning; L¹ alone is explicitly stated as insufficient. TabKAN (2504.06559): frequency-weighted ℓ₂ penalty on Chebyshev/Fourier coefficients to suppress high-order terms. | `tabkan.py:77-126` implements L¹ + entropy regularisation (parameters `sparsity_lambda`, `l1_weight`, `entropy_weight`) and wires it into `training_step()`. However, `sparsity_lambda` is `0.0` in the experiment configs that are actually run, so regularisation is disabled in practice. `kan_pruning.py:29-209` applies post-hoc L¹-threshold pruning on the dense trained model. |
| **Feature importance** | TabKAN (Section 5.7): compute absolute values of Chebyshev/Fourier coefficients (Eq. 8 and Eq. 14) as the model-native importance ranking. The paper shows separate bar charts per architecture; cross-model scale comparison is never attempted. | `utils/kan_coefficients.py:26-93` correctly implements `sum(abs(coefficients))` over outputs and basis terms per input feature, matching Eq. 8 / Eq. 14. Used throughout comparison scripts. Both models are plotted on the same bar chart axis in `comparison_per_risk.py`, which the paper does not do (see §3.3). |
| **Function inspection** | Plot every active 1D edge function on the normalized input domain to reveal monotonicity, periodicity, saturation. ChebyKAN: x-axis is tanh-normalized [-1,1] (Eq. 4). FourierKAN: x-axis is standardized input. | `kan_symbolic.py` samples surviving edges and renders top-10 feature grids. `comparison_side_by_side.py` and `feature_risk_influence.py` now use `sample_feature_function` (mean aggregation across hidden outputs). The linear residual term (`base_weight`) is applied to `tanh(x)` in visualization but to raw `x` in the forward pass — a confirmed bug (see §3.4). |
| **Symbolification** | After simplification, replace functions with human-readable formulas and document quality (R² tiers). | Symbolic fits recorded with mean/median R² and tier flags (`≥0.99` clean, `0.90–0.99` acceptable, `<0.90` flagged). `r2_pipeline.py:25-116` generalises the quality report. Correctly implemented. |
| **Human workflow** | Iterate across all layers: train with sparsity → visualise → prune nodes → inspect paths → recover symbolic expressions. The compositional graph is the primary differentiator of KANs over GAMs. | All comparison helpers and `utils/kan_coefficients.py` hardcode the first KAN layer only. No multi-layer composition is visible. The pipeline treats KAN as a flat attribution method equivalent to a GAM. |

---

## 2. Alignment Highlights

1. **Activation-based pruning metric** — The pruning stage measures the L¹ magnitude of learned activation functions, matching Liu et al. Eq. 2.17–2.18 (`kan_pruning.py:29-139`).
2. **Symbolic recovery tooling** — Surviving edges are exported to CSVs, fitted with formulae, and assigned quality tiers. Matches the symbolification workflow (`kan_symbolic.py:391-470`, `r2_pipeline.py:25-116`).
3. **Original-scale visualisations** — Figures invert preprocessing to show KAN outputs on raw feature scales (`comparison_side_by_side.py`, `feature_risk_influence.py`).
4. **Regularisation infrastructure exists** — `tabkan.py:77-126` correctly implements the full L¹ + entropy formula from Eq. 2.20 of the original KAN paper.
5. **Paper-native coefficient importance** — `utils/kan_coefficients.py` now correctly derives feature importance from `sum(abs(cheby_coeffs))` and `sum(abs(fourier_a) + abs(fourier_b))` per TabKAN Section 5.7.
6. **KAN-native feature selection** — `feature_risk_influence.py` uses coefficient-magnitude ranking as the primary sort key for which features to show, consistent with TabKAN Section 5.7's empirical finding that KAN-native selection outperforms SHAP.
7. **Global KAN bars honestly labelled** — `comparison_per_risk.py` now labels KAN bars as "(global)" and includes a figure caption stating they are repeated reference values, not conditional estimates.

---

## 3. Open Divergences and Bugs

### 3.1 Regularisation is implemented but disabled (Critical)

`tabkan.py:87` returns `0.0` when `sparsity_lambda == 0.0`. Both experiment configs run with this default. Training without sparsity pressure means the post-hoc pruning cuts an already-dense model whose activations were never encouraged to separate into important and unimportant edges. The pruned graph does not reflect learned sparsity; it reflects an arbitrary threshold applied to a uniformly-dense model.

**Fix:** Set `sparsity_lambda > 0` in `configs/chebykan_experiment.yaml` and `configs/fourierkan_experiment.yaml`. No code changes needed. One ablation run required to find a lambda that preserves QWK while reducing edge density.

---

### 3.2 Layer-0 tunnel vision invalidates compositional claims (Critical)

`utils/kan_coefficients.py:16-23` (`get_first_kan_layer`) always returns the input-adjacent layer. Every downstream importance and visualization function inherits this restriction:

- `coefficient_importance_from_layer` / `coefficient_importance_from_module` — layer 0 only
- `sample_feature_function` — layer 0 only
- `top_features_by_coefficients` — layer 0 only

KANs differ from GAMs precisely because information flows through composed layers. Restricting all analysis to layer 0 is equivalent to reading only the input embeddings of a neural network. Any claim about KAN "interpretability" from this pipeline describes only the first-layer mappings. There is no justification for KAN's added complexity over a GAM if the compositional structure is never inspected.

**Fix:** Extend all comparison helpers and `kan_coefficients.py` to cover all layers. Label hidden-layer edges as `h{i}` (already done in `kan_symbolic.py:448`). Optionally add a pruned-network diagram (nodes as circles, surviving edges weighted by L¹ magnitude).

---

### 3.3 Cross-model importance bars on an incomparable scale (Moderate)

`comparison_per_risk.py` plots ChebyKAN and FourierKAN importance bars on the same axis with raw (unnormalized) scores. The TabKAN paper (Figures 3 and 4) shows them in **separate** charts. The raw scores are not comparable across architectures:

- ChebyKAN with `degree=3`: 4 basis terms per edge → `sum(abs(coeffs))` over 4 terms
- FourierKAN with `grid_size=4`: 8 basis terms per edge (4 cosine + 4 sine) → `sum(abs(a) + abs(b))` over 8 terms

All else being equal, FourierKAN scores are ~2× larger purely as an artifact of having more basis terms. The feature-selection union step (lines 118–121) correctly normalizes each column to [0,1] before voting, so the **ranking** logic is sound. But the **displayed bars** in the figure are on incomparable scales.

**Fix:** Either normalize each model's bar heights by `n_basis_terms × n_hidden_outputs` before plotting, or show ChebyKAN and FourierKAN in separate subplots as the paper does.

---

### 3.4 `base_weight` visualization uses wrong input scale (Moderate)

**Confirmed by reading `kan_layers.py:41-42` and `kan_layers.py:88`.**

Both layer forward passes apply `base_weight` to **raw x**:
```python
base_out = nn.functional.linear(x, self.base_weight)  # raw x, not tanh(x)
```

But `utils/kan_coefficients.py:168` applies it to `x_norm = tanh(x)`:
```python
base = layer.base_weight[:, feature_idx].detach() * x_norm.unsqueeze(-1)  # wrong
```

The same error exists in `kan_symbolic.py` (`_sample_chebykan_edge`): `base_w * x_norm` should be `base_w * x`.

The consequence: the plotted feature function systematically flattens the linear component in the tails (where `tanh'(x) ≈ 0`). The visualization does not faithfully reproduce what the model actually computes. For features with a strong linear trend, the displayed curve underestimates the model's sensitivity at extreme values.

**Fix:** In `sample_feature_function`, replace `x_norm` with `x` for the `base_weight` term. Apply the same fix to `_sample_chebykan_edge`.

---

### 3.5 Kendall τ annotations mislead when KAN series is constant (Low)

`comparison_per_risk.py:168-196` computes `τ_GLM↔Cheby` and `τ_SHAP↔Cheby` inside the per-panel loop. Because the KAN importance series is **identical** across all 8 panels, τ variation across panels is entirely driven by the changing `top_feats` subset (itself influenced by the per-panel SHAP values). A reader may interpret the variation in `τ_GLM↔Cheby` as evidence that the KAN captures risk-level structure. It does not.

**Fix:** Either remove the KAN τ annotations from the per-risk panels and report them once in a global summary table, or add "(KAN ranking is identical across all panels)" explicitly to the annotation text.

---

### 3.6 No graph or path visualisation after pruning (Low)

After pruning, `final_comparison.py` records only scalar counts (non-zero weights, active edges). The compositional graph — which nodes survive, how they connect, which input-to-output paths dominate — is never drawn. Without this, the concept of "structural compactness" is unverifiable and the main visual differentiator of KANs over MLPs is absent.

**Fix:** Add a minimal pruned-network diagram after pruning using matplotlib or networkx. Nodes as circles, surviving edges as lines weighted by L¹ magnitude.

---

### 3.7 Broken test and missing coverage for `kan_coefficients.py` (Low, but blocks CI)

- `tests/interpretability/test_comparison_per_risk.py` still imports `_kan_importance_from_variance`, which was deleted in commit c66108f. The test suite has an `ImportError` on this file.
- `utils/kan_coefficients.py` is the central new utility (all importance and visualization code depends on it) but has zero test coverage.

**Fix:** Replace the broken test with a test covering `_kan_importance_global` using `coefficient_importance_from_layer`. Add unit tests for `kan_coefficients.py`: both layer types, `layer=None`, sort order, and `sample_feature_function` shape/range.

---

### 3.8 Symbolic lock-in step is missing (Moderate)

The original KAN paper (arXiv:2404.19756, Section 2.5.1 and Figure 2.4) defines symbolification as a two-step process:

1. **Fit** a symbolic formula to the learned activation curve (R² assessment).
2. **Replace** the learned spline/basis activation with the locked symbolic function, then fine-tune the affine parameters (a, b, c, d) to machine precision.

Only step 1 is implemented. `kan_symbolic.py` fits formulae and records R² tiers, but the symbolic formula is never written back into the model. The activations remain as learned Chebyshev/Fourier expansions. This means:

- The final model is not a symbolic expression — it cannot be written down as a closed-form formula.
- The R² tiers (`clean`, `acceptable`, `flagged`) report fit quality but have no downstream effect on the model.
- The core claim of symbolic regression in KANs — that the network *becomes* the formula — is not realized.

This is not a requirement for reporting feature importance or visualizing learned functions, but it is the step that distinguishes KAN interpretability from post-hoc curve fitting on a black-box model.

**Fix:** After identifying edges with R² ≥ 0.99 (`clean` tier), replace those edges' activations in the model state with the fitted symbolic function (evaluated as a fixed tensor lookup or as a PyTorch expression). Fit affine correction parameters (a, b, c, d) by linear regression on the pre- and post-activation samples, as described in the paper. Save the symbolified checkpoint separately from the pruned checkpoint.

---

### 3.9 PySR is a fallback rather than the primary symbolic backend (Low)

The original KAN paper uses PySR (evolutionary symbolic regression) as the **primary** method for discovering symbolic formulas on edges. The codebase inverts this:

- `kan_symbolic.py:185-190`: scipy fixed-candidate library runs first.
- PySR is invoked only when scipy achieves R² < 0.90 (`pysr_fallback_threshold`).
- PySR requires Julia and is gated behind `--use-pysr`.

The fixed-candidate library (linear, polynomial up to degree 3, sin, cos, log, exp) will miss formulas outside its vocabulary. An edge that encodes, e.g., `a * x^1.5 + b` or `a / (1 + exp(-bx))` will be flagged as R² < 0.90 and reported as unresolved even though PySR would find it. The BIC-penalized scipy approach is a reasonable engineering shortcut, but it should be documented as a limitation rather than presented as equivalent to the paper's method.

**Fix (minimal):** Lower `pysr_fallback_threshold` from 0.90 to 0.95 so PySR is attempted on edges where scipy finds an acceptable but not clean fit. Document in the report that scipy is used instead of PySR for computational reasons.

---

## 4. Priority Summary

| # | Severity | Item |
|---|----------|------|
| 3.1 | Critical | Sparsity regularisation disabled in configs |
| 3.2 | Critical | All analysis restricted to layer 0; compositional structure never inspected |
| 3.4 | Moderate | `base_weight` visualization uses `tanh(x)` but forward pass uses raw `x` |
| 3.3 | Moderate | ChebyKAN and FourierKAN importance bars on incomparable scales in same plot |
| 3.8 | Moderate | Symbolic lock-in step (replace activation with formula) not implemented |
| 3.6 | Low | No pruned-network graph visualization |
| 3.5 | Low | Per-panel Kendall τ annotations misleading when KAN ranking is constant |
| 3.7 | Low | Broken test (`ImportError`) + no tests for `kan_coefficients.py` |
| 3.9 | Low | PySR is fallback rather than primary; fixed-candidate library misses formulas outside its vocabulary |

Items 3.1 and 3.2 must be fixed before any result from this pipeline can support a published claim about KAN interpretability. Items 3.3, 3.4, and 3.8 affect the scientific accuracy of figures and claims in reports. Items 3.5–3.7 and 3.9 are quality and documentation issues that do not block correctness.
