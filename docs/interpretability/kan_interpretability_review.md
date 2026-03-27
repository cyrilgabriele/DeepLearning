# KAN Interpretability Review

This note documents how the current `src/interpretability/` pipeline implements (or diverges from) the interpretability workflow described in the two reference papers (arXiv:2404.19756v5 and arXiv:2504.06559v3).

---

## 1. Overview of the Implemented Pipeline

| Stage | What the papers recommend | What the codebase does |
| --- | --- | --- |
| **Sparsification** | Liu et al. (2404.19756): L¹ + entropy penalties during training so the sparse graph emerges before pruning; L¹ alone is explicitly stated as insufficient. TabKAN (2504.06559): frequency-weighted ℓ₂ penalty on Chebyshev/Fourier coefficients to suppress high-order terms. | `tabkan.py:77-126` implements L¹ + entropy regularisation (parameters `sparsity_lambda`, `l1_weight`, `entropy_weight`) and wires it into `training_step()`. However, `sparsity_lambda` is `0.0` in the experiment configs that are actually run, so regularisation is disabled in practice. `kan_pruning.py:29-209` then applies post-hoc variance-threshold pruning on the dense trained model. |
| **Function inspection** | Plot every active 1D edge function so humans can read monotonicity, periodicity, saturation. For TabKAN specifically: inspect Chebyshev/Fourier coefficient magnitudes directly — the paper shows this outperforms SHAP-based ranking on AUC (Section 4.7). | `kan_symbolic.py` samples surviving edges, fits symbolic surrogates, and renders top-10 feature grids on original scales. Coefficients are only used to compute L¹ norms (importance proxy); the actual coefficient values are never displayed to the analyst. |
| **Symbolification** | After simplification, replace functions by human-readable formulas and document quality (R² tiers). | Symbolic fits are recorded with mean/median R² and tier flags (`≥0.99` clean, `0.90–0.99` acceptable, `<0.90` flagged). `r2_pipeline.py:25-116` generalises this quality report. Correctly implemented. |
| **Human workflow** | Iterate across all layers: train with sparsity → visualise → prune nodes → inspect paths → recover symbolic expressions. The compositional graph across layers is the primary differentiator of KANs over per-feature GAMs. | The repo stacks comparison plots (`comparison_side_by_side.py`, `feature_risk_influence.py`, `comparison_per_risk.py`) to juxtapose GLM/XGBoost/KAN outputs. All comparison helpers hard-filter to layer 0; no multi-layer composition is visible. The pipeline treats the KAN as a flat attribution method equivalent to SHAP rather than as a compositional model. |

---

## 2. Alignment Highlights

1. **Activation-based pruning metric** – The pruning stage measures the L¹ magnitude of learned activation functions matching Liu et al. and uses it as both a pruning and importance signal (`kan_pruning.py:29-139`, `kan_symbolic.py:220-232`).
2. **Symbolic recovery tooling** – Surviving edges are exported to CSVs, fitted with formulae, and assigned quality tiers. This matches the symbolification workflow (`kan_symbolic.py:391-470`, `r2_pipeline.py:25-116`).
3. **Original-scale visualisations** – GLM/SHAP/KAN figures invert preprocessing to show KAN edge outputs on raw feature scales (`comparison_side_by_side.py:74-205`, `feature_risk_influence.py:60-240`).
4. **Regularisation infrastructure exists** – `tabkan.py:77-126` implements the full L¹ + entropy formula from Eq. 2.20 of the original KAN paper. It is correctly connected to the training loop.

---

## 3. Divergences and Gaps

### 3.1 Regularisation is implemented but disabled (Critical)

`tabkan.py:87` reads `if sparsity_lambda == 0.0: return 0.0`. The configs run with this default. The mechanism is correct; it just needs to be turned on. Training without sparsity pressure means the post-hoc pruning cuts an already-dense model whose weights were never encouraged to separate into "important" and "unimportant" — the pruning is working against itself.

### 3.2 Layer-0 tunnel vision invalidates compositional claims (Critical)

Every comparison helper filters to layer 0 at read time:
- `comparison_side_by_side.py:68`: `sym_df[(sym_df["layer"] == 0) & ...]`
- `feature_risk_influence.py:281`: `sym_df[sym_df["layer"] == 0]`
- `comparison_per_risk.py:104`: `layer0 = sym_df[sym_df["layer"] == 0]`

KANs differ from GAMs precisely because information flows through composed layers. Showing only layer 0 is equivalent to showing only the input embeddings of a neural network and claiming you have inspected the model. Any result claiming KAN "interpretability" based on this pipeline is only describing first-layer mappings.

### 3.3 Per-risk comparison presents global scores as conditional — a direct methodological error (Critical)

`comparison_per_risk.py:76-96` (`_kan_importance_by_risk()`) computes KAN importance once globally, then copies the identical vector into all eight risk-level subplots. The figure looks like a conditional analysis but is not. A reader will draw wrong conclusions. This is not a gap; it is an error.

### 3.4 Basis coefficients never surfaced (Moderate)

TabKAN's primary interpretability claim (Section 4.7, arXiv:2504.06559) is that Chebyshev/Fourier coefficient magnitudes serve as a model-native, deterministic importance ranking that outperforms SHAP. The codebase uses coefficients internally for L¹ norm computation but never renders them for an analyst. The frequency-weighted ℓ₂ penalty that would suppress high-order terms (making the coefficients more readable) is also unused. This means the main interpretability contribution of TabKAN is entirely absent from the pipeline.

### 3.5 No graph or path visualisation after pruning (Moderate)

After pruning, `final_comparison.py` records scalar counts (non-zero weights, active edges). The compositional graph — which nodes survive, how they connect, which paths dominate — is never drawn. Without this, the concept of "structural compactness" is unverifiable.

### 3.6 Feature ordering driven by baselines (Low)

`feature_risk_influence.py` selects which features to show using `_top_by_type(shap_rank, ...)`. KAN-native rankings are available but not used as the primary sort key. This is a framing issue rather than an error, but it biases the narrative toward confirming baseline results rather than discovering KAN-specific structure.

---

## 4. Changes Required

The following is ordered by impact. Items marked **Critical** are needed for the pipeline to make valid scientific claims; the rest improve quality but are not blocking.

---

### 4.1 Enable sparsity regularisation during training

**What:** Set `sparsity_lambda > 0` (and tune `l1_weight`, `entropy_weight`) in the experiment configs. No new code is needed — the implementation in `tabkan.py:77-126` is correct.

**Work:** Low. Config change + one ablation run to find a `lambda` that preserves QWK while reducing edge density.

**Gain:** The trained model arrives at pruning with a sparse activation structure that reflects actual learned importance, rather than uniform density. All downstream interpretability results become more meaningful.

**Priority: Critical.**

---

### 4.2 Fix the per-risk KAN comparison (remove the methodological error)

**What:** In `comparison_per_risk.py:76-96`, replace the global-score copy with actual per-risk computation: filter the dataset to samples in each risk bucket, run a forward pass, and compute edge L¹ norms on that subset. If that is too expensive, remove the KAN bars from the per-risk plot entirely and note that KAN importance is global-only.

**Work:** Low–Medium. Either delete the misleading bars or add a masked forward-pass computation.

**Gain:** Eliminates an active error in the figures. Any claim about "KAN importance per risk level" currently in the paper or report is wrong and must not be published.

**Priority: Critical.**

---

### 4.3 Extend all comparison helpers to include deeper layers

**What:** Remove the `layer == 0` hard-filter from `comparison_side_by_side.py:68`, `feature_risk_influence.py:281`, and `comparison_per_risk.py:104`. Add a layer index label to each subplot. For hidden-layer edges, label inputs as `h{i}` (already done in `kan_symbolic.py:448`). Optionally add a pruned-network diagram (nodes as circles, surviving edges as lines weighted by L¹ magnitude).

**Work:** Medium. Removing the filter and adjusting subplot layout. A minimal network diagram (matplotlib or networkx).

**Gain:** This is the core scientific contribution — demonstrating that the KAN learned a multi-step composition rather than a flat set of univariate mappings. Without it, the KAN produces results indistinguishable from a GAM and there is no justification for the added complexity.

**Priority: Critical.**

---

### 4.4 Surface Chebyshev/Fourier coefficient magnitudes as the primary importance metric

**What:** In `kan_symbolic.py` and `comparison_per_risk.py`, add a ranking computed directly from coefficient magnitudes (the `T_{bik}` and `W_{ik}` tensors) per the TabKAN paper's method. Display as a bar chart alongside (or replacing) the variance-based L¹ proxy. This is the paper's own stated interpretability mechanism.

**Work:** Low–Medium. The coefficients are already accessible from the model state dict. Computing their norms and plotting.

**Gain:** Validates the TabKAN paper's core claim in the actual experimental context. Provides a deterministic, basis-grounded importance ranking that can be reported in the paper rather than a proxy measure.

**Priority: Moderate.**

---

### 4.5 Use KAN-native ranking as the primary sort key in comparison plots

**What:** In `feature_risk_influence.py`, replace `_top_by_type(shap_rank, ...)` with KAN coefficient-magnitude ranking as the primary sort, and overlay SHAP for contrast.

**Work:** Low. One-line change to the sort key; layout adjustment for any reordering.

**Gain:** Keeps the interpretability framing centred on what the KAN itself learned rather than confirming what SHAP already found. Consistent with the TabKAN paper's empirical finding that KAN-native selection outperforms SHAP on AUC.

**Priority: Low** (framing issue; not blocking scientific validity).

---

## 5. Summary

The pipeline's symbolic fitting, R² quality tiers, and original-scale visualisations are correctly implemented. The regularisation infrastructure is also correctly coded. The blocking issues are operational (regularisation is off) and analytical (per-risk KAN scores are fabricated copies of global scores; only layer 0 is ever inspected). Fixing items 4.1–4.3 is required before any result from this pipeline can support a claim about KAN interpretability. Items 4.4–4.5 strengthen the paper's grounding in the TabKAN manuscript but are not prerequisites for correctness.
