# KAN Interpretability Review

This note documents how the current `src/interpretability/` pipeline implements (or diverges from) the interpretability workflow described in the two reference papers (arXiv:2404.19756v5 and arXiv:2504.06559v3).

## 1. Overview of the Implemented Pipeline

| Stage | What the papers recommend | What the codebase does |
| --- | --- | --- |
| **Sparsification** | Penalise activation magnitudes (L¹ plus entropy) during training, then prune edges whose learned univariate functions have low magnitude; prune nodes/paths to expose the compositional graph. | `kan_pruning.py` computes the L¹ magnitude of each learned edge post‑training and removes edges below a threshold, retrying with smaller thresholds if QWK drops too much (`src/interpretability/kan_pruning.py:29-209`). No activation-level regularisers are applied while training, and pruning stops at masking edge weights. |
| **Function inspection** | Plot every active 1D edge function (basis dependent) so humans can read monotonicity, periodicity, saturation, etc.; optionally convert edge functions to symbolic forms. | `kan_symbolic.py` samples surviving edges, fits BIC-penalised symbolic surrogates (poly/abs/log/exp + Fourier harmonics), dumps CSVs, and renders “top-10 feature” grids on original scales (`src/interpretability/kan_symbolic.py:28-324,391-486`). |
| **Symbolification** | After simplification, replace functions by human-readable formulas and document the quality (R² tiers). | Symbolic fits are recorded with mean/median R² and tier flags; `r2_pipeline.py` generalises this quality report (`src/interpretability/r2_pipeline.py:25-116`). |
| **Human workflow** | Iterate: train with sparsity, visualise, prune nodes, inspect paths, recover symbolic expressions, highlight top compositional motifs. | The repo stacks several comparison plots (`comparison_side_by_side.py`, `feature_risk_influence.py`, `comparison_per_risk.py`) to juxtapose GLM/XGBoost/KAN outputs, but mostly at the first-layer feature level; structural graphs or node-level analyses are absent. |

## 2. Alignment Highlights

1. **Activation-based pruning metric** – The pruning stage measures the L¹ magnitude of the learned activation functions exactly like Liu et al. describe and uses it as both a pruning and importance signal (`src/interpretability/kan_pruning.py:29-139`, `kan_symbolic.py:220-232`).
2. **Symbolic recovery tooling** – Surviving edges are exported to CSVs, fitted with formulae, and assigned quality tiers. This matches the “symbolification” workflow and can feed back into documentation or downstream modelling (`src/interpretability/kan_symbolic.py:391-470`, `r2_pipeline.py:25-116`).
3. **Original-scale visualisations** – The mixed GLM/SHAP/KAN figures invert the preprocessing to show KAN edge outputs on raw feature scales, which aligns with the “human-inspectable function” goal (`src/interpretability/comparison_side_by_side.py:74-205`, `feature_risk_influence.py:60-240`).

## 3. Divergences and Gaps

1. **No training-time sparsity pressure** – The papers explicitly combine L¹ and entropy penalties during optimisation so the model discovers a sparse compositional graph before pruning. Here, sparsification is entirely post hoc: a dense TabKAN is trained without activation penalties, then masks are applied afterward. This means the “external DOFs” (graph structure) never receive gradient-level encouragement towards interpretability.
2. **Layer-0 tunnel vision** – Every inspection helper filters to layer 0 (`sym_df[sym_df["layer"] == 0]` in `comparison_side_by_side.py`; `_get_kan_edge` and feature ranking logic likewise). Consequently, deeper learned compositions—the very structure that differentiates KANs from per-feature GAMs—remain hidden, violating the “inspectable compositional graph” promise.
3. **Global importance reused per risk level** – The per-risk comparison replicates the same global first-layer L¹ feature ranking for each of the eight risk buckets instead of recomputing edge activations conditional on class. That misses the papers’ caution that deeper layers blend univariate effects into class-specific behaviours.
4. **Basis information discarded** – TabKAN emphasises interpreting learned Chebyshev/Fourier coefficients directly (e.g., magnitude of specific harmonics). In this repo those coefficients are only used to generate samples before fitting generic surrogate formulas, so analysts cannot read which basis terms matter.
5. **No graph/path visualisation** – After pruning, the workflow should expose which hidden nodes remain and how they connect. Here, “structural compactness” is reduced to scalar counts (non-zero weights, leaves, active edges) in `final_comparison.py`, with no depiction of pathways or node pruning as showcased in the original KAN paper.
6. **Feature selection driven by baselines** – Plots choose features via GLM/SHAP rankings and only overlay KAN curves afterward. This keeps the interpretability narrative centred on baseline explanations rather than letting the KAN dictate which features/edges deserve attention.

## 4. Recommended Improvements

1. **Integrate activation-level regularisation** – Add L¹ + entropy penalties on edge activations to the TabKAN training loop so sparsity emerges before pruning. That will better separate the “external” (graph) and “internal” (edge shape) DOFs as discussed in the papers.
2. **Expose deeper layers** – Extend symbolic export and plotting to every KAN layer, add graph visualisations (e.g., network diagrams with edge magnitudes), and allow node-level pruning/inspection so users can see the compositional structure uncovered after sparsification.
3. **Compute class-conditional edge usage** – Replace the replicated global importances with per-risk statistics derived from actual edge activations on samples belonging to each risk class.
4. **Surface basis-native parameters** – In addition to fitted surrogates, display the learned Chebyshev/Fourier/Pade coefficients (or reconstructed functions in their native bases) so analysts can reason about monotonicity/periodicity directly.
5. **Automate the simplification workflow** – Script the full “train → sparsify → visualise → prune nodes → symbolify” loop, including optional human-in-the-loop symbol replacement, to match the canonical workflow described in the first paper.
6. **Let KAN drive feature selection** – When producing comparison plots, start from KAN-derived rankings (L¹ sums or symbolic-fit counts) and then overlay GLM/SHAP for contrast, keeping the emphasis on the intrinsically learned functions.

Implementing these adjustments would align the repository much more closely with the interpretability claims made in the original KAN and TabKAN manuscripts, moving from “KAN curves plotted next to baseline explainers” to the intended sparse, human-auditable compositional models.
