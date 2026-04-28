# Paper LaTeX Handoff: Pareto KAN vs XGBoost Interpretability

Author/context owner: Cyril Gabriele  
Date: 2026-04-26  
Purpose: pass this Markdown file, together with the referenced plots/data files, to a ChatGPT instance so it can draft the LaTeX results/interpretability section for the paper.

## What The Other Model Should Produce

Write a LaTeX-ready paper subsection about the interpretability results. The section should explain how the project compares:

1. XGBoost feature importance and SHAP explanations.
2. ChebyKAN and FourierKAN native feature rankings.
3. XGBoost SHAP feature effects versus KAN partial-dependence effects.
4. The interpretability tradeoff between post-hoc SHAP and KAN-native PDP/symbolic edge explanations.

The recommended framing is feature-first rather than method-first: discuss the same important features across methods and then compare what each explanation method says about those features.

## Files To Attach To The ChatGPT Session

Attach this Markdown file and the following artifacts.

### Primary Figure For The Paper

- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/figures/feature_effect_comparison.pdf`
- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/figures/feature_effect_comparison.png`

Use the PDF for the paper if possible. The PNG is useful for visual inspection in chat.

This figure is organized as:

- columns: `BMI`, `Product_Info_4`, `Wt`, `Medical_Keyword_3`
- row 1: XGBoost SHAP dependence/effect panels
- row 2: ChebyKAN PDP panels
- row 3: FourierKAN PDP panels

### Primary Data Tables / Metadata

- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/data/model_summary.json`
- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/data/feature_overlap_summary.json`
- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/data/feature_ranking_comparison.csv`
- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/data/selected_features.json`
- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/reports/feature_effect_comparison.md`

These are the source of truth for the numbers in the paper text.

### Optional Supporting Figures

Use these only if the LaTeX section needs supplementary figures or if a reviewer asks how the individual model artifacts look.

XGBoost:

- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/figures/shap_xgb_beeswarm.pdf`
- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/figures/shap_xgb_dependence_BMI.pdf`
- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/figures/shap_xgb_dependence_Product_Info_4.pdf`
- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/figures/shap_xgb_dependence_Wt.pdf`
- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/figures/shap_xgb_dependence_Medical_History_15.pdf`
- `outputs/interpretability/xgboost_paper/stage-c-xgboost-best/figures/shap_xgb_dependence_Medical_History_4.pdf`

ChebyKAN:

- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/figures/chebykan_feature_ranking.pdf`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/figures/chebykan_partial_dependence.pdf`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/figures/chebykan_activations.pdf`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/figures/feature_validation_curves.pdf`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/figures/chebykan_r2_distribution.pdf`

FourierKAN:

- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/figures/fourierkan_feature_ranking.pdf`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/figures/fourierkan_partial_dependence.pdf`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/figures/fourierkan_activations.pdf`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/figures/feature_validation_curves.pdf`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/figures/fourierkan_r2_distribution.pdf`

### Optional Supporting Reports

ChebyKAN:

- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/reports/chebykan_pruning_summary.json`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/reports/chebykan_r2_report.json`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/reports/chebykan_symbolic_formulas.md`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/reports/chebykan_exact_closed_form.md`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94/reports/chebykan_closed_form_surrogate.md`

FourierKAN:

- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/reports/fourierkan_pruning_summary.json`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/reports/fourierkan_r2_report.json`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/reports/fourierkan_symbolic_formulas.md`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/reports/fourierkan_exact_closed_form.md`
- `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76/reports/fourierkan_closed_form_surrogate.md`

## Models Compared

The comparison uses the pruned Pareto KAN runs, not the dense KAN runs.

| Model | Run | Recipe | Main explanation source |
| --- | --- | --- | --- |
| XGBoost | `stage-c-xgboost-best` | `xgboost_paper` | predicted-class Tree SHAP |
| ChebyKAN | `stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94` | `kan_paper` | KAN-native ranking + PDP + symbolic edge recovery |
| FourierKAN | `stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76` | `kan_paper` | KAN-native ranking + PDP + symbolic edge recovery |

The Pareto KANs were chosen because they preserve or improve QWK relative to the dense KAN artifacts while using fewer active edges. This is a reasonable interpretability choice because the sparse/pruned graph is smaller and easier to inspect.

## Current Performance Numbers

Use these exact numbers unless newer artifacts are regenerated.

| Model | QWK to report | Active edges | Mean per-edge symbolic R2 |
| --- | ---: | ---: | ---: |
| XGBoost | `0.558721` | not applicable | not applicable |
| ChebyKAN Pareto | `0.617183` | `3302` | `1.000000` |
| FourierKAN Pareto | `0.616046` | `18147` | `1.000000` |

Notes:

- The KAN QWK values above are post-pruning, threshold-consistent QWK values from the interpretability/pruning artifacts.
- `model_summary.json` also contains run-summary QWK values before the final pruning-artifact number:
  - ChebyKAN run-summary QWK: `0.6197720207866612`
  - FourierKAN run-summary QWK: `0.6162702507838511`
- For the paper interpretability section, prefer the post-pruning values because the plotted PDPs and active-edge counts come from the pruned modules.

## Feature Ranking Results

The feature-ranking comparison is in:

- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/data/feature_ranking_comparison.csv`
- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/data/feature_overlap_summary.json`

Shared top-20 features across all three methods:

- count: `6`
- features: `BMI`, `Wt`, `Product_Info_4`, `Ins_Age`, `Medical_Keyword_3`, `Medical_History_30`

Pairwise overlap with XGBoost:

- ChebyKAN vs XGBoost:
  - shared top-20 count: `9`
  - rank-scaled overlap score: `4.3825`
  - Kendall-like top-union agreement: `0.012195`
- FourierKAN vs XGBoost:
  - shared top-20 count: `7`
  - rank-scaled overlap score: `3.76`
  - Kendall-like top-union agreement: `-0.111111`

Interpretation:

- There is meaningful agreement about the dominant underwriting variables, especially `BMI`, `Wt`, `Product_Info_4`, `Ins_Age`, `Medical_Keyword_3`, and `Medical_History_30`.
- The exact ranking order differs, so do not claim that the models produce the same explanation. The fair claim is that they converge on a small shared set of important features while differing in rank order and effect geometry.
- `BMI` is rank 1 for all three methods.

## Selected Feature Panels

The main comparison figure uses:

- `BMI`
- `Product_Info_4`
- `Wt`
- `Medical_Keyword_3`

These were selected because they are available in all three models, rank highly across methods, and include both continuous and binary semantics.

Important ranks from `feature_ranking_comparison.csv`:

| Feature | Type | XGBoost rank | ChebyKAN rank | FourierKAN rank |
| --- | --- | ---: | ---: | ---: |
| `BMI` | continuous | 1 | 1 | 1 |
| `Product_Info_4` | continuous | 5 | 4 | 2 |
| `Wt` | continuous | 4 | 3 | 8 |
| `Medical_Keyword_3` | binary | 11 | 2 | 6 |

## How To Interpret The Main Figure

Main figure:

- `outputs/interpretability/comparison/pareto_kan_vs_xgboost/figures/feature_effect_comparison.pdf`

The figure compares explanation geometry, not only feature rank.

### XGBoost SHAP Row

- Each point is one applicant.
- The y-axis is the SHAP contribution for the applicant's predicted class.
- The black curve is a binned mean for continuous features or a state mean for discrete features.
- This is post-hoc: it explains the trained tree ensemble after fitting.
- SHAP is local-additive and class-specific here, so it is good for sample-level contribution analysis.

### KAN PDP Rows

- Each curve is a partial dependence curve from the pruned KAN module.
- The y-axis is the average predicted risk score when sweeping one feature and holding all other applicant features at their observed values.
- For discrete features, the PDP uses observed states only.
- This is model-native: it evaluates the learned KAN function directly.
- PDPs are global marginal effects, so they should not be described as local applicant attributions.

### Direct Comparison Caveat

Do not say that SHAP values and PDP values have the same y-axis meaning. They do not.

The correct comparison is qualitative and structural:

- SHAP shows the distribution of local feature contributions to the predicted class.
- PDP shows the global marginal response of the KAN risk score.
- Both can reveal whether a feature is influential and whether its effect is monotone, nonlinear, threshold-like, or interaction-sensitive, but the vertical scales are not directly interchangeable.

## Suggested Paper Structure

Use this structure for the LaTeX section.

### 1. Global Feature Agreement

Introduce the three explanation sources:

- XGBoost: mean absolute predicted-class SHAP values.
- ChebyKAN/FourierKAN: KAN-native first-layer coefficient/edge importance after pruning.

Then report:

- `BMI` is rank 1 across all models.
- Six features appear in the top 20 for all three methods.
- ChebyKAN shares 9 top-20 features with XGBoost; FourierKAN shares 7.

Possible message:

> The models agree on a core set of risk-related variables, but do not impose the same ranking. This supports using the comparison as a stability check rather than treating any single ranking as the unique ground truth explanation.

### 2. Feature-Level Effects

Discuss the four figure columns:

- `BMI`: highest-ranked feature for all three models; use it as the anchor example.
- `Product_Info_4`: highly ranked by all three, especially FourierKAN.
- `Wt`: important across all models, but with rank differences.
- `Medical_Keyword_3`: binary feature; useful because it demonstrates that the PDP code uses observed states rather than continuous interpolation.

Describe what the figure lets the reader compare:

- SHAP: sample-level spread and binned/state mean contribution.
- ChebyKAN/FourierKAN: smooth or discrete global response curves.

Do not overstate that one method is universally more interpretable. The stronger claim is:

- SHAP is better for local additive decomposition of a black-box tree ensemble.
- KAN PDPs are better for reading a learned functional response directly from the model, especially after pruning.

### 3. Sparsity And Paper-Readiness

Report the Pareto KANs:

- ChebyKAN: QWK `0.617183`, `3302` active edges.
- FourierKAN: QWK `0.616046`, `18147` active edges.

Suggested interpretation:

- The Pareto ChebyKAN is the most compact of the two KANs and retains strong predictive performance.
- The FourierKAN has similar QWK but many more active edges, so it is less compact as an interpretability object.
- It is reasonable to present both because they represent different KAN bases, but ChebyKAN is the cleaner sparse interpretability candidate.

### 4. SHAP vs KAN Interpretability

Recommended balanced conclusion:

- XGBoost SHAP provides mature post-hoc explanations and clear local attributions.
- KAN explanations are model-native and expose feature-response functions directly, but the plotted PDPs are global marginal summaries rather than per-applicant additive explanations.
- The KAN symbolic edge recovery is exact at the edge level in these artifacts (`mean R2 = 1.0`), which supports the fidelity of edge-level formulas.
- However, these LayerNorm/Fourier/pruned models should not be described as fully end-to-end closed-form symbolic models.

## Important Caveats To Preserve

These caveats should appear somewhere in the paper text, footnote, or caption.

1. SHAP and PDP y-axes are not directly comparable.
2. XGBoost SHAP is post-hoc and class-specific; KAN PDP is model-native and global.
3. KAN per-edge symbolic recovery is exact in these artifacts, but the full model is not a simple end-to-end closed-form expression because the deployed/pruned models can include architectural components such as LayerNorm and, for FourierKAN, basis functions that are not collapsed into one compact formula.
4. The feature-ranking overlap supports shared feature importance, not identical model logic.
5. The selected features are chosen for cross-model availability, high rank, and interpretability diversity; they are not the only important features.

## Suggested Figure Caption

Use or adapt:

```latex
\caption{
Feature-level explanation comparison between the XGBoost baseline and the Pareto-pruned KAN models. Columns show four features selected for high cross-model relevance and mixed feature semantics. The XGBoost row reports predicted-class SHAP values for individual applicants with a binned or state-wise mean overlay. The ChebyKAN and FourierKAN rows report model-native partial-dependence curves from the pruned KAN modules. The panels should be interpreted as comparing explanation geometry rather than sharing a common vertical scale: SHAP values are local additive contributions, whereas KAN PDPs are global marginal responses of the predicted risk score.
}
```

## Suggested LaTeX Table

Use or adapt:

```latex
\begin{table}[t]
\centering
\caption{Predictive performance and interpretability size of the compared models.}
\label{tab:interpretability-model-summary}
\begin{tabular}{lrrr}
\toprule
Model & QWK & Active edges & Mean edge $R^2$ \\
\midrule
XGBoost & 0.558721 & -- & -- \\
Pareto ChebyKAN & 0.617183 & 3302 & 1.000000 \\
Pareto FourierKAN & 0.616046 & 18147 & 1.000000 \\
\bottomrule
\end{tabular}
\end{table}
```

Feature-overlap table:

```latex
\begin{table}[t]
\centering
\caption{Top-20 feature-ranking overlap between XGBoost SHAP and KAN-native importance.}
\label{tab:feature-overlap}
\begin{tabular}{lrr}
\toprule
Comparison & Shared top-20 features & Rank-scaled overlap \\
\midrule
ChebyKAN vs. XGBoost & 9 & 4.3825 \\
FourierKAN vs. XGBoost & 7 & 3.7600 \\
All three models & 6 & -- \\
\bottomrule
\end{tabular}
\end{table}
```

## Suggested Narrative Paragraph

Use this as a starting point, but rewrite in the paper's voice:

```text
The interpretability comparison shows that the tree baseline and the two KAN variants agree on a compact set of influential underwriting variables while assigning them different relative order and effect geometry. BMI is the top-ranked feature for all three methods, and six variables appear in the top 20 across XGBoost SHAP, ChebyKAN native importance, and FourierKAN native importance. The Pareto ChebyKAN shares nine top-20 features with XGBoost, whereas the Pareto FourierKAN shares seven. This level of overlap suggests that the models recover a common signal core, while the remaining rank differences reflect architecture-specific representations rather than simple explanation noise.

The feature-effect panels further distinguish the explanation modalities. XGBoost SHAP describes local class-specific contributions, producing a distribution of applicant-level effects for each feature. In contrast, the KAN panels show global partial-dependence curves obtained directly from the pruned learned function. Thus, SHAP is more suitable for local additive explanations, whereas the KAN PDPs make the learned response geometry easier to inspect as a model-native function. Among the KANs, ChebyKAN provides the more compact interpretable object, achieving QWK 0.617183 with 3302 active edges, compared with QWK 0.616046 and 18147 active edges for FourierKAN.
```

## Do Not Claim

Avoid these claims:

- Do not claim that the KAN PDP values are SHAP values.
- Do not claim that the KAN models are fully symbolic end-to-end closed forms.
- Do not claim that the selected four features exhaust the interpretability story.
- Do not claim that FourierKAN is less accurate; it is similar in QWK but less compact.
- Do not claim direct numerical comparability between the SHAP y-axis and KAN PDP y-axis.

## Regeneration Command

If the artifacts need to be regenerated:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.interpretability.paper_comparison
```

Expected output directory:

```text
outputs/interpretability/comparison/pareto_kan_vs_xgboost/
```

## Verification State

The current interpretability test suite passed after creating the comparison package and artifacts:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/interpretability/
100 passed, 3 warnings
```

