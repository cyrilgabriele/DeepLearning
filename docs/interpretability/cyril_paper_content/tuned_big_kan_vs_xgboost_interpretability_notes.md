# Tuned Big KAN vs XGBoost: Interpretability Section Notes

## Core Claim

The tuned KAN models achieve performance in the same predictive regime as the XGBoost baseline, while offering model-native feature-effect inspection through the learned KAN structure. XGBoost requires an additional post-hoc explanation method, SHAP, to obtain comparable feature-level explanations.

This should be presented carefully: the tuned KANs are not compact symbolic models. Their advantage is not that they are automatically simple, but that their architecture provides native learned univariate components and response curves that can be inspected without adding a separate attribution framework.

## Models Compared

Artifacts used:

| Model | Artifact / run | Explanation source |
| --- | --- | --- |
| XGBoost | `stage-c-xgb-best` | Predicted-class Tree SHAP |
| ChebyKAN tuned big | `stage-c-chebykan-pareto-sparsity-trial-009` | KAN-native feature ranking + PDP |
| FourierKAN tuned big | `stage-c-fourierkan-pareto-sparsity-trial-009` | KAN-native feature ranking + PDP |

Comparison output:

```text
outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/
```

Main figure:

```text
outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/figures/feature_effect_comparison.pdf
```

## Performance and Interpretability Summary

| Model | QWK | QWK after pruning / interpreted module | Accuracy | Feature count | Active edges | Mean per-edge R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.645505 | n/a | 0.378378 | 126 | n/a | n/a |
| ChebyKAN tuned big | 0.625136 | 0.616137 | 0.329544 | 140 | 17,482 | 1.000000 |
| FourierKAN tuned big | 0.633108 | 0.633216 | 0.363223 | 140 | 17,855 | 1.000000 |

Interpretation:

- XGBoost has the highest absolute QWK.
- FourierKAN is close to XGBoost in the same predictive regime.
- ChebyKAN is somewhat lower but still competitive.
- The tuned KANs are dense models with many active edges, so they should not be described as compact symbolic explanations.
- The per-edge symbolic recovery is perfect in these artifacts, but this does not mean the entire dense model is a simple closed-form expression.

## Feature Ranking Overlap

Top-20 overlap results:

| Comparison | Shared top-20 features | Rank-scaled overlap score |
| --- | ---: | ---: |
| All three models | 7 | n/a |
| ChebyKAN vs XGBoost | 12 | 5.1675 |
| FourierKAN vs XGBoost | 9 | 4.1725 |

Interpretation:

- The models recover a meaningful shared signal core.
- ChebyKAN shares more top-20 features with XGBoost than FourierKAN in this comparison.
- Feature overlap supports the claim that KAN-native rankings identify underwriting variables similar to those found by XGBoost + SHAP.
- However, overlap does not imply identical effect shapes.

## Selected Feature-Effect Panels

The figure compares four selected features:

| Feature | Type | XGBoost rank | ChebyKAN rank | FourierKAN rank |
| --- | --- | ---: | ---: | ---: |
| `BMI` | continuous | 1 | 1 | 2 |
| `Medical_History_15` | continuous | 2 | 8 | 3 |
| `Product_Info_4` | continuous | 5 | 5 | 5 |
| `Medical_Keyword_3` | binary | 7 | 3 | 6 |

## How to Interpret the Figure

The rows are different explanation objects:

| Row | Method | Y-axis meaning |
| --- | --- | --- |
| XGBoost | SHAP | Predicted-class SHAP value |
| ChebyKAN | Native KAN PDP | Average predicted risk score |
| FourierKAN | Native KAN PDP | Average predicted risk score |

Important caveat:

The y-axes are not directly numerically comparable. SHAP values are local additive attributions for the predicted class. KAN PDPs are global marginal response curves based on sweeping one feature while holding other applicant features fixed.

The correct comparison is qualitative:

- Are the same features important?
- Are the effect directions broadly similar?
- Are the learned response shapes smooth, monotonic, or nonlinear?
- Do the KANs expose model-native response curves without SHAP?

## Feature-Level Interpretation

### BMI

- XGBoost ranks `BMI` first.
- ChebyKAN also ranks `BMI` first.
- FourierKAN ranks `BMI` second.
- XGBoost and ChebyKAN both show a broadly decreasing response at higher BMI values.
- FourierKAN shows a more oscillatory nonlinear response.

Interpretation:

`BMI` is the strongest shared feature across the models. ChebyKAN provides a cleaner monotonic-ish response, while FourierKAN captures a less straightforward shape.

### Medical_History_15

- XGBoost ranks it second.
- ChebyKAN ranks it eighth.
- FourierKAN ranks it third.
- XGBoost shows an increasing SHAP trend.
- The KAN PDPs show a sharp low-end rise followed by a decline across much of the range.

Interpretation:

The models agree that the feature is important, but they disagree in effect geometry. This is useful evidence that feature ranking overlap alone is not enough; feature-effect curves are necessary.

### Product_Info_4

- All three models rank it fifth.
- This is one of the strongest agreement features by rank.
- The curves show nonlinear behavior rather than a purely linear effect.

Interpretation:

`Product_Info_4` is a good example of cross-model agreement in feature importance, with KANs providing native nonlinear response curves.

### Medical_Keyword_3

- XGBoost ranks it seventh.
- ChebyKAN ranks it third.
- FourierKAN ranks it sixth.
- XGBoost and ChebyKAN show a strong change between state 0 and state 1.
- FourierKAN shows a much smaller marginal change despite ranking the feature highly.

Interpretation:

The feature is important across models, but the learned effect magnitude differs. This should be presented as a model-behavior difference, not as a contradiction.

## Defensible Paper Framing

Use language like:

> Although XGBoost remains the strongest baseline in absolute QWK, the tuned KAN variants operate in the same predictive regime while exposing feature effects through their learned KAN structure. This enables model-native feature-effect inspection without requiring a separate attribution framework. By contrast, XGBoost requires SHAP to produce comparable feature-level explanations. The KAN explanations should be interpreted as native response diagnostics rather than compact closed-form models, since the tuned variants remain dense.

Alternative shorter version:

> The tuned KAN models achieve competitive QWK relative to XGBoost and recover a substantial overlap in top-ranked underwriting features. Unlike XGBoost, whose feature effects are obtained through post-hoc SHAP values, KANs provide model-native response curves through their learned univariate transformations. This gives KANs a structural interpretability advantage, although the dense tuned models should not be treated as compact symbolic formulas.

## Claims That Are Safe

- The tuned KANs are competitive with XGBoost, especially FourierKAN.
- KANs provide model-native feature-effect diagnostics.
- XGBoost requires SHAP for comparable feature-level explanations.
- The models share a meaningful set of important features.
- KAN PDPs reveal nonlinear learned response shapes.
- The tuned big KANs are not compact, but their architecture still gives a direct route to explanation.

## Claims To Avoid

- Do not claim the KANs outperform XGBoost.
- Do not claim ChebyKAN has negligible QWK difference unless explicitly qualifying the split and metric.
- Do not claim KAN PDP values are SHAP values.
- Do not directly compare SHAP y-values to KAN PDP y-values.
- Do not claim the tuned big KANs are fully symbolic or fully transparent.
- Do not claim per-edge R2 = 1.0 means the full model is a simple closed-form expression.
- Do not claim feature importance alone proves identical model reasoning.

## Suggested Final Message

The main result is a tradeoff:

- XGBoost gives the best absolute QWK.
- FourierKAN is close in predictive performance.
- ChebyKAN is somewhat lower but still competitive.
- KANs provide native feature-effect curves from the model structure.
- XGBoost requires SHAP as an external post-hoc explanation framework.
- Dense KANs are not automatically simple, so the interpretability claim should be about model-native diagnostics, not full transparency.

## One-Sentence Takeaway

The tuned KANs do not replace XGBoost on raw performance, but they provide competitive predictive models whose learned structure enables native feature-effect inspection, reducing reliance on external post-hoc explanation frameworks such as SHAP.
