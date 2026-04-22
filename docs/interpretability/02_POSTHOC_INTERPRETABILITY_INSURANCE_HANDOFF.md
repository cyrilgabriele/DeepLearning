# Post-Hoc Interpretability Handoff for Insurance-Facing KAN Explanations

Date: 2026-04-21

This document is a root-level handoff for another Codex instance. It is focused on the user's specific question:

- train primarily for predictive quality, especially QWK
- extract as much useful interpretability as possible afterward
- make the final explanation package usable by someone working in insurance

This handoff is intentionally practical. It references the current repo structure, the current artifacts, the main technical constraints, and the recommended implementation path.

## 1. Primary User Goal

The user's stated objective is:

- optimize KAN training for predictive performance rather than interpretability
- then recover interpretability post hoc
- ideally reach an "XGBoost + SHAP" style workflow, but with stronger structural interpretability because of the KAN architecture
- potentially use something like partial derivatives / "Greeks" for local explanations

The right framing is:

- KAN is not automatically interpretable just because it is a KAN
- post-hoc interpretability must still be designed carefully
- the explanation package should be split into:
  - model-level explanations
  - feature-effect explanations
  - case-level explanations
  - governance / validation outputs

## 2. Existing Handoffs and Context Files

Read these first before changing anything:

- [INTERPRETABILITY_HANDOFF.md](INTERPRETABILITY_HANDOFF.md)
- [docs/interpretability/human_action_needed.md](docs/interpretability/human_action_needed.md)
- [docs/interpretability/2026-04-09-selection-pipeline-findings.md](docs/interpretability/2026-04-09-selection-pipeline-findings.md)
- [docs/project_setup/experiment_stages/experiment_stages.md](docs/project_setup/experiment_stages/experiment_stages.md)

This document complements those files. The older handoff is centered on implementation state; this one is centered on the best post-hoc strategy for the user's insurance use case.

## 3. Verified Current State

### 3.1 The run the user just executed is not a good flagship explanation target

User-run config:

- [configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml)

Run summary:

- [artifacts/stage-c-chebykan-best/run-summary-20260421-095203.json](artifacts/stage-c-chebykan-best/run-summary-20260421-095203.json)

Recorded metrics:

- `mae = 1.8822`
- `accuracy = 0.1793`
- `f1_macro = 0.0959`
- `qwk = 0.1672`

Why this matters:

- if the base model is weak, the explanation package is not credible for business use
- an insurance user needs explanations of a competitive model, not explanations of noise
- therefore this exact checkpoint should not be the main case study unless the performance drop is explicitly explained and accepted

### 3.2 The "best" sweep result and the user run are not aligned

Sweep summary:

- [sweeps/chebykan_best.json](sweeps/chebykan_best.json)

Important value:

- `best_qwk = 0.6254`

Important caveat:

- the sweep summary only captures the tuned parameters listed there
- the user-run config in [configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml) includes `sparsity_lambda: 0.1`
- that is much stronger regularization than the sparse Pareto config below and may be one reason performance collapsed

Practical instruction for the next agent:

- do not assume `chebykan_best.yaml` is the real best-performance explanation target
- verify whether this file is legacy, manually edited, or intentionally over-regularized
- do not present this checkpoint as "the best ChebyKAN" without investigation

### 3.3 There is no current selection manifest in `artifacts/selection/`

At the time of this handoff:

- there is no populated `artifacts/selection/` directory

This means:

- the static stage-C YAMLs should not automatically be treated as authoritative outputs of the retrain/select pipeline
- if the next agent needs a defensible "best-performance" or "best-interpretable" config, it should verify selection logic rather than trust the YAML names

Selection code:

- [src/selection/pipeline.py](src/selection/pipeline.py)
- [src/selection/materialize_config.py](src/selection/materialize_config.py)

Important caveat in current selector logic:

- [src/selection/pipeline.py](src/selection/pipeline.py) loads `qwk_after_pruning`
- but candidate eligibility still uses pre-pruning `mean_qwk`
- for the user's use case, the post-pruning QWK should still be checked manually

## 4. The Best Currently Available Explanation Target in This Repo

### 4.1 Dense model: not suitable for business-facing explanation

Dense config:

- [configs/experiment_stages/stage_c_explanation_package/chebykan_best_top20.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_best_top20.yaml)

Dense interpretability artifacts:

- [outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/reports/chebykan_pruning_summary.json](outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/reports/chebykan_pruning_summary.json)
- [outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/data/chebykan_symbolic_fits.csv](outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/data/chebykan_symbolic_fits.csv)
- [outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/figures/chebykan_partial_dependence.pdf](outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/figures/chebykan_partial_dependence.pdf)

Verified pruning summary:

- `edges_before = 26112`
- `edges_after = 26048`
- `sparsity_ratio = 0.0025`
- `qwk_before = 0.538848`
- `qwk_after = 0.537906`

Verified symbolic-fit summary from the CSV:

- `26048` symbolic edges total
- layer counts: `17856` on layer 0, `8192` on layer 1
- only about `34.95%` of edges have `R^2 >= 0.90`
- only about `0.22%` are `clean` under the current tiering

Conclusion:

- this model is too dense to explain well
- it is not suitable for an insurance-facing explanation package
- it is still useful as a predictive reference model

### 4.2 Sparse Pareto model: best current explanation target

Sparse config:

- [configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml)

Sparse eval and interpretability artifacts:

- [outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/](outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/)
- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/reports/chebykan_pruning_summary.json](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/reports/chebykan_pruning_summary.json)
- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/data/chebykan_symbolic_fits.csv](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/data/chebykan_symbolic_fits.csv)
- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/figures/chebykan_partial_dependence.pdf](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/figures/chebykan_partial_dependence.pdf)
- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/figures/feature_validation_curves.pdf](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/figures/feature_validation_curves.pdf)

Verified pruning summary:

- `edges_before = 26112`
- `edges_after = 666`
- `sparsity_ratio = 0.9745`
- `qwk_before = 0.582132`
- `qwk_after = 0.575298`
- `qwk_drop = 0.006833`

Verified symbolic-fit summary from the CSV:

- `323` symbolic edges total
- layer counts: `52` on layer 0, `271` on layer 1
- about `15.79%` of edges have `R^2 >= 0.90`
- about `3.72%` are `clean`

Conclusion:

- this is the best currently available explanation target in the repo
- the post-pruning QWK stays strong enough to defend
- the graph is sparse enough to inspect
- symbolic quality is still limited, so symbolic explanations must be selective rather than universal

## 5. Main Recommendation

The best post-hoc approach is not "one method." It is a layered explanation package.

Use this stack:

1. Choose a pruned sparse KAN on the QWK-vs-sparsity Pareto front.
2. Use KAN-native feature ranking plus feature-retention validation as the global importance backbone.
3. Use ALE as the primary global effect plot, with PDP/ICE only as supporting views.
4. Use local finite-difference sensitivities and counterfactual what-if analyses for applicant-level explanations.
5. Use symbolic formulas only for a limited set of clean or acceptable high-value first-layer edges.
6. If business users insist on a single policy-readable formula, fit a surrogate and report fidelity explicitly.

This gives something closer to:

- XGBoost + SHAP for operational usability
- plus KAN-native edge/function views for scientific interpretability

## 6. Why This Is Better Than Relying on a Single Closed Form

### 6.1 The current `TabKAN` architecture is not cleanly reducible to one honest final formula

Model architecture:

- [src/models/tabkan.py](src/models/tabkan.py)

Important facts:

- every hidden KAN layer is followed by `LayerNorm`
- the network ends with a final linear `head`

Therefore:

- the current symbolic composition is not an exact end-to-end formula for the actual trained predictor
- this is already noted in [INTERPRETABILITY_HANDOFF.md](INTERPRETABILITY_HANDOFF.md)

Current composition code:

- [src/interpretability/formula_composition.py](src/interpretability/formula_composition.py)

Why this matters:

- the user's "Black-Scholes / Greeks / closed-form" intuition is valuable
- but with the current architecture, a single final symbolic formula is not the safest main story

Recommended position:

- use exact symbolic forms only for individual edge functions where fit quality is strong
- do not claim the composed formula is the exact production scoring function for the current `TabKAN`
- if a final closed form is required for communication, make it a surrogate with measured fidelity

### 6.2 Partial derivatives are useful, but raw gradients are not enough for insurance users

The "Greeks" analogy is directionally good.

However:

- raw derivatives on encoded features are too technical
- some features are binary or categorical, so derivatives are not the right object
- insurance users need effect sizes in meaningful changes, not just slopes

Better translation:

- for continuous features: use local finite-difference sensitivities over percentile or IQR-sized moves
- for binary/categorical features: use discrete counterfactual toggles
- express effects in model output delta and predicted risk-class delta

Example framing:

- "moving BMI from the applicant's current percentile to the cohort median decreases predicted risk score by 0.34"
- "changing feature X from absent to present increases predicted risk class from 4 to 5 in this profile"

## 7. The Recommended Insurance-Facing Explanation Package

The target package should have three layers.

### 7.1 Portfolio-level / governance layer

Audience:

- model risk
- analytics leads
- actuarial stakeholders

Outputs:

- overall QWK and calibration-style summaries
- pruning Pareto curve
- retained-edge count and sparsity ratio
- top features from KAN-native ranking
- feature-retention validation curve
- global effect plots for top features

Existing reusable modules:

- [src/interpretability/quality_figures.py](src/interpretability/quality_figures.py)
- [src/interpretability/feature_validation.py](src/interpretability/feature_validation.py)
- [src/interpretability/partial_dependence.py](src/interpretability/partial_dependence.py)
- [src/interpretability/kan_pruning.py](src/interpretability/kan_pruning.py)

### 7.2 Feature-level scientific layer

Audience:

- technical reviewers
- thesis/paper readers
- internal ML reviewers

Outputs:

- selected first-layer edge functions
- symbolic fits for the most important clean / acceptable edges
- small KAN graph view
- feature-effect shape summaries such as monotone / U-shaped / threshold-like / interaction-suspect

Existing reusable modules:

- [src/interpretability/kan_symbolic.py](src/interpretability/kan_symbolic.py)
- [src/interpretability/kan_network_diagram.py](src/interpretability/kan_network_diagram.py)
- [src/interpretability/utils/kan_coefficients.py](src/interpretability/utils/kan_coefficients.py)

### 7.3 Applicant-level decision support layer

Audience:

- someone working in insurance on individual cases

Outputs:

- predicted score and rounded risk class
- top positive contributors
- top negative contributors
- 2-5 actionable what-if scenarios
- sensitivity summary on a business scale

This layer does not exist cleanly yet. It should be added.

Good starting files for style and data handling:

- [src/interpretability/feature_risk_influence.py](src/interpretability/feature_risk_influence.py)
- [src/interpretability/comparison_side_by_side.py](src/interpretability/comparison_side_by_side.py)
- [src/interpretability/utils/style.py](src/interpretability/utils/style.py)

## 8. Best Methods for Each Explanation Layer

### 8.1 Global importance: KAN-native ranking plus retention validation

This should be the backbone.

Why:

- KAN feature ranking is architecture-native
- it avoids importing a fully external explanation method as the main story
- the ranking can be validated empirically by retaining only top-k features and checking QWK

Relevant code:

- [src/interpretability/feature_validation.py](src/interpretability/feature_validation.py)
- [src/interpretability/pipeline.py](src/interpretability/pipeline.py)

Recommendation:

- keep the current feature-retention validation
- make it a first-class output in the final report
- compare KAN ranking against XGBoost SHAP and, if possible, a simple GLM baseline

This is much more defensible than showing only an importance bar chart.

### 8.2 Global feature effects: use ALE as primary, PDP as secondary

Current PDP code:

- [src/interpretability/partial_dependence.py](src/interpretability/partial_dependence.py)

Why plain PDP is not enough:

- insurance tabular features are correlated
- PDP can show unrealistic averages because it sweeps one feature while leaving correlated companions untouched

Recommendation:

- add `ALE` plots as the primary effect view
- keep PDP only as a supplementary figure
- optionally add a small number of ICE curves for representative applicants

Suggested new module:

- `src/interpretability/accumulated_local_effects.py`

Minimum deliverables:

- 1D ALE for top 10-20 features
- optional 2D ALE for the most plausible feature pairs
- one-page summary of turning points and monotonicity

### 8.3 Local explanations: finite differences and counterfactuals

This is the most useful layer for an insurance user.

Recommended local explanation primitives:

- applicant baseline prediction
- feature-wise delta from moving one feature to a reference value
- sensitivity per IQR move for continuous features
- discrete toggle effect for binary/categorical features

This is preferable to exposing raw gradients directly.

Suggested new module:

- `src/interpretability/local_case_explanations.py`

Recommended outputs:

- `case_summary.json`
- `case_summary.md`
- `case_waterfall.pdf`
- `case_what_if_table.csv`

### 8.4 "Greeks" / derivatives: useful, but expose them as business sensitivities

If you implement derivatives, do it in this form:

- continuous features:
  - local slope around the applicant
  - local delta for a small percentile move
  - local delta for a one-IQR move
- binary/categorical features:
  - no derivative
  - only discrete counterfactual changes

Do not present raw gradients on encoded coordinates as the business-facing artifact.

Use them internally to rank local sensitivity if helpful.

### 8.5 Symbolic explanations: only for selected high-value edges

Current symbolic modules:

- [src/interpretability/kan_symbolic.py](src/interpretability/kan_symbolic.py)
- [src/interpretability/r2_pipeline.py](src/interpretability/r2_pipeline.py)
- [src/interpretability/formula_composition.py](src/interpretability/formula_composition.py)

Current limitation:

- symbolic recovery quality is not high enough to make "full symbolic model" the main interpretation story

Recommended policy:

- only show symbolic formulas for:
  - first-layer edges
  - high-importance features
  - edges with at least acceptable fit
- everything else should stay as:
  - plotted function
  - fitted-shape label
  - or "no reliable symbolic approximation"

This is a strength, not a weakness. It is more honest.

### 8.6 Optional communication surrogate

If the user or a business stakeholder wants one compact formula or scorecard:

- fit a surrogate model on the pruned KAN's predictions
- limit it to the top features
- report fidelity to the KAN explicitly

Possible surrogate families:

- sparse GAM
- symbolic regressor
- monotone scorecard-like surrogate

Important:

- label it clearly as a surrogate of the KAN, not the original model

## 9. Important Data-Scale Caveat

Raw eval export:

- [src/training/trainer.py](src/training/trainer.py)
- [outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/X_eval_raw.parquet](outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/X_eval_raw.parquet)

Raw-scale mapping utility:

- [src/interpretability/utils/style.py](src/interpretability/utils/style.py)

Important constraint already documented in [INTERPRETABILITY_HANDOFF.md](INTERPRETABILITY_HANDOFF.md):

- several "raw" features in the Prudential CSV are already normalized to `0..1`
- that includes examples like `BMI`, `Wt`, `Ht`, and `Ins_Age`

Therefore:

- do not label plots with real-world units like `BMI = 25` unless you recover the original physical-unit mapping from outside the current dataset
- for business communication, prefer:
  - cohort percentile
  - low / median / high buckets
  - observed raw normalized scale, if necessary

This is critical for honesty.

## 10. Recommended Implementation Plan for the Next Agent

### Phase 1: choose the correct explanation target

1. Verify whether [configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml) is intentionally over-regularized or stale.
2. Treat [configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml) as the default explanation target unless a better candidate is established.
3. If needed, rerun selection and materialize a proper `best_interpretable` config.

### Phase 2: make the global explanation package insurance-safe

1. Add ALE support.
2. Keep PDP, but demote it to secondary status.
3. Add summary text for each top feature:
   - effect direction
   - whether the curve is monotone / non-monotone
   - whether there is a visible threshold or turning point

Recommended file additions:

- `src/interpretability/accumulated_local_effects.py`
- optionally `src/interpretability/prediction_profiles.py`

### Phase 3: add applicant-level explanations

1. Implement local what-if analysis.
2. Implement local finite-difference sensitivities.
3. Build an applicant report with top contributors and what-if scenarios.

Recommended new file:

- `src/interpretability/local_case_explanations.py`

Potential integration point:

- [src/interpretability/pipeline.py](src/interpretability/pipeline.py)

### Phase 4: tighten symbolic messaging

1. Keep edge-level symbolic regression.
2. Do not present the current composed formula as the exact final score function.
3. Gate or annotate formula composition when `LayerNorm` and the final head are present.

Relevant file:

- [src/interpretability/formula_composition.py](src/interpretability/formula_composition.py)

Reason:

- this avoids overclaiming interpretability

### Phase 5: build a final insurance-facing report bundle

Produce a bundle with:

- model summary
- top drivers
- ALE plots
- applicant what-if report
- selected symbolic edge appendix
- governance appendix with sparsity and validation

This can be markdown plus PDFs and CSVs under a single experiment directory.

## 11. Suggested Concrete Deliverables

The next agent should try to produce these outputs for the sparse Pareto ChebyKAN experiment.

Under something like:

- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/`

Suggested artifacts:

- `figures/chebykan_ale.pdf`
- `figures/chebykan_pdp_secondary.pdf`
- `figures/chebykan_case_waterfall_<id>.pdf`
- `reports/chebykan_business_summary.md`
- `reports/chebykan_governance_summary.md`
- `reports/chebykan_case_summary_<id>.md`
- `data/chebykan_case_what_if_<id>.csv`
- `data/chebykan_local_sensitivities_<id>.csv`

## 12. Recommended Narrative for the User

If the next agent needs to explain the strategy back to the user, the safest phrasing is:

- train for QWK first
- select a sparse high-QWK checkpoint on the Pareto front
- explain the pruned model globally with KAN-native ranking and ALE
- explain individual applicants locally with finite-difference what-if effects
- use symbolic formulas only where the fit is reliable

That is the most defensible interpretation of "post-hoc interpretability for KANs" in this repo.

## 13. What Not To Do

Do not do these things:

- do not use the user's `qwk = 0.1672` run as the main explanation showcase
- do not claim the current composed symbolic formula is the exact final `TabKAN` scoring rule
- do not use only PDP and call the job done
- do not expose only raw gradients on encoded variables to insurance users
- do not label normalized raw features with real-world units that the dataset does not actually provide
- do not present thousands of edge formulas as if they are human-usable

## 14. Exact Files Most Likely Needed Next

Core model and pipeline:

- [src/models/tabkan.py](src/models/tabkan.py)
- [src/interpretability/pipeline.py](src/interpretability/pipeline.py)
- [main.py](main.py)

Current explanation components:

- [src/interpretability/kan_pruning.py](src/interpretability/kan_pruning.py)
- [src/interpretability/kan_symbolic.py](src/interpretability/kan_symbolic.py)
- [src/interpretability/r2_pipeline.py](src/interpretability/r2_pipeline.py)
- [src/interpretability/formula_composition.py](src/interpretability/formula_composition.py)
- [src/interpretability/partial_dependence.py](src/interpretability/partial_dependence.py)
- [src/interpretability/feature_validation.py](src/interpretability/feature_validation.py)
- [src/interpretability/feature_risk_influence.py](src/interpretability/feature_risk_influence.py)
- [src/interpretability/comparison_side_by_side.py](src/interpretability/comparison_side_by_side.py)
- [src/interpretability/quality_figures.py](src/interpretability/quality_figures.py)
- [src/interpretability/utils/style.py](src/interpretability/utils/style.py)
- [src/interpretability/utils/kan_coefficients.py](src/interpretability/utils/kan_coefficients.py)

Configs and evidence:

- [configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml)
- [configs/experiment_stages/stage_c_explanation_package/chebykan_best_top20.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_best_top20.yaml)
- [configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml)
- [artifacts/stage-c-chebykan-best/run-summary-20260421-095203.json](artifacts/stage-c-chebykan-best/run-summary-20260421-095203.json)
- [sweeps/chebykan_best.json](sweeps/chebykan_best.json)

Best current target artifacts:

- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/reports/chebykan_pruning_summary.json](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/reports/chebykan_pruning_summary.json)
- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/data/chebykan_symbolic_fits.csv](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/data/chebykan_symbolic_fits.csv)
- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/figures/chebykan_partial_dependence.pdf](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/figures/chebykan_partial_dependence.pdf)
- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/figures/feature_validation_curves.pdf](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/figures/feature_validation_curves.pdf)

## 15. Minimal Command Set for Reproduction

If the next agent wants to reproduce the current interpretability package for the sparse Pareto model:

```bash
uv run python main.py \
  --stage interpret \
  --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml \
  --pruning-threshold 0.01 \
  --qwk-tolerance 0.01 \
  --candidate-library scipy \
  --max-features 20
```

If the next agent instead wants to continue from the existing outputs, the main working directory is:

```text
outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/
```

## 16. Final Recommendation in One Paragraph

The best path is to explain a sparse high-QWK ChebyKAN, not the user's weak `qwk = 0.1672` run; use KAN-native feature ranking plus retention validation as the global importance backbone, replace PDP-first reporting with ALE-first reporting, implement applicant-level what-if and sensitivity explanations for insurance users, and treat symbolic formulas as selective edge-level evidence rather than a single exact end-to-end scoring equation.
