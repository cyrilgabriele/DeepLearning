# Experiment Stages

This note defines the concrete experiment structure for the project.

The goal is to separate predictive tuning from interpretability-driven selection and to keep the final results defensible against the KAN and TabKAN papers.

## Overall Principle

- Keep preprocessing fixed during the experiment stages.
- First optimize for predictive quality.
- Then optimize for interpretability within a small accuracy gap.
- Only generate the final explanation package from the selected pruned models.

## Core Models

- KAN models: `ChebyKAN`, `FourierKAN`
- Optional extension: `BSplineKAN`
- Performance baseline: `XGBoost`
- Interpretability baselines: `XGBoost`, `GLM`

## Selection Rule

- Primary metric: validation `QWK`
- Interpretability tolerance: `<= 0.01` validation QWK from the best model in the same family

## Stage A: Performance Tuning

- Use fixed preprocessing.
- Tune `ChebyKAN` and `FourierKAN` on validation `QWK`.
- Tune `XGBoost` as the main predictive baseline.
- Search shallow architectures first.
- For KANs, prioritize the proposal-motivated range of `1–3` layers.
- Tune depth, width, and basis complexity.

Why:

- This stage identifies the strongest predictive candidates before interpretability constraints are imposed.
- Starting shallow is consistent with the proposal and keeps the models easier to analyze later.

Outputs:

- best validation-QWK runs for `ChebyKAN`
- best validation-QWK runs for `FourierKAN`
- best validation-QWK run for `XGBoost`
- a small shortlist of strong KAN configurations for the next stage

## Stage B: Interpretability Tuning

- Take the top few Stage-A KAN configurations.
- Retrain each with sparsification and entropy regularization enabled.
- Run `3–5` seeds per candidate.
- Prune each trained model with one fixed pruning rule.
- If needed, briefly fine-tune after pruning.
- Select the smallest pruned model within `<= 0.01` validation `QWK` of the best model in that KAN family.

Prefer:

- fewer active nodes and edges after pruning
- lower basis complexity
- cleaner learned feature-response curves
- better seed stability

Why:

- This stage turns the KAN paper logic into an actual model-selection rule.
- The final interpretable model should not be an arbitrary dense winner.

Outputs:

- best-performance `ChebyKAN`
- best-interpretable `ChebyKAN`
- best-performance `FourierKAN`
- best-interpretable `FourierKAN`

## Stage C: Explanation Package

Generate the final explanation artifacts only for the selected final models.

For KANs:

- active nodes and edges after pruning
- feature-wise learned function plots
- coefficient-based global feature importance
- feature-reduction ablation using top-k features
- symbolic fit only for the clearest `1D` activations, with `R^2`

For `XGBoost`:

- SHAP summary outputs
- dependence-style inspection for important features

For `GLM`:

- coefficient magnitudes
- coefficient signs

Why:

- This gives a compact and consistent explanation package across model families.
- It keeps symbolic fitting in the right role: useful, but not the main success criterion.

## How To Read The Stages

- Stage A answers: which models are competitive?
- Stage B answers: which KAN models are still strong after enforcing simplicity?
- Stage C answers: what do the selected final models actually learn and how interpretable are they?

## Scope Note

- `BSplineKAN` remains optional.
- It can be added only after the `ChebyKAN` and `FourierKAN` pipeline is stable and reproducible.
