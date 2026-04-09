# Project Steps

This is the recommended execution order for the project.

The goal is to reduce avoidable pipeline errors, retrain only after the workflow is correct, and keep the research story aligned with the KAN and TabKAN papers.

## Models In Scope

- Main KAN models: `ChebyKAN`, `FourierKAN`
- Optional extension: `BSplineKAN`
- Performance baseline: `XGBoost`
- Interpretability baselines: `XGBoost`, `GLM`
- Optional predictive-only baseline: `MLP`

## Selection Rule

- Primary metric: validation `QWK`
- For each KAN family, keep:
- best-performance model
- best-interpretable model within `<= 0.01` validation QWK of the best, preferring the smaller and simpler pruned model

## Recommended Workflow

### 1. Freeze the scientific scope

- Final core comparison: `ChebyKAN` vs `FourierKAN` vs `XGBoost`
- Use `GLM` only as an interpretability baseline
- Keep `BSplineKAN` optional

Why:

- This is enough for a defensible project without overextending the scope.
- `ChebyKAN` and `FourierKAN` already capture the main KAN comparison we care about:
- `ChebyKAN` as the smoother and usually easier-to-read basis
- `FourierKAN` as the more flexible basis that can capture more oscillatory structure
- `BSplineKAN` would add more implementation and interpretation work, but it would not change the core scientific question as much as getting the `ChebyKAN` and `FourierKAN` pipeline fully correct.
- The current interpretability workflow is already centered on `ChebyKAN` and `FourierKAN`, so keeping `BSplineKAN` optional helps control scope and reduce avoidable errors.

### 2. Finalize the pipeline before final retraining

- Make the interpretability pipeline dynamic and artifact-aware for the final workflow.
- Ensure the interpret stage reads the trained model setup correctly and does not assume fixed KAN width/depth patterns.
- Standardize one final artifact format for the models we will retrain.

Why:

- Retraining is cheap here.
- It is better to retrain once cleanly than to support many legacy checkpoint edge cases.

### 3. Lock preprocessing

- Choose the final preprocessing recipe before the final interpretability study.
- Keep the preprocessing fixed during the comparison runs.

Why:

- Otherwise explanation changes cannot be separated from preprocessing changes.

### 4. Run performance tuning

- Tune `ChebyKAN` and `FourierKAN` first for predictive quality.
- Tune `XGBoost` as the main predictive baseline.
- Search depth, width, and basis complexity.
- Keep the search shallow and interpretable first.

Why:

- We need competitive models before interpretability analysis is meaningful.
- KANs are expressive enough that shallow models are a reasonable starting point.

### 5. Pick candidate configurations

- For each KAN family, keep the top few validation-QWK configurations.
- Do not keep only one winner immediately.

Why:

- The final interpretable choice should come from a small candidate set, not from one dense best-score model.

### 6. Run interpretability-aware retraining

- Retrain the selected KAN candidates with sparsity regularization enabled.
- Run multiple seeds for stability checks.
- Keep the same data split and preprocessing recipe.

Why:

- The papers’ interpretability logic depends on sparsification before pruning.
- Seed stability matters for a credible result.

### 7. Prune the retrained KANs

- Prune after sparsified training.
- Use one fixed pruning rule across the KAN comparison.
- If needed, briefly fine-tune after pruning.

Why:

- Interpretability should be reported from the reduced model, not the dense one.

### 8. Select the final KANs

- For each KAN family, select:
- the best-performance model
- the best-interpretable model within `<= 0.01` validation QWK of the best

Prefer the model with:

- fewer active nodes and edges after pruning
- lower basis complexity
- cleaner feature-response curves
- better seed stability

### 9. Generate the explanation package

- For `ChebyKAN` and `FourierKAN`, report:
- active nodes and edges after pruning
- coefficient-based global feature importance
- learned feature-response curves for the most important features
- optional symbolic fits with `R^2` for the clearest activations

- For `XGBoost`, report:
- SHAP summary and dependence-style outputs

- For `GLM`, report:
- coefficient magnitudes and signs

Why:

- This is the cleanest cross-model explanation set for the current project.

### 10. Write the results with the right claim strength

- Main claim:
- KAN-style models can produce compact, function-level explanations while staying competitive in QWK after pruning.

- Do not overclaim:
- do not treat symbolic recovery as the main success criterion
- do not present KAN coefficient importance as true per-risk importance in the same sense as SHAP

## Model-Specific Guidance

### ChebyKAN

- Main KAN model for interpretability discussion
- Best candidate for smooth feature-response curves
- Symbolic fitting is more meaningful here than for FourierKAN

### FourierKAN

- Keep as the flexible KAN comparison model
- Use the same pipeline as ChebyKAN
- Treat symbolic fitting as more optional
- Do not directly compare raw coefficient magnitudes on the same scale as ChebyKAN

### BSplineKAN

- Optional only
- Include only if the main pipeline is already stable and there is time left

Why it is optional and not core:

- It is not excluded because it is uninteresting. It is excluded from the core scope because it is not necessary to answer the main project question.
- The proposal already motivates the main KAN comparison through `ChebyKAN` vs `FourierKAN`: smooth basis vs more flexible basis. That is already a coherent and defensible KAN comparison.
- The current interpretability setup is more naturally aligned with `ChebyKAN` and `FourierKAN`, especially for coefficient-based feature importance and basis-specific explanation.
- Adding `BSplineKAN` would require more implementation, more testing, and more explanation work, while giving less marginal value than first making the `ChebyKAN` and `FourierKAN` analysis fully correct and reproducible.
- If the main pipeline is stable early and results are strong, `BSplineKAN` becomes a good extension or robustness check.

### XGBoost

- Main predictive baseline
- Main post-hoc interpretability baseline via SHAP

### GLM

- Main simple interpretability baseline
- Useful reference for signs, linear direction, and coefficient ranking
