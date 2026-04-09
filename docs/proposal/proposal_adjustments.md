# Proposal Adjustments

This note captures the small adjustments we should make to the original project proposal after gaining more practical understanding of the models and the interpretability workflow.

These are refinements, not a major scope change.

## Keep As Is

- Keep the main research question: compare TabKAN-style models against traditional baselines on ordinal life-insurance risk prediction using QWK.
- Keep `ChebyKAN` and `FourierKAN` as the main KAN models in scope.
- Keep the focus on tuning depth, width, and basis complexity.
- Keep QWK as the primary model-selection metric.
- Keep pruning, compactness, and function-level inspection as central parts of the interpretability story.

## Adjust Slightly

### 1. Baselines

- Use `XGBoost` as the main predictive baseline.
- Use `XGBoost` and `GLM` as the main interpretability baselines.
- Treat `MLP` as optional for predictive comparison, but not as a required interpretability baseline.

Why:

- `XGBoost` is the strongest practical performance baseline in this project.
- `XGBoost` plus `GLM` gives a cleaner interpretability comparison than forcing `MLP` into the same role.

### 2. KAN Interpretability Framing

- Do not frame KAN interpretability primarily as full symbolic recovery of the whole model.
- Frame it primarily as:
    - compact pruned architecture
    - learned feature-response curves
    - coefficient-based global feature importance
    - optional symbolic fits for the clearest 1D activations

Why:

- This is better aligned with how `ChebyKAN` and `FourierKAN` are actually used on tabular data.
- Full symbolic recovery is more realistic as an optional bonus than as the core success criterion.

### 3. Per-Risk Wording

- Avoid claiming that KAN gives paper-native per-risk feature importance in the same sense as SHAP.
- Rephrase this as:
    - global KAN-native feature ranking
    - feature-response curves for important features
    - optional risk-stratified inspection of those learned functions

Why:

- For the current KAN setup, coefficient-based importance is fundamentally global.
- This change makes the proposal more scientifically accurate without changing the project goal.

### 4. Model Selection Rule

- Explicitly state that the final interpretability analysis will not use only the single best-QWK run.
- Use two selected targets per KAN family:
- best-performance model
- best-interpretable model within `<= 0.01` validation QWK of the best, preferring the smaller and simpler model after pruning

Why:

- This matches the actual research tradeoff between predictive quality and interpretability.
- It avoids overclaiming from one dense winner model.

### 5. BSplineKAN Scope

- Keep `BSplineKAN` explicitly optional.
- It may be included later as an exploratory extension, but it is not required for the core deliverable.

Why:

- This keeps scope under control while preserving flexibility.

## Suggested One-Sentence Update

If we need one concise update for the supervisor, the clean version is:

`ChebyKAN` and `FourierKAN` remain the core KAN models, with `XGBoost` as the main predictive baseline and `XGBoost` plus `GLM` as interpretability baselines; interpretability will be assessed mainly through pruning, compactness, coefficient-based feature ranking, and learned feature-response curves, with symbolic fits treated as optional follow-up analysis.
