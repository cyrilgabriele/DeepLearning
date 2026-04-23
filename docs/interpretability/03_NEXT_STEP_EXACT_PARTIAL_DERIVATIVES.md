# Next Step: Exact Partials And Discrete Effects

Date: 2026-04-21

We have now validated that the no-`LayerNorm` ChebyKAN can be exported as an exact symbolic model.

This makes the next artifact feasible:

- exact partial derivatives for continuous selected features and exact discrete effects for discrete observed-state selected features of the final symbolic model

## Canonical Target

Use the 20-feature no-`LayerNorm` ChebyKAN run as the source model:

- `artifacts/stage-c-chebykan-pareto-q0583-top20-noln`

## Intended Artifact

Create:

- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/reports/chebykan_exact_partials.json`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/reports/chebykan_exact_partials.md`

This should follow the existing per-run interpretability artifact layout rather than introducing a new repo-root `reports/` folder.

## Preprocessing Findings For This Target

- The canonical target run uses `kan_paper` preprocessing.
- For the selected 20-feature target run, the four continuous selected features are:
  - `BMI`
  - `Product_Info_4`
  - `Wt`
  - `Ins_Age`
- For these four continuous selected features, the model-input values are effectively identical to the raw exported eval values, up to float32 rounding.
- No nontrivial affine rescaling is needed for these four continuous selected features in this target run.
- Therefore, for this target run, the raw-space derivative and transformed-space derivative are the same exact quantity for the continuous selected features.
- The selected top-20 feature set also contains discrete-coded features:
  - `7` binary / two-level discrete features
  - `9` categorical features
  - `0` ordinal-step features
  - `0` missingness-indicator features
- For discrete observed-state features, the canonical observed state set and modal reference state should be derived from the run-specific preprocessed outer training split after feature subsetting, not from the eval split.

## Scope

- Differentiate the nested symbolic model, not the fully expanded scalar expression.
- Compute exact partials with respect to the model-input features.
- For the four continuous selected features in this target run, expose both transformed-space and raw-space derivatives, noting that they coincide exactly under the deployed `kan_paper` preprocessing used here.
- For binary and categorical selected features, do not claim a classical derivative. Treat them as discrete observed-state features and report exact discrete effects instead.

## Traceability: Why Differentiate The Nested Symbolic Model

This instruction is about representation, not about changing the mathematics.

- The nested symbolic model keeps the layer-by-layer symbolic composition intact, so each hidden node remains a symbolic function of upstream nodes and the final output remains a symbolic function of those intermediate nodes.
- The fully expanded scalar expression substitutes every intermediate node into the final output until only the original input-feature symbols remain.
- These two representations are mathematically equivalent. If both are differentiated exactly, they yield the same partial derivative.
- The implementation should therefore differentiate through the symbolic graph by repeated chain-rule application rather than algebraically expanding the whole model first.
- This is the right choice for the canonical target because the existing exact closed-form artifact for `stage-c-chebykan-pareto-q0583-top20-noln` is already too large to be a practical human-facing object:
  - `operation_count = 707824`
  - `expression_length = 6568142`
  - recorded in `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/reports/chebykan_exact_closed_form.md`
- Keeping the derivative nested preserves traceability from a derivative term back to a layer, node, or edge family and produces an artifact that is still inspectable in JSON and Markdown.

Illustrative example:

$$
h(x) = a x + b T_2(\tanh(x))
$$

$$
y(x) = c h(x) + d T_3(\tanh(h(x)))
$$

Differentiate this as a nested composition:

$$
\frac{dy}{dx}
=
\left(c + d \frac{d}{dh} T_3(\tanh(h))\right)\frac{dh}{dx},
\qquad
\frac{dh}{dx}
=
a + b \frac{d}{dx} T_2(\tanh(x))
$$

Do not first substitute `h(x)` into `y(x)` and flatten everything into one giant scalar expression before differentiating.

## Discrete Feature Effects

- For a discrete observed-state feature $x_j \in \mathcal{S}_j$, choose a reference state $r \in \mathcal{S}_j$.
- Define $\mathcal{S}_j$ and the modal default reference state from the run-specific preprocessed outer training split after feature subsetting.
- Store exact effects relative to that reference:

$$
\Delta_{r \to k}(x_{-j}) := f(x_j = k, x_{-j}) - f(x_j = r, x_{-j})
\quad \text{for all } k \in \mathcal{S}_j,\; k \neq r
$$

- Here, $x_{-j}$ means all other features are held fixed.
- For a feature with $k$ observed states, store exactly $k - 1$ contrasts.
- Any pairwise contrast can be reconstructed from the stored reference-based contrasts:

$$
\Delta_{a \to b} = \Delta_{r \to b} - \Delta_{r \to a}
$$

- Use the modal observed state as the default reference state unless a domain-specific baseline is later declared.
- Store these reference-based effects as exact nested symbolic expressions in the JSON artifact.
- In the Markdown artifact, summarize the same symbolic effects in a readable report.
- Numeric evaluations may be included only as an optional secondary example block; they are illustrative and are not the primary artifact contract.
- If that optional numeric example block is included, it should use the same canonical applicant row already used by the per-run case-summary workflow for this target run.
- This rule applies uniformly to:
  - binary features with states like `{0,1}`
  - two-level non-boolean features with states like `{1,2}` or `{2,3}`
  - multi-level categorical features with states like `{1,2,3}`

## Why This Is The Right Next Step

- The exact closed-form model now exists for the no-`LayerNorm` candidates.
- The fully expanded expression is too large to be a human-facing artifact.
- Exact partial derivatives are still mathematically valid and directly useful for local sensitivity analysis.
- A nested symbolic derivative representation is much more practical than relying on the giant expanded formula.

## Minimum Contents

- feature name
- representation type: `continuous_exact_partial` or `reference_based_discrete_effect`
- derivative with respect to transformed feature
- derivative with respect to raw feature when valid
- validity note explaining why raw-space rescaling is exact for the continuous selected features in this target run
- for each discrete observed-state feature:
  - observed state set
  - chosen reference state
  - stored exact symbolic reference-based effects $\Delta_{r \to k}$ for all non-reference states
  - note that arbitrary pairwise contrasts are derivable from the stored representation
- optional secondary example block with numeric evaluations for the canonical applicant row already used by the per-run case-summary workflow for this target run, if included
- use one unified per-feature record in which non-applicable fields are explicitly `null` or empty lists rather than omitted

## Acceptance

- the artifact is generated for the 20-feature no-`LayerNorm` ChebyKAN target run
- the artifact is emitted in the run-scoped interpretability `reports/` directory for that target run
- exact symbolic partials are generated for continuous features of the no-`LayerNorm` exact model
- raw-space derivatives are exposed for `BMI`, `Product_Info_4`, `Wt`, and `Ins_Age`, with an explicit note that they are identical to transformed-space derivatives in this target run
- binary and categorical selected features are handled as discrete observed-state features using the reference-based representation
- the observed state set and modal default reference state are derived from the run-specific preprocessed outer training split after feature subsetting
- discrete effects are stored symbolically as the primary contract, with numeric examples included only optionally as secondary illustrations
