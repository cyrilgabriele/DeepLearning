# 04 Underwriter-Facing Underwriting Artifact

This note captures how to turn the exact symbolic backend artifact

- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/reports/chebykan_exact_partials.json`

into something an insurance company can actually consume.

The exact-partials JSON should remain a backend technical artifact. It contains the exact symbolic information needed to generate a useful underwriter-facing explanation, but it is not itself an operational report.

## Scope For V1

For the first implementation, this document is scoped only to:

- `TabKAN`
- `flavor=chebykan`
- the deployed no-`LayerNorm` target run represented by `chebykan_exact_partials.json`

Out of scope for this first version:

- FourierKAN
- spline-KAN / generic original KAN symbolic-snapping workflows
- JacobiKAN, PadéRKAN, fKAN, fast-KAN, or any other KAN-family variants
- a generic symbolic-regression layer that tries to rediscover explanations from sampled edge curves

This matters because the explanation path for V1 should be model-specific and exact:

- use the exact ChebyKAN backend artifact already exported for the deployed run
- use exact partials, exact discrete substitution effects, and exact threshold-aware class mapping
- do not make the underwriter-facing artifact depend on the generic symbolic-fitting / symbolic-snapping workflow from `kan_symbolic.py`

The original KAN paper uses spline activations plus optional symbolic snapping as an interactive simplification workflow. That is useful background, but it is not the primary explanation contract for this V1 artifact. For `TabKAN` with `chebykan`, the explanation source should instead be the exact ChebyKAN / exact-partials path already materialized in this repo.

## What The Exact JSON Already Gives Us

The exact backend artifact already contains the hard part:

- the shared nested symbolic model for the deployed no-`LayerNorm` ChebyKAN
- the exact model-specific ChebyKAN explanation contract for this run, rather than a post-hoc symbolic approximation layer
- exact local partial-derivative traces for the 4 continuous selected features:
  - `BMI`
  - `Product_Info_4`
  - `Wt`
  - `Ins_Age`
- exact reference-based discrete state effects for the 16 discrete selected features
- observed-state counts and modal reference states derived from the reconstructed run-specific outer training split after feature subsetting
- explicit confirmation that, for the 4 continuous selected features in this target run, raw-space and transformed-space derivatives coincide exactly

This means the current JSON is best understood as a symbolic source artifact from which operational artifacts can be generated.

## Recommended Artifact Strategy

Use a three-layer approach:

1. Underwriter Case Sheet
2. Feature Playbook
3. Technical Appendix

### 1. Underwriter Case Sheet

This should be the first artifact to implement.

Audience:

- underwriting analysts
- business stakeholders reviewing a specific applicant

Purpose:

- explain why the applicant received the current predicted score / class assignment
- show which features are most locally influential
- show what one-feature changes would move the score
- clearly flag weakly supported states

Suggested contents:

- applicant ID when available; otherwise explicit eval row position
- predicted score
- risk class using stored optimized ordinal thresholds when available; otherwise explicitly labeled rounded-score class
- distance to next lower / higher class threshold when stored optimized thresholds are available; otherwise mark this as unavailable and say why
- top features increasing risk
- top features decreasing risk
- continuous-feature sensitivities in raw business units
- discrete-feature exact state-change effects
- support / rarity warnings for uncommon discrete states
- caveats stating that the artifact describes model behavior, not causality

The important design choice is that the underwriter should not see symbolic derivative expressions directly. Those should be translated into plain-language local effect statements.

For this V1, the sheet should also not expose or depend on the generic symbolic-fit library from the broader KAN workflow. The operational artifact should be derived from the exact ChebyKAN backend, not from sampled curve fitting.

Examples:

- "At this applicant's current profile, a +1 BMI change changes the model score by about X."
- "Changing `Medical_History_5` from state 1 to state 2 changes the model score by exactly Y, holding all other model inputs fixed."

### 2. Feature Playbook

Audience:

- underwriting managers
- model governance readers
- internal users who need reusable feature interpretation guidance

Purpose:

- explain how to interpret each feature's local effect in the deployed model
- document the meaning of modal reference states for discrete features
- provide stable guidance outside a single applicant case

Suggested contents:

- per-feature summary
- feature type
- continuous vs discrete handling rule
- raw-space interpretation note
- observed-state set for discrete features
- modal default reference state
- support counts and rarity flags
- examples of how to read the case-sheet statements

### 3. Technical Appendix

Audience:

- model risk
- compliance
- technical reviewers

Purpose:

- document that the underwriter-facing artifact is derived from exact symbolic backend quantities
- preserve traceability from user-facing explanations back to the exact symbolic representation

Suggested contents:

- model identifier and run identifier
- preprocessing contract
- explicit V1 scope note: `TabKAN` with `flavor=chebykan` only
- explanation of nested symbolic graph representation
- note that this artifact is derived from the exact ChebyKAN backend and not from generic symbolic regression or symbolic snapping
- explanation of continuous exact partials
- explanation of reference-based discrete effects
- exactness caveats and scope notes
- note that the underwriter-facing sheet is a derived artifact, not a separate approximation layer

## What Underwriters Actually Need

The underwriter-facing artifact should answer four operational questions.

### Why is this case scored where it is?

Show the top upward and downward contributors for the current applicant.

This likely needs a derived ranking layer on top of the exact backend artifact, because the exact JSON provides per-feature derivative / state-effect contracts but not yet a ready-made ordered applicant-specific explanation summary.

Recommended MVP driver definition:

- use one coherent single-feature reference rule across all features
- define a feature's applicant-specific driver value as:

$$
\text{driver}_j(x) := f(x) - f(x_j = r_j,\; x_{-j})
$$

- here $r_j$ is the chosen reference value or reference state for feature $j$
- for discrete features, use the modal reference state already stored in the exact-partials artifact
- for continuous features, use a configurable reference strategy with `median` as the default for V1
- the continuous reference strategy should be exposed as an interchangeable parameter in the generator function or config file so an insurer can later replace the default median with a custom business baseline

This gives one ranking rule for both continuous and discrete features and avoids mixing incompatible notions such as derivative magnitude for some features and state contrasts for others.

### Which features are locally most influential right now?

For continuous features, do not rank by raw derivative magnitude alone.

A derivative by itself is too abstract and too unit-sensitive. Instead, rank by one of:

- exact score change for a fixed business delta
- exact score change from current value to a policy reference value
- exact score change needed to cross the next class threshold

For discrete features, rank by the absolute effect of moving from the current state to alternative observed states.

For the first implementation, choose one rule rather than mixing several:

- for continuous features, use exact score change from the current value to a configurable reference value
- for V1, set that reference to the median by default
- keep the reference strategy configurable so a future deployment can substitute a custom insurer baseline
- future work: add alternative continuous scenario policies such as fixed business deltas and quantile-shift scenarios as optional modes

### What one-feature changes would move the case materially?

Show exact what-if scenarios, especially where a single-feature change would move the applicant across a score or class boundary.

For continuous features, those class-changing scenarios should be computed by exact re-evaluation of the stored symbolic model with all other inputs held fixed. They should not be inferred by linearly extrapolating from the local derivative.

V1 search contract for continuous class-changing scenarios:

- search over an explicit allowed interval for that feature
- by default, use the observed min / max range from the exported eval data for that run
- find the nearest feasible class-boundary crossing by bounded one-dimensional bracket-and-refine search on the exact score function
- use local derivative values only as descriptive sensitivity context, not as the scenario generator itself
- if no one-feature class change exists inside the allowed interval, say so explicitly rather than fabricating a scenario

Examples:

- "Reducing BMI from 31.2 to 29.8 would lower the predicted score from 5.61 to 5.47, moving from class 6 to class 5."
- "If `Medical_History_5` were state 2 instead of state 1, the predicted score would increase by 0.23."

### How reliable is this explanation?

The artifact should expose support warnings and clear caveats:

- local explanation, not causal explanation
- all other features are held fixed
- rare discrete states should be flagged
- exact for the deployed model, but still descriptive of model behavior rather than real-world intervention effects

## Class Definition Policy

The underwriter-facing sheet should make the class-definition rule explicit.

Preferred rule:

- treat `outputs/eval/<recipe>/<experiment>/ordinal_thresholds.json` as the canonical threshold input for the generator
- for the current target run, that resolves to `outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/ordinal_thresholds.json`
- treat the copies persisted in `artifacts/<experiment>/run-summary-<timestamp>.json` and `checkpoints/<experiment>/model-<timestamp>.manifest.json` as audit mirrors, not alternate generator inputs
- use the exact optimized ordinal thresholds associated with the deployed run when that canonical eval-sidecar file is available
- when those thresholds are available, report:
  - predicted score
  - threshold-based risk class
  - margin to next lower / higher threshold

Fallback rule:

- if the exact run-specific optimized thresholds are not available to the generator, fall back to:

$$
\text{rounded class} = \operatorname{clip}(\operatorname{round}(\text{score}), 1, 8)
$$

- in that fallback mode, the sheet must explicitly say that it is using rounded-score classes because stored optimized thresholds were unavailable
- the sheet should not silently present rounded classes as if they were threshold-based classes

Practical note:

- threshold optimization already exists in the broader project pipeline
- new runs already persist the eval-sidecar threshold file, so threshold-based classes should be the normal path rather than an optional convenience
- the rounded-score fallback should remain available only for historical runs where the canonical eval-sidecar threshold file does not exist

## How To Translate The Exact JSON Into Underwriter Language

### Continuous Features

Backend source:

- exact local partial derivative in raw space
- exact ChebyKAN model evaluation for one-feature scenario generation

Underwriter-facing translation:

- plain-language local slope or local change statement in business units

Do not expose:

- symbolic derivative formulas
- hidden-node derivative traces

Instead expose:

- "per 1 unit"
- "per 5 units"
- "from current value to configurable reference value"
- "from current value to class-changing threshold", when threshold-based class boundaries are available

For V1:

- use the configurable reference-value mode with `median` as the default continuous reference
- keep the reference strategy replaceable through a function parameter or config field
- future work: add quantile-shift scenario generation for continuous features

Implementation note:

- do not route these continuous explanations through sampled edge-curve symbolic fitting
- for `chebykan`, the model-specific exact backend is the canonical source of truth

Because the 4 continuous features are raw-space identical in this target run, the output can be written directly in business units without additional rescaling logic.

### Discrete Features

Backend source:

- exact reference-based discrete effect contracts
- observed-state sets
- reference state
- state counts

Underwriter-facing translation:

- current state
- alternative observed states
- exact score delta for each alternative
- support warning where relevant

This is already close to underwriter language and is likely the easiest part of the underwriter-facing artifact to operationalize.

## Feature Naming Layer

If a human-readable feature label dictionary exists, prefer displaying that label in the underwriter-facing sheet while preserving the raw technical feature id for traceability.

Example:

- underwriter-facing label: "Medical history flag 5"
- raw model feature id: `Medical_History_5`

For the first internal MVP, raw feature ids are acceptable if no stable label dictionary is yet available, but a naming layer should be treated as a high-value follow-up for real insurer delivery.

For now, raw feature naming is fine for the first implementation.

## Support / Rarity Layer

The observed-state counts in the exact JSON should be used operationally.

V1 support contract:

- always display raw observed-state counts for the applicant's current discrete state and for any alternative states shown in the report
- qualitative support labels such as `well-supported`, `limited support`, and `rare` are optional in V1 and should remain config-driven until a business-approved bucket policy exists
- if a qualitative labeling policy is enabled, the artifact metadata should state the thresholds used
- if no labeling policy is configured, omit the qualitative label rather than inventing one

Possible future labels, if a business-approved bucket policy is later added:

- well-supported
- limited support
- rare

This keeps the first implementation complete without forcing the implementer to invent unsupported bucket thresholds.

## Recommended MVP

The first underwriter-facing artifact should be a single-applicant Underwriter Case Sheet.

Suggested output names:

- `chebykan_underwriting_case_<row_slug>.json`
- `chebykan_underwriting_case_<row_slug>.md`

Here `<row_slug>` should resolve to applicant ID when available, otherwise the explicit `eval_row_position`.

Later:

- PDF export
- API payload

Suggested MVP sections:

- applicant summary
- current score and class
- margin to neighboring classes
- top upward drivers
- top downward drivers
- continuous local sensitivities in raw units
- exact discrete alternative-state effects
- one-feature scenarios most likely to change class
- raw support counts and any configured support warnings
- explanation caveats

## Implementation Direction

The current exact-partials JSON should remain the canonical backend source for symbolic effects, but the generator also needs an explicit V1 input contract.

For clarity, the V1 generator should be tied to the exact `TabKAN` `chebykan` backend only. It should not try to support all KAN-family variants behind one generic interpretability abstraction.

Required V1 inputs:

- `outputs/interpretability/<recipe>/<experiment>/reports/chebykan_exact_partials.json`
- `outputs/eval/<recipe>/<experiment>/X_eval.parquet`
- `outputs/eval/<recipe>/<experiment>/X_eval_raw.parquet` when raw-value display or `applicant_id` lookup is needed
- `outputs/eval/<recipe>/<experiment>/ordinal_thresholds.json` when threshold-based classes are available

Row-selection contract:

- accept exactly one selector: zero-based `eval_row_position` or `applicant_id`
- `eval_row_position` refers to the exported row order in `X_eval.parquet`
- `applicant_id` lookup is available only when `X_eval_raw.parquet` exists and contains `Id`
- if selector resolution is unavailable, ambiguous, or missing, fail fast rather than silently picking a row

The next implementation layer should be a separate generator that:

- reads `chebykan_exact_partials.json`
- resolves one applicant row through the explicit selector contract above
- evaluates the exact symbolic contracts at that applicant row
- loads optimized ordinal thresholds from the canonical eval sidecar when available; otherwise falls back explicitly to rounded-score classes
- computes applicant-specific drivers using the unified reference-based rule
- uses exact full-model re-evaluation plus bounded one-dimensional search for continuous class-changing scenarios
- uses `median` as the default continuous reference strategy for V1, while keeping that strategy configurable through a parameter or config field
- surfaces raw observed-state counts and any configured support-label policy in the output
- ranks and formats the results into an underwriter-facing artifact

It should explicitly not:

- read generic symbolic-fit outputs such as `*_symbolic_fits.csv` as the primary explanation source
- depend on `fix_symbolic`, `suggest_symbolic`, or `auto_symbolic` style workflows
- infer underwriter-facing effects from approximate fitted edge formulas when exact ChebyKAN quantities are already available

This keeps the architecture clean:

- exact symbolic backend artifact
- derived underwriter-facing artifact

## Short Conclusion

The exact backend artifact is already sufficient to support a useful insurance-company deliverable.

The right next step is not to expose that JSON directly, but to build a derived Underwriter Case Sheet that translates:

- exact continuous local derivatives into raw-unit business statements
- exact discrete substitution effects into plain-language what-if contrasts
- state counts into mandatory support-count disclosures and optional configurable support warnings
- explicit class-definition logic into either threshold-based classes or clearly labeled rounded-score fallback classes
- a unified reference-based driver definition into ordered upward and downward driver lists

For this first implementation, that translation layer should be explicitly limited to `TabKAN` with `chebykan` flavor and should rely on the exact model-specific backend, not on generic symbolic fitting.

That is the shortest path from the current symbolic backend to an artifact an underwriter can actually use.


## Things to Clarify Before Implementation
Before this file can be used as an implementation guide we need to define: 
### Assessment
The big architectural questions are now in good shape:

- scope is cleanly limited to TabKAN + chebykan only
- the exact ChebyKAN backend is the canonical explanation source
- threshold source is pinned
- row selection is pinned
- generic symbolic fitting is explicitly excluded

So I would call the document sound, but not fully complete. The remaining work is mostly contract closure, not conceptual redesign.
What I’d close before implementation

### to be defined:
  - define the continuous reference source and add the needed input/artifact if it is not already available
  - fix exact display counts for drivers, sensitivities, and scenarios
  - define the discrete alternative-state display rule
  - add one explicit sentence on score/class risk direction
  - define a minimal stable JSON schema