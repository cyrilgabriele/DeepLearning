# Ready-to-Implement Brief: Closed-Form ChebyKAN for Insurance

Date: 2026-04-21

This file is the narrow execution brief derived from [POSTHOC_INTERPRETABILITY_INSURANCE_HANDOFF.md](POSTHOC_INTERPRETABILITY_INSURANCE_HANDOFF.md).

It is intentionally specific. Another agent should be able to start implementing from this file without re-deciding the strategy.

## 1. Single Working Objective

Build a ChebyKAN-based framework for insurance that:

- keeps QWK at least near the current XGBoost reference
- prioritizes high post-hoc interpretability over raw model complexity
- is much more interpretable than the current dense KAN
- produces either:
  - an exact closed-form expression for the final simplified deployed KAN, or
  - a closed-form surrogate with explicitly measured fidelity if exact symbolic composition still fails
- provides applicant-level "Greek-style" local sensitivities and what-if explanations

Important clarification:

- visual interpretability is not the main goal in this brief
- the main goal is a mathematically inspectable model plus usable local post-hoc explanations
- the 140-feature model is a discovery baseline, not the intended final symbolic model

## 2. Exact Baseline to Use

Use this as the starting ChebyKAN config:

- [configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml)

Why this one:

- it is the strongest current ChebyKAN artifact in the repo
- it is already sparse enough to be a serious interpretability candidate
- it performs better than the other checked-in ChebyKAN stage-C runs

Verified baseline evidence:

- [artifacts/stage-c-chebykan-pareto-q0583-top20/run-summary-20260419-142332.json](artifacts/stage-c-chebykan-pareto-q0583-top20/run-summary-20260419-142332.json)
- [outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/reports/chebykan_pruning_summary.json](outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/reports/chebykan_pruning_summary.json)

Current baseline numbers:

- base QWK: `0.5821316591780568`
- post-pruning QWK: `0.575298`
- active edges after pruning: `666`
- sparsity ratio: `0.9745`

Important interpretation:

- this is the best current **140-feature discovery baseline**
- it is **not** the intended final closed-form model
- its job is to provide:
  - the best currently verified sparse ChebyKAN checkpoint
  - a KAN-native feature ranking from the full feature universe
  - the hyperparameters to reuse when retraining on a smaller train-time feature subset

Do not use these as the starting point:

- [artifacts/stage-c-chebykan-best/run-summary-20260421-095203.json](artifacts/stage-c-chebykan-best/run-summary-20260421-095203.json)

Reason:

- that specific run has `qwk = 0.1672`
- it is not a credible baseline for deployment or explanation

## 2.1 Mandatory Reduced-Feature Rule

The final symbolic candidate must **not** be trained on all 140 features.

Use this rule:

- keep the hyperparameters from the 140-feature sparse Pareto baseline
- derive a KAN-native feature ranking from that 140-feature baseline
- retrain on the **best 20 features only** as the first reduced candidate

This is a train-time restriction, not just an interpret-time restriction.

Reason:

- training on 140 features is acceptable for discovery
- it is not the right target for a final closed-form model or business-facing local sensitivities
- interpret-time zeroing alone is weaker than a true retrain on the reduced feature set

## 3. XGBoost Reference to Beat or Match

Use this as the verified current XGBoost reference from repo artifacts:

- [artifacts/xgboost-paper/run-summary-20260321-150553.json](artifacts/xgboost-paper/run-summary-20260321-150553.json)

Verified XGBoost QWK:

- `0.5682451534450409`

There are also sweep summaries:

- [sweeps/xgb_paper_xgb_paper_best.json](sweeps/xgb_paper_xgb_paper_best.json)
- [sweeps/xgb_xgb_paper_best.json](sweeps/xgb_xgb_paper_best.json)

But for implementation acceptance, use the verified artifact run first, not only sweep metadata.

## 4. Concrete Success Criteria

The project is successful if all of these are true.

### 4.1 Predictive target

For the final simplified ChebyKAN candidate:

- `qwk_after_pruning >= 0.50`
- and ideally `qwk_after_pruning >= 0.56`

This keeps the model at least near the verified XGBoost reference in the repo.

### 4.2 Interpretability target

The final simplified model must satisfy all of:

- no `LayerNorm` in the deployed symbolic candidate
- train-time restriction to a reduced feature set
- interpret-time restriction can still be applied additionally if useful
- active edges after pruning remain `<= 1000`
- closed-form output exists either exactly or as a surrogate

### 4.3 "10x better interpretability" proxy

Use this operational definition:

- dense ChebyKAN reference has `26048` active edges after pruning:
  - [outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/reports/chebykan_pruning_summary.json](outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/reports/chebykan_pruning_summary.json)
- final symbolic candidate should have at most `2604` active edges after pruning

This is a strict `10x` simplification proxy relative to the dense reference.

The current sparse Pareto model already satisfies this with `666` edges, so the main risk is preserving QWK after removing `LayerNorm` and restricting features.

### 4.4 Closed-form target

Primary target:

- export an exact symbolic formula for the final simplified no-`LayerNorm` KAN including the final linear head

Fallback target:

- export a closed-form surrogate over the final top-20 feature set
- require fidelity `R^2 >= 0.95` to the final simplified KAN predictions on the eval set

Realistic feature budget for high-probability exact closed form:

- recommended target: **12 features**
- realistic range: **10-12 features**
- treat **20 features** as the first reduced performance-preserving retrain, not as the high-probability exact-symbolic budget

Reason:

- with the current 2-layer `[128, 64]` ChebyKAN, 20 train-time features is still likely to produce an exact formula that is technically possible but operationally too large
- 10-12 features is a more realistic budget for a usable exact closed form while still keeping meaningful predictive power
- use **12** as the default exact-symbolic target if the top-20 retrain still produces an unwieldy formula

### 4.5 Local explanation target

For at least one applicant / eval row, export:

- top positive contributors
- top negative contributors
- finite-difference sensitivities
- what-if table with at least 3 counterfactual changes

## 5. Important Implementation Fact

The current checked-in code does **not** expose a `LayerNorm` toggle in config.

Relevant files:

- [src/models/tabkan.py](src/models/tabkan.py)
- [src/config/model/model_config.py](src/config/model/model_config.py)

What is true right now:

- `TabKAN` always appends `nn.LayerNorm(...)` after each KAN layer
- `ModelConfig` does not currently define a `use_layernorm` or similar field

Therefore:

- removing `LayerNorm` is not currently a YAML-only change
- it requires code changes first

## 5.1 Second Important Implementation Fact

The current checked-in code also does **not** support a train-time selected-feature subset through config.

Relevant files:

- [src/config/preprocessing/preprocessing_config.py](src/config/preprocessing/preprocessing_config.py)
- [src/training/trainer.py](src/training/trainer.py)

What is true right now:

- the preprocessing config only defines `contract_version` and `recipe`
- the trainer always uses the full column set returned by preprocessing
- `--max-features` exists only for the `interpret` stage

Therefore:

- retraining on the best 20 features requires code changes
- this must be implemented before the "top-20 retrain" can exist as a true model

## 6. Scope of This Implementation

Do not try to build a broad interpretability suite first.

Only implement these 5 things:

1. make train-time selected-feature retraining possible through config
2. make no-`LayerNorm` ChebyKAN trainable through config
3. make exact symbolic composition possible for the no-`LayerNorm` final model
4. add local "Greek-style" finite-difference sensitivities and what-if explanations
5. add a surrogate fallback if exact symbolic composition is still inadequate

Everything else is secondary.

## 7. Execution Plan

### Phase A: export a reproducible feature ranking from the 140-feature baseline

Use the current 140-feature sparse baseline as the ranking source:

- [configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml)
- [artifacts/stage-c-chebykan-pareto-q0583-top20/run-summary-20260419-142332.json](artifacts/stage-c-chebykan-pareto-q0583-top20/run-summary-20260419-142332.json)

Ranking utility:

- [src/interpretability/utils/kan_coefficients.py](src/interpretability/utils/kan_coefficients.py)

Implement exactly this:

- export the full KAN-native feature ranking from `coefficient_importance_from_module(...)`
- save it as a stable artifact, for example:
  - `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/data/chebykan_feature_ranking.csv`
- from that ranking, materialize two feature lists:
  - top-20 features for the reduced performance candidate
  - top-12 features for the exact-symbolic candidate

Suggested artifact names:

- `configs/experiment_stages/stage_c_explanation_package/feature_lists/chebykan_pareto_q0583_top20_features.json`
- `configs/experiment_stages/stage_c_explanation_package/feature_lists/chebykan_pareto_q0583_top12_features.json`

### Phase B: add train-time selected-feature support

Goal:

- allow training and evaluation on a curated subset of preprocessed features

Edit these files:

- [src/config/preprocessing/preprocessing_config.py](src/config/preprocessing/preprocessing_config.py)
- [src/training/trainer.py](src/training/trainer.py)

Implement exactly this:

- add a preprocessing config field such as `selected_features_path` with default `null`
- when provided, load the feature list after preprocessing and before model fit
- subset all of these consistently:
  - `PreparedDataset.X_train`
  - `PreparedDataset.X_eval`
  - `PreparedDataset.X_train_inner`
  - `PreparedDataset.X_val_inner`
  - `PreparedDataset.X_eval_raw` when columns overlap
  - `PreparedDataset.feature_names`
- ensure test-time transformation respects the reduced feature list:
  - [src/training/trainer.py](src/training/trainer.py:383)
- ensure exported eval artifacts also reflect the reduced feature set:
  - [src/training/trainer.py](src/training/trainer.py:275)
- persist the selected-feature contract in the run summary

Required output:

- a config can now point to a top-20 or top-12 feature list
- training, eval export, and test prediction generation all use the same reduced feature universe

### Phase C: make `LayerNorm` optional

Goal:

- allow `TabKAN` to be trained with or without `LayerNorm`

Edit these files:

- [src/config/model/model_config.py](src/config/model/model_config.py)
- [src/models/tabkan.py](src/models/tabkan.py)
- [src/models/registry.py](src/models/registry.py) only if needed

Implement exactly this:

- add a boolean config field named `use_layernorm` with default `true`
- pass it through `registry_kwargs()`
- pass it through `build_tabkan_model(...)`
- pass it through `TabKANClassifier`
- in `TabKAN.__init__`, only append `nn.LayerNorm(...)` when `use_layernorm` is `true`

Required output:

- existing configs continue to work unchanged
- a new no-`LayerNorm` config becomes possible

### Phase D: create the top-20 reduced no-`LayerNorm` baseline config

Create a new config derived directly from the 140-feature best baseline:

- `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml`

It should match:

- [configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml)

except for:

- experiment name
- `selected_features_path` pointing to the top-20 feature list
- `use_layernorm: false`

Do not change:

- hidden widths `[128, 64]`
- `degree: 6`
- optimizer hyperparameters
- `sparsity_lambda: 0.0108`

Reason:

- reuse the best known 140-feature hyperparameters
- isolate the effect of train-time feature reduction plus `LayerNorm` removal
- do not broaden the search space immediately

### Phase E: train the top-20 reduced no-`LayerNorm` baseline

Command target:

```bash
uv run python main.py \
  --stage train \
  --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml
```

Acceptance for Phase E:

- training completes successfully
- run summary is written
- QWK is recorded

Decision rule:

- if the top-20 reduced no-`LayerNorm` run drops below `0.56` QWK before pruning, stop and document failure
- otherwise continue

### Phase F: run pruning and optional interpret-time feature restriction on the top-20 model

Use the current interpret pipeline with feature restriction:

```bash
uv run python main.py \
  --stage interpret \
  --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml \
  --pruning-threshold 0.01 \
  --qwk-tolerance 0.01 \
  --candidate-library scipy \
  --max-features 20
```

Relevant files:

- [main.py](main.py)
- [src/interpretability/pipeline.py](src/interpretability/pipeline.py)
- [src/interpretability/kan_pruning.py](src/interpretability/kan_pruning.py)

Acceptance for Phase F:

- post-pruning QWK remains `>= 0.50`
- active edges after pruning are `<= 1000`
- feature restriction is applied through `--max-features 20`

Important note:

- the exact symbolic formula, if produced later, will then correspond to the final pruned and feature-restricted model variant
- that is acceptable if this simplified model variant is the one you choose to communicate or deploy

### Phase G: attempt exact symbolic composition on the top-20 model

Current composition code:

- [src/interpretability/formula_composition.py](src/interpretability/formula_composition.py)

Current problem:

- it ignores `LayerNorm`
- it ignores the final linear head

Implement exactly this:

1. detect whether `LayerNorm` exists inside `module.kan_layers`
2. if any `LayerNorm` exists:
   - refuse to claim exact end-to-end composition
   - emit a report stating exact composition is unavailable
3. if `LayerNorm` does not exist:
   - compose through all KAN layers
   - then append the final `head` exactly as a linear combination of the final hidden node expressions plus bias

Required new output:

- a final markdown or JSON report with one exact closed-form expression for the model output

Suggested output names:

- `reports/chebykan_exact_closed_form.md`
- `reports/chebykan_exact_closed_form.json`

Acceptance for Phase G:

- for the no-`LayerNorm` model, the report is generated
- the formula includes the final head
- the formula is clearly labeled as exact for the simplified deployed model variant

Decision rule after Phase G:

- if the top-20 exact formula is technically correct and still usable, keep top-20 as the final model
- if the top-20 exact formula is technically correct but too large to be useful, move immediately to the top-12 exact-symbolic candidate below
- if exact composition is still unavailable for structural reasons, use the surrogate fallback path

### Phase H: create and train the top-12 exact-symbolic candidate if needed

Create a second reduced config that reuses the same hyperparameters but points to the top-12 feature list.

Suggested config:

- `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top12_noln.yaml`

Use the same pattern as the top-20 config, except:

- `selected_features_path` points to the top-12 feature list

Reason:

- 12 features is the realistic exact-symbolic target with high probability
- this is the candidate most likely to yield a usable exact closed form

Acceptance:

- train and interpret it only if the top-20 model is still too complex for a usable exact expression
- prefer top-20 if it already satisfies both QWK and exact-symbolic usability

### Phase I: add local "Greek-style" explanations

Create:

- `src/interpretability/local_case_explanations.py`

Goal:

- for one row or a selected list of rows, compute feature-level local effects

Implement only these explanation modes:

- continuous features:
  - local finite-difference slope around the applicant
  - delta for moving to median
  - delta for a one-IQR move
- binary / missing-indicator features:
  - discrete toggle effect
- categorical / ordinal encoded features:
  - discrete step change or percentile move, whichever is honest for that feature

Use:

- model output delta
- rounded risk-class delta

Do not expose:

- raw autograd gradients as the only artifact

Required outputs for one case:

- `reports/chebykan_case_summary_<rowid>.md`
- `data/chebykan_case_what_if_<rowid>.csv`
- `data/chebykan_local_sensitivities_<rowid>.csv`

### Phase J: build surrogate fallback only if exact composition is still not usable

If Phase E succeeds technically but the final exact expression is too unstable or too large to be useful, add a fallback:

- `src/interpretability/closed_form_surrogate.py`

Input:

- predictions from the final simplified no-`LayerNorm`, pruned, top-20-feature KAN

Output:

- a closed-form surrogate over the same top-20 feature set

Acceptance:

- surrogate fidelity `R^2 >= 0.95` to the final simplified KAN predictions on eval data
- report both surrogate fidelity and QWK of surrogate-rounded predictions

Do not let the surrogate replace the exact model silently.

It must be labeled:

- "surrogate of the simplified KAN"

## 8. Files to Edit

Core model/config:

- [src/config/model/model_config.py](src/config/model/model_config.py)
- [src/models/tabkan.py](src/models/tabkan.py)
- [src/config/preprocessing/preprocessing_config.py](src/config/preprocessing/preprocessing_config.py)
- [src/training/trainer.py](src/training/trainer.py)

Interpretability:

- [src/interpretability/formula_composition.py](src/interpretability/formula_composition.py)
- [src/interpretability/pipeline.py](src/interpretability/pipeline.py)

New files to add:

- `configs/experiment_stages/stage_c_explanation_package/feature_lists/chebykan_pareto_q0583_top20_features.json`
- `configs/experiment_stages/stage_c_explanation_package/feature_lists/chebykan_pareto_q0583_top12_features.json`
- `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml`
- `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top12_noln.yaml`
- `src/interpretability/local_case_explanations.py`
- optionally `src/interpretability/closed_form_surrogate.py`

Tests to update or add:

- [tests/interpretability/test_formula_composition.py](tests/interpretability/test_formula_composition.py)
- [tests/test_pipeline_integration.py](tests/test_pipeline_integration.py)
- [tests/training/test_trainer.py](tests/training/test_trainer.py)

Recommended new tests:

- selected-feature config and dataset subsetting test
- no-`LayerNorm` config round-trip test
- exact closed-form export test for a tiny no-`LayerNorm` TabKAN with a final head
- local case explanation smoke test

## 9. Explicit Non-Goals

Do not spend time on these before the core closed-form path works:

- broad ALE/PDP figure redesign
- full insurance dashboard UX
- multi-model comparison suite
- PySR expansion for every edge
- large ablation sweeps

Those can come later.

## 10. Definition of Done

This task is done when all of the following exist:

1. `use_layernorm: false` works in config and training.
2. Train-time selected-feature support works through config and training.
3. [configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml](configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml) exists.
4. A trained top-20 reduced no-`LayerNorm` ChebyKAN run summary exists and shows `qwk >= 0.56`.
5. An interpret run with `--max-features 20` exists for that top-20 reduced model.
6. Post-pruning QWK is still `>= 0.56`.
7. Active edges after pruning are `<= 1000`.
8. An exact closed-form report exists either for the top-20 model or, if needed, for the top-12 exact-symbolic candidate; otherwise a surrogate fallback report exists with `R^2 >= 0.95`.
9. A case-level local sensitivity / what-if report exists for at least one applicant.

## 11. Final Decision Rule

If the top-20 reduced no-`LayerNorm` model preserves QWK and allows an exact composition that is still usable, that becomes the main insurance-facing framework.

If the top-20 reduced model preserves QWK but the exact formula is still too large, then:

- retrain the top-12 no-`LayerNorm` candidate with the same hyperparameters
- prefer that top-12 model for the exact closed-form insurance-facing version

If the reduced no-`LayerNorm` path loses too much QWK, then:

- keep the current sparse Pareto ChebyKAN with `LayerNorm` as the predictive backbone
- use local finite-difference sensitivities for case explanations
- and produce a closed-form surrogate over the top-20 features instead of forcing an invalid exact formula

That is the fallback path. It is still acceptable, but the preferred outcome is an exact closed-form no-`LayerNorm` simplified KAN.
