# Project Pipeline Audit

Date: 2026-04-09

Purpose: dense handoff note for a fresh Codex instance. This document captures the current project state relative to the intended workflow in `docs/project_setup/project_steps.md`, with emphasis on what is actually executable, what is stale, and what must be adjusted implementation-wise before the pipeline can be run step-by-step in a clean and defensible way.

## Scope and framing

The user originally referred to `docs/process_pipeline/project_steps.md`, but in the current repo the actual file is:

- `docs/project_setup/project_steps.md`

That is the workflow reference used for this audit.

Main question answered here:

- assuming preprocessing is frozen, what is missing in the current project setup and pipeline?
- what must be adjusted so the project can be executed step-by-step?

Bottom line:

- the core `main.py` path for `train`, `tune`, and `interpret` is usable
- the full research workflow described in `docs/project_setup/project_steps.md` is not yet implemented end-to-end
- the biggest issues are workflow completeness, artifact consistency, stale side scripts, and stale docs
- this is not primarily a preprocessing failure at this point

## What was verified

The following targeted test suite was run:

```bash
uv run python -m pytest tests/test_main.py tests/training/test_trainer.py tests/tune/test_sweep.py tests/interpretability/test_pipeline.py -q
```

Result:

- `11 passed`

Interpretation:

- `main.py` dispatch works
- the current `Trainer` runs
- the current `tune` path runs
- the current `interpret` entrypoint runs for the tested cases

This means the repo is not in a generally broken state. The problems are more structural than catastrophic.

## Current supported core path

The supported orchestrator is:

- `main.py`

Supported stages there:

- `train`
- `tune`
- `interpret`

Key files:

- `main.py`
- `src/training/trainer.py`
- `src/tune/sweep.py`
- `src/interpretability/pipeline.py`

Current config system:

- `configs/config_loader.py`
- `configs/train/trainer_config.py`
- `configs/preprocessing/preprocessing_config.py`
- `configs/model/model_config.py`
- `configs/tune/tune_config.py`

Current preprocessing recipes actually supported by the typed config:

- `xgboost_paper`
- `kan_paper`
- `kan_sota`

Those are defined in:

- `configs/preprocessing/preprocessing_config.py`

## Main conclusion

Relative to `docs/project_setup/project_steps.md`, the project currently supports:

- freezing scope conceptually
- running one training job
- running one Optuna sweep
- exporting eval artifacts namespaced by preprocessing recipe and experiment
- running model-specific interpretability for `glm`, `xgboost-paper`, `chebykan`, and `fourierkan`

What it does not yet support cleanly:

- encoding a frozen preprocessing setup beyond just the recipe name
- keeping top KAN candidates instead of only the single best sweep result
- running multi-seed retraining and stability checks
- selecting final KANs according to the documented rule
- artifact-aware KAN reconstruction for non-uniform architectures
- running the final comparison on the current namespaced artifact layout
- relying on the README and older utility scripts without confusion

## Detailed findings

### 1. Frozen preprocessing is only partially encoded

The project steps say preprocessing should be locked before final comparison.

Current reality:

- `PreprocessingConfig` only stores `recipe`
- no additional preprocessing parameters are part of the validated typed config

Relevant file:

- `configs/preprocessing/preprocessing_config.py`

However, several YAML configs still contain old fields such as:

- `eval_size`
- `missing_threshold`
- `stratify`
- `use_stratified_kfold`
- `kan_n_splits`

Examples:

- `configs/model/chebykan_experiment.yaml`
- `configs/model/fourierkan_experiment.yaml`
- `configs/model/glm_experiment.yaml`

These fields are currently ignored by the typed config loader rather than enforced. This is dangerous because it gives the impression that preprocessing is explicitly frozen when in fact only the recipe is being used.

Practical implication:

- the repo cannot yet say, in a strong reproducibility sense, that preprocessing is fully frozen by config
- there is config drift between what users think matters and what the runtime actually consumes

### 2. There are still two different pipeline eras in the repo

The active pipeline is the `main.py` + `Trainer` + namespaced outputs path.

But older scripts still exist and reflect a different workflow:

- `src/submit.py`
- `src/evaluate.py`
- `src/preprocessing/dataset.py`

Problems with those scripts:

- they rely on the older `PrudentialDataModule` path
- they assume older checkpoint conventions such as `.ckpt`
- they bypass the current trainer artifact conventions
- they do not align with the current namespaced eval and interpretability outputs

Practical implication:

- a fresh agent or user can easily start from the wrong execution path
- the repo currently contains both the new path and legacy path, without a single strongly enforced one

### 3. The README is stale enough to mislead execution

The README references config paths that do not exist anymore, such as:

- `configs/smoke_experiment.yaml`
- `configs/experiments/kan_cheby_single.yaml`
- `configs/experiments/xgboost_paper_experiment.yaml`

But the actual current config files live under:

- `configs/model/`
- `configs/tune/`

Relevant files:

- `README.md`
- `configs/model/chebykan_experiment.yaml`
- `configs/model/fourierkan_experiment.yaml`
- `configs/model/glm_experiment.yaml`
- `configs/model/xgboost_paper_experiment.yaml`

Practical implication:

- even though the core code works, the documented step-by-step workflow is currently misleading

### 4. Tuning exists, but the project-step workflow is only partially covered

The tune stage is implemented in:

- `src/tune/sweep.py`

This stage is functional and uses the same trainer path as training.

But relative to the documented workflow, gaps remain:

- only a ChebyKAN tune config exists under `configs/tune/kan_cheby/`
- no corresponding FourierKAN tune config is present
- `run_tune()` persists only the single best config for later training
- the project steps explicitly require keeping a small candidate set, not just one winner

Relevant files:

- `src/tune/sweep.py`
- `configs/tune/kan_cheby/kan_cheby_tune.yaml`
- `configs/tune/xgboost_paper/xgboost_paper_tune.yaml`

Practical implication:

- step 4 is partially implemented
- step 5 is not implemented in the intended sense

### 5. Candidate management is missing

The project steps require:

- keep the top few validation-QWK KAN candidates
- do not collapse immediately to one winner

Current behavior:

- `src/tune/sweep.py` writes one `*_best.json`
- `src/tune/sweep.py` writes one `*_best.yaml`
- it prints top trials, but there is no structured candidate manifest emitted for downstream retraining

Practical implication:

- there is no clean machine-readable bridge from predictive tuning to interpretability-aware retraining

### 6. Multi-seed retraining and seed stability checks are not implemented

The project steps require:

- retrain selected KAN candidates with sparsity regularization enabled
- run multiple seeds
- compare stability

Current behavior:

- `TrainerConfig` supports only one seed per run
- there is no stage that expands one selected config across multiple seeds
- there is no aggregation logic for stability summaries
- no candidate-level or family-level selection artifact exists

Relevant files:

- `configs/train/trainer_config.py`
- `src/training/trainer.py`
- `src/tune/sweep.py`

Practical implication:

- step 6 is not implemented
- the current pipeline cannot yet support a credible stability-based final selection

### 7. Final KAN selection logic is not implemented

The project steps require selecting for each KAN family:

- best-performance model
- best-interpretable model within `<= 0.01` validation QWK of the best

Current reality:

- pruning and symbolic pipelines exist
- there is no stage that consumes multiple KAN candidates and applies that rule
- there is no selector that combines predictive score, post-pruning compactness, basis complexity, curve cleanliness, and seed stability

Relevant files:

- `docs/project_setup/project_steps.md`
- `src/interpretability/kan_pruning.py`
- `src/interpretability/kan_symbolic.py`
- `src/interpretability/r2_pipeline.py`

Practical implication:

- step 8 is still a manual interpretation task, not a reproducible pipeline stage

### 8. KAN interpretability is only partially artifact-aware

The high-level interpret stage in:

- `src/interpretability/pipeline.py`

is in much better shape than older code because it:

- uses namespaced eval directories
- resolves config from checkpoint-linked run summaries
- writes outputs under recipe and experiment namespaces

However, the KAN-specific loaders still rebuild architectures from:

- scalar `depth`
- scalar `width`

Examples:

- `src/interpretability/kan_pruning.py`
- `src/interpretability/kan_symbolic.py`
- `src/interpretability/r2_pipeline.py`
- `src/interpretability/final_comparison.py`

This is a problem because:

- old and tuned KAN artifacts may use non-uniform hidden widths
- `ModelConfig` currently cannot represent `hidden_widths: list[int]`
- several interpretability scripts assume `widths = [width] * depth`

Practical implication:

- the current interpret pipeline is acceptable for runs whose training config truly used uniform widths
- it is not a robust general solution for final artifact-aware KAN analysis
- step 2 from `project_steps.md` is therefore only partially satisfied

### 9. Standardized final artifact format is still missing

The project steps explicitly call for one final artifact format before retraining.

Current outputs are split across several locations and conventions:

- run summaries under `artifacts/<experiment>/run-summary-*.json`
- checkpoints under `checkpoints/<experiment>/model-*.pt` or `.joblib`
- eval data under `outputs/eval/<recipe>/<experiment>/`
- interpret outputs under `outputs/interpretability/<recipe>/<experiment>/`

This is already much better than a flat output space, but still incomplete as a final artifact contract because there is no single manifest that says:

- exact preprocessing identity
- exact architecture
- exact basis parameters
- seed
- checkpoint path
- eval artifact path
- pruning summary path
- symbolic output path
- selection metadata

Practical implication:

- artifact resolution is still spread across conventions rather than one canonical manifest

### 10. Final comparison is still wired to an older artifact layout

The file:

- `src/interpretability/final_comparison.py`

still assumes older paths and conventions such as:

- flat `outputs` expectations
- older checkpoint locations and names
- older experiment names like `xgb-baseline`
- scalar-width KAN reconstruction

It does not naturally consume the current namespaced outputs produced by:

- `src/training/trainer.py`
- `src/interpretability/pipeline.py`

Practical implication:

- step 9 can be run per model
- the cross-model final assembly step is not yet aligned with the current pipeline

### 11. The current core path is usable, but only for a narrower workflow

What can be done today, using the intended modern path:

1. Train one config with `main.py --stage train`
2. Tune one config with `main.py --stage tune`
3. Interpret one trained checkpoint with `main.py --stage interpret`

What cannot yet be done cleanly as a reproducible pipeline:

1. freeze a complete preprocessing contract
2. tune all in-scope model families under one clean convention
3. carry top-k KAN candidates forward automatically
4. retrain those candidates across multiple seeds
5. select final KANs by the documented performance vs interpretability rule
6. generate the final comparison package from the current artifact layout without manual glue

## Recommended implementation changes

Priority is ordered by what most directly unblocks the documented workflow.

### Priority 1: make config and documentation truthful

Do this first because it reduces user and agent confusion immediately.

1. Make config validation strict for stale keys.
2. Remove dead keys from the YAMLs or formally add them to typed config if they are intended to matter.
3. Update the README to the real config paths and real command examples.
4. Explicitly point users to `main.py` as the only supported orchestration path.
5. Mark `src/submit.py` and `src/evaluate.py` as legacy or rewrite them.

Recommended concrete change:

- switch Pydantic config handling from permissive ignore behavior to a stricter mode where unexpected keys fail loudly, unless there is a strong reason to keep backward compatibility

### Priority 2: encode frozen preprocessing as an actual artifact and config contract

If preprocessing is declared frozen, that needs to be machine-verifiable.

Suggested change:

1. Extend `PreprocessingConfig` beyond just `recipe`.
2. Add only the parameters that genuinely define the frozen recipe.
3. Persist a preprocessing fingerprint in the run summary and checkpoint-side manifest.

Minimum useful fields would likely include:

- `recipe`
- any recipe-specific knobs that materially affect the produced feature space
- a derived feature-space fingerprint or hash

### Priority 3: add explicit KAN architecture representation

This is the main interpretability-side structural gap.

Suggested change:

1. Add `hidden_widths: list[int]` to the model config and artifact payload.
2. Keep `depth` and `width` only as convenience inputs if needed.
3. Normalize to one canonical internal representation before training and before interpretation.

Then update all KAN reconstruction points:

- `src/interpretability/kan_pruning.py`
- `src/interpretability/kan_symbolic.py`
- `src/interpretability/r2_pipeline.py`
- `src/interpretability/final_comparison.py`

Goal:

- no interpretability code should ever assume `widths = [width] * depth` unless that was explicitly the trained architecture

### Priority 4: add missing tune configs and top-k candidate export

This unblocks steps 4 and 5 from `project_steps.md`.

Suggested change:

1. Add a FourierKAN tune config mirroring the ChebyKAN tune path.
2. Extend `run_tune()` to emit a structured candidate manifest.

That manifest should contain, for example:

- trial number
- QWK
- architecture
- basis parameters
- sparsity parameters
- checkpoint path if kept
- summary path

Instead of only:

- one best JSON
- one best YAML

Recommended output:

- `sweeps/<study_name>_candidates.json`

with the top `k` complete trials, where `k` is configurable

### Priority 5: add a retraining stage for selected KAN candidates

This is the largest missing workflow stage.

Suggested new stage under `main.py`:

- `retrain`

Inputs:

- candidate manifest
- selected candidate ids or top-k filter
- seed list
- optional output experiment prefix

Behavior:

1. materialize one run per candidate per seed
2. enforce sparsity regularization for the retrain stage
3. keep the same preprocessing recipe
4. export a structured retraining manifest

Recommended output:

- `artifacts/retrain/<family>/<selection_name>/manifest.json`

### Priority 6: add a selector stage implementing the documented rule

Suggested new stage under `main.py`:

- `select`

Inputs:

- retraining manifest
- interpretability outputs

Behavior:

1. compute best-performance model per KAN family
2. compute candidate set within `<= 0.01` QWK of the family best
3. rank those by interpretability criteria
4. emit final chosen models and supporting metrics

Selection criteria should include:

- validation QWK
- QWK after pruning
- active edges after pruning
- sparsity ratio
- basis complexity
- symbolic fit quality summary
- seed stability summary

Recommended output:

- `artifacts/selection/<family>_selection.json`

### Priority 7: rewrite final comparison to consume current namespaced outputs

The current `final_comparison.py` should be treated as legacy until rewritten.

Rewrite it so it consumes:

- selected-model manifests
- namespaced eval artifacts
- namespaced interpretability outputs
- current checkpoint conventions

It should not:

- guess old checkpoint paths
- assume flat outputs
- assume older experiment names
- rebuild KANs using scalar width and depth only

### Priority 8: decide what to do with legacy utility scripts

Two acceptable options:

1. Rewrite `src/submit.py` and `src/evaluate.py` to use current artifacts.
2. Mark them clearly as legacy and remove them from current workflow documentation.

At the moment they are context traps.

## Recommended step-by-step execution model after the fixes

This is the intended clean workflow once the missing pieces above are implemented.

### Stage A: lock final setup

1. choose final preprocessing recipe
2. encode it fully in typed config
3. choose final model scope: `ChebyKAN`, `FourierKAN`, `XGBoost`, `GLM`

### Stage B: predictive tuning

1. tune `ChebyKAN`
2. tune `FourierKAN`
3. tune `XGBoost`
4. export top-k KAN candidates per family

### Stage C: interpretability-aware retraining

1. retrain top KAN candidates with sparsity enabled
2. run multiple seeds
3. keep artifacts structured

### Stage D: post-training interpretation

1. prune each retrained KAN
2. run symbolic fits and R2 summaries
3. run XGBoost SHAP
4. run GLM coefficients

### Stage E: final selection

1. select best-performance KAN per family
2. select best-interpretable KAN within `<= 0.01` QWK of the best
3. compare against `XGBoost` and `GLM`

### Stage F: reporting

1. assemble final cross-model comparison from current artifact namespaces
2. write the final narrative with the proper claim strength

## Suggested immediate next implementation slice

If a fresh agent should continue from here, the most leverage-heavy first slice is:

1. make config validation strict and remove stale YAML fields
2. update the README to the real paths and commands
3. add a FourierKAN tune config
4. extend `run_tune()` to export top-k candidate manifests
5. add explicit `hidden_widths` support and update all KAN interpretability loaders

Why this slice first:

- it removes the most confusion
- it aligns the repo with the documented project steps
- it creates the bridge from tuning to retraining
- it fixes the largest remaining architecture-level interpretability mismatch

## Files most relevant for the next agent

Core orchestration:

- `main.py`
- `src/training/trainer.py`
- `src/tune/sweep.py`
- `src/interpretability/pipeline.py`

Config system:

- `configs/config_loader.py`
- `configs/train/trainer_config.py`
- `configs/preprocessing/preprocessing_config.py`
- `configs/model/model_config.py`
- `configs/tune/tune_config.py`

Experiment configs:

- `configs/model/chebykan_experiment.yaml`
- `configs/model/fourierkan_experiment.yaml`
- `configs/model/glm_experiment.yaml`
- `configs/model/xgboost_paper_experiment.yaml`
- `configs/tune/kan_cheby/kan_cheby_tune.yaml`
- `configs/tune/xgboost_paper/xgboost_paper_tune.yaml`

Interpretability loaders and outputs:

- `src/interpretability/kan_pruning.py`
- `src/interpretability/kan_symbolic.py`
- `src/interpretability/r2_pipeline.py`
- `src/interpretability/shap_xgboost.py`
- `src/interpretability/glm_coefficients.py`
- `src/interpretability/final_comparison.py`
- `src/interpretability/utils/paths.py`

Legacy or potentially misleading:

- `README.md`
- `src/submit.py`
- `src/evaluate.py`
- `src/preprocessing/dataset.py`

Workflow reference:

- `docs/project_setup/project_steps.md`

Related compatibility context:

- `docs/interpretability/model_pipeline_compatibility.md`

## Final condensed summary

If a fresh agent needs the shortest possible version:

- the current `train/tune/interpret` core path works
- preprocessing is not fully frozen in config, only the recipe is
- stale YAML fields are silently ignored
- README and legacy scripts still point to an older pipeline era
- tune currently exports only one winner, not top-k KAN candidates
- no multi-seed retraining stage exists
- no final selection stage exists
- KAN interpretation still reconstructs models from scalar `depth` and `width`, which is not robust enough
- final comparison still expects older artifact layouts
- the next best implementation slice is: strict configs, doc cleanup, Fourier tune config, top-k candidate export, and explicit `hidden_widths` support
