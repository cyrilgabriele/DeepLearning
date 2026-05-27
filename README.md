# Interpretable KANs for Prudential Life Insurance Risk Prediction

This repository contains the code, configuration files, runbook, and paper-supporting scripts for a deep-learning course research project on applicant-level risk prediction for life insurance. The project compares Kolmogorov-Arnold Network style tabular models against strong tabular baselines on the Prudential Life Insurance Assessment dataset, with a focus on whether KANs can provide compact, function-level explanations while remaining competitive on ordinal prediction quality.

The central research question is:

> Can ChebyKAN and FourierKAN models produce interpretable feature-response structure for ordinal life-insurance risk prediction without losing too much predictive performance relative to XGBoost and other tabular baselines?

The target is Kaggle's `Response` label, an ordinal risk class from 1 to 8. The primary metric is quadratic weighted kappa (`QWK`), because it rewards near misses more than far-off ordinal predictions. The training pipeline also records mean absolute error, accuracy, and macro F1.

## What Is In This Repository

- `main.py`: the supported orchestration entry point for training, tuning, retraining, selection, interpretation, and comparison.
- `src/training/`: preprocessing dispatch, model fitting, metric computation, checkpointing, run summaries, evaluation exports, and Kaggle-style test predictions.
- `src/models/`: model registry and wrappers for TabKAN variants, XGBoost, GLM, MLP-related code, and TabPFN.
- `src/preprocessing/`: frozen preprocessing recipes for KAN, XGBoost, and TabPFN experiments.
- `src/tune/`: Optuna search orchestration and candidate manifest export.
- `src/retrain/` and `src/selection/`: robust multi-seed candidate validation and final candidate selection.
- `src/interpretability/`: KAN pruning, coefficient importance, symbolic fitting, partial dependence, local explanations, SHAP-style XGBoost analysis, final comparisons, and paper figures/tables.
- `configs/experiment_stages/`: executable experiment configurations organized into Stage A, Stage B, and Stage C.
- `sweeps/`: committed sweep summaries, best configs, candidate manifests, and Optuna databases from previous runs.
- `scripts/`: one-off research scripts used to build paper figures, tables, TabPFN analyses, and additional diagnostics.
- `docs/`: project plans, preprocessing notes, interpretability handoffs, proposal material, and the detailed stage-by-stage runbook.
- `tests/`: unit and integration tests for preprocessing, model configuration, metrics, tuning, retraining, selection, and interpretability utilities.

Generated data, checkpoints, logs, and output artifacts are intentionally not committed. See [Generated Artifacts](#generated-artifacts).

## Data

The project uses the Kaggle Prudential Life Insurance Assessment dataset. Because the dataset is distributed by Kaggle, it is not included in this repository.

Download the competition data and place the extracted files here:

```text
data/prudential-life-insurance-assessment/train.csv
data/prudential-life-insurance-assessment/test.csv
```

The expected training file contains:

- `Id`: applicant identifier.
- `Response`: ordinal target class in `{1, ..., 8}`.
- Prudential product, medical, family-history, employment, and insurance-history feature columns.

The expected test file contains the same feature columns and `Id`, but no `Response`.

Verify the files before running experiments:

```bash
test -f data/prudential-life-insurance-assessment/train.csv
test -f data/prudential-life-insurance-assessment/test.csv
```

## Environment

The project uses Python 3.11 or newer and `uv` for reproducible dependency management. The lockfile `uv.lock` is committed.

Install dependencies:

```bash
uv sync
```

Run commands from the repository root. `main.py` automatically loads `.env` if present, which is useful for optional secrets such as TabPFN-related tokens. The core KAN and XGBoost workflows do not require a `.env` file.

GPU acceleration is used when available through PyTorch. CPU execution is supported but slower, especially for KAN tuning and multi-seed retraining.

## Supported Entry Point

Use `main.py` for the current pipeline:

```bash
uv run python main.py --stage <stage> [options]
```

Supported stages:

- `train`: train one concrete YAML configuration.
- `tune`: run an Optuna sweep from a YAML configuration.
- `retrain`: retrain selected sweep candidates across one or more seeds.
- `select`: choose robust candidates from a retrain manifest.
- `interpret`: generate model-specific explanation artifacts.
- `compare`: assemble final cross-model comparison artifacts from completed runs.

Legacy entry points such as `src/evaluate.py` and `src/submit.py` are retained for historical context but are not the supported workflow.

## Models

Registered model names are defined in `src/models/registry.py`:

| Registry name | Role | Notes |
| --- | --- | --- |
| `tabkan-tiny`, `tabkan-small`, `tabkan-base` | KAN models | Use with `flavor: chebykan`, `fourierkan`, or `bsplinekan`. The main paper workflow focuses on ChebyKAN and FourierKAN. |
| `xgboost-paper` | XGBoost classifier baseline | Paper-style multiclass XGBoost baseline. |
| `xgb` | XGBoost regressor baseline | Regression model with ordinal thresholding for QWK-oriented predictions. |
| `glm` | Simple interpretable baseline | Regularized GLM-style baseline for coefficient signs and magnitudes. |
| `tabpfn-auto` | Predictive-only TabPFN baseline | Optional comparison; uses the `tabpfn_paper` recipe. |

Main KAN variants:

- `ChebyKAN`: smoother polynomial basis, preferred for readable feature-response curves and symbolic summaries.
- `FourierKAN`: more flexible basis for oscillatory structure, useful as the second core KAN comparison.
- `BSplineKAN`: implemented as an optional extension, but not central to the current research claim.

## Preprocessing Recipes

Preprocessing is part of the experiment contract and is recorded in run summaries. The supported recipes are defined by `src/config/preprocessing/preprocessing_config.py` and dispatched in `src/training/trainer.py`.

| Recipe | Intended models | Summary |
| --- | --- | --- |
| `kan_paper` | ChebyKAN, FourierKAN, BSplineKAN | KAN-aligned paper preprocessing and scaling. |
| `kan_sota` | KAN experiments | Alternative KAN-focused recipe with additional missingness handling. |
| `xgboost_paper` | XGBoost baselines | Paper-style preprocessing for XGBoost. |
| `tabpfn_paper` | TabPFN | Thin recipe shim aligned with the XGBoost-style preprocessing convention. |

Some Stage C configs use `preprocessing.selected_features_path` to restrict training and interpretation to a fixed top-feature list under `configs/experiment_stages/stage_c_explanation_package/feature_lists/`.

## Experiment Design

The recommended workflow is staged:

1. Stage A: broad predictive tuning for ChebyKAN, FourierKAN, and XGBoost.
2. Stage B: robust optimizer validation across shared seeds and candidate selection.
3. Stage C: sparsity, pruning, compact 20-feature variants, interpretability artifacts, and final comparison.

The detailed executable runbook lives in `docs/project_setup/experiment_stages/experiment_stages.md`. The commands below summarize the main path and keep all required paths visible from this README.

### Stage A: Predictive Tuning

ChebyKAN:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_chebykan/chebykan_tune.yaml
uv run python main.py --stage train --config sweeps/stage_a/chebykan/stage-a-chebykan-tune_best.yaml
```

FourierKAN:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_fourierkan/fourierkan_tune.yaml
uv run python main.py --stage train --config sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_best.yaml
```

XGBoost:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_xgboost/xgboost_tune.yaml
uv run python main.py --stage train --config sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.yaml
```

Stage A emits best configs and candidate manifests under `sweeps/stage_a/<family>/`, run summaries under `artifacts/stage_a/`, and checkpoints under `checkpoints/<experiment>/`.

### Stage B: Robust Candidate Validation

Stage B narrows optimizer settings and validates candidates across shared seeds (`13`, `29`, `47`) with a `0.01` QWK tolerance.

ChebyKAN:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/chebykan_optimizer_tune.yaml
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-chebykan-optimizer-shortlist --output-experiment-prefix stage-b-chebykan
uv run python main.py --stage select --retrain-manifest artifacts/stage_b/retrain/chebykan/stage-b-chebykan-optimizer-shortlist/manifest.json --qwk-tolerance 0.01
```

FourierKAN:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/fourierkan_optimizer_tune.yaml
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-fourierkan-optimizer-shortlist --output-experiment-prefix stage-b-fourierkan
uv run python main.py --stage select --retrain-manifest artifacts/stage_b/retrain/fourierkan/stage-b-fourierkan-optimizer-shortlist/manifest.json --qwk-tolerance 0.01
```

XGBoost:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/xgboost_optimizer_tune.yaml
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune_candidates.json --top-k 1 --seeds 13 29 47 --selection-name stage-b-xgboost-optimizer-shortlist --output-experiment-prefix stage-b-xgboost
```

To convert selected Stage B candidates into concrete Stage C YAML configs, use:

```bash
uv run python -m src.selection.materialize_config --selection-manifest artifacts/stage_b/selection/chebykan_selection.json --role best_performance_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_performance.yaml
uv run python -m src.selection.materialize_config --selection-manifest artifacts/stage_b/selection/chebykan_selection.json --role best_interpretable_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_interpretable.yaml
uv run python -m src.selection.materialize_config --selection-manifest artifacts/stage_b/selection/fourierkan_selection.json --role best_performance_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_performance.yaml
uv run python -m src.selection.materialize_config --selection-manifest artifacts/stage_b/selection/fourierkan_selection.json --role best_interpretable_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_interpretable.yaml
```

The `materialized/` configs are generated artifacts. If they are missing, regenerate them from the Stage B selection manifests.

### Stage C: Explanation Package

Train and interpret the GLM baseline:

```bash
uv run python main.py --stage train --config configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml
```

Interpret the Stage A XGBoost winner:

```bash
uv run python main.py --stage interpret --config sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.yaml
```

Interpret materialized KAN selections:

```bash
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_performance.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_interpretable.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_performance.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_interpretable.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
```

Stage C can also run predefined full-feature and 20-feature configs directly from:

```text
configs/experiment_stages/stage_c_explanation_package/cheby/
configs/experiment_stages/stage_c_explanation_package/fourier/
configs/experiment_stages/stage_c_explanation_package/xgboost/
configs/experiment_stages/stage_c_explanation_package/tabpfn/
```

For example, the compact top-20 ChebyKAN run used by later interpretability work can be trained with:

```bash
uv run python main.py --stage train --config configs/experiment_stages/stage_c_explanation_package/cheby/20_features/train/chebykan_pareto_q0583_top20.yaml
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/cheby/20_features/train/chebykan_pareto_q0583_top20.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
```

### Final Comparison

After the relevant models have been trained and interpreted, assemble the final comparison package:

```bash
uv run python -m src.interpretability.final_comparison \
  --artifacts-dir artifacts \
  --outputs-dir outputs \
  --glm-checkpoint checkpoints/<glm-experiment>/model-<timestamp>.joblib \
  --xgb-checkpoint checkpoints/<xgboost-experiment>/model-<timestamp>.joblib \
  --chebykan-checkpoint checkpoints/<chebykan-experiment>/model-<timestamp>.pt \
  --fourierkan-checkpoint checkpoints/<fourierkan-experiment>/model-<timestamp>.pt \
  --chebykan-config configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_interpretable.yaml \
  --fourierkan-config configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_interpretable.yaml \
  --chebykan-pruning-summary outputs/interpretability/kan_paper/<chebykan-experiment>/pruning_summary.json \
  --fourierkan-pruning-summary outputs/interpretability/kan_paper/<fourierkan-experiment>/pruning_summary.json \
  --eval-features outputs/eval/kan_paper/<experiment>/features.parquet \
  --eval-labels outputs/eval/kan_paper/<experiment>/labels.parquet
```

The placeholder paths must be replaced with the concrete checkpoint, config, pruning-summary, and evaluation files from the runs being compared. `main.py --stage compare` exposes the same explicit checkpoint/config arguments; inspect `uv run python main.py --stage compare --help` for that interface.

## Configuration Format

Experiment YAML files have three required sections and one optional section:

```yaml
trainer:
  experiment_name: example-run
  train_csv: data/prudential-life-insurance-assessment/train.csv
  test_csv: data/prudential-life-insurance-assessment/test.csv
  seed: 42

preprocessing:
  contract_version: 1
  recipe: kan_paper
  selected_features_path: null

model:
  name: tabkan-base
  flavor: chebykan
  hidden_widths: [128, 64]
  degree: 6
  use_layernorm: true
  params:
    max_epochs: 100
    lr: 0.001
    weight_decay: 0.00001
    batch_size: 256
    sparsity_lambda: 0.0
    l1_weight: 1.0
    entropy_weight: 1.0

tune:
  name: example-tune
  storage: sweeps/example/example-tune.db
  n_trials: 50
  sampler: tpe
  top_k_candidates: 5
  search_space:
    width:
      type: categorical
      choices: [32, 64, 128]
```

Important config rules:

- `trainer.experiment_name` determines artifact and checkpoint names.
- `trainer.train_csv` is required. `trainer.test_csv` is optional but needed for Kaggle-style prediction exports.
- KAN configs require `flavor`, hidden widths or `depth`/`width`, and all required training parameters.
- ChebyKAN requires `degree`; FourierKAN and BSplineKAN require `grid_size`; BSplineKAN also requires `spline_order`.
- `tune.search_space` keys must be supported by the selected model. Invalid keys fail fast during config loading.
- `selected_features_path` may point to JSON or text feature-list files for fixed top-feature experiments.

## Generated Artifacts

The pipeline writes timestamped artifacts so repeated runs do not overwrite previous results.

Training artifacts:

```text
artifacts/<experiment>/run-summary-*.json
artifacts/stage_a/<experiment>/run-summary-*.json
artifacts/stage_b/runs/<experiment>/run-summary-*.json
checkpoints/<experiment>/model-*.pt
checkpoints/<experiment>/model-*.joblib
checkpoints/<experiment>/model-*.manifest.json
```

Evaluation exports:

```text
outputs/eval/<recipe>/<experiment>/features.parquet
outputs/eval/<recipe>/<experiment>/labels.parquet
outputs/eval/<recipe>/<experiment>/raw_features.parquet
```

Interpretability outputs:

```text
outputs/interpretability/<recipe>/<experiment>/
outputs/final_comparison/
```

Sweep and selection outputs:

```text
sweeps/<stage>/<family>/*_best.json
sweeps/<stage>/<family>/*_best.yaml
sweeps/<stage>/<family>/*_candidates.json
sweeps/<stage>/<family>/*.db
artifacts/stage_b/retrain/<family>/<selection_name>/manifest.json
artifacts/stage_b/selection/<family>_selection.json
```

The repository `.gitignore` excludes local data, checkpoints, logs, outputs, runs, and local files. This means a fresh clone can inspect the code and committed configs, but must regenerate local artifacts or receive them separately.

## Reproducing The Research

For the closest reproduction of the full workflow:

1. Install dependencies with `uv sync`.
2. Place the Kaggle CSVs under `data/prudential-life-insurance-assessment/`.
3. Run Stage A tuning and train the emitted best configs.
4. Run Stage B optimizer tuning, multi-seed retraining, and selection.
5. Materialize Stage C best-performance and best-interpretable KAN configs.
6. Run Stage C baseline training, KAN interpretation, pruning, and symbolic fitting.
7. Build final comparison outputs.
8. Run the relevant tests before changing code or reporting results.

For a faster reproduction or extension, start from committed best/candidate YAMLs under `sweeps/` and concrete Stage C configs under `configs/experiment_stages/stage_c_explanation_package/`. This avoids rerunning broad Stage A searches.

For a narrow interpretability extension, use the 20-feature configs and feature lists in Stage C. These are faster and easier to inspect than full-feature KANs.

## Tests

Run the full test suite:

```bash
uv run python -m pytest tests -v
```

Run focused tests while developing:

```bash
uv run python -m pytest tests/metrics/test_qwk.py -v
uv run python -m pytest tests/tune/test_sweep.py -v
uv run python -m pytest tests/selection tests/retrain -v
uv run python -m pytest tests/interpretability -v
```

Most tests use synthetic fixtures and do not require Kaggle data. Full training, tuning, and interpretation commands do require the Kaggle CSVs.

## Paper And Documentation Map

- `paper.txt`: paper text draft or extracted paper content.
- `docs/proposal/`: original project proposal material.
- `docs/project_setup/project_steps.md`: scientific workflow rationale and model-scope decisions.
- `docs/project_setup/experiment_stages/experiment_stages.md`: detailed executable stage runbook.
- `docs/preprocessing/`: preprocessing notes and missing-data handling rationale.
- `docs/interpretability/`: interpretability plans, handoffs, closed-form notes, ordinal calibration explanation, and paper-supporting material.
- `docs/interpretability/paper_draft/`: generated LaTeX snippets and figures used in the paper draft.
- `scripts/bootstrap_qwk_table1.py`: bootstrap QWK table helper.
- `scripts/build_figure1_interpretability.py`, `scripts/build_figure3_waterfall.py`, and plotting scripts: paper figure helpers.
- `scripts/tabpfn_*`: TabPFN comparison, validation QWK, attention, permutation importance, and overlap helpers.

## Extending The Project

Use the existing pipeline contracts when adding a new experiment:

- Add or copy a YAML config under `configs/experiment_stages/`.
- Keep preprocessing fixed when comparing models, unless the experiment is explicitly about preprocessing.
- Use `main.py --stage train` for concrete runs and `main.py --stage tune` for Optuna searches.
- Record new top-feature lists under `configs/experiment_stages/stage_c_explanation_package/feature_lists/` when a compact model depends on them.
- Add tests when changing shared config validation, preprocessing, model registry behavior, metric computation, artifact paths, or interpretability logic.
- Avoid relying on notebook state or local-only files for paper claims; every claim should trace back to a config, script, artifact manifest, or documented command.

## Known Practical Constraints

- Full Optuna tuning and multi-seed retraining can be slow on CPU.
- TabPFN runs can be resource-intensive and are optional for the core KAN-vs-XGBoost claim.
- The committed repository does not include Kaggle data or local checkpoints.
- Timestamped artifacts make reruns append new files; when comparing results, check the run summary and checkpoint manifest for the exact config, seed, preprocessing recipe, and timestamp.
- KAN coefficient importance and symbolic approximations are explanation tools, not causal effects. SHAP-style XGBoost explanations and GLM coefficients are separate baselines with different semantics.
