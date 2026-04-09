# ParrotLabs

Deep Learning course @ HSG.

## Dataset

Download the Prudential Kaggle data and place the extracted CSVs at:

```text
data/prudential-life-insurance-assessment/train.csv
data/prudential-life-insurance-assessment/test.csv
```

## Setup

```bash
uv sync
```

## Supported Orchestration

`main.py` is the only supported orchestration entrypoint for the current pipeline.

Supported stages:

- `train`
- `tune`
- `interpret`
- `retrain`
- `select`

## Config Entry Points

- Stage A performance tuning:
  - `configs/experiment_stages/stage_a_performance_tuning/chebykan_tune.yaml`
  - `configs/experiment_stages/stage_a_performance_tuning/fourierkan_tune.yaml`
  - `configs/experiment_stages/stage_a_performance_tuning/xgboost_tune.yaml`
- Stage B interpretability tuning plans:
  - `configs/experiment_stages/stage_b_interpretability_tuning/chebykan_retrain_plan.yaml`
  - `configs/experiment_stages/stage_b_interpretability_tuning/fourierkan_retrain_plan.yaml`
- Stage C explanation package:
  - `configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml`
  - `configs/experiment_stages/stage_c_explanation_package/explanation_package_plan.yaml`

## Commands

Install dependencies and verify the dataset:

```bash
uv sync
test -f data/prudential-life-insurance-assessment/train.csv
test -f data/prudential-life-insurance-assessment/test.csv
```

Use the full stage-by-stage runbook in `docs/project_setup/experiment_stages/experiment_stages.md`.

## Artifact Layout

Training artifacts:

- `artifacts/<experiment>/run-summary-*.json`
- `checkpoints/<experiment>/model-*.pt`
- `checkpoints/<experiment>/model-*.joblib`
- `checkpoints/<experiment>/model-*.manifest.json`

Namespaced evaluation and interpretability outputs:

- `outputs/eval/<recipe>/<experiment>/`
- `outputs/interpretability/<recipe>/<experiment>/`

Tune artifacts:

- `sweeps/*_best.json`
- `sweeps/*_best.yaml`
- `sweeps/*_candidates.json`

Retrain and selection artifacts:

- `artifacts/retrain/<family>/<selection_name>/manifest.json`
- `artifacts/selection/<family>_selection.json`

## Legacy Scripts

`src/evaluate.py` and `src/submit.py` are legacy entrypoints. They are intentionally not part of the supported workflow anymore. Use `main.py` instead.

## Tests

```bash
uv run python -m pytest tests -v
```
