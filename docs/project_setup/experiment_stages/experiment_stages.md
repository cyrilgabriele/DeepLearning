# Experiment Stages

This is the executable runbook for the current experiment pipeline.

Run every command from the repository root.

## Stage Config Tree

- Stage A runnable tune configs:
  - `configs/experiment_stages/stage_a_performance_tuning/stage_a_chebykan/chebykan_tune.yaml`
  - `configs/experiment_stages/stage_a_performance_tuning/stage_a_fourierkan/fourierkan_tune.yaml`
  - `configs/experiment_stages/stage_a_performance_tuning/stage_a_xgboost/xgboost_tune.yaml`
- Stage B administrative plans:
  - `configs/experiment_stages/stage_b_interpretability_tuning/chebykan_retrain_plan.yaml`
  - `configs/experiment_stages/stage_b_interpretability_tuning/fourierkan_retrain_plan.yaml`
  - `configs/experiment_stages/stage_b_interpretability_tuning/xgboost_retrain_plan.yaml`
- Stage C runnable and materialized configs:
  - `configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml`
  - `configs/experiment_stages/stage_c_explanation_package/explanation_package_plan.yaml`
  - `configs/experiment_stages/stage_c_explanation_package/materialized/*.yaml`

## Preconditions

Install dependencies and verify the Kaggle CSVs are present:

```bash
uv sync
test -f data/prudential-life-insurance-assessment/train.csv
test -f data/prudential-life-insurance-assessment/test.csv
```

The pipeline creates `sweeps/`, `artifacts/`, `checkpoints/`, and `outputs/` automatically.

## Stage A: Performance Tuning

Tune each family, then immediately materialize the best full-data run from the generated `_best.yaml`.

### A1. ChebyKAN

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_chebykan/chebykan_tune.yaml
uv run python main.py --stage train --config sweeps/stage_a/chebykan/stage-a-chebykan-tune_best.yaml
```

Main outputs:

- `sweeps/stage_a/chebykan/stage-a-chebykan-tune_best.yaml`
- `sweeps/stage_a/chebykan/stage-a-chebykan-tune_candidates.json`
- `artifacts/stage_a/stage-a-chebykan-tuned/run-summary-*.json`
- `checkpoints/stage-a-chebykan-tuned/model-*.pt`

### A2. FourierKAN

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_fourierkan/fourierkan_tune.yaml
uv run python main.py --stage train --config sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_best.yaml
```

Main outputs:

- `sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_best.yaml`
- `sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_candidates.json`
- `artifacts/stage_a/stage-a-fourierkan-tuned/run-summary-*.json`
- `checkpoints/stage-a-fourierkan-tuned/model-*.pt`

### A3. XGBoost

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_xgboost/xgboost_tune.yaml
uv run python main.py --stage train --config sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.yaml
```

Main outputs:

- `sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.yaml`
- `artifacts/stage_a/stage-a-xgboost-tuned/run-summary-*.json`
- `checkpoints/stage-a-xgboost-tuned/model-*.joblib`

## Stage B: Robust Performance Tuning

The Stage B YAMLs are run-control records. The actual pipeline input is the candidate manifest emitted by Stage A.

The fixed controls are:

- `seeds=13 29 47`
- `qwk_tolerance=0.01`

ChebyKAN and FourierKAN use `top_k=5`; XGBoost uses `top_k=1`.

### B1. ChebyKAN shortlist and robust selection

```bash
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_a/chebykan/stage-a-chebykan-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-chebykan-shortlist --output-experiment-prefix stage-b-chebykan
uv run python main.py --stage select --retrain-manifest artifacts/stage_b/retrain/chebykan/stage-b-chebykan-shortlist/manifest.json --qwk-tolerance 0.01
uv run python -m src.selection.materialize_config --selection-manifest artifacts/stage_b/selection/chebykan_selection.json --role best_performance_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_performance.yaml
```

### B2. FourierKAN shortlist and robust selection

```bash
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-fourierkan-shortlist --output-experiment-prefix stage-b-fourierkan
uv run python main.py --stage select --retrain-manifest artifacts/stage_b/retrain/fourierkan/stage-b-fourierkan-shortlist/manifest.json --qwk-tolerance 0.01
uv run python -m src.selection.materialize_config --selection-manifest artifacts/stage_b/selection/fourierkan_selection.json --role best_performance_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_performance.yaml
```

### B3. XGBoost robust validation

```bash
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_a/xgboost/stage-a-xgboost-tune_candidates.json --top-k 1 --seeds 13 29 47 --selection-name stage-b-xgboost-shortlist --output-experiment-prefix stage-b-xgboost
```

Main Stage B outputs:

- `artifacts/stage_b/retrain/chebykan/stage-b-chebykan-shortlist/manifest.json`
- `artifacts/stage_b/retrain/fourierkan/stage-b-fourierkan-shortlist/manifest.json`
- `artifacts/stage_b/retrain/xgboost-paper/stage-b-xgboost-shortlist/manifest.json`
- `artifacts/stage_b/selection/chebykan_selection.json`
- `artifacts/stage_b/selection/fourierkan_selection.json`
- `configs/experiment_stages/stage_c_explanation_package/materialized/*.yaml`

## Stage C: Explanation Package

Interpret every selected KAN config plus the two baselines. The materialized KAN YAMLs from Stage B make these commands stable and repeatable.

### C1. Train and interpret the GLM baseline

```bash
uv run python main.py --stage train --config configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml
```

### C2. Interpret the Stage A XGBoost winner

```bash
uv run python main.py --stage interpret --config sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.yaml
```

### C3. Interpret the selected ChebyKAN and FourierKAN runs

```bash
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_performance.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_interpretable.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_performance.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_interpretable.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
```

### C4. Assemble the final comparison package

```bash
uv run python -m src.interpretability.final_comparison \
  --selection-manifest artifacts/stage_b/selection/chebykan_selection.json \
  --selection-manifest artifacts/stage_b/selection/fourierkan_selection.json \
  --baseline-config sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.yaml \
  --baseline-config configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml \
  --output-root outputs
```

Main Stage C outputs:

- `outputs/interpretability/kan_paper/stage-c-glm-baseline/`
- `outputs/interpretability/xgboost_paper/stage-a-xgboost-tuned/`
- `outputs/interpretability/kan_paper/<selected-kan-experiment>/`
- `outputs/final_comparison/final_comparison.json`
- `outputs/final_comparison/final_comparison.md`

## Practical Notes

- Stage A uses predictive tuning only. `sparsity_lambda` is fixed to `0.0` there on purpose.
- Stage B is where sparsity regularization matters; the retrain stage enforces it if a candidate comes in dense.
- Stage C uses `scipy` symbolic fitting by default so the workflow does not require Julia or PySR.
- If you rerun any stage, the pipeline will append new timestamped artifacts; the commands above always target the latest matching summary/checkpoint for the chosen experiment name.
