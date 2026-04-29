# Stage B Robust Performance Tuning

Stage B takes the Stage A architecture shortlist, tunes narrowed optimizer intervals, and checks which candidates perform robustly across shared retraining seeds. The goal is dense-model performance stability, not sparsity or symbolic interpretability.

## Scope

These plans should control robust candidate retraining and performance selection only:

- Select the top Stage A candidates from each model family.
- Tune narrowed optimizer intervals on the fixed Stage A architecture.
- Retrain each selected candidate with the same seed list.
- Compare mean validation QWK and seed-to-seed variability.
- Use `qwk_tolerance` only as a performance-selection tolerance around the best mean QWK.

Stage B should not tune or evaluate interpretability-specific controls. Pruning thresholds, sparsity penalties, symbolic candidate libraries, and QWK-vs-sparsity tradeoffs belong to Stage C.

## Shared Controls

All model families use the same retraining seeds:

- `13`
- `29`
- `47`

ChebyKAN and FourierKAN keep `top_k: 5` because Stage A exports several candidate architectures for robust comparison. XGBoost keeps `top_k: 1` because the Stage A XGBoost config exports one best candidate.

## Optimizer Search Spaces

ChebyKAN and FourierKAN tune only dense optimizer hyperparameters after Stage A has fixed architecture:

- `lr`: log-uniform from `0.0001` to `0.01`
- `weight_decay`: log-uniform from `0.000001` to `0.001`

The XGBoost stage configs use the `xgb` registry model, backed by `src/models/xgb_baseline.py` (`XGBRegressor` plus optimized ordinal thresholds). XGBoost has no parameter literally named `weight_decay`; its L2 regularization analogue is `reg_lambda`. Its Stage B optimizer search is:

- `learning_rate`: log-uniform from `0.01` to `0.3`
- `reg_lambda`: log-uniform from `0.1` to `10.0`

## Files

- `chebykan_optimizer_tune.yaml`: ChebyKAN optimizer tune config with Stage A architecture fixed.
- `chebykan_retrain_plan.yaml`: ChebyKAN robust retraining plan.
- `fourierkan_optimizer_tune.yaml`: FourierKAN optimizer tune config with Stage A architecture fixed.
- `fourierkan_retrain_plan.yaml`: FourierKAN robust retraining plan.
- `xgboost_optimizer_tune.yaml`: XGBoost optimizer tune config with Stage A tree architecture fixed.
- `xgboost_retrain_plan.yaml`: XGBoost robust retraining plan.

## Commands

Run these commands from the project root after the Stage A candidate manifests exist:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/chebykan_optimizer_tune.yaml
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-chebykan-optimizer-shortlist --output-experiment-prefix stage-b-chebykan
uv run python main.py --stage select --retrain-manifest artifacts/stage_b/retrain/chebykan/stage-b-chebykan-optimizer-shortlist/manifest.json --qwk-tolerance 0.01
uv run python -m src.selection.materialize_config --selection-manifest artifacts/stage_b/selection/chebykan_selection.json --role best_performance_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_performance.yaml

uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/fourierkan_optimizer_tune.yaml
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-fourierkan-optimizer-shortlist --output-experiment-prefix stage-b-fourierkan
uv run python main.py --stage select --retrain-manifest artifacts/stage_b/retrain/fourierkan/stage-b-fourierkan-optimizer-shortlist/manifest.json --qwk-tolerance 0.01
uv run python -m src.selection.materialize_config --selection-manifest artifacts/stage_b/selection/fourierkan_selection.json --role best_performance_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_performance.yaml

uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/xgboost_optimizer_tune.yaml
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune_candidates.json --top-k 1 --seeds 13 29 47 --selection-name stage-b-xgboost-optimizer-shortlist --output-experiment-prefix stage-b-xgboost
```

## Expected Outputs

Running the Stage B retrain command writes per-run summaries under `artifacts/stage_b/runs/{experiment_name}/` and a retrain manifest under `artifacts/stage_b/retrain/{family}/{selection_name}/manifest.json`. For KAN families, the downstream selection step can also write `artifacts/stage_b/selection/{family}_selection.json` for choosing the robust performance candidate used by later stages.
