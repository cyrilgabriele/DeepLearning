# Stage B Robust Performance Tuning

Stage B takes the Stage A architecture shortlist and checks which candidates perform robustly across shared retraining seeds. The goal is performance stability, not sparsity or symbolic interpretability.

## Scope

These plans should control robust candidate retraining and performance selection only:

- Select the top Stage A candidates from each model family.
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

## Files

- `chebykan_retrain_plan.yaml`: ChebyKAN robust retraining plan.
- `fourierkan_retrain_plan.yaml`: FourierKAN robust retraining plan.
- `xgboost_retrain_plan.yaml`: XGBoost robust retraining plan.

## Commands

Run these commands from the project root after the Stage A candidate manifests exist:

```bash
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_a/chebykan/stage-a-chebykan-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-chebykan-shortlist --output-experiment-prefix stage-b-chebykan
uv run python main.py --stage select --retrain-manifest artifacts/retrain/chebykan/stage-b-chebykan-shortlist/manifest.json --qwk-tolerance 0.01
uv run python -m src.selection.materialize_config --selection-manifest artifacts/selection/chebykan_selection.json --role best_performance_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_performance.yaml

uv run python main.py --stage retrain --candidate-manifest sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-fourierkan-shortlist --output-experiment-prefix stage-b-fourierkan
uv run python main.py --stage select --retrain-manifest artifacts/retrain/fourierkan/stage-b-fourierkan-shortlist/manifest.json --qwk-tolerance 0.01
uv run python -m src.selection.materialize_config --selection-manifest artifacts/selection/fourierkan_selection.json --role best_performance_candidate --output configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_performance.yaml

uv run python main.py --stage retrain --candidate-manifest sweeps/stage_a/xgboost/stage-a-xgboost-tune_candidates.json --top-k 1 --seeds 13 29 47 --selection-name stage-b-xgboost-shortlist --output-experiment-prefix stage-b-xgboost
```

## Expected Outputs

Running the Stage B retrain command writes a retrain manifest under `artifacts/retrain/{family}/{selection_name}/manifest.json`. For KAN families, the downstream selection step can also write `artifacts/selection/{family}_selection.json` for choosing the robust performance candidate used by later stages.
