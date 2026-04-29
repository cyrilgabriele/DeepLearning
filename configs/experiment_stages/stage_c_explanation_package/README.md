# Stage C Explanation Package Configs

Stage C uses the robust candidates from Stage B to build the explanation package and to explore the QWK-vs-sparsity tradeoff. At this point, architecture and ordinary performance hyperparameters should already be narrowed or fixed; Stage C should focus on sparsity, pruning, feature-restricted explanation runs, and final comparison artifacts.

## Scope

Stage C configs should cover:

- Full-feature sparsity tuning for the large KAN models.
- Pareto-front runs that optimize validation QWK and sparsity together.
- Post-training pruning and symbolic analysis during interpretation.
- 20-feature follow-up runs for compact explanation targets.
- Baseline training and interpretation for GLM and XGBoost comparisons.

Stage C should not reopen broad architecture search. Architecture belongs to Stage A, and robust performance validation across seeds belongs to Stage B.

## Layout

- `cheby/all_features/train/`: full-feature ChebyKAN train configs.
- `cheby/all_features/tune/`: full-feature ChebyKAN sparsity/Pareto tune configs.
- `cheby/20_features/train/`: 20-feature ChebyKAN train configs.
- `cheby/20_features/tune/`: 20-feature ChebyKAN tune configs.
- `fourier/all_features/train/`: full-feature FourierKAN train configs.
- `fourier/all_features/tune/`: full-feature FourierKAN sparsity/Pareto tune configs.
- `fourier/20_features/train/`: 20-feature FourierKAN train configs.
- `fourier/20_features/tune/`: 20-feature FourierKAN tune configs.
- `xgboost/all_features/train/`: full-feature XGBoost baseline train configs.
- `xgboost/20_features/train/`: 20-feature XGBoost baseline train configs.
- `xgboost/20_features/tune/`: 20-feature XGBoost tune configs.
- `feature_lists/`: selected feature lists used by 20-feature configs.

## Full-Feature Tuning Commands

Run these commands from the project root after Stage A and Stage B have established the robust architecture/performance settings:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_c_explanation_package/cheby/all_features/tune/chebykan_pareto_sparsity.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_c_explanation_package/fourier/all_features/tune/fourierkan_pareto_sparsity.yaml
```

Optional wider sparsity searches for the full-feature KAN models:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_c_explanation_package/cheby/all_features/tune/chebykan_sparsity_tune.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_c_explanation_package/fourier/all_features/tune/fourierkan_sparsity_tune.yaml
```

## Full-Feature Baseline Commands

Run these from the project root when regenerating full-feature baselines:

```bash
uv run python main.py --stage train --config configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml

uv run python main.py --stage train --config configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgboost_best.yaml
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgboost_best.yaml
```

## Interpretation Commands

After selecting or materializing final KAN configs, run interpretation with pruning and symbolic fitting controls:

```bash
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_performance.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
uv run python main.py --stage interpret --config configs/experiment_stages/stage_c_explanation_package/materialized/fourierkan_best_performance.yaml --pruning-threshold 0.01 --qwk-tolerance 0.01 --candidate-library scipy
```

## Stage Dependency

Yes: run Stage A first, then Stage B, before finalizing Stage C intervals. Stage A identifies plausible architectures, and Stage B checks which candidates are robust across seeds. Stage C should then freeze or narrowly adjust those robust settings and tune sparsity/Pareto behavior on top of them.
