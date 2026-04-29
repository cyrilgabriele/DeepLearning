# Stage A Performance Tuning Configs

Stage A configs are used to select the model architecture for each model family under a fixed, established training setup. The intent is to compare architecture capacity while avoiding early tuning of optimizer, regularization, sampling, or other training hyperparameters.

## Scope

These configs should tune architecture-defining parameters only:

- ChebyKAN: network depth, network width, and optionally Chebyshev degree when treated as basis capacity.
- FourierKAN: network depth, network width, and optionally Fourier grid size when treated as basis capacity.
- XGBoost: tree-ensemble capacity parameters such as number of trees and tree depth.

All non-architecture hyperparameters should remain fixed at reasonable field defaults during Stage A. Examples include learning rate, weight decay, batch size, epoch budget, subsampling, column sampling, and regularization strengths. Those parameters belong in Stage B if they need robust performance tuning across seeds, or Stage C if they define the QWK-sparsity tradeoff.

## Search Ranges

The Stage A ranges are intentionally broad enough to cover plausible tabular model capacity without spending trials on extreme configurations:

- ChebyKAN depth: `1..3`; width: `32, 64, 128, 256`; degree: `2..8`.
- FourierKAN depth: `1..3`; width: `32, 64, 128, 256`; grid size: `2..8`.
- XGBoost estimators: `200..1000` in steps of `50`; max depth: `3..15`.

These ranges should produce the Stage A shortlist. Stage B should then use narrower intervals around the best Stage A region and evaluate them across multiple seeds.

## Files

- `stage_a_chebykan/chebykan_tune.yaml`: ChebyKAN architecture sweep.
- `stage_a_fourierkan/fourierkan_tune.yaml`: FourierKAN architecture sweep.
- `stage_a_xgboost/xgboost_tune.yaml`: XGBoost architecture/capacity sweep.

## Commands

Run these commands from the project root:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_chebykan/chebykan_tune.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_fourierkan/fourierkan_tune.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_xgboost/xgboost_tune.yaml
```

## Expected Outputs

Running a Stage A tune config writes the Optuna database, best full config, result JSON, and candidate manifest under `sweeps/stage_a/{model_name}/`. Trial run summaries and predictions are written under `artifacts/stage_a/{experiment_name}/`. The candidate manifest is the handoff into Stage B, where shortlisted architectures can be retrained and evaluated for stability.
