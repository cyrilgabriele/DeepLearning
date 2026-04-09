# Model-Pipeline Compatibility Assessment

This note captures the current mismatch between the saved model artifacts under `models/` and what the interpretability pipeline currently expects.

## Bottom Line

At the moment, this looks primarily like a code-compatibility problem, not a retraining problem.

- The KAN interpretability loaders rebuild the wrong architecture from config.
- The XGBoost SHAP loader expects a different serialization format than the saved artifacts use.
- There is also a preprocessing-space mismatch between KAN and XGBoost artifacts that should be handled explicitly in the pipeline.

So the clean first step is to make the pipeline artifact-aware before deciding to retrain anything.

## Verified Findings

### 1. KAN mismatch is architecture reconstruction, not obviously bad weights

Several interpretability entry points reconstruct `TabKAN` from experiment config values such as:

- `src/interpretability/r2_pipeline.py`
- `src/interpretability/kan_symbolic.py`
- `src/interpretability/kan_pruning.py`
- `src/interpretability/comparison_side_by_side.py`
- `src/interpretability/comparison_per_risk.py`
- `src/interpretability/feature_risk_influence.py`
- `src/interpretability/final_comparison.py`

The current reconstruction pattern assumes uniform widths via:

```python
widths = [cfg.model.width] * cfg.model.depth
```

That does not match the saved KAN artifacts under `models/`.

Example: `models/cheby/chebykan_kan_paper_20260401_103120.json`

- saved widths: `[128, 64]`
- saved degree: `6`

But `configs/chebykan_experiment.yaml` currently says:

- depth: `2`
- width: `128`
- degree: `3`

The saved `.pt` checkpoint also confirms a non-uniform architecture:

- `140 -> 128 -> 64`

Example: `models/fourier/fourierkan_kan_paper_20260401_103745.json`

- saved widths: `[64, 256, 64]`
- saved grid size: `8`

The saved checkpoint confirms:

- `140 -> 64 -> 256 -> 64`

Conclusion:

- The KAN artifacts themselves appear structurally valid.
- The current interpretability code is reconstructing the wrong model before loading weights.
- This points to loader fixes, not immediate retraining.

### 2. XGBoost mismatch is serialization/loading convention

`src/interpretability/shap_xgboost.py` currently assumes a `joblib`-serialized wrapper object and loads `wrapper.model`.

The saved artifacts under `models/xgb/` are different:

- raw XGBoost `.json` model files
- separate artifact metadata `.json` files

This is still usable.

Verified behavior:

- `xgb_paper_xgb_paper_20260401_104011.json` loads cleanly as `xgboost.XGBClassifier`
- `xgb_xgb_paper_20260401_105856.json` loads cleanly as `xgboost.XGBRegressor`

For the baseline `xgb` artifact, the companion artifact JSON also contains the threshold vector needed to convert regression outputs back to ordinal predictions.

Conclusion:

- The saved XGBoost files are valid.
- The SHAP loader is just too narrow about accepted checkpoint formats.
- This is also fixable code-side.

### 3. There is a real preprocessing-space mismatch

The saved models were not all trained on the same encoded feature space.

KAN artifacts use `kan_paper` preprocessing:

- 140 features

XGBoost paper-faithful artifacts use `xgb_paper` preprocessing:

- 126 features

This matters because some interpretability code currently assumes one shared exported eval matrix such as:

- `outputs/data/X_eval.parquet`
- `outputs/reports/feature_types.json`

That is not a clean abstraction if different model families were trained on different feature spaces.

Conclusion:

- KAN pruning and symbolic analysis should use KAN-space eval features.
- XGBoost SHAP should use XGBoost-space eval features.
- Cross-model comparison should happen on aligned derived outputs or raw feature identities, not by forcing every model onto one encoded matrix.

### 4. Current KAN interpretability support is narrower than the set of saved models

The current interpretability code is mainly written around:

- `chebykan`
- `fourierkan`

Artifacts also exist for:

- `bsplinekan`
- `mlp`

Those are not first-class citizens in the current interpretability pipeline.

## Recommendation

I would not retrain yet.

The cleaner sequence is:

1. Add one shared artifact-aware loader for KAN checkpoints.
2. Add one shared artifact-aware loader for XGBoost checkpoints.
3. Make interpretability scripts prefer artifact metadata first and config files second.
4. Namespace exported eval artifacts by preprocessing recipe instead of assuming one global `outputs/data/X_eval.parquet`.
5. Run the interpretability workflow first on `ChebyKAN`.

## What the Code Fix Should Do

For KAN:

- read widths from artifact metadata when available
- read `degree` or `grid_size` from artifact metadata when available
- reconstruct the exact saved architecture before calling `load_state_dict`

For XGBoost:

- detect whether the checkpoint is:
  - a `joblib` wrapper
  - a raw `XGBClassifier` JSON
  - a raw `XGBRegressor` JSON
- if using regression XGBoost artifacts, also load thresholds from the companion artifact metadata when ordinal predictions are needed

For exported eval data:

- separate KAN eval exports from XGBoost eval exports
- avoid assuming that one encoded feature matrix works for all models

## When Retraining Might Still Make Sense

Retraining becomes reasonable if the goal is not just "make current checkpoints usable", but instead:

- standardize one checkpoint/artifact format for all future runs
- standardize one preprocessing space across model families
- simplify the interpretability code so it does not need backward-compatibility branches

That is a workflow simplification argument, not evidence that the current saved models are unusable.

## Practical Decision

At this stage, the best assumption is:

- the saved checkpoints are probably usable
- the pipeline needs to be made artifact-aware
- retraining should only be considered after those compatibility fixes are in place

If the artifact-aware loader still fails after reconstructing the exact saved architecture and preprocessing context, then retraining becomes a stronger possibility. Right now, the evidence points to pipeline mismatch first.
