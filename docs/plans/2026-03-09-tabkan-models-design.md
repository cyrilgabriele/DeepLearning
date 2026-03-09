# TabKAN Model Training Framework Design

**Date:** 2026-03-09
**Branch:** `feature/tabkan-models`
**Based on:** `dev-SOTA-preprocessing`

## Decisions

- **Config management:** Hydra (composable YAML configs)
- **Training framework:** PyTorch Lightning
- **KAN implementation:** `efficient-kan` foundation + custom ChebyKAN/FourierKAN layers
- **Baselines:** MLP (Lightning), XGBoost (sklearn-style wrapper)
- **Preprocessing:** Reuse `SOTAPreprocessor` from `dev-SOTA-preprocessing`
- **Target strategy:** Regression (MSE) + optimized rounding thresholds for QWK
- **Logging:** TensorBoard (fallback: CSV)
- **Primary metric:** Quadratic Weighted Kappa (QWK)

## Architecture

```
configs/
  config.yaml              # Hydra root — defaults list
  model/
    chebykan.yaml           # ChebyKAN: depth, widths, degree
    fourierkan.yaml         # FourierKAN: depth, widths, grid_size
    mlp.yaml                # MLP: depth, widths, dropout
    xgb.yaml                # XGBoost: n_estimators, max_depth, lr
  train/
    default.yaml            # lr, max_epochs, early_stopping_patience, batch_size
  data/
    default.yaml            # use_sota, missing_threshold, val_split, seed

src/
  data/
    sota_preprocessing.py   # (exists) SOTAPreprocessor
    dataset.py              # LightningDataModule wrapping SOTAPreprocessor
  models/
    kan_layers.py           # ChebyKANLayer, FourierKANLayer (nn.Module)
    tabkan.py               # TabKAN LightningModule — configurable KAN type
    mlp.py                  # MLP LightningModule baseline
    xgb_baseline.py         # XGBoost wrapper (non-Lightning)
  metrics/
    qwk.py                  # QWK computation + threshold optimizer
  train.py                  # Hydra entry point
  evaluate.py               # Post-training threshold optimization + final metrics
```

## Component Specifications

### KAN Layers (`kan_layers.py`)

Two `nn.Module` classes sharing the same interface:

```python
class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=3)

class FourierKANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=4)
```

**ChebyKANLayer:** Each edge learns a linear combination of Chebyshev polynomials T_0(x) through T_d(x). Input is clamped to [-1, 1] (compatible with our preprocessing). Parameters: weight matrix of shape `(out_features, in_features, degree+1)`.

**FourierKANLayer:** Each edge learns Fourier coefficients for cos/sin basis. Parameters: `a_coeffs` and `b_coeffs` of shape `(out_features, in_features, grid_size)`.

Both layers include a residual linear path (`base_weight`) for stability, following efficient-kan convention.

### TabKAN Model (`tabkan.py`)

- Stacks N KAN layers based on config `widths` list
- Layer normalization between KAN layers
- Single scalar output (regression)
- MSE loss
- Logs train/val loss + validation QWK each epoch
- KAN layer type selected via `kan_type` config field

### MLP Baseline (`mlp.py`)

- Configurable depth/widths, ReLU, optional dropout
- Same MSE loss and logging as TabKAN
- Fair comparison baseline

### XGBoost Baseline (`xgb_baseline.py`)

- Wraps `xgboost.XGBRegressor`
- `fit(X, y)` and `predict(X)` interface
- Uses same train/val split and threshold optimizer

### DataModule (`dataset.py`)

- Loads Prudential CSV
- Calls `SOTAPreprocessor.fit_transform`
- Stratified train/val split (configurable ratio, default 80/20)
- Missingness masks concatenated to feature tensor
- Returns `DataLoader`s with configurable batch size

### QWK + Threshold Optimizer (`qwk.py`)

- `quadratic_weighted_kappa(y_true, y_pred)` — standard QWK
- `optimize_thresholds(y_true, y_pred_continuous)` — uses `scipy.optimize.minimize` to find 7 thresholds mapping continuous predictions to ordinal 1-8, maximizing QWK

### Entry Point (`train.py`)

```bash
# Single run
python src/train.py model=chebykan

# Compare all models
python src/train.py model=chebykan,fourierkan,mlp -m

# Tune ChebyKAN degree
python src/train.py model=chebykan model.degree=2,3,4,5 -m
```

XGBoost runs separately (not Lightning):
```bash
python src/train.py model=xgb
```

## Training Flow

1. Hydra composes config from YAML files
2. DataModule loads data, preprocesses, creates splits
3. For neural models: Lightning Trainer runs training with early stopping on val loss
4. Post-training: threshold optimizer finds optimal rounding on validation predictions
5. Final QWK reported on validation set with optimized thresholds
6. Checkpoints and logs saved to `outputs/` (Hydra default)
