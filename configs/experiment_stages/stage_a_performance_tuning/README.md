# Stage A Performance Tuning Configs

Stage A configs are used to select the model architecture for each model family under a fixed, established training setup. The intent is to compare architecture capacity while avoiding early tuning of optimizer, regularization, sampling, or other training hyperparameters.

## Scope

These configs should tune architecture-defining parameters only:

- ChebyKAN: network depth, network width, and optionally Chebyshev degree when treated as basis capacity.
- FourierKAN: network depth, network width, and optionally Fourier grid size when treated as basis capacity.
- XGBoost: tree-ensemble capacity parameters such as number of trees and tree depth.

All non-architecture hyperparameters should remain fixed at reasonable field defaults during Stage A. Examples include learning rate, weight decay, batch size, epoch budget, subsampling, column sampling, and regularization strengths. Those parameters belong in later tuning stages if they need to be optimized.

## Files

- `stage_a_cheby/chebykan_tune.yaml`: ChebyKAN architecture sweep.
- `stage_a_fourier/fourierkan_tune.yaml`: FourierKAN architecture sweep.
- `stage_a_xgboost/xgboost_tune.yaml`: XGBoost architecture/capacity sweep.

## Expected Outputs

Running a Stage A tune config writes the best full config and candidate manifest to `sweeps/`. The candidate manifest is the handoff into Stage B, where shortlisted architectures can be retrained and evaluated for stability and interpretability.
