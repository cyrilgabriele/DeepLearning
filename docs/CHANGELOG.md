# Changelog

Track what was changed, why it was changed, and any important notes.

## Entry Format

```markdown
### [YYYY-MM-DD] - [Contributor Name]

#### What
- List changes here

#### Why
- Explain reasoning

#### Remarks
- Optional notes, issues, or future work
```

### [2026-03-09] - Christof Steiner

#### What
- Implemented complete TabKAN training framework 
- Added three KAN layer architectures: B-Spline KAN (Liu et al. 2024 original), ChebyKAN (Chebyshev polynomial basis), FourierKAN (Fourier series basis)
- Added MLP baseline (same loss/metric/training setup for fair neural comparison) and XGBoost baseline (tree-based reference with same QWK evaluation)
- Added QWK (Quadratic Weighted Kappa) metric with Nelder-Mead threshold optimizer that finds optimal rounding boundaries to maximize QWK
- Added PrudentialDataModule (Lightning DataModule wrapping SOTAPreprocessor with stratified train/val splits)
- Added Hydra configuration system: YAML configs for all 5 models, training parameters, and data settings — hyperparameters are changed via config or CLI override, not code
- Added summary results table that prints after each training run
- Added 37 unit tests covering all layers, models, metrics, and data loading

#### Why
- Need a reproducible, fair comparison framework to test the paper's hypothesis: do KAN architectures outperform traditional methods on tabular insurance risk prediction?
- Regression + threshold optimization (not classification) because the target is ordinal (1-8) and this approach won the original Kaggle competition — MSE respects ordinal distance, threshold optimizer directly maximizes QWK
- Three KAN basis types to test smooth (Chebyshev) vs flexible (Fourier) vs reference (B-Spline) basis expansions
- Hydra configs ensure every experiment is fully reproducible and hyperparameter sweeps are one CLI flag
- All models share identical preprocessing, data splits, loss function, and evaluation so performance differences isolate the architecture

#### Remarks
- Initial baseline results (5 epochs, no tuning, CPU): B-Spline KAN 0.6029, FourierKAN 0.6011, XGBoost 0.5997, ChebyKAN 0.5894, MLP 0.5752
- Next steps: hyperparameter tuning (Stage 1: depth/width, Stage 2: basis complexity), longer training runs with early stopping, interpretability analysis

---

### [2026-03-04] - Gian Seifert

#### What
- implemented SOTA methods for preprocessing so that we can test them on for the KAN, the methods where choosen by looking at new sota developments in preprocessing and in KANs 
- Methods implemented: Iterative Imputation (MICE) for missing values, CatBoost Encoding to handle categorical variables without expanding the feature space, and Quantile Transformation to map continuous features to a uniform distribution optimized for KAN spline grids.

#### Why
- To have the sota preprocessing ready for the experiments and to look if they work, simultaneously also normal already battleproof methods are beeing implemented so that we see the difference 

#### Remarks
- not connected to main.py at the moment needs to be done when testing, there is also a test test_preprocessing.py to see if the preprocessing fits the expected outcome


---

### [2026-03-03] - Cyril Gabriele

#### What
- Rebuilt `src/data/playground/data_insights.ipynb` into a full Prudential EDA playbook covering feature semantics, structural checks, missingness analysis, cardinality scans, distribution stats, target relationships, redundancy checks, and optional matplotlib plots.

#### Why
- Provide every teammate with an accurate, self-serve overview of the dataset along with deeper diagnostics aligned to the 30%/50% missingness guidance cited in recent literature.

#### Remarks
- Install `matplotlib` if you plan to run the optional plotting section.
