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

### [2026-03-26] - Gian Seifert

#### What
- Added full interpretability pipeline for GLM, XGBoost, ChebyKAN, and FourierKAN.
- New modules: GLM coefficient extraction, SHAP for XGBoost, KAN edge pruning, symbolic regression (scipy curve_fit with 11 candidates), R² evaluation pipeline, per-risk-level comparison, side-by-side visualization, and final comparison matrix.
- Added GLM baseline model

#### Why
- Provide interpretability analysis to compare how each model explains its predictions and to assess whether KAN activations can be approximated by closed-form symbolic expressions.
- the design decisions for the kan pruning and regressin can be found in more detail and with sources under the kan design decisions md

#### Remarks
- 

---

### [2026-03-25] - Christof Steiner

#### What
- Added `src/sweep.py`: Optuna-based Bayesian hyperparameter sweep with SQLite persistence, supporting all 5 model types (ChebyKAN, FourierKAN, BSplineKAN, MLP, XGBoost) with per-model search spaces and objectives.
- Generalized the sweep to ensure a fair comparison: reverted KAN-only enhancements (residual blocks, ordinal loss, cosine schedule) so all models compete on equal footing.
- Added Optuna-tuned YAML configs (`chebykan_tuned.yaml`, `fourierkan_tuned.yaml`, `bsplinekan_tuned.yaml`, `xgboost_regression_tuned.yaml`) containing best hyperparameters from 50-trial sweeps.
- Fixed validation leakage in `XGBoostPaperModel` sweep: now reports tuning QWK before refit on train+val, not post-refit evaluation on the leaked validation set.
- Added `src/submit.py` for Kaggle submission generation from best sweep parameters.
- Stored sweep results as JSON in `sweeps/` for all 6 configurations (3 KAN variants, MLP, XGBoost regression, XGBoost classifier).

#### Why
- Need a rigorous, fair hyperparameter budget across all architectures so the paper comparison reflects model capacity rather than tuning effort.
- The original ChebyKAN-only sweep was not comparable — extending Optuna to all models with identical trial budgets and preprocessing (`kan_paper`) ensures apples-to-apples results.
- The XGBoost leakage fix was critical: reporting post-refit QWK inflated the tree baseline, making KAN variants look worse than they are.

#### Remarks
- Kaggle private leaderboard results: XGBoost Regression 0.660, BSplineKAN 0.630, ChebyKAN 0.633, FourierKAN 0.619, MLP 0.606, XGBoost Classifier 0.591.


---

### [2026-03-21] - Cyril Gabriele

#### What
- Added `XGBoostPaperModel`, a PrudentialModel wrapper that reproduces the ICSITech 2019 sequential tuning pipeline (label encoding, paper grids, refit on train+val) and exposes it through the registry.
- Registered the model under `xgboost-paper`, wired it into the Trainer config surface, and added dedicated unit tests plus a Trainer integration test.
- Added explicit configs (`configs/model/xgboost_paper.yaml` for Hydra overrides and `configs/xgboost_paper_experiment.yaml` for the Trainer CLI) so the paper setup can run without editing code.
- Added optional `seed_trials` support so the estimator can train several random seeds and keep the highest-validation-QWK run.

#### Why
- The preprocessing refactor already mirrors the paper splits; we now have the matching estimator so we can benchmark KAN variants against the exact XGBoost procedure documented in the case study.

#### Remarks
- The tuning grid is customisable (used for lightweight tests) but defaults to the values reported in the paper, and predictions remain 1–8 ordinal classes for downstream QWK metrics.

---

### [2026-03-16] - Christof Steiner

#### What
- Merged origin/main preprocessing refactor into feature/tabkan-models and fixed broken imports (`SOTAPreprocessor` → `PrudentialKANPreprocessor`, added `build_tabkan_model` factory for the Trainer pipeline)
- Added pipeline diagnostics: each training run saves to a timestamped `runs/` directory with human-readable logs (`train.log`) and machine-readable metrics (`metrics.jsonl`, `epoch_metrics.csv`) so we can verify results and compare runs
- Added per-run data snapshots: `raw_sample.csv`, `processed_sample.csv`, `feature_stats.csv`, `config.json` — everything needed to reproduce and inspect a run
- Added 18 integration tests and a diagnostic trace script to verify the full pipeline end-to-end

#### Why
- Need to track whether changes to hyperparameters, preprocessing, or architecture actually improve QWK — every run is now saved with its config and metrics so we can compare tactics
- Need to verify preprocessing correctness for the paper: the logs prove all features are in [-1, 1], zero NaN, and each feature type is processed correctly

#### Remarks
- none

---

### [2026-03-18] - Cyril Gabriele

#### What
- Replaced the legacy `Prudential*Preprocessor` stack with the paper-faithful `preprocess_xgboost_paper.py` and the minimally adapted `preprocess_kan_paper.py`, wiring Trainer/Evaluate/DataModule directly into those pipelines.
- Simplified `PreprocessingConfig`/`TrainerConfig` (no more ad-hoc eval splits or K-fold knobs) and removed the obsolete dataset splitter plus helper modules/tests.
- Updated diagnostics, configs, and regression tests to rely on the shared outer/inner splits and to exercise inference-time transforms via the new helper APIs; all suites now pass against the refactored pipeline.

#### Why
- Ensure every experiment—XGBoost or KAN—runs on identical, paper-aligned splits while keeping the exact preprocessing steps auditable and free of silent leakage.
- Reduce configuration surface to only the recipe choice and eliminate dead code paths that diverged from the documented preprocessing assumptions.
- Keep tooling/tests honest by hitting the same deterministic preprocessing routines that production training now uses.

#### Remarks
- All scripts that previously expected `PrudentialKANPreprocessor` now import the new modules; see `preprocess_*.py` docstrings for the explicit paper vs. KAN deviations.

---

### [2026-03-19] - Cyril Gabriele

#### What
- Added `PaperPreprocessingBase`, centralising the paper-faithful loading/encoding/splitting flow.
- Refactored `preprocess_xgboost_paper.py` to wrap the shared base (with an explicit log noting the XGBoost recipe) and rebuilt `preprocess_kan_paper.py` as a subclass that applies only the median-imputation/Product_Info_2 tweaks required for TabKAN tensors.
- Updated dependent modules/tests to call the new wrappers so both pipelines stay independent while sharing the exact baseline logic.

#### Why
- Keep KAN and XGBoost preprocessors fully separated yet guaranteed to stay in lockstep on baseline behaviour, making future audits/changes more maintainable.

#### Remarks
- Logging now makes it obvious which preprocessing path was invoked when reading experiment traces.

---

### [2026-03-20] - Cyril Gabriele

#### What
- Added `preprocess_kan_sota.py`, a CatBoost-encoding + MICE-imputation + quantile-scaling pipeline derived from `PaperPreprocessingBase` for TabKAN experiments.
- Extended the preprocessing config (`kan_sota` recipe), trainer logic, and dataset regression tests so the SOTA pipeline can be selected like the existing paper recipes.
- Documented the upgrade in tests/changelog to keep the new recipe auditable.
- Routed every preprocessing pipeline through a single seed produced by `set_global_seed` (now configured via `TrainerConfig.seed`), removing stray `RANDOM_SEED` constants and updating the trainer/DataModule/evaluation scripts to thread the same seed everywhere.

#### Why
- Provide a ready-to-use, state-of-the-art preprocessing path that leverages dataset insights (target encodings, covariate-aware imputations, bounded feature scales) without disturbing the faithful paper baselines.
- Ensure reproducibility is controlled from one place (the CLI config), avoiding hidden defaults in preprocessors or data modules.

#### Remarks
- Outputs remain deterministic and float32-bounded, making them drop-in replacements for downstream TabKAN models.
- Tests now explicitly seed via `set_global_seed`, so local runs mirror the main entrypoint's determinism.

---

### [2026-03-10] - Cyril Gabriele

#### What
- Split the previous `SOTAPreprocessor` into two explicit modules: `PrudentialPaperPreprocessor` and `PrudentialKANPreprocessor`.
- Added `prudential_features.py` so both preprocessors share a single source of feature-taxonomy truth.
- Updated preprocessing tests and documentation references to the new class names and [-1, 1] expectations.
- Added `prudential_dataset.py` plus regression tests to ensure we split the Kaggle training file before fitting preprocessors (no leakage) and only carve an optional eval subset from that training data.
- Enabled stratified k-fold target encoding inside `PrudentialKANPreprocessor` and covered it with synthetic regression tests.
- Introduced a typed CLI (`main.py`) powered by Pydantic along with the new `Trainer` orchestration module, synthetic TabKAN placeholders, and regression tests for the trainer pipeline.
- Removed CLI overrides so the only way to run experiments is via fully specified YAML configs, and added `configs/smoke_experiment.yaml` as a reproducible template.
- Enforced a single global random seed (42) and automatic device detection (CUDA/MPS/CPU) so every run is deterministic aside from the available hardware.
- Trainer now dumps a JSON run summary (config, random seed, device, metrics) into `artifacts/<experiment-name>/run-summary-<timestamp>.json` and surfaces the path via CLI output so every experiment leaves a reproducible breadcrumb.
- `TrainingArtifacts` exposes the resolved config + seed and hands the CLI the summary path, ensuring downstream consumers/tests can assert against the exact setup that ran.
- TabKAN registry builder now accepts arbitrary keyword arguments so runtime-detected knobs such as `device` and any upcoming model overrides pass cleanly from the config through the trainer into the placeholder estimator.
- Fixed the trainer dataclass field ordering so we can inject the resolved device without tripping the frozen dataclass initializer.
- Added Quadratic Weighted Kappa (QWK) to the trainer's evaluation metrics and run-summary payload to mirror the Prudential competition's official score.
- Trainer now stores Torch checkpoints alongside JSON summaries, writing `checkpoints/<experiment-name>/model-<timestamp>.pt` whenever the estimator exposes a Torch module.
- Trainer optionally loads Kaggle's `test.csv`, runs inference with the fitted preprocessor/model, and saves predictions to `artifacts/<experiment-name>/test-predictions-<timestamp>.csv`.
- Fixed the dataset splitter to always fit the preprocessor on the training slice before transforming the optional evaluation split, removing a long-standing ordering bug.

#### Why
- Keep a faithful reproduction of the paper baseline separate from the experimental, KAN-optimized pipeline while preventing feature assignment drift.
- Provide a publication-ready data-handling workflow where every preprocessing statistic is learned on the train subset only, respecting Kaggle's existing train/test split.
- Mimic best-practice leakage safeguards cited in the literature by ensuring categorical encodings are computed out-of-fold with respect to stratified splits.
- Ensure experiments are reproducible through a single entry point with strongly typed arguments shared between the CLI and trainer.
- Make the configuration file the single source of truth so no experiment runs with accidental defaults.
- Keep hardware-awareness limited to runtime device detection while everything else remains fixed.
- Preserve experiment provenance by persisting the resolved configuration/seed/device with the reported metrics so historical runs can be replayed even after YAML edits.
- Ensure the placeholder model remains smokable while preparing the pipeline to accept real TabKAN implementations that need the runtime device and future hyper-parameters.
- Align local evaluation with the competition's metric (QWK) and avoid dataclass regressions when threading runtime-only fields like `device`.
- Persist the final set of learned weights for reproducibility and downstream fine-tuning without rerunning the entire training job.
- Provide ready-to-submit predictions so the held-out Kaggle test split is exercised every run even when evaluation splits are disabled.
- Ensure evaluation splits actually reuse the trained preprocessing statistics instead of transforming before fit, which previously raised errors on small experiments.

#### Remarks
- The new naming makes it straightforward to pick the desired preprocessing recipe inside training scripts.

---

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

### [2026-03-09] - Gian Seifert

#### What
- Refined feature categorization: moved low-cardinality "continuous" features (e.g., `Product_Info_3`, `Employment_Info_2`) to the `ordinal` group.
- Switched ordinal scaling from `QuantileTransformer` to `MinMaxScaler` mapped to `[-1, 1]`.
- Enabled automated missingness indicators (`missing_` flags) in `PrudentialKANPreprocessor` (formerly `SOTAPreprocessor`).
- Implemented explicit `[-1, 1]` clipping and stored scaling parameters for consistency across train/test splits.

#### Why
- `QuantileTransformer` on discrete/low-cardinality/ordinal data creates steep "staircase" gradients that destabilize B-spline activations in KANs. `MinMaxScaler` preserves the linear relationship and discrete steps, allowing KAN splines to learn smoother, more stable mappings.
- Explicit clipping prevents out-of-bounds instability in B-spline evaluations, and missingness indicators ensure the model can explicitly learn from informative data gaps.

#### Remarks


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
