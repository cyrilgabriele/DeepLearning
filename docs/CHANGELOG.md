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

### [2026-04-18] - Gian Seifert

#### What
- Ran a narrow-architecture experiment to test whether reducing first-layer width enables feature elimination via edge pruning. Trained 3 ChebyKAN variants at `sparsity_lambda=0.01`, `degree=6`, 100 epochs, then pruned at threshold=0.01:
  - **[140→16→8→1]**: QWK=0.556, pruned QWK=0.529, 88.6% sparsity, **107/140 features survive**
  - **[140→32→16→1]**: QWK=0.592, pruned QWK=0.559, 94.0% sparsity, **112/140 features survive**
  - **[140→64→32→1]**: QWK=0.565, pruned QWK=0.568, 97.9% sparsity, **111/140 features survive**
- **Key finding**: even with only 16 neurons in the first hidden layer (giving each feature just 16 edges instead of 128), 107 of 140 features survive pruning. Feature elimination via edge pruning is not viable for this 140-feature dataset at any reasonable architecture width.
- The narrow models lose 8–13% QWK compared to the wide [140→128→64→1] reference (QWK=0.605) without achieving meaningful feature reduction. This is a dataset property, not an architecture limitation.

#### Why
- The wide [140→128→64→1] Pareto sweep (2026-04-17) showed that even at 99.9% edge sparsity, 28+ features survive because each input gets 128 chances to keep one edge. The hypothesis was that narrowing the first layer would force the model to be selective. The data disproves this: the Prudential dataset genuinely distributes signal across most of its 140 features.
- This result strengthens the paper's "glass box" narrative: KAN interpretability on high-dimensional tabular data comes from coefficient importance rankings and activation curve inspection, not from feature elimination or symbolic formula extraction.

#### Remarks
- Configs saved under `configs/experiment_stages/stage_c_explanation_package/chebykan_narrow_{16_8,32_16,64_32}.yaml`.
- Results saved to `outputs/narrow_architecture_experiment.json`.
- Runner script: `scripts/run_narrow_experiment.py`.

### [2026-04-17] - Gian Seifert

#### What
- Ran the full interpretability pipeline (`kan_pruning` → `kan_symbolic` → `r2_pipeline` → `formula_composition` → `kan_network_diagram` → `feature_validation`) on all 8 selected Pareto trials (4 ChebyKAN, 4 FourierKAN).
- 2/8 completed fully (ChebyKAN trial 4/μ, FourierKAN trial 18/μ-1σ). The other 6 completed through R² reports but stalled in SymPy formula composition due to high edge counts (228–16,823 edges). Killed after 6+ hours; core artifacts (pruning, coefficient importance, symbolic fits, activation curves, feature ranking, R² reports, pruned checkpoints) were preserved for all 8.
- **Key interpretability findings that change the paper narrative**:
  1. **Edge pruning ≠ feature elimination**: even at 99.9% edge sparsity, 28+ of 140 input features retain at least one active edge. The [140→128→64] architecture gives each feature 128 chances to survive.
  2. **Symbolic fits are poor at scale**: 82.8% of edges have R² < 0.90 for symbolic approximation. The composed SymPy formulas are page-long expressions with 50-digit coefficients — not human-readable.
  3. **The model genuinely needs many features**: feature validation (TabKAN §5.7) shows QWK ≈ 0 with 5–25 features, ~0.25 at 50, and full performance only at 140. There is no small interpretable core.
  4. **The original KAN interpretability promise (prune → symbolify → extract formulas) does not scale** from the 2–5 input problems in Liu et al. (2024) to 140-feature tabular insurance data.
- **Revised interpretability framing for the paper**: KANs provide a "glass box" rather than a "white box" for high-dimensional tabular data:
  - **Native feature importance** from Chebyshev coefficient magnitudes (no post-hoc SHAP needed)
  - **Inspectable per-feature activation curves** showing the learned nonlinear relationship
  - **Sparsity-accuracy Pareto front** demonstrating architectural robustness (ChebyKAN) vs fragility (FourierKAN)
  - But NOT closed-form symbolic formulas or feature-count reduction

#### Why
- The formula composition step is O(edges²) or worse due to SymPy simplification of multi-layer composed expressions. With 795 edges (trial 4) it took ~5 min; with 1,443+ edges the remaining runs exceeded 6 hours without completing. This is a fundamental scalability limitation of the symbolification approach, not a bug.
- The honest finding — that KAN interpretability degrades with input dimensionality — is more valuable to the paper than forcing clean formulas. It positions our contribution as: "KANs on real-world high-dimensional tabular data: what works (native importance, activation curves, sparsity robustness) and what doesn't (symbolic reduction, feature elimination)."
- Deprioritizing formula composition, network diagrams, and feature validation for the 6 incomplete runs. The core artifacts (importance rankings, activation curves, R² reports) are sufficient for the paper figures.

#### Remarks
- Saved artifacts per run: `{data,figures,models,reports}/` under `outputs/interpretability/kan_paper/stage-c-{chebykan,fourierkan}-pareto-sparsity-pareto-*/`.
- The 2 fully complete runs (ChebyKAN trial 4, FourierKAN trial 18) have all 15 artifacts including network diagram, R² distribution plot, and feature validation curves. The other 6 have 9 artifacts each (everything except formula composition and downstream).

### [2026-04-17] - Gian Seifert

#### What
- Retrained all 9 ChebyKAN and 10 FourierKAN Pareto-optimal trials locally via `src/interpretability/pareto_select.py`, pruning each at threshold=0.01 and counting surviving input features.
- **Key finding**: edge pruning at threshold=0.01 does not eliminate input features — even at 99.9% edge sparsity, 28+ of 140 features survive because each feature only needs one edge above the threshold into any of the 128 first-layer hidden neurons. The pruning removes redundant edges *within* features, not features themselves.
- **Revised interpretability approach**: feature-count cutoffs (5/10/15/20) are not achievable through sparsity regularization alone with this architecture. Instead, interpretability is delivered through the existing `kan_symbolic.py` pipeline: rank features by coefficient importance, visualize the top-k learned activation curves, and fit symbolic expressions. The sparsity sweep demonstrates the accuracy-sparsity tradeoff, not the accuracy-interpretability tradeoff directly.
- Selected 4 representative points per model spread across the Pareto front using mean ± 1σ and +2σ of pruned QWK to show the full tradeoff curve:
  - **ChebyKAN** (SD=0.027, tight — robust to pruning):
    - μ+2σ: trial 12, λ=0.0029, QWK=0.596, sparsity=94.5%, 1443 edges, 132 features
    - μ+1σ: trial 19, λ=0.0019, QWK=0.578, sparsity=91.3%, 2272 edges, 135 features
    - μ:    trial 4,  λ=0.0056, QWK=0.562, sparsity=97.0%, 795 edges, 130 features
    - μ-1σ: trial 13, λ=0.0162, QWK=0.534, sparsity=99.1%, 228 edges, 109 features
  - **FourierKAN** (SD=0.087, wide — collapses under pruning):
    - μ+2σ: trial 29, λ=0.0036, QWK=0.481, sparsity=59.7%, 16823 edges, 140 features
    - μ+1σ: trial 26, λ=0.0085, QWK=0.380, sparsity=71.9%, 11745 edges, 134 features
    - μ:    trial 1,  λ=0.025,  QWK=0.309, sparsity=91.1%, 3734 edges, 129 features
    - μ-1σ: trial 18, λ=0.212,  QWK=0.022, sparsity=99.7%, 137 edges, 55 features
- Results saved to `outputs/interpretability/kan_paper/pareto-select-{chebykan,fourierkan}/reports/`.

#### Why
- The initial plan assumed that high edge sparsity would translate to few surviving input features, making a "how many features can the actuary review" cutoff viable. The data shows this assumption was wrong for the [140→128→64→8] architecture: the wide first hidden layer (128 neurons) means each input feature has 128 chances to keep at least one edge above the pruning threshold.
- The feature-count-based selection was replaced with a statistical spread (mean ± SD) across the Pareto front. This gives 4 evenly distributed points that capture the full tradeoff curve without requiring an arbitrary cutoff.
- ChebyKAN is confirmed as the stronger interpretability candidate: its QWK degrades gracefully under pruning (SD=0.027 across the front), while FourierKAN collapses rapidly (SD=0.087, with most trials producing near-zero QWK after pruning).

#### Remarks
- ChebyKAN's best pruned variant (trial 12, λ=0.0029) retains 94.5% sparsity with only 1.6% relative QWK drop — this is the recommended starting point for the full interpretability pipeline (symbolic fitting, activation curve visualization, feature ranking).
- FourierKAN's best pruned variant (trial 29, λ=0.0036) only achieves 59.7% sparsity and still loses 12.2% QWK — significantly worse than ChebyKAN at every point on the frontier.

### [2026-04-17] - Gian Seifert

#### What
- Defined the Pareto tradeoff selection criterion for "best interpretable" KAN variant: instead of a fixed sparsity percentage, select the highest-QWK trial where the number of surviving input features after pruning is small enough for manual actuarial review.
- Added `src/interpretability/pareto_select.py`: script that takes a Pareto JSON manifest, loads each trial checkpoint, prunes at threshold=0.01, counts surviving input features (features with ≥1 active edge into the first hidden layer), and outputs a ranked table plus per-trial pruned checkpoints for actuary review.
- Target range for surviving input features: 5–20, derived from cognitive science and actuarial practice literature (see references below).

#### Why
- The original 50% sparsity threshold was arbitrary and not grounded in what actuaries actually need. An actuary must be able to review each surviving feature's learned activation curve and confirm it is "actuarially reasonable."
- Literature basis for the feature-count criterion:
  - **Miller (1956)** "The Magical Number Seven, Plus or Minus Two" — human working memory holds ~7 items. **Cowan (2001)** revised this to ~3–5 chunks.
  - **Lage et al. (2019)** "Human Evaluation of Models Built for Interpretability" (AAAI HCOMP) — explicitly connects Miller's law to ML model interpretability; fewer features = better human simulatability.
  - **Liu et al. (2024)** "KAN: Kolmogorov-Arnold Networks" (ICLR 2025, arXiv:2404.19756) — defines the L1+entropy pruning procedure; in KAN 2.0 (arXiv:2408.10205), `prune_input` on 100 features retained only 5.
  - **Zhang & Zhuang (2026)** "What KAN Mortality Say" (ASTIN Bulletin 56(1), doi:10.1017/asb.2025.10079) — validates KANs for actuarial modeling with intrinsic interpretability through smooth activation curves. Uses architectural interpretability (shallow KAN[2,1]) rather than pruning, but confirms that inspectable univariate activation functions are the right unit of explanation for actuaries.
  - **Rudin (2019)** "Stop Explaining Black Box ML Models for High Stakes Decisions" (Nature Machine Intelligence, doi:10.1038/s42256-019-0048-x) — for high-stakes domains, use inherently interpretable models; the Rashomon set argument shows the accuracy/interpretability tradeoff is often a false dilemma.
  - **ASOP No. 56** (Actuarial Standards Board, 2020) — requires model documentation sufficient for "another actuary qualified in the same practice area" to assess reasonableness.
  - **Kuo & Lupton (2023)** "Towards Explainability of ML Models in Insurance Pricing" (Variance 16(1), doi:10.66573/001c.68374) — ML adoption in P&C ratemaking is limited by lack of transparency vs GLMs; variable importance + response curves are the minimum explanation.
- The selection logic: filter Pareto trials to those with ≤N surviving input features, then pick max QWK among those. Multiple N values (5, 10, 15, 20) are generated so an actuary can review and state which level is still interpretable.

#### Remarks
- The Zhang & Zhuang paper uses a fundamentally different approach (architectural interpretability with 2 input features) vs our pruning-based approach (140 input features pruned down). Both are valid KAN interpretability strategies for different problem scales.

### [2026-04-15] - Christof Steiner

#### What
- Added `grid` search-space type and `grid` sampler to the tuning layer (`src/config/tune/tune_config.py`, `src/tune/sweep.py`), and fixed the sampler-selection rule so explicit `sampler: grid` is honoured for multi-objective studies.
- Added `chebykan_pareto_sparsity.yaml` and `fourierkan_pareto_sparsity.yaml` under `configs/experiment_stages/stage_c_explanation_package/`: freeze Stage A winners, sweep only `sparsity_lambda` over a 30-point geometric grid in `[1e-3, 0.5]`, directions `[maximize, maximize]` (QWK, sparsity_ratio).
- Ran both sweeps and committed the Pareto manifests + per-trial configs under `sweeps/`.

#### Why
- Stage A winners trained at `sparsity_lambda = 0`, so pruning at threshold 0.01 removed <1% of edges. A dense KAN is not a defensible base for the interpretability story.
- Single-objective tuning would drive λ to 0; the tradeoff only surfaces as a multi-objective problem. A deterministic grid is preferred over NSGA-II for bit-reproducibility of the paper figure.
- The frontier itself is the contribution, it lets us claim "X% prunable at <Y% QWK drop" rather than picking one λ by hand.

#### Remarks


### [2026-04-09] - Cyril Gabriele

#### What
- Implemented the pipeline-audit follow-up across config validation, orchestration, tuning, retraining, selection, artifact contracts, final comparison, and legacy-script cleanup.
- Made the typed config layer strict in `configs/config_loader.py`, `configs/train/trainer_config.py`, `configs/preprocessing/preprocessing_config.py`, `configs/model/model_config.py`, and `configs/tune/tune_config.py`, so stale keys now fail loudly instead of being silently ignored.
- Removed the dead preprocessing knobs from the active experiment YAMLs, updated the model configs to the current schema, and refreshed `README.md` so `main.py` is the only supported orchestration path.
- Kept preprocessing behavior frozen and tightened the surrounding artifact contract: run summaries and checkpoint-adjacent manifests now persist the effective preprocessing payload in `src/training/trainer.py`, without runtime fingerprint enforcement.
- Added canonical KAN architecture support through `hidden_widths` in `configs/model/model_config.py` and `src/models/tabkan.py`, then updated the KAN interpretability loaders in `src/interpretability/kan_pruning.py`, `src/interpretability/kan_symbolic.py`, and `src/interpretability/r2_pipeline.py` to reconstruct from the saved architecture rather than assuming uniform `[width] * depth`.
- Extended the tune stage in `src/tune/sweep.py` to emit top-k candidate manifests (`*_candidates.json`) and added the missing FourierKAN tune config under `configs/tune/kan_fourier/kan_fourier_tune.yaml`.
- Added the new `retrain` stage in `src/retrain/pipeline.py` and wired it into `main.py`, so selected KAN candidates can be materialized across multiple seeds with sparsity regularization enforced and persisted under `artifacts/retrain/<family>/<selection_name>/manifest.json`.
- Added the new `select` stage in `src/selection/pipeline.py` and wired it into `main.py`, implementing family-wise best-performance vs. best-interpretable selection under the documented QWK tolerance rule and persisting results under `artifacts/selection/<family>_selection.json`.
- Rewrote `src/interpretability/final_comparison.py` to consume the current manifest-driven, namespaced artifact layout instead of the older flat output conventions.
- Marked `src/evaluate.py` and `src/submit.py` as legacy entrypoints that fail loudly and point users back to `main.py`.
- Added regression coverage for the new behavior in `tests/test_main.py`, `tests/training/test_trainer.py`, `tests/tune/test_sweep.py`, `tests/retrain/test_pipeline.py`, `tests/selection/test_selection_pipeline.py`, `tests/interpretability/test_final_comparison.py`, `tests/models/test_tabkan.py`, and related pipeline tests.

#### Why
- The audit showed that the core `train`/`tune`/`interpret` path was usable, but the repo still mixed two workflow eras, silently ignored stale config fields, lacked a reliable preprocessing/artifact contract, collapsed tuning to one winner too early, and was missing the documented retraining/selection bridge.
- Strict configs and cleaned-up docs reduce operator error immediately.
- Persisting the effective preprocessing payload and explicit hidden-layer layouts makes artifacts easier to audit and keeps downstream interpretation aligned with what was actually trained.
- Top-k candidate export plus dedicated `retrain` and `select` stages turns the previously manual KAN candidate workflow into a reproducible pipeline.
- Rewriting final comparison and disabling legacy scripts removes old-path ambiguity and aligns the repo with the current namespaced artifact layout.

#### Remarks
- Preprocessing logic itself was intentionally left unchanged; only the surrounding config/artifact contract was tightened.
- Verified with targeted pytest runs covering config loading, trainer artifacts, KAN architecture handling, tune manifests, retrain/select stages, interpretability helpers, and the pipeline integration path (`57 passed` in the focused suite).

### [2026-04-09] - Cyril Gabriele

#### What
- Audited the preprocessing hierarchy against the project proposal (`docs/proposal/KAN-2026-DL-Project-Proposal.pdf`), the cited XGBoost baseline paper (`docs/Analysis_Accuracy_of_XGBoost_Model_for_Multiclass_Classification_-_A_Case_Study_of_Applicant_Level_Risk_Prediction_for_Life_Insurance.pdf`), and the Prudential EDA notebook in `src/preprocessing/playground/data_insights.ipynb`.
- Kept `src/preprocessing/preprocess_xgboost_paper.py` as the paper-faithful baseline and confirmed that `src/preprocessing/preprocess_kan_paper.py` remains the minimal KAN-safe adaptation of that same preprocessing.
- Reworked `src/preprocessing/preprocess_kan_sota.py` so the stronger KAN pipeline now uses leakage-safe out-of-fold CatBoost encoding on training rows, retains explicit missingness indicators, drops only ultra-sparse numeric value channels above the 50% missingness threshold while keeping their masks, and separates continuous (`QuantileTransformer`) from ordinal (`MinMaxScaler`) scaling.
- Exposed dropped-value-column metadata through `src/preprocessing/dataset.py`, updated `src/training/trainer.py` so encoded/scaled SOTA feature names are mapped back to the correct feature types during eval export, and added regression coverage for the sparse-column behavior in `tests/data/test_dataset_pipeline.py`.

#### Why
- The intended comparison is `XGBoost paper baseline` vs `KAN with the same preprocessing except for the minimum changes needed to remove raw NaNs`, followed by a clearly stronger `KAN SOTA` recipe. The audit was needed to make sure those three preprocessing paths are scientifically distinct and internally consistent.
- The previous `kan_sota` implementation still encoded training categoricals in a leakage-prone way and applied one scaling strategy too broadly. The revised version aligns better with both the notebook evidence on structured missingness and the repo's earlier notes about ordinal features behaving poorly under quantile scaling.
- Surfacing dropped-feature metadata and preserving correct feature typing keeps downstream diagnostics and interpretability artifacts honest once SOTA features are prefixed as `cb_`, `qt_`, `mm_`, or `missing_`.

#### Remarks
- On the real Prudential training data, the updated `kan_sota` path now drops the six value channels above the 50% missingness threshold (`Family_Hist_3`, `Family_Hist_5`, `Medical_History_10`, `Medical_History_15`, `Medical_History_24`, `Medical_History_32`) while keeping all corresponding missingness masks.
- Verified with `./.venv/bin/python -m pytest tests/data/test_dataset_pipeline.py tests/training/test_trainer.py -q` and with direct real-data runs of the three preprocessing pipelines.

### [2026-04-06] - Cyril Gabriele

#### What
- Added a typed `tune:` config surface (`configs/tune_config.py`) so Optuna study settings and hyperparameter intervals are declared in the experiment YAML instead of being hardcoded in `src/tune/sweep.py`.
- Refactored `src/tune/sweep.py` to sample exclusively from `config.tune.search_space`, honor `tune.name`, `tune.storage`, `tune.n_trials`, `tune.timeout`, and `tune.sampler`, and fail fast when `--stage tune` is run without a `tune:` block.
- Removed the internal sequential tuning path from `src/models/xgboost_paper.py`; the model now trains fixed parameters only, with tuning handled externally through `main.py --stage tune`.
- Migrated the active sweep configs (`configs/smoke_experiment.yaml`, `configs/experiments/kan_cheby_single.yaml`, `configs/xgboost_paper_experiment.yaml`, `configs/experiments/xgboost_paper_experiment.yaml`) so their search intervals live under `tune.search_space`.
- Added `optuna` to `pyproject.toml` so the config-driven tune stage is installable in the repo environment.
- Removed the obsolete Hydra-era config fragments under `configs/model/`, `configs/data/`, `configs/train/`, plus the unsupported `configs/xgb_experiment.yaml`, because they no longer match the current `ExperimentConfig`/registry-backed workflow.
- Updated `src/submit.py`, `README.md`, and the regression tests to follow the new config-driven sweep flow and the fixed-parameter XGBoost paper model.

#### Why
- The previous sweep runner hardcoded all Optuna intervals in Python, which made the search space opaque, harder to audit, and inconsistent with the config-driven tuning structure already used in the `ParrotLLM` project.
- Keeping XGBoost's paper-inspired sequential tuner inside the model created two competing tuning systems in the repository. Removing the internal tuner makes `main.py --stage tune` the single source of truth for hyperparameter search.
- Explicit YAML search spaces make sweep intent reviewable, reproducible, and easy to adjust without changing code.

#### Remarks
- Existing experiment YAMLs that still rely on legacy XGBoost self-tuning flags such as `auto_tune` or `seed_trials` need to be updated before they can be used with the current `xgboost-paper` model.

### [2026-04-06] - Cyril Gabriele

#### What
- Updated `main.py --stage interpret` so `--config` is no longer required when a checkpoint is provided and the experiment config can be recovered automatically.
- Added config resolution in `src/interpretability/pipeline.py` that loads the saved experiment config from the checkpoint-linked `artifacts/<experiment_name>/run-summary-<timestamp>.json` file and rejects mismatches when both `--config` and `--checkpoint` are passed.
- Refactored the KAN interpretability path (`kan_pruning.py`, `kan_symbolic.py`, `r2_pipeline.py`) to consume the resolved `ExperimentConfig` directly instead of reloading the YAML path again.
- Added regression coverage for checkpoint-only interpret dispatch and automatic config recovery.

#### Why
- The interpret stage was redundantly asking for a YAML config even though the run artifacts already preserved the effective experiment configuration for that checkpoint.
- Recovering the config from the saved run summary keeps the interpret workflow aligned with the trained artifact, reduces CLI friction, and prevents accidental drift between a checkpoint and a manually re-supplied config file.

#### Remarks
- The current auto-recovery source is the run summary paired to the checkpoint timestamp; the raw checkpoint payload itself still does not embed the full config.

### [2026-04-06] - Cyril Gabriele

#### What
- Namespaced eval artifact export by preprocessing recipe and experiment name, so training now writes `X_eval.parquet`, `y_eval.parquet`, `X_eval_raw.parquet`, `feature_names.json`, and `feature_types.json` under `outputs/eval/<recipe>/<experiment_name>/` instead of the previous global `outputs/data/` and `outputs/reports/` locations.
- Added shared path helpers in `src/interpretability/utils/paths.py` for both eval artifacts and interpretability outputs.
- Added a new `interpret` stage to `main.py` and implemented a dedicated orchestration module in `src/interpretability/pipeline.py`.
- Made the interpret stage model-aware: `glm` runs coefficient extraction, `xgboost-paper` runs the SHAP pipeline, and supported TabKAN flavors (`chebykan`, `fourierkan`) run pruning, symbolic fitting, and the R² report.
- Namespaced interpretability outputs under `outputs/interpretability/<recipe>/<experiment_name>/` so derived figures, reports, tables, and pruned checkpoints do not collide across preprocessing spaces.
- Extended `src/interpretability/shap_xgboost.py` so it can load the trainer-saved XGBoost wrapper objects cleanly and collapse multiclass SHAP outputs into the predicted-class view expected by the downstream plots.
- Relaxed the common model `fit()` interface so trainer-forwarded kwargs such as validation splits do not break models that do not use them.
- Restored the missing `_kan_importance_from_variance()` compatibility helper in `comparison_per_risk.py` so the interpretability test suite is green again.
- Added regression coverage for the new export namespace and `main.py --stage interpret`, and updated `README.md` to document the new stage and directory layout.

#### Why
- One global `outputs/data/X_eval.parquet` was no longer a valid abstraction because different model families in this repository were trained on different preprocessing spaces. Namespacing the eval artifacts by recipe removes that ambiguity and gives each interpretability run the correct feature space.
- Adding `main.py --stage interpret` keeps interpretability inside the same CLI contract as training and tuning, which makes the workflow reproducible and easier to run end to end from config.
- The SHAP loader and model-interface fixes were required so the new stage actually works with the artifacts the current trainer persists, instead of only with older or narrower assumptions.

#### Remarks
- The new interpret stage currently supports `glm`, `xgboost-paper`, and TabKAN flavors `chebykan` / `fourierkan`; `bsplinekan` is still not wired into this stage yet.

### [2026-04-06] - Cyril Gabriele

#### What
- Fixed the regression in `src/training/trainer.py` where `Trainer.run()` called `self._export_eval_data(splits)` with an undefined variable, causing the run to crash after fitting and checkpointing.
- Completed the eval-export path in `Trainer` by aligning it with the actual `PreparedDataset` contract, adding raw evaluation feature reconstruction (`X_eval_raw`) and exporting the feature-type metadata needed by the interpretability scripts.
- Added a shared `run_train()` helper in `src/training/trainer.py` so training can be invoked through one code path from both the CLI and the tuning workflow.
- Reworked `main.py` into a stage-based entrypoint with `--stage {train,tune}`, following the same pattern used in the other project.
- Replaced the old `src/sweep.py` implementation with a trainer-backed Optuna runner that materializes `ExperimentConfig` trial variants and evaluates them via `Trainer.run()` instead of the legacy Lightning/Hydra training path.
- Removed the obsolete `src/train.py` entrypoint and updated `README.md` so the documented workflow uses `main.py --stage train` and `main.py --stage tune`.

#### Why
- The broken `splits` call meant the current trainer could fit a model but still fail before returning artifacts, which made the main experiment path unreliable.
- The repository had drifted into two incompatible training systems: the modern registry-backed `Trainer` path and the older Hydra/Lightning `src/train.py` path. Consolidating around `Trainer.run()` removes duplicated logic and keeps training and tuning behavior consistent.
- Stage-based dispatch in `main.py` keeps the CLI surface explicit and makes the entrypoint match the project structure used elsewhere.

#### Remarks
- The new sweep only targets model families that the current registry-backed trainer actually supports (`glm`, `xgboost-paper`, and TabKAN flavors); the old MLP / legacy XGBoost regression sweep path was intentionally removed with `src/train.py`.

### [2026-03-29] - Cyril Gabriele

#### What
- Aligned the interpretability pipeline with the KAN and TabKAN papers in `docs/interpretability/papers/2404.19756v5.pdf` and `docs/interpretability/papers/2504.06559v3.pdf`, cross-checking against the TabKAN reference notebooks in `https://github.com/aseslamian/TAbKAN/tree/main/Interpretability`.
- Added `src/interpretability/kan_coefficients.py` to compute paper-native layer-0 feature importance from Chebyshev/Fourier coefficient magnitudes and to reconstruct aggregated first-layer feature functions.
- Updated `kan_symbolic.py` to rank features by coefficient magnitude, export `*_coefficient_importance.csv`, and relabel the KAN feature-importance plot accordingly.
- Updated `comparison_side_by_side.py` and `feature_risk_influence.py` so KAN curves are drawn from aggregated first-layer feature functions and feature selection is driven by KAN-native coefficient ranking instead of SHAP-only or edge-count heuristics.
- Updated `comparison_per_risk.py` so KAN importance is treated as global, paper-native coefficient importance rather than falsely duplicated as if it were per-risk; also added optional Fourier checkpoint/config loading so both KAN variants can use the same coefficient-based path.
- Updated `final_comparison.py` to use coefficient-based KAN feature rankings and fixed the KAN retention-curve prediction path so it rounds the single regression output correctly instead of treating it like multiclass logits.
- Clarified the pruning/R² documentation (`kan_pruning.py`, `r2_pipeline.py`) to refer to activation L1 magnitude instead of variance, matching the implemented criterion.
- Enabled sparsity regularization in `configs/chebykan_experiment.yaml` and `configs/fourierkan_experiment.yaml` via `sparsity_lambda`, `l1_weight`, and `entropy_weight` so future retraining follows the simplification scheme described in the original KAN paper.

#### Why
- The existing KAN interpretability code was only partially paper-faithful: it relied too heavily on L1/edge-count proxies, selected features through SHAP-first framing, and presented KAN values in the per-risk plots as if they were conditional when the TabKAN paper defines feature importance globally from first-layer basis coefficients.
- The goal was to make the framework scientifically defensible and consistent with the papers' stated methodology, while preserving SHAP as the XGBoost explanation mechanism and ensuring we can extract the most important risk-prediction features from the KAN models themselves.

#### Remarks
- none

---

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
