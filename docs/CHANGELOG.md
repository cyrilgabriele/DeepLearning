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

### [2026-04-26] - Cyril Gabriele

#### What
- Added shared ordinal class-mapping helpers in `src/interpretability/ordinal.py` so interpretability code now maps continuous TabKAN scores through the stored optimized ordinal-threshold contract when available.
- Updated the interpretability pipeline, local case explanations, closed-form surrogate reports, feature-validation curves, pruning helpers, quality-figure utilities, and final-comparison retention helper to use the same threshold contract instead of silently falling back to `round(score)`.
- Added regression coverage for threshold-aware score-to-class conversion and threshold-aware local/surrogate interpretability outputs.
- Retrained `stage-c-chebykan-best` so the dense ChebyKAN comparison checkpoint now persists `ordinal_thresholds.json`.
- Regenerated threshold-consistent interpretability artifacts for:
  - `stage-c-chebykan-best`
  - `stage-c-fourierkan-best`
  - `stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94`
  - `stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76`
- Removed stale pre-threshold dense-Cheby artifacts and the obsolete `stage-c-chebykan-best-top20` eval/interpretability outputs.
- Added the standalone paper-comparison package `src/interpretability/paper_comparison/`.
- The package compares three already-materialized interpretability runs without coupling the comparison to the single-run `src/interpretability/pipeline.py`:
  - XGBoost baseline: `outputs/interpretability/xgboost_paper/stage-c-xgboost-best`
  - Pareto ChebyKAN: `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-pareto-q0.601-s0.94`
  - Pareto FourierKAN: `outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-pareto-q0.579-s0.76`
- Implemented `python -m src.interpretability.paper_comparison` as the CLI entrypoint for regenerating the paper comparison bundle.
- Set the default output directory to `outputs/interpretability/comparison/pareto_kan_vs_xgboost/`.
- Implemented cross-model feature-ranking comparison from:
  - mean absolute predicted-class XGBoost SHAP values in `shap_xgb_values.parquet`
  - native ChebyKAN feature rankings in `chebykan_feature_ranking.csv`
  - native FourierKAN feature rankings in `fourierkan_feature_ranking.csv`
- Exported the combined ranking table to `data/feature_ranking_comparison.csv`.
- Exported top-k overlap diagnostics to `data/feature_overlap_summary.json`, including the three model top-20 lists, shared top-20 features, rank-scaled overlap scores, and a Kendall-like top-union rank agreement diagnostic.
- Implemented deterministic feature-panel selection that prefers features available in all three models, high-ranking across all three rankings, and a mix of continuous and non-continuous feature semantics.
- Exported the selected paper-panel features to `data/selected_features.json`.
- Reconstructed the pruned ChebyKAN and FourierKAN modules from their run summaries and `models/*_pruned_module.pt` state dicts so the comparison figure uses the actual post-pruning models.
- Generated the feature-effect comparison figure at `figures/feature_effect_comparison.pdf` and `figures/feature_effect_comparison.png`.
- The figure is organized feature-first: columns are selected features, rows are XGBoost SHAP, ChebyKAN PDP, and FourierKAN PDP.
- XGBoost panels show sample-wise SHAP values for each applicant's predicted class, with a binned/state mean overlay.
- KAN panels show native partial-dependence curves from the pruned model modules, using observed states for discrete inputs and the preprocessing-aware display-domain helpers for axis labels.
- Exported model-level metadata to `data/model_summary.json`, including run-summary QWK, accuracy, feature count, preprocessing recipe, random seed, post-pruning QWK, active edge count, and mean per-edge symbolic R² where applicable.
- Exported a compact paper-facing Markdown summary to `reports/feature_effect_comparison.md`.
- Added regression coverage in `tests/interpretability/test_paper_comparison.py` for ranking overlap construction, deterministic feature-panel selection, and end-to-end artifact writing with mocked KAN loaders.
- Added `docs/interpretability/2026-04-26-paper-tex-handoff.md`, a self-contained handoff for generating the paper's LaTeX interpretability section from the comparison artifacts.

#### Why
- The training and pruning stages had been updated to persist and reuse optimized ordinal thresholds, but later interpretability stages still rounded raw KAN scores for feature-validation QWK, local class deltas, and surrogate QWK.
- That made regenerated feature-retention curves and local class explanations inconsistent with the active model metric contract.
- Using one score-to-class contract across training, pruning, and interpretability makes paper-facing QWK and class-delta artifacts comparable.

#### Remarks
- Verified with:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/interpretability/` → `100 passed`
- Regenerated artifact audit:
  - dense ChebyKAN: `qwk_after=0.596386`, `edges_after=21519`, feature-validation full-feature QWK `0.596386`
  - dense FourierKAN: `qwk_after=0.591532`, `edges_after=41713`, feature-validation full-feature QWK `0.591532`
  - Pareto ChebyKAN: `qwk_after=0.617183`, `edges_after=3302`, feature-validation full-feature QWK `0.617183`
  - Pareto FourierKAN: `qwk_after=0.616046`, `edges_after=18147`, feature-validation full-feature QWK `0.616046`
- Paper comparison bundle under `outputs/interpretability/comparison/pareto_kan_vs_xgboost/`:
  - selected feature-effect panels: `BMI`, `Product_Info_4`, `Wt`, `Medical_Keyword_3`
  - shared top-20 features across XGBoost, ChebyKAN, and FourierKAN: 6
  - ChebyKAN vs XGBoost shared top-20 count: 9
  - FourierKAN vs XGBoost shared top-20 count: 7
  - XGBoost QWK: `0.558721`
  - Pareto ChebyKAN post-pruning QWK: `0.617183`, active edges: `3302`, mean per-edge symbolic R²: `1.000000`
  - Pareto FourierKAN post-pruning QWK: `0.616046`, active edges: `18147`, mean per-edge symbolic R²: `1.000000`
### [2026-04-25] - Gian Seifert

#### What
- **Tightened 20-feature methodology** by re-deriving each model's top-20 feature list from the *specific* implementation that ends up in Table 2:
  - Trained tuned FourierKAN ([256, 256], grid 8, λ=0.0249) on full 140 features (`stage-c-fourierkan-tuned-sparse-fullft`); extracted top-20 by coefficient importance into `feature_lists/fourierkan_tuned_top20_features.json` (15/20 overlap with old Pareto-derived list).
  - Computed SHAP `TreeExplainer` on the new `xgb-best` (XGBRegressor) checkpoint; saved top-20 by mean(|SHAP|) into `feature_lists/xgb_tuned_top20_features.json` (16/20 overlap with the old `xgboost-paper`-XGBClassifier-derived list).
  - Re-ran both 50-trial Optuna sweeps on the corrected feature lists; retrained winners; recomputed bootstrap CIs.
- **New Table 2 sparse-regime numbers** (outer-test, threshold-calibrated, 95 % bootstrap CI over n = 11 877):
  - XGBoost, tuned: 0.611 [0.598, 0.624] (was 0.601 [0.587, 0.614])
  - FourierKAN, tuned: 0.608 [0.596, 0.622] (was 0.604 [0.590, 0.618])
  - ChebyKAN sparse hero: 0.592 [0.579, 0.606] (unchanged)
  - All three remain statistically tied (CIs overlap pairwise).
- **FourierKAN architecture flipped** with the new feature list: Optuna's winner is now [64, 64] grid 7 (vs the prior [256, 256] grid 8). KAN-edge count drops from 70 656 → 5 376 — same order of magnitude as the sparse ChebyKAN hero (806 edges) instead of 10× larger.
- **Polished §3.2 prose** in `local_files/main (1).tex` for accuracy + concision:
  - Tightened the methodology paragraph (5 packed topics → one focused paragraph).
  - Replaced "polynomial of degree ≤ 6" with ≤ 7 (current tuned ChebyKAN config uses degree 7).
  - Replaced "single closed-form polynomial in the inputs" with "single closed-form expression" (the composition has nested `tanh` terms — finite closed form, but not a polynomial in `x`).
  - Removed "no surrogate, no sampling, no post-hoc approximation" (IG uses path integration, so "no sampling" over-claimed).
  - Mentioned R² = 1.000 exactly once (in the methodology paragraph) instead of three times.
  - Removed bold from FourierKAN's QWK cell — the prose said "tied", but the bold visually implied FourierKAN was the winner.
  - Fixed FourierKAN edge count (70 912 → 70 656 → 5 376 across iterations) — the final linear head is not a KAN edge.
  - Tightened the §3.2.2 summary by removing redundancy with paragraph 1.
- **Blanked out full-feature QWK cells in Table 2** with `\textcolor{red}{fill from Cyril}` placeholders for the XGBoost / dense-ChebyKAN / dense-FourierKAN rows (those numbers belong to Cyril's §3.2.1 dense-regime sub-section). Added `\usepackage{xcolor}` to the preamble. Caption + summary paragraph updated to point to §3.2.1 for full-feature numbers.

#### Why
- Reviewer-likely objection: in the previous 20-feature comparison, the XGBoost top-20 was derived from SHAP of the *deprecated* `xgboost-paper` (XGBClassifier — a different model class), and the FourierKAN top-20 was derived from the OLD Pareto-sparsity FourierKAN (different architecture and λ). Each row's feature ranking should come from the same model implementation that gets reported, otherwise the comparison silently mixes implementations.
- The architecture flip for FourierKAN-20 also revealed that with the *correct* feature list, Optuna prefers a much smaller network — strengthening rather than weakening the "compact KAN" claim.
- The polish round addressed honest issues (technically wrong claims about polynomial composition, over-claimed IG semantics, visual bolding contradicting prose) that would have drawn reviewer pushback.

#### Remarks
- Verified statistical equivalence between the three sparse-regime rows: XGB − Fourier gap = +0.003, XGB − Cheby gap = +0.019, Fourier − Cheby gap = +0.016, all within 0.013–0.026 bootstrap-CI-overlap regions.
- Generated `outputs/reports/table1_bootstrap_qwk.json` with the full updated bootstrap CI ledger for all six models.
- `paper_draft/interpretability_section.tex` synced with `local_files/main (1).tex` after every edit so future merges have a clean reference.
- Final commit: `5a3ef73`. Branch `gian-interpretability` is ready to ship modulo Cyril filling §3.2.1 + the three full-feature QWK cells.

### [2026-04-24] - Gian Seifert

#### What
- **Re-derived the sparse ChebyKAN hero end-to-end from the Optuna-tuned config** (no mix-and-match between Pareto-sweep features and Optuna-tuned hyperparameters):
  1. Trained tuned ChebyKAN ([64, 64], degree 7, lr 0.00095) on all 140 features with L1 sparsity at λ = 0.1 (`stage-c-chebykan-tuned-sparse-fullft`).
  2. Extracted top-20 features from *that* model's coefficient importance → `feature_lists/chebykan_tuned_top20_features.json` (15/20 overlap with the prior Pareto-derived list).
  3. 5-point sparsity-λ scan via `scripts/sparse_hero_lambda_scan.py`; λ = 0.1 selected as the Pareto sweet spot (85 % within-architecture pruning at essentially zero QWK loss).
  4. Retrained on the new top-20 with no LayerNorm at λ = 0.1 → 5 376 trainable KAN edges.
  5. L1-pruned at threshold 0.002 → **806 active edges**, outer-test QWK **0.592** (up from the original sparse hero's 0.533).
- **Per-budget Optuna tuning for all three 20-feature baselines** (`stage-c-{xgb,chebykan,fourierkan}-top20-tune`, 50 trials each):
  - Equalised the search spaces: added `gamma` to XGBoost; added `lr`/`weight_decay`/`batch_size` to both KAN configs (the existing tune YAMLs only varied architecture).
  - Created `xgb_top20_tune.yaml`, `chebykan_top20_tune.yaml`, `fourierkan_top20_tune.yaml` with matched search spaces.
  - Extended `src/tune/sweep.py` `_resolve_model_family` to recognise `"xgb"` (mapped to `"xgboost-paper"` family for sweep pipeline purposes).
- **Fixed two real XGBoost reproducibility bugs** uncovered while investigating the 0.10 inner-val → outer-test gap:
  1. `XGBoostPaperModel.__init__` did not accept `gamma` — Optuna-best `gamma = 3.8958` was being silently dropped.
  2. `_build_estimator` constructed `xgb.XGBClassifier` without forwarding `gamma` even when present in `_base_params`.
  3. `ModelConfig.allowed_param_keys()` rejected `gamma` as unknown — added a new `_XGBOOST_OPTIONAL_PARAMS` set so existing configs without `gamma` continue to validate.
  4. Even after both fixes, `xgboost-paper` (XGBClassifier) peaks at ~ 0.55 outer-test, not 0.65 — because the §3.1 Table 1 number 0.6546 came from a *different* implementation (`model = xgb`, XGBRegressor + threshold calibration) that was orphaned by the April 9 refactor.
- **Re-registered `XGBBaseline` (XGBRegressor + thresholds) under `"xgb"`** in `src/models/registry.py`:
  - Updated `XGBBaseline.fit` to accept trainer-style kwargs (`validation_data`, `validation_splits`).
  - Updated `build_xgb_model` to forward all tuned hyperparameters (was only passing `n_estimators`/`max_depth`/`learning_rate` and silently dropping the rest).
  - Added `_XGB_REQUIRED_PARAMS` and `_XGB_OPTIONAL_PARAMS` to the config validator.
  - New `xgb_best.yaml` and `xgb_top20.yaml` configs that point to the proper XGBoost implementation.
  - Outer-test QWK on the full feature set: 0.642 [0.629, 0.655] — matches §3.1's 0.6546 within selection-bias variance (single retrain vs 100-trial Optuna best).
- **Fair 20-feature XGBoost baseline added to Table 2** with retuned hyperparameters (max_depth 6, 300 trees, gamma 1) so XGBoost is no longer compared at default settings tuned for 126 features.
- **Replaced the 2×2 edge-activation Figure 3 with an integrated-gradient waterfall** for applicant 55728 on the sparse ChebyKAN hero (`scripts/build_figure3_waterfall.py`); IG sums to f(applicant) − f(reference) exactly by construction. Reference applicant: lowest-scoring class-1 applicant (Id 44235, score −0.18); applicant 55728 score 4.58 → predicted class 5.
- **Multi-seed dense KAN training** (3 seeds each: 42, 0, 1):
  - ChebyKAN dense: 0.595 ± 0.027
  - FourierKAN dense: 0.599 ± 0.007
  - Replaces the previous single-seed dense numbers (0.607 / 0.592) which were Optuna-selection-biased.
- **Trimmed Table 3 closed-form rows to top-3 terms** (was full 7-term expansion which overflowed in the spconf single-column layout).

#### Why
- Original sparse ChebyKAN hero was a mix of (a) features from the OLD Pareto-sparsity sweep and (b) hyperparameters from the new Optuna sweep. End-to-end re-derivation from one consistent source closes that loophole.
- 50-trial per-budget Optuna sweeps replace ad-hoc hand-tuning; matched search spaces avoid asymmetric tuning advantages.
- The XGBoost gap was the user's "are these numbers right?" prompt — turned out to be both a real bug (gamma dropped silently) and a deeper architecture mismatch (XGBClassifier vs the original XGBRegressor pipeline). Restoring `xgb_baseline.XGBBaseline` returns XGBoost to its proper baseline strength.
- The waterfall figure visually demonstrates the unique sparse-ChebyKAN claim ("exact analytic per-feature decomposition") that no other row in Table 2 supports — much stronger than the old R² = 1 redundancy figure.

#### Remarks
- All commits touching this work: `7206c6e` (waterfall + multi-seed + Table 3 trim), `8666e4c` (XGBoost-20 fair baseline), `1e3a229` (gamma fix), `5627961` (re-register XGBBaseline), `00b438d` (per-budget tuning), `2d4e75a` (re-derive sparse hero).
- All 25 interpretability tests pass after the changes.
- `scripts/xgb_gamma_sweep.py` left in the tree — quick gamma grid for confirming the gamma-bug fix doesn't regress the baseline.

### [2026-04-23] - Gian Seifert

#### What
- **Added native FourierKAN symbolic extractor** mirroring the ChebyKAN-native path:
  - `_compose_exact_fourierkan_edge` in `src/interpretability/formula_composition.py` builds the exact symbolic edge `base_weight · x + Σₖ aₖ cos(k·π·(tanh(x)+1)) + bₖ sin(k·π·(tanh(x)+1))` (matches the runtime forward in `_sample_fourierkan_edge`).
  - `fit_symbolic_edge_fourierkan_native` in `src/interpretability/kan_symbolic.py` reads `fourier_a`, `fourier_b`, `base_weight` directly and emits the closed form (R² = 1.000 by construction).
  - Wired into `_build_edge_records` so FourierKAN rows now use `fit_mode = "fourierkan_native"`.
  - Updated `lock_in_symbolic_edges` to skip both `chebykan_native` and `fourierkan_native` rows (projection is a no-op).
  - All 25 existing interpretability tests pass.
- **Trained the first sparse FourierKAN hero** (`stage-c-fourierkan-pareto-top20-noln`) using a top-20 feature list derived from the existing Pareto-sparsity FourierKAN run; outer-test QWK 0.562 with 7 158 active edges. Symbolic recovery R² jumped from 0.27 (scipy fallback) to 1.000 (native extractor).
- **Built the paper content recommendation document** at `docs/interpretability/PAPER_CONTENT_INTERPRETABILITY.md` capturing the planned interpretability section: headline claim, hero models locked in, six-row Table 1 layout, single Figure 1 recommendation, appendix dump pointers, caveats, follow-up checklist.
- **Iterated the doc through five design rounds**:
  1. Initial recommendation (small-hero focus only).
  2. **Option B** — six-row Pareto framing (full-feature dense baselines + sparse heroes in one table).
  3. **Greeks integration** — added the exact-symbolic-derivative claim as a single-row addition + Table 2 cell update + appendix line.
  4. **Gian / Cyril split** — explicit ownership map; new §11 hand-off block for Cyril's dense-regime work + six coordination items.
  5. **Outer-test QWK convention** — locked the QWK regime; switched all six row numbers; flagged the 0.10 XGBoost gap as Optuna-selection bias (later disproved — see 2026-04-24 entry).
- **Threshold-calibration fairness pass**:
  - Discovered the dense ChebyKAN and FourierKAN runs (manifests dated 2026-04-12) used default `round()` rather than threshold calibration, while the sparse heroes used calibrated thresholds. This made the comparison apples-to-oranges.
  - Re-trained `stage-c-chebykan-best` and `stage-c-fourierkan-best` on the current codebase so they use the same inner-validation threshold-fitting procedure as the sparse heroes.
  - Reverted a stray `sparsity_lambda: 0.1` that had been inserted in `chebykan_best.yaml` → back to 0.0 (the original 2026-04-12 dense-baseline value).
  - Trained GLM baseline (`stage-c-glm-baseline`) for the Table 2 row that was previously blank.
- **Bootstrap CIs** for outer-test QWK across all six Table 1 rows via `scripts/bootstrap_qwk_table1.py` (n = 11 877, 1 000 resamples, seed 42).
- **Worked Greek for applicant 55728** via `scripts/worked_greek_applicant_55728.py`: ∂score/∂BMI computed three ways:
  - Symbolic chain rule (SymPy via `exact_partials.compose_exact_chebykan_symbolic_graph` + `build_continuous_partial_trace` + `evaluate_continuous_partial_trace_row`): −0.6505
  - Autograd (PyTorch backward): −0.6505 (agrees to 3 × 10⁻⁶)
  - Central finite difference (ε = 10⁻³): −0.6499 (agrees to 5 × 10⁻⁴, O(ε²) truncation)
- **First Gian deliverables for the paper**: 2×2 edge-activation figure (BMI / Wt × ChebyKAN / FourierKAN sparse heroes), simplified closed-form table (top-3 terms × 5 representative edges from the sparse ChebyKAN hero), and the LaTeX section file `docs/interpretability/paper_draft/interpretability_section.tex` matching the spconf template at `local_files/main (1).tex`.

#### Why
- The previous comparison was unfair: FourierKAN edges were being fit by scipy candidates that only covered harmonics k = 1..4 while the model used `grid_size = 8`, giving R² = 0.27. Native readout closes that gap.
- The PAPER_CONTENT doc started the editorial conversation; the five rounds reflect actual user feedback iterating from a single-row claim to the full Pareto comparison.
- Threshold-calibration fairness was discovered while pulling Table 2 numbers — the dense-row "0.543 / 0.520" had nothing to do with model quality, only with whether `predict()` had been threshold-calibrated.
- Bootstrap CIs are the standard reviewer-defence number for "is this gap real or noise?".
- The worked Greek concretises the exact-Greeks claim: a number a reviewer can verify against the artifact, not just an abstract algebraic argument.

#### Remarks
- Branch created: `gian-interpretability` (from `main`).
- All commits chronologically: `dd690c1`, `c46476d`, `2945814`, `bb46406`, `8d5967d`, `af5547f`, `3264038`, `b835782`, `0404717`.
- New scripts under `scripts/`: `bootstrap_qwk_table1.py`, `worked_greek_applicant_55728.py`, `build_figure1_interpretability.py` (later renamed to `build_figure3_waterfall.py`), `simplified_closed_forms_table.py`.
- New configs under `configs/experiment_stages/stage_c_explanation_package/`: `fourierkan_pareto_top20_noln.yaml`, plus per-seed dense KAN configs (`{cheby,fourier}kan_best_seed{0,1}.yaml`) used for the multi-seed averaging on 2026-04-24.
- Open at end of day: full-feature comparison still showed XGBoost weaker than KAN heroes, which led to the 2026-04-24 deep-dive into the gamma bug and the `xgb`/`xgboost-paper` registry mismatch.

### [2026-04-22] - Gian Seifert

#### What
- Replaced per-edge scipy candidate fitting with the native ChebyKAN extractor for the `symbolic_fits.csv` and `r2_report.json` artifacts.
- Added `fit_symbolic_edge_chebykan_native` in `src/interpretability/kan_symbolic.py` — reuses `_compose_exact_chebykan_edge` to read `cheby_coeffs` and `base_weight` directly and emit the exact edge formula `base_weight·x + Σ_k c_k · T_k(tanh(x))`.
- Extracted `_build_edge_records` so `kan_symbolic.run()` and `r2_pipeline.evaluate_symbolic_fit()` share a single per-edge dispatcher that branches ChebyKAN → native, FourierKAN → scipy.
- Added a `fit_mode` column to `symbolic_fits.csv` and gated `lock_in_symbolic_edges` to skip `chebykan_native` rows (projection onto the same basis is a no-op).
- Trained `stage-c-chebykan-pareto-q0583-top20-noln` — the no-LayerNorm variant referenced in `docs/interpretability/04_CLIENT_FACING_UNDERWRITING_ARTIFACT.md` — so the graph-level exact closed-form path (`compose_exact_chebykan_model`) becomes `exact_available=True`.
- Added regression coverage in `tests/interpretability/test_kan_symbolic_native.py` for:
  - native per-edge fit matches layer forward on an isolated edge
  - zero-coefficient edge returns constant zero
  - `_build_edge_records` is fully native for a ChebyKAN module
  - `r2_pipeline.evaluate_symbolic_fit` reports R² ≈ 1 for all ChebyKAN edges

#### Why
- The scipy candidate library (polynomials up to cubic, sin/cos harmonics, exp/log/sqrt) cannot represent a ChebyKAN edge, which is a polynomial in `tanh(x)` with a `base_weight · x` residual. On the `stage-c-chebykan-pareto-q0583-top20` run only 97 of 589 active edges (16.5 %) reached R² ≥ 0.9 under the scipy path, and the reported mean per-edge R² was 0.47.
- The exact form is already stored as trained parameters on the layer, so "fitting" is just reading the coefficients. That yields R² = 1 by construction and means every downstream consumer — the KAN network diagram, activation grid, quality-figure distribution, feature-risk influence, `r2_report.json`, and ultimately the underwriter-facing artifact — speaks the same exact-formula language.
- Training `-noln` unblocks the graph-level exact closed-form report for the V1 artifact contract in `docs/interpretability/04_CLIENT_FACING_UNDERWRITING_ARTIFACT.md`; LayerNorm would otherwise short-circuit `compose_exact_chebykan_model` to `exact_available=False`.

#### Remarks
- Verified with:
  - `uv run pytest tests/interpretability/` → `86 passed`
  - Re-ran `--stage interpret` on `stage-c-chebykan-pareto-q0583-top20` (with LayerNorm): `symbolic_fits.csv` now reports mean R² = 1.0 across 976 edges (was 0.47 on the same run before). Graph-level exact correctly reports `exact_available=False, reason=layernorm_present` because composing through LayerNorm is not algebraic.
  - Trained and interpreted `stage-c-chebykan-pareto-q0583-top20-noln`: `exact_available=True`, `has_layernorm=False`. The fully expanded closed form is 605 907 operations / 5.6 M characters, so the report flags `usable=False` (too large to display). The exact symbolic structure is materialized; `end_to_end_r2` is skipped for the same size reason — a numeric-only verification via layer re-evaluation is future work.
- No change to the FourierKAN path: it still uses the scipy candidate library.

### [2026-04-24] - Cyril Gabriele

#### What
- Refactored the KAN interpretability plotting layer to be preprocessing-aware instead of assuming that model-input feature names always match raw feature names.
- Added shared feature-domain resolution in `src/interpretability/utils/style.py` so interpretability code now derives:
  - the preprocessing recipe from the resolved checkpoint-linked config
  - the raw feature name corresponding to each model-input feature
  - the transform family (`identity`, `cb_`, `qt_`, `mm_`, `missing_`)
  - whether a feature should be treated as continuous or discrete in plots
- Fixed `src/interpretability/partial_dependence.py` so PDPs now:
  - use observed states for discrete inputs instead of sweeping impossible in-between values
  - stop dropping valid categories through percentile clipping on discrete features
  - label axes according to the true model-input / raw-input semantics instead of hardcoding `[-1,1]`
- Updated KAN feature-function / activation plotting paths in:
  - `src/interpretability/kan_symbolic.py`
  - `src/interpretability/comparison_side_by_side.py`
  - `src/interpretability/feature_risk_influence.py`
  so they evaluate curves on actual model-input grids rather than conflating the internal `tanh` domain with the external input axis.
- Added regression coverage in:
  - `tests/interpretability/test_style.py`
  - `tests/interpretability/test_partial_dependence.py`

#### Why
- The previous plots were only really safe for a narrow case where `X_eval` and `X_eval_raw` had the same feature names and the input domain could be read as if it were already normalized. That is true only for some current runs and breaks as soon as preprocessing uses prefixed transformed features such as `cb_`, `qt_`, or `mm_`.
- The dense ChebyKAN PDP artifact was materially misleading because many plotted features were discrete coded variables, yet the code treated them as continuous sweeps and in some cases removed real categories via percentile clipping.
- Making the interpretability layer preprocessing-aware now avoids a second rewrite later if the project switches from `kan_paper` to a transformed recipe such as `kan_sota`.

#### Remarks
- This change does not alter training, preprocessing, model weights, or the feature-importance metric itself. It changes how interpretability artifacts are generated and labeled.
- The source of truth for the preprocessing recipe is the checkpoint-linked config resolved in `src/interpretability/pipeline.py`, not heuristic inspection of existing output files.
- Existing files under `outputs/interpretability/...` remain historical artifacts until the interpret stage is rerun for the target checkpoints.

### [2026-04-22] - Cyril Gabriele

#### What
- Fixed `src/interpretability/kan_pruning.py` so pruning-stage QWK now reuses the stored ordinal-threshold contract from the corresponding training run instead of silently falling back to naive rounded scores.
- The pruning step now:
  - first looks for `outputs/eval/<recipe>/<experiment>/ordinal_thresholds.json`
  - falls back to the matching `artifacts/<experiment>/run-summary-<timestamp>.json` `ordinal_calibration` payload when the eval sidecar is absent
  - hydrates the reconstructed `TabKANClassifier` wrapper with those thresholds before computing `qwk_before` / `qwk_after`
  - records the active QWK contract in `chebykan_pruning_summary.json` via `qwk_metric` and `qwk_metric_source_split`
- Added regression coverage in `tests/interpretability/test_kan_pruning.py` for:
  - run-summary fallback loading
  - pruning-stage QWK evaluation using the stored threshold sidecar

#### Why
- After threshold calibration was introduced in training, the interpret/pruning stage was still reconstructing a fresh wrapper with no thresholds attached.
- That caused a metric-contract mismatch:
  - training summary QWK was threshold-calibrated
  - pruning summary QWK was still based on `round(score)`
- The result was misleading comparisons such as `train qwk = 0.5464` versus `pruning qwk = 0.4930`, where most of the apparent gap came from different class-mapping rules rather than pruning damage.
- Rehydrating the stored thresholds makes training, pruning, and the later underwriter-facing artifact speak the same ordinal-class language.

#### Remarks
- Verified with:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/interpretability/test_kan_pruning.py`
  - Result: `2 passed`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/interpretability/test_pipeline.py tests/test_pipeline_integration.py`
  - Result: `21 passed`
- This change does not alter the pruning criterion itself. It fixes how pruning-stage performance is reported.

### [2026-04-22] - Cyril Gabriele

#### What
- Added explicit ordinal-threshold persistence to the training artifact contract.
- Updated `src/models/tabkan.py` so `TabKANClassifier` now:
  - keeps the continuous score path available internally
  - calibrates optimized ordinal thresholds after fit
  - uses the inner validation split as the preferred threshold-calibration source when available, otherwise falls back to the training split
  - predicts ordinal classes via stored optimized thresholds instead of naive rounded scores for newly trained runs
- Added shared ordinal-calibration hooks in `src/models/base.py` and exposed threshold metadata from the threshold-based baseline wrappers as well.
- Updated `src/training/trainer.py` so newly trained runs now persist the threshold contract in three places:
  - `artifacts/<experiment>/run-summary-<timestamp>.json`
  - `checkpoints/<experiment>/model-<timestamp>.manifest.json`
  - `outputs/eval/<recipe>/<experiment>/ordinal_thresholds.json`
- Added regression coverage for:
  - TabKAN validation-split threshold calibration in `tests/models/test_tabkan.py`
  - trainer-level threshold persistence and eval-export sidecar emission in `tests/training/test_trainer.py`

#### Why
- The underwriter-facing artifact needs a stable class-definition contract if it is going to report threshold-based classes and margins to neighboring class boundaries.
- Thresholds are not preprocessing state. They are post-fit ordinal calibration metadata derived from model scores, so they belong in run/eval artifacts, not in the preprocessing pipeline itself.
- Persisting the thresholds directly alongside the saved run and eval artifacts removes the ambiguity where older no-`LayerNorm` KAN reports had exact symbolic score behavior available but no durable threshold metadata to map those scores back to ordinal classes.
- Because there is no production contract to preserve yet, it is cleaner to fix the KAN training/evaluation contract now rather than keep propagating rounded-score fallback behavior.

#### Remarks
- Verified with:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/models/test_tabkan.py tests/training/test_trainer.py`
  - Result: `19 passed`
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/interpretability/test_pipeline.py tests/test_pipeline_integration.py`
  - Result: `21 passed`
- New runs will carry optimized threshold metadata. Older already-materialized artifacts remain valid as historical outputs, but they do not gain the new threshold sidecars until the corresponding training/eval pipeline is rerun.

### [2026-04-22] - Cyril Gabriele

#### What
- Added a standalone exact-partials and discrete-effects generator in `src/interpretability/exact_partials.py` for the no-`LayerNorm` ChebyKAN target.
- The generator reconstructs the run-specific preprocessed outer training split under `kan_paper` using the config seed and selected-feature list, then uses that split to derive:
  - exact partial-derivative traces for the 4 continuous selected features
  - exact reference-based discrete effects for the 16 discrete selected features
  - modal reference states and observed-state counts from the reconstructed outer training split after feature subsetting
- Emitted the new run-scoped reports for `stage-c-chebykan-pareto-q0583-top20-noln`:
  - `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/reports/chebykan_exact_partials.json`
  - `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/reports/chebykan_exact_partials.md`
- Added regression coverage in `tests/interpretability/test_exact_partials.py` for:
  - exact nested graph evaluation
  - continuous partials against autograd
  - discrete substitution-effect contracts
  - report persistence

#### Why
- The exact closed-form export proved the no-`LayerNorm` ChebyKAN is symbolically exact, but the fully expanded formula is too large to serve as a practical insurance-facing artifact.
- A nested symbolic derivative/effect representation preserves exactness while staying traceable back to hidden nodes and layer structure.
- Discrete selected features should not be forced into a classical derivative framing; reference-based exact state contrasts are the mathematically correct and operationally cleaner representation.

#### Remarks
- Verified with:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q tests/interpretability/test_formula_composition.py tests/interpretability/test_exact_partials.py`
  - Result: `22 passed`
- The standalone generator was also executed on the real target config:
  - `UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.interpretability.exact_partials --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml`
- TODO:
  - the exact-partials generator is not yet wired into the main interpretability stage / pipeline
  - the current JSON/Markdown artifact is still backend-facing and should be used as the source for a client-facing underwriting artifact rather than served directly

### [2026-04-21] - Cyril Gabriele

#### What
- Added config-driven train-time feature subsetting through `preprocessing.selected_features_path` in `src/config/preprocessing/preprocessing_config.py` and `src/training/trainer.py`.
- Added config-driven `use_layernorm` support for TabKAN in `src/config/model/model_config.py` and `src/models/tabkan.py`, and propagated that flag through all downstream TabKAN reconstruction sites used by tuning and interpretability.
- Extended the interpretability pipeline to export stable KAN-native feature rankings and materialized top-k feature lists, including:
  - repo-tracked lists under `configs/experiment_stages/stage_c_explanation_package/feature_lists/`
  - run artifacts under `outputs/interpretability/.../data/{chebykan_feature_ranking,chebykan_top20_features,chebykan_top12_features}.json/csv`
- Added exact end-to-end closed-form export for no-`LayerNorm` ChebyKAN models in `src/interpretability/formula_composition.py`, including the final linear head and explicit refusal metadata when exact export is structurally unavailable.
- Added local applicant-level finite-difference and what-if explanations in `src/interpretability/local_case_explanations.py`.
- Added a closed-form surrogate fallback in `src/interpretability/closed_form_surrogate.py`, wired so it is emitted only when the exact closed-form report is unavailable.
- Added the new stage configs:
  - `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml`
  - `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top12_noln.yaml`
- Added regression coverage for the new feature-subset, no-`LayerNorm`, exact-export, local-explanation, and surrogate-fallback paths.

#### Why
- The previous repo state could not actually train the intended reduced-feature symbolic candidates because feature restriction existed only as an interpret-time knob (`--max-features`), not as a train-time contract.
- `LayerNorm` blocked exact end-to-end symbolic composition. Making it optional was required before a mathematically exact deployed ChebyKAN variant could exist.
- The project objective was narrowed to a business-auditable model artifact, not just visual interpretability. That required three new deliverables:
  - an exact closed-form report when structurally possible
  - a clearly labeled fallback surrogate only when exact export is unavailable
  - case-level sensitivities and what-if outputs in business-facing terms
- The feature-list configs and stage YAMLs keep `main.py` and the config system as the single source of truth instead of introducing one-off scripts or hardcoded experiment branches.

#### Remarks
- Verified with:
  - `UV_CACHE_DIR=.uv-cache uv run pytest tests/training/test_trainer.py tests/test_pipeline_integration.py tests/interpretability/test_pipeline.py tests/interpretability/test_formula_composition.py tests/interpretability/test_local_case_explanations.py tests/interpretability/test_closed_form_surrogate.py -q`
  - Result: `46 passed`
- Real-data stage runs were also executed through `main.py`:
  - `stage-c-chebykan-pareto-q0583-top20-noln`: train QWK `0.5048`, pruned QWK `0.5119`, active edges `651`
  - `stage-c-chebykan-pareto-q0583-top12-noln`: train QWK `0.4792`, pruned QWK `0.4739`, active edges `459`
- The no-`LayerNorm` candidates now produce mathematically exact end-to-end formulas, but both are still operationally unusable as business artifacts under the current report thresholds:
  - top-20 exact formula: `707824` symbolic operations, `6568142` characters
  - top-12 exact formula: `343941` symbolic operations, `2986921` characters
- In this changelog and the corresponding reports, "too large" means the formula is exact but too big to be realistically read, reviewed, documented, or manually reasoned about by a human user. SymPy can still differentiate it in principle (`sympy_derivable=true`), but it is not a practical insurance-facing artifact in its current expanded form.
- The surrogate fallback is now correctly gated:
  - emitted for the old LayerNorm baseline because exact export is unavailable
  - not emitted for the no-`LayerNorm` candidates because exact export exists, even when that exact formula is still marked `usable=false`

### [2026-04-21] - Cyril Gabriele

#### What
- Fixed the TabKAN wrapper in `src/models/tabkan.py` so config-provided `lr`, `weight_decay`, and `batch_size` are now actually used during training instead of silently falling back to wrapper defaults or hardcoded dataloader sizes.
- Removed registry-side default hyperparameters for TabKAN, XGBoost, and GLM builders where the pipeline is expected to read concrete values from config files.
- Tightened config validation in `src/config/model/model_config.py` and `src/config/config_loader.py`:
  - unsupported `model.params` keys now fail loudly
  - required model hyperparameters must be present in `model.params` or explicitly supplied by `tune.search_space`
  - `Trainer.run()` now rejects train/retrain configs that still leave required model parameters unresolved
- Updated the affected experiment YAMLs and regression tests so the config files remain the single source of truth for active model hyperparameters.
- Updated TabKAN reconstruction sites (`src/tune/sweep.py`, `src/interpretability/kan_pruning.py`) to rebuild wrappers from the effective config payload instead of reintroducing ad-hoc defaults.

#### Why
- The previous TabKAN wrapper ignored some config values (`lr`, `weight_decay`) and hardcoded others (`batch_size=256`), which made older sweep results partly unreliable and allowed silent config drift.
- Tune configs should be allowed to omit parameters only when those parameters are explicitly defined in `tune.search_space`; train/retrain configs must carry a complete concrete model contract.
- The repo already moved toward strict config-driven orchestration; this change closes one of the remaining loopholes where model code could still override or invent hyperparameters at runtime.

#### Remarks
- Historical TabKAN sweeps produced before this fix should be treated cautiously, especially any result that depends on `lr`, `weight_decay`, or `batch_size`.
- Verified with:
  - `UV_CACHE_DIR=.uv-cache uv run pytest tests/models/test_tabkan.py tests/tune/test_sweep.py tests/training/test_trainer.py tests/test_pipeline_integration.py -q`
  - `UV_CACHE_DIR=.uv-cache uv run pytest tests/interpretability/test_pipeline.py tests/test_main.py -q`
### [2026-04-20] - Gian Seifert

#### What
- Added `use_layernorm` parameter to `TabKAN` and `TabKANClassifier` to allow training without LayerNorm, enabling exact mathematical composition of edge functions into closed-form input→output formulas.
- Ran a systematic interpretable KAN search across 66 configurations (2 sweeps): varied feature counts (5–20), hidden widths ([4]–[32]), degrees (3–4), and sparsity regularization (0–0.01). Key findings:
  - **Degree 3 is critical** for interpretability — degree 4 drops symbolic fit quality from 100% to 65–84%.
  - **Sparsity regularization (λ=0.005)** boosts clean symbolic fits from ~16% to ~88% with minimal QWK loss.
  - **[4,2] depth-2 architectures always collapse** (QWK=0).
  - Best with LayerNorm: 20 features, width [16], degree 3, λ=0.005 → QWK=0.466, 88% clean edges.
- Trained the final interpretable model **without LayerNorm** (20 features, width [8], degree 3, λ=0.005), producing a pure **generalized additive model (GAM)**: `prediction = 0.30 + Σ fᵢ(tanh(xᵢ))` where each fᵢ is a closed-form function (cubic polynomial, exponential, or trigonometric). QWK=0.427 with 19 active features and 0 flagged edges (all R² > 0.97).
- Verified that symbolic formula predictions match the KAN model (QWK gap of only 0.007 when applying tanh normalization correctly).
- Generated a full interpretability plot suite:
  - Per-feature risk contribution functions (input→output, 4×5 grid)
  - Individual prediction decomposition (patient-level waterfall charts)
  - Binary keyword toggle effects (present vs absent bar chart)
  - Symbolic fit fidelity (exact curve vs formula overlay)
  - Prediction distribution by risk class (violin + confusion matrix)
  - Cumulative feature importance (QWK vs feature count)
  - KAN network grid diagram (activation function per edge in matrix layout)
  - Feature importance waterfall (average contributions)

#### Why
- The previous interpretability results (2026-04-17/18) showed that KAN symbolic formulas on the full [128,64] architecture were not human-readable — page-long expressions with hundreds of edges. The goal was to find the largest model that an actuary can actually inspect as closed-form equations.
- Removing LayerNorm was necessary because it breaks the additive composition: with LayerNorm, edge functions cannot be collapsed into one formula per feature. Without it, the linear head weights directly compose with edge functions, giving exact input→output formulas.
- This proves that KANs *can* deliver on their interpretability promise for tabular data, but only with narrow architectures (width ≤16), low degree (3), sparsity regularization, and no LayerNorm — at the cost of ~32% QWK relative to the black-box baseline (0.427 vs 0.625).

#### Remarks
- Scripts: `scripts/interpretable_kan_search.py` (sweep 1), `scripts/interpretable_kan_search_v2.py` (sweep 2), `scripts/interpretable_kan_no_layernorm.py` (no-LN training + formula extraction), `scripts/plot_input_to_output_formulas.py`, `scripts/plot_interpretability_suite.py`, `scripts/plot_network_diagram_v3.py`.
- Config: `configs/experiment_stages/stage_c_explanation_package/chebykan_interpretable_best.yaml`.
- Results: `outputs/interpretable_kan_search/`, `outputs/interpretable_kan_search_v2/`, `outputs/interpretable_kan_no_layernorm/`.
- The `use_layernorm` parameter defaults to `True` so existing models and checkpoints are unaffected.

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
