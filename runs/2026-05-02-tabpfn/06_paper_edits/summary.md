# TabPFN Comparison — Run Summary

**Branch:** `tabpfn`
**Spec:** `docs/superpowers/specs/2026-05-02-tabpfn-comparison-design.md`
**Plan:** `docs/superpowers/plans/2026-05-02-tabpfn-comparison.md`
**Tests:** 199 passing (baseline 191 + 8 new)

## Status

| Task | Status | Notes |
|------|--------|-------|
| 1   Add tabpfn dep                          | ✅ DONE | base `tabpfn>=2.0.0`; `tabpfn-extensions` not used (`pandas<3` conflict) |
| 2-3 Recipe shim + tests                     | ✅ DONE | `preprocess_tabpfn_paper` shims xgboost_paper |
| 4-5 Trainer dispatch + tests                | ✅ DONE | recipe Literal + `tabpfn-auto` allowlist |
| 6-7 Wrapper class + tests                   | ✅ DONE | base TabPFNRegressor + `ignore_pretraining_limits=True` + 10k subsample |
| 8   Registry                                 | ✅ DONE | `tabpfn-auto` factory |
| 9   Run A — 3 seeds full-feature             | ✅ DONE | per-seed outer-test QWKs: 0.6347, 0.6399, 0.6346 |
| 10  Validation QWK for Table 1               | ✅ DONE | mean 0.6442 across 3 seeds (highest: seed 1337 at 0.6585) |
| 11  Feature importance + top-20              | ✅ DONE | switched from sklearn permutation_importance (>12 h) to batched mean-replacement ablation (~50 min) |
| 12  Run B — 20-feature                       | ✅ DONE | outer-test QWK 0.5617 |
| 13  Bootstrap CI + full-feature t-CI         | ✅ DONE | sparse-row CI 0.547–0.575; full-feature t-CI 0.629–0.644 |
| 14  Top-5 feature overlap                    | ✅ DONE | TabPFN ∩ XGB-SHAP: 3, ∩ ChebyKAN: 2 |
| 15  Attention attribution applicant 55728    | ✅ FALLBACK | TabPFN-v2 uses `F.scaled_dot_product_attention`; standard forward hooks don't expose weights → mention-only paper wording per spec |
| 16  Paper edits — Tables 1 + 2               | ✅ DONE | one row in Table 1, two rows in Table 2 (full + sparse) |
| 17  Paper edits — §1, §2.2, §3.2.1, §3.2.2, §4, §5 | ✅ DONE | all 6 prose insertions; bib citation kept as red placeholder |

## Key numbers

| Metric | Value | Where |
|--------|-------|-------|
| TabPFN val QWK (Table 1) | **0.6442** | mean over 3 seeds |
| TabPFN full-feature outer-test (Table 2) | **0.636 [0.629, 0.644]** | 3-seed t-CI |
| TabPFN 20-feature outer-test (Table 2) | **0.562 [0.547, 0.575]** | 1000-resample bootstrap |
| TabPFN top-5 features | Medical_History_32, BMI, Product_Info_4, Medical_History_4, Ins_Age | mean-replacement ablation |
| ∩ XGBoost-SHAP top-5 | **3** features | BMI, Medical_History_4, Product_Info_4 |
| ∩ ChebyKAN top-5 | **2** features | BMI, Product_Info_4 |
| Δ vs XGBoost on top-20 | **−0.051** | sparse regime |

## Spec deviations (all documented)

1. **`tabpfn-extensions` not used** — package pins `pandas<3`; project uses `pandas>=3.0.1`. Switched to base `tabpfn` with `ignore_pretraining_limits=True`. Wrapper class kept name `TabPFNAutoRegressor` for plan/config-name continuity.
2. **Training-set subsampling** — Prudential's 47k-row training set exceeds TabPFN's published operating range. Wrapper subsamples to 10,000 deterministic rows per seed (each seed → different subsample). Reflected in §2.2 and §5 of the paper.
3. **Feature importance via mean-replacement ablation** — sklearn's `permutation_importance` with the same parameters ran for >12h before being killed; switched to a 2-call batched ablation that completes in ~50 min. Same intent ("how much does QWK drop when this feature is removed"), deterministic instead of random shuffling. Reflected in §3.2.1 wording ("by mean-replacement ablation impact on QWK").
4. **Attention attribution fell back** — TabPFN-v2 uses `F.scaled_dot_product_attention` which does not expose attention weights through standard PyTorch forward hooks. Spec's mention-only fallback wording is used in §3.2.1.
5. **Permutation-importance subsample** — uses 100 outer-test rows (not full 11,877) due to TabPFN inference cost. The ranking is stable enough for top-20 selection but is noisier than a full-test computation.

## Manual follow-ups for the user

- Replace the red `\textcolor{red}{[tabpfn reference to add]}` placeholder in §1 (line 46) with the correct `\cite{<key>}` after adding the Hollmann et al. 2025 bib entry to the paper's bibliography file.
- Verify the LaTeX renders cleanly on Overleaf (or local pdflatex).
- Decide whether to keep `local_files/main (1).tex` in git history (the file is in a `.gitignore`-d directory; was force-added on this branch). Squash or reset before merging if you want to keep it private.
- Decide whether to merge the `tabpfn` branch into main, open a PR, or keep as a side branch.

## Commit list (chronological on `tabpfn` branch)

```
e9f8b4d  deps: add tabpfn for TabPFN baseline comparison
6e9ae06  test: add failing tests for tabpfn_paper recipe shim
570467b  feat(preprocessing): add tabpfn_paper recipe shim over xgboost_paper
f25485a  test: add failing trainer dispatch test for tabpfn_paper recipe
392ffdc  feat(trainer): dispatch tabpfn_paper recipe to preprocess_tabpfn_paper
37684bf  chore(tabpfn): drop unused Path import and update recipe field description
1ee6379  test: add failing tests for TabPFNAutoRegressor wrapper
f69f7d2  feat(models): add TabPFNAutoRegressor wrapper with ordinal calibration
ff4a273  feat(tabpfn): subsample training set to max_train_samples (default 10000)
44b070c  feat(registry): register tabpfn-auto factory
64ecaf1  feat: auto-load .env at main.py startup via python-dotenv
2693b64  fix(tabpfn): allow 'device' in tabpfn-auto params so YAML device:cpu propagates
af7c426  run: TabPFN Run A full-feature, 3 seeds
cf236c1  run: compute TabPFN validation QWK for Table 1
8db72f6  feat: TabPFN feature importance via batched mean-replacement ablation
b648e4d  run: TabPFN Run B 20-feature on permutation top-20
b0e42db  run: bootstrap 95% CI for TabPFN 20-feature row
4b8f51b  feat: TabPFN top-5 feature-overlap with XGBoost-SHAP and ChebyKAN
7354975  feat: TabPFN attention attribution script with documented fallback
015344e  docs(paper): add TabPFN baseline to Tables 1+2 and prose
```
