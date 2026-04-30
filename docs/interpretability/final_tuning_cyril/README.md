# Final Tuning Record - Cyril

This document records the final three-stage tuning chain for the full-feature ChebyKAN, FourierKAN, and XGBoost models. Paths are written relative to the project root.

## Stage Purposes

- Stage A: tune architecture/capacity only.
- Stage B: tune optimizer/performance hyperparameters on the Stage A architecture, then retrain shortlisted candidates over seeds `13`, `29`, and `47`.
- Stage C: freeze Stage A/B settings and build the explanation package. For KANs this includes a QWK-vs-sparsity Pareto grid. For XGBoost this is the full-feature baseline train run.

## Final Summary

| Model | Stage A architecture | Stage B robust optimizer result | Stage C output |
| --- | --- | --- | --- |
| ChebyKAN | `hidden_widths=[128]`, `degree=5` | trial 016, mean QWK `0.624024`, `lr=0.0008814400571473658`, `weight_decay=0.00009472373311076866` | Pareto grid complete: 33 trials, 10 Pareto points, best QWK point `0.625136` at `sparsity_lambda=0.0`, sparsity `0.0244` |
| FourierKAN | `hidden_widths=[128]`, `grid_size=5` | trial 014, mean QWK `0.629917`, `lr=0.0011202234168345851`, `weight_decay=0.0000010952927106776856` | Pareto grid complete: 33 trials, 17 Pareto points, best QWK point `0.633108` at `sparsity_lambda=0.0`, sparsity `0.0036` |
| XGBoost (`xgb`) | `n_estimators=950`, `max_depth=3` | trial 025, mean QWK `0.648360`, `learning_rate=0.08367544058924387`, `reg_lambda=7.25884254519818` | full-feature train QWK `0.645505` |

## Interpretability Stage Summary

These are the post-training interpretation runs used for the tuned big model comparison. The KAN interpret runs use the trial `009` checkpoints selected above. Despite the Stage C sweep name containing `pareto-sparsity`, both referenced KAN configs use `sparsity_lambda=0.0`, `selected_features_path=null`, and width `128`; they are the tuned full-feature, non-sparse big models.

| Model | Interpret config | Interpret output | Main artifacts |
| --- | --- | --- | --- |
| XGBoost | [`configs/interpretability_stage/stage_c_best/xgboost.yaml`](../../../configs/interpretability_stage/stage_c_best/xgboost.yaml) | [`outputs/interpretability/xgboost_paper/stage-c-xgb-best/`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/) | SHAP values, beeswarm, dependence plots |
| ChebyKAN | [`configs/interpretability_stage/stage_c_best/chebykan.yaml`](../../../configs/interpretability_stage/stage_c_best/chebykan.yaml) | [`outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/) | feature ranking, PDP, pruned module, symbolic fits, local case explanation |
| FourierKAN | [`configs/interpretability_stage/stage_c_best/fourierkan.yaml`](../../../configs/interpretability_stage/stage_c_best/fourierkan.yaml) | [`outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/) | feature ranking, PDP, pruned module, symbolic fits, local case explanation |

Interpret-stage metrics and model sizes:

| Model | QWK before pruning / train artifact | QWK after pruning / interpreted module | Active edges before | Active edges after | Sparsity ratio | Mean per-edge R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost | `0.645505` | n/a | n/a | n/a | n/a | n/a |
| ChebyKAN | `0.625136` | `0.616137` | `17920` | `17482` | `0.0244` | `1.000000` |
| FourierKAN | `0.633108` | `0.633216` | `17920` | `17855` | `0.0036` | `1.000000` |

Interpretation note:

- XGBoost has the highest absolute QWK and uses predicted-class Tree SHAP for feature-level explanations.
- FourierKAN is closest to XGBoost in QWK among the KANs and retains nearly all active edges under the interpret-stage pruning threshold.
- ChebyKAN remains competitive but has a larger QWK gap to XGBoost and drops by `0.008999` under the pruning threshold.
- The tuned big KANs provide model-native feature rankings and PDPs, but they are dense explanation objects. They should not be described as compact symbolic models.
- The per-edge symbolic recovery reports `R2=1.0` because the native KAN edge functions can be represented exactly in the chosen basis. This does not imply that the full dense model is a simple closed-form model.

Cross-model comparison bundle:

- Output root: [`outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/`](../../../outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/)
- Main figure: [`feature_effect_comparison.pdf`](../../../outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/figures/feature_effect_comparison.pdf)
- Report: [`feature_effect_comparison.md`](../../../outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/reports/feature_effect_comparison.md)
- Ranking comparison: [`feature_ranking_comparison.csv`](../../../outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/data/feature_ranking_comparison.csv)
- Overlap summary: [`feature_overlap_summary.json`](../../../outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/data/feature_overlap_summary.json)
- Model summary: [`model_summary.json`](../../../outputs/interpretability/comparison/tuned_big_kan_vs_xgboost/data/model_summary.json)

Comparison result:

| Comparison | Shared top-20 features | Rank-scaled overlap score | Kendall tau over top-union |
| --- | ---: | ---: | ---: |
| All three models | `7` | n/a | n/a |
| ChebyKAN vs XGBoost | `12` | `5.1675` | `0.142857` |
| FourierKAN vs XGBoost | `9` | `4.1725` | `-0.026829` |

Shared top-20 features across all three models:

```text
BMI, Medical_History_15, Product_Info_4, Medical_Keyword_3, Wt, Ins_Age, Medical_History_30
```

Selected feature-effect panels in the comparison figure:

| Feature | Type | XGBoost rank | ChebyKAN rank | FourierKAN rank |
| --- | --- | ---: | ---: | ---: |
| `BMI` | continuous | `1` | `1` | `2` |
| `Medical_History_15` | continuous | `2` | `8` | `3` |
| `Product_Info_4` | continuous | `5` | `5` | `5` |
| `Medical_Keyword_3` | binary | `7` | `3` | `6` |

Paper-facing note:

- Detailed text-generation notes for ChatGPT or paper drafting are in [`docs/interpretability/cyril_paper_content/tuned_big_kan_vs_xgboost_interpretability_notes.md`](../cyril_paper_content/tuned_big_kan_vs_xgboost_interpretability_notes.md).

## ChebyKAN

### Stage A

- Config: [`configs/experiment_stages/stage_a_performance_tuning/stage_a_chebykan/chebykan_tune.yaml`](../../../configs/experiment_stages/stage_a_performance_tuning/stage_a_chebykan/chebykan_tune.yaml)
- Study DB: [`sweeps/stage_a/chebykan/stage-a-chebykan-tune.db`](../../../sweeps/stage_a/chebykan/stage-a-chebykan-tune.db)
- Best JSON: [`sweeps/stage_a/chebykan/stage-a-chebykan-tune_best.json`](../../../sweeps/stage_a/chebykan/stage-a-chebykan-tune_best.json)
- Best materialized config: [`sweeps/stage_a/chebykan/stage-a-chebykan-tune_best.yaml`](../../../sweeps/stage_a/chebykan/stage-a-chebykan-tune_best.yaml)
- Candidate manifest: [`sweeps/stage_a/chebykan/stage-a-chebykan-tune_candidates.json`](../../../sweeps/stage_a/chebykan/stage-a-chebykan-tune_candidates.json)

Outcome:

- Completed trials: `50`
- Best trial: `46`
- Best QWK: `0.625650`
- Best architecture: `width=128`, `depth=1`, `degree=5`

### Stage B

- Optimizer config: [`configs/experiment_stages/stage_b_robust_performance_tuning/chebykan_optimizer_tune.yaml`](../../../configs/experiment_stages/stage_b_robust_performance_tuning/chebykan_optimizer_tune.yaml)
- Retrain plan: [`configs/experiment_stages/stage_b_robust_performance_tuning/chebykan_retrain_plan.yaml`](../../../configs/experiment_stages/stage_b_robust_performance_tuning/chebykan_retrain_plan.yaml)
- Study DB: [`sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune.db`](../../../sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune.db)
- Best JSON: [`sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune_best.json`](../../../sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune_best.json)
- Candidate manifest: [`sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune_candidates.json`](../../../sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune_candidates.json)
- Retrain manifest: [`artifacts/stage_b/retrain/chebykan/stage-b-chebykan-optimizer-shortlist/manifest.json`](../../../artifacts/stage_b/retrain/chebykan/stage-b-chebykan-optimizer-shortlist/manifest.json)

Optimizer-tune best:

- Best trial: `23`
- Best QWK: `0.626592`
- Best tune params: `lr=0.001081440662876616`, `weight_decay=0.00025811664980013316`

Robust retrain-selected candidate:

- Candidate: `stage-b-chebykan-optimizer-tune-trial-016`
- Tune rank: `3`
- Seed QWKs: `0.619726`, `0.625567`, `0.626780`
- Mean QWK: `0.624024`
- Final Stage C dense settings: `hidden_widths=[128]`, `degree=5`, `lr=0.0008814400571473658`, `weight_decay=0.00009472373311076866`, `batch_size=256`, `sparsity_lambda=0.0`

### Stage C

- Dense full-feature config: [`configs/experiment_stages/stage_c_explanation_package/cheby/all_features/train/chebykan_best.yaml`](../../../configs/experiment_stages/stage_c_explanation_package/cheby/all_features/train/chebykan_best.yaml)
- Pareto tune config: [`configs/experiment_stages/stage_c_explanation_package/cheby/all_features/tune/chebykan_pareto_sparsity.yaml`](../../../configs/experiment_stages/stage_c_explanation_package/cheby/all_features/tune/chebykan_pareto_sparsity.yaml)
- Pareto study DB: [`sweeps/stage_c/chebykan/all_features/stage-c-chebykan-pareto-sparsity.db`](../../../sweeps/stage_c/chebykan/all_features/stage-c-chebykan-pareto-sparsity.db)
- Pareto JSON: [`sweeps/stage_c/chebykan/all_features/stage-c-chebykan-pareto-sparsity_pareto.json`](../../../sweeps/stage_c/chebykan/all_features/stage-c-chebykan-pareto-sparsity_pareto.json)
- Pareto materialized configs: [`sweeps/stage_c/chebykan/all_features/`](../../../sweeps/stage_c/chebykan/all_features/)
- Stage C run artifacts: [`artifacts/stage-c-chebykan-pareto-sparsity-trial-009/`](../../../artifacts/stage-c-chebykan-pareto-sparsity-trial-009/)
- Stage C checkpoint: [`checkpoints/stage-c-chebykan-pareto-sparsity-trial-009/model-20260429-134803.pt`](../../../checkpoints/stage-c-chebykan-pareto-sparsity-trial-009/model-20260429-134803.pt)
- Stage C eval outputs: [`outputs/eval/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/`](../../../outputs/eval/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/)

Pareto result:

- Completed trials: `33`
- Pareto points: `10`
- Highest-QWK Pareto point: trial `9`, QWK `0.625136`, sparsity ratio `0.0244`, `sparsity_lambda=0.0`
- Strong sparse reference point: trial `12`, QWK `0.619056`, sparsity ratio `0.9488`, `sparsity_lambda=0.0015351`

### Interpret Stage

- Interpret config wrapper: [`configs/interpretability_stage/stage_c_best/chebykan.yaml`](../../../configs/interpretability_stage/stage_c_best/chebykan.yaml)
- Resolved model config: [`configs/interpretability_stage/stage_c_best/chebykan_trial009.yaml`](../../../configs/interpretability_stage/stage_c_best/chebykan_trial009.yaml)
- Input checkpoint: [`checkpoints/stage-c-chebykan-pareto-sparsity-trial-009/model-20260429-134803.pt`](../../../checkpoints/stage-c-chebykan-pareto-sparsity-trial-009/model-20260429-134803.pt)
- Eval artifacts: [`outputs/eval/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/`](../../../outputs/eval/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/)
- Interpret output: [`outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/)

Interpret metrics:

- Pruning summary: [`chebykan_pruning_summary.json`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/reports/chebykan_pruning_summary.json)
- Pruning threshold: `0.01`
- QWK before pruning: `0.625136`
- QWK after pruning: `0.616137`
- QWK drop: `0.008999`
- Active edges before pruning: `17920`
- Active edges after pruning: `17482`
- Sparsity ratio: `0.0244`
- R2 report: [`chebykan_r2_report.json`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/reports/chebykan_r2_report.json)
- Mean per-edge R2: `1.000000`
- Clean symbolic edges in R2 report: `2549`

Main outputs:

- Pruned module: [`chebykan_pruned_module.pt`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/models/chebykan_pruned_module.pt)
- Feature ranking data: [`chebykan_feature_ranking.csv`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/data/chebykan_feature_ranking.csv)
- Feature ranking figure: [`chebykan_feature_ranking.pdf`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/figures/chebykan_feature_ranking.pdf)
- Partial dependence figure: [`chebykan_partial_dependence.pdf`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/figures/chebykan_partial_dependence.pdf)
- KAN diagram: [`chebykan_kan_diagram.pdf`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/figures/chebykan_kan_diagram.pdf)
- Activation figure: [`chebykan_activations.pdf`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/figures/chebykan_activations.pdf)
- R2 distribution figure: [`chebykan_r2_distribution.pdf`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/figures/chebykan_r2_distribution.pdf)
- Symbolic fits data: [`chebykan_symbolic_fits.csv`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/data/chebykan_symbolic_fits.csv)
- Symbolic formulas report: [`chebykan_symbolic_formulas.md`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/reports/chebykan_symbolic_formulas.md)
- Exact closed form report: [`chebykan_exact_closed_form.md`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/reports/chebykan_exact_closed_form.md)
- Closed-form surrogate report: [`chebykan_closed_form_surrogate.md`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/reports/chebykan_closed_form_surrogate.md)
- Local case summary: [`chebykan_case_summary_55728.md`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/reports/chebykan_case_summary_55728.md)
- Feature validation curves: [`feature_validation_curves.pdf`](../../../outputs/interpretability/kan_paper/stage-c-chebykan-pareto-sparsity-trial-009/figures/feature_validation_curves.pdf)

Top ChebyKAN native feature rankings:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `BMI` | `241.099442` |
| 2 | `Wt` | `130.339279` |
| 3 | `Medical_Keyword_3` | `112.814529` |
| 4 | `Medical_History_5` | `105.701393` |
| 5 | `Product_Info_4` | `104.145538` |

## FourierKAN

### Stage A

- Config: [`configs/experiment_stages/stage_a_performance_tuning/stage_a_fourierkan/fourierkan_tune.yaml`](../../../configs/experiment_stages/stage_a_performance_tuning/stage_a_fourierkan/fourierkan_tune.yaml)
- Study DB: [`sweeps/stage_a/fourierkan/stage-a-fourierkan-tune.db`](../../../sweeps/stage_a/fourierkan/stage-a-fourierkan-tune.db)
- Best JSON: [`sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_best.json`](../../../sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_best.json)
- Best materialized config: [`sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_best.yaml`](../../../sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_best.yaml)
- Candidate manifest: [`sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_candidates.json`](../../../sweeps/stage_a/fourierkan/stage-a-fourierkan-tune_candidates.json)

Outcome:

- Completed trials: `50`
- Best trial: `23`
- Best QWK: `0.632383`
- Best architecture: `width=128`, `depth=1`, `grid_size=5`

### Stage B

- Optimizer config: [`configs/experiment_stages/stage_b_robust_performance_tuning/fourierkan_optimizer_tune.yaml`](../../../configs/experiment_stages/stage_b_robust_performance_tuning/fourierkan_optimizer_tune.yaml)
- Retrain plan: [`configs/experiment_stages/stage_b_robust_performance_tuning/fourierkan_retrain_plan.yaml`](../../../configs/experiment_stages/stage_b_robust_performance_tuning/fourierkan_retrain_plan.yaml)
- Study DB: [`sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune.db`](../../../sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune.db)
- Best JSON: [`sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune_best.json`](../../../sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune_best.json)
- Candidate manifest: [`sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune_candidates.json`](../../../sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune_candidates.json)
- Retrain manifest: [`artifacts/stage_b/retrain/fourierkan/stage-b-fourierkan-optimizer-shortlist/manifest.json`](../../../artifacts/stage_b/retrain/fourierkan/stage-b-fourierkan-optimizer-shortlist/manifest.json)

Optimizer-tune best:

- Best trial: `28`
- Best QWK: `0.635243`
- Best tune params: `lr=0.00041495195451630713`, `weight_decay=0.0000023494880310381893`

Robust retrain-selected candidate:

- Candidate: `stage-b-fourierkan-optimizer-tune-trial-014`
- Tune rank: `3`
- Seed QWKs: `0.616853`, `0.636752`, `0.636145`
- Mean QWK: `0.629917`
- Final Stage C dense settings: `hidden_widths=[128]`, `grid_size=5`, `lr=0.0011202234168345851`, `weight_decay=0.0000010952927106776856`, `batch_size=256`, `sparsity_lambda=0.0`

### Stage C

- Dense full-feature config: [`configs/experiment_stages/stage_c_explanation_package/fourier/all_features/train/fourierkan_best.yaml`](../../../configs/experiment_stages/stage_c_explanation_package/fourier/all_features/train/fourierkan_best.yaml)
- Pareto tune config: [`configs/experiment_stages/stage_c_explanation_package/fourier/all_features/tune/fourierkan_pareto_sparsity.yaml`](../../../configs/experiment_stages/stage_c_explanation_package/fourier/all_features/tune/fourierkan_pareto_sparsity.yaml)
- Pareto study DB: [`sweeps/stage_c/fourierkan/all_features/stage-c-fourierkan-pareto-sparsity.db`](../../../sweeps/stage_c/fourierkan/all_features/stage-c-fourierkan-pareto-sparsity.db)
- Pareto JSON: [`sweeps/stage_c/fourierkan/all_features/stage-c-fourierkan-pareto-sparsity_pareto.json`](../../../sweeps/stage_c/fourierkan/all_features/stage-c-fourierkan-pareto-sparsity_pareto.json)
- Pareto materialized configs: [`sweeps/stage_c/fourierkan/all_features/`](../../../sweeps/stage_c/fourierkan/all_features/)
- Stage C run artifacts: [`artifacts/stage-c-fourierkan-pareto-sparsity-trial-009/`](../../../artifacts/stage-c-fourierkan-pareto-sparsity-trial-009/)
- Stage C checkpoint: [`checkpoints/stage-c-fourierkan-pareto-sparsity-trial-009/model-20260429-134914.pt`](../../../checkpoints/stage-c-fourierkan-pareto-sparsity-trial-009/model-20260429-134914.pt)
- Stage C eval outputs: [`outputs/eval/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/`](../../../outputs/eval/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/)

Pareto result:

- Completed trials: `33`
- Pareto points: `17`
- Highest-QWK Pareto point: trial `9`, QWK `0.633108`, sparsity ratio `0.0036`, `sparsity_lambda=0.0`
- Strong sparse reference point: trial `17`, QWK `0.628962`, sparsity ratio `0.8831`, `sparsity_lambda=0.0005`

### Interpret Stage

- Interpret config wrapper: [`configs/interpretability_stage/stage_c_best/fourierkan.yaml`](../../../configs/interpretability_stage/stage_c_best/fourierkan.yaml)
- Resolved model config: [`configs/interpretability_stage/stage_c_best/fourierkan_trial009.yaml`](../../../configs/interpretability_stage/stage_c_best/fourierkan_trial009.yaml)
- Input checkpoint: [`checkpoints/stage-c-fourierkan-pareto-sparsity-trial-009/model-20260429-134914.pt`](../../../checkpoints/stage-c-fourierkan-pareto-sparsity-trial-009/model-20260429-134914.pt)
- Eval artifacts: [`outputs/eval/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/`](../../../outputs/eval/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/)
- Interpret output: [`outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/)

Interpret metrics:

- Pruning summary: [`fourierkan_pruning_summary.json`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/reports/fourierkan_pruning_summary.json)
- Pruning threshold: `0.01`
- QWK before pruning: `0.633108`
- QWK after pruning: `0.633216`
- QWK drop: `-0.000108`
- Active edges before pruning: `17920`
- Active edges after pruning: `17855`
- Sparsity ratio: `0.0036`
- R2 report: [`fourierkan_r2_report.json`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/reports/fourierkan_r2_report.json)
- Mean per-edge R2: `1.000000`
- Clean symbolic edges in R2 report: `2560`

Main outputs:

- Pruned module: [`fourierkan_pruned_module.pt`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/models/fourierkan_pruned_module.pt)
- Feature ranking data: [`fourierkan_feature_ranking.csv`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/data/fourierkan_feature_ranking.csv)
- Feature ranking figure: [`fourierkan_feature_ranking.pdf`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/figures/fourierkan_feature_ranking.pdf)
- Partial dependence figure: [`fourierkan_partial_dependence.pdf`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/figures/fourierkan_partial_dependence.pdf)
- KAN diagram: [`fourierkan_kan_diagram.pdf`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/figures/fourierkan_kan_diagram.pdf)
- Activation figure: [`fourierkan_activations.pdf`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/figures/fourierkan_activations.pdf)
- R2 distribution figure: [`fourierkan_r2_distribution.pdf`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/figures/fourierkan_r2_distribution.pdf)
- Symbolic fits data: [`fourierkan_symbolic_fits.csv`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/data/fourierkan_symbolic_fits.csv)
- Symbolic formulas report: [`fourierkan_symbolic_formulas.md`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/reports/fourierkan_symbolic_formulas.md)
- Exact closed form report: [`fourierkan_exact_closed_form.md`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/reports/fourierkan_exact_closed_form.md)
- Closed-form surrogate report: [`fourierkan_closed_form_surrogate.md`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/reports/fourierkan_closed_form_surrogate.md)
- Local case summary: [`fourierkan_case_summary_55728.md`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/reports/fourierkan_case_summary_55728.md)
- Feature validation curves: [`feature_validation_curves.pdf`](../../../outputs/interpretability/kan_paper/stage-c-fourierkan-pareto-sparsity-trial-009/figures/feature_validation_curves.pdf)

Top FourierKAN native feature rankings:

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | `Medical_History_11` | `742.513855` |
| 2 | `BMI` | `630.719482` |
| 3 | `Medical_History_15` | `628.363709` |
| 4 | `Medical_History_17` | `609.689087` |
| 5 | `Product_Info_4` | `608.433105` |

## XGBoost

The current XGBoost model is `xgb`, backed by `src/models/xgb_baseline.py`, not `xgboost-paper`.

### Stage A

- Config: [`configs/experiment_stages/stage_a_performance_tuning/stage_a_xgboost/xgboost_tune.yaml`](../../../configs/experiment_stages/stage_a_performance_tuning/stage_a_xgboost/xgboost_tune.yaml)
- Study DB: [`sweeps/stage_a/xgboost/stage-a-xgboost-tune.db`](../../../sweeps/stage_a/xgboost/stage-a-xgboost-tune.db)
- Best JSON: [`sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.json`](../../../sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.json)
- Best materialized config: [`sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.yaml`](../../../sweeps/stage_a/xgboost/stage-a-xgboost-tune_best.yaml)
- Candidate manifest: [`sweeps/stage_a/xgboost/stage-a-xgboost-tune_candidates.json`](../../../sweeps/stage_a/xgboost/stage-a-xgboost-tune_candidates.json)

Outcome:

- Completed trials: `50`
- Best trial: `49`
- Best QWK: `0.647765`
- Best architecture: `n_estimators=950`, `max_depth=3`

### Stage B

- Optimizer config: [`configs/experiment_stages/stage_b_robust_performance_tuning/xgboost_optimizer_tune.yaml`](../../../configs/experiment_stages/stage_b_robust_performance_tuning/xgboost_optimizer_tune.yaml)
- Retrain plan: [`configs/experiment_stages/stage_b_robust_performance_tuning/xgboost_retrain_plan.yaml`](../../../configs/experiment_stages/stage_b_robust_performance_tuning/xgboost_retrain_plan.yaml)
- Study DB: [`sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune.db`](../../../sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune.db)
- Best JSON: [`sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune_best.json`](../../../sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune_best.json)
- Candidate manifest: [`sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune_candidates.json`](../../../sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune_candidates.json)
- Retrain manifest: [`artifacts/stage_b/retrain/xgb/stage-b-xgboost-optimizer-shortlist/manifest.json`](../../../artifacts/stage_b/retrain/xgb/stage-b-xgboost-optimizer-shortlist/manifest.json)

Optimizer-tune best:

- Best trial: `25`
- Best QWK: `0.645505`
- Best tune params: `learning_rate=0.08367544058924387`, `reg_lambda=7.25884254519818`

Robust retrain-selected candidate:

- Candidate: `stage-b-xgboost-optimizer-tune-trial-025`
- Tune rank: `1`
- Seed QWKs: `0.638769`, `0.653639`, `0.652673`
- Mean QWK: `0.648360`
- Final Stage C settings: `n_estimators=950`, `max_depth=3`, `learning_rate=0.08367544058924387`, `reg_lambda=7.25884254519818`, `subsample=1.0`, `colsample_bytree=1.0`, `reg_alpha=1.0`

### Stage C

- Full-feature config: [`configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgb_best.yaml`](../../../configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgb_best.yaml)
- Alias config with same model settings: [`configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgboost_best.yaml`](../../../configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgboost_best.yaml)
- Run artifacts: [`artifacts/stage-c-xgb-best/`](../../../artifacts/stage-c-xgb-best/)
- Run summary: [`artifacts/stage-c-xgb-best/run-summary-20260429-133411.json`](../../../artifacts/stage-c-xgb-best/run-summary-20260429-133411.json)
- Checkpoint: [`checkpoints/stage-c-xgb-best/model-20260429-133411.joblib`](../../../checkpoints/stage-c-xgb-best/model-20260429-133411.joblib)
- Eval outputs: [`outputs/eval/xgboost_paper/stage-c-xgb-best/`](../../../outputs/eval/xgboost_paper/stage-c-xgb-best/)

Train result:

- QWK: `0.645505`
- Accuracy: `0.378378`
- Macro F1: `0.246685`
- MAE: `1.282310`

### Interpret Stage

- Interpret config wrapper: [`configs/interpretability_stage/stage_c_best/xgboost.yaml`](../../../configs/interpretability_stage/stage_c_best/xgboost.yaml)
- Resolved model config: [`configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgb_best.yaml`](../../../configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgb_best.yaml)
- Input checkpoint: [`checkpoints/stage-c-xgb-best/model-20260429-133411.joblib`](../../../checkpoints/stage-c-xgb-best/model-20260429-133411.joblib)
- Eval artifacts: [`outputs/eval/xgboost_paper/stage-c-xgb-best/`](../../../outputs/eval/xgboost_paper/stage-c-xgb-best/)
- Interpret output: [`outputs/interpretability/xgboost_paper/stage-c-xgb-best/`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/)

Main outputs:

- SHAP values: [`shap_xgb_values.parquet`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/data/shap_xgb_values.parquet)
- SHAP beeswarm: [`shap_xgb_beeswarm.pdf`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/figures/shap_xgb_beeswarm.pdf)
- BMI dependence: [`shap_xgb_dependence_BMI.pdf`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/figures/shap_xgb_dependence_BMI.pdf)
- Medical History 15 dependence: [`shap_xgb_dependence_Medical_History_15.pdf`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/figures/shap_xgb_dependence_Medical_History_15.pdf)
- Medical History 23 dependence: [`shap_xgb_dependence_Medical_History_23.pdf`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/figures/shap_xgb_dependence_Medical_History_23.pdf)
- Medical History 4 dependence: [`shap_xgb_dependence_Medical_History_4.pdf`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/figures/shap_xgb_dependence_Medical_History_4.pdf)
- Product Info 4 dependence: [`shap_xgb_dependence_Product_Info_4.pdf`](../../../outputs/interpretability/xgboost_paper/stage-c-xgb-best/figures/shap_xgb_dependence_Product_Info_4.pdf)

The XGBoost interpretation is based on predicted-class Tree SHAP values. Unlike the KAN runs, it does not produce native KAN feature rankings, pruned modules, or symbolic edge fits.

## Stage C Pareto Fronts

### ChebyKAN Pareto Points

| Trial | QWK | Sparsity ratio | `sparsity_lambda` |
| --- | ---: | ---: | ---: |
| 9 | 0.625136 | 0.0244 | 0.0 |
| 12 | 0.619056 | 0.9488 | 0.0015351 |
| 1 | 0.616623 | 0.9923 | 0.0130862 |
| 3 | 0.607193 | 0.9935 | 0.0200887 |
| 27 | 0.604836 | 0.9946 | 0.038208 |
| 25 | 0.601745 | 0.9971 | 0.111556 |
| 20 | 0.584680 | 0.9974 | 0.17125 |
| 18 | 0.583261 | 0.9981 | 0.262885 |
| 22 | 0.578566 | 0.9983 | 0.325712 |
| 28 | 0.564376 | 0.9986 | 0.5 |

### FourierKAN Pareto Points

| Trial | QWK | Sparsity ratio | `sparsity_lambda` |
| --- | ---: | ---: | ---: |
| 9 | 0.633108 | 0.0036 | 0.0 |
| 16 | 0.632749 | 0.4384 | 0.0001 |
| 17 | 0.628962 | 0.8831 | 0.0005 |
| 19 | 0.627330 | 0.9403 | 0.001 |
| 10 | 0.626247 | 0.9485 | 0.00123899 |
| 5 | 0.625996 | 0.9768 | 0.00361749 |
| 30 | 0.622510 | 0.9812 | 0.00448204 |
| 1 | 0.621685 | 0.9890 | 0.0130862 |
| 6 | 0.616755 | 0.9939 | 0.030838 |
| 27 | 0.615923 | 0.9946 | 0.038208 |
| 23 | 0.615705 | 0.9952 | 0.0586531 |
| 21 | 0.612562 | 0.9956 | 0.0726706 |
| 2 | 0.609185 | 0.9968 | 0.138217 |
| 20 | 0.608535 | 0.9970 | 0.17125 |
| 18 | 0.605316 | 0.9978 | 0.262885 |
| 0 | 0.587129 | 0.9982 | 0.403554 |
| 28 | 0.576814 | 0.9983 | 0.5 |

## Commands Used

Stage A:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_chebykan/chebykan_tune.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_fourierkan/fourierkan_tune.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_a_performance_tuning/stage_a_xgboost/xgboost_tune.yaml
```

Stage B tune:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/chebykan_optimizer_tune.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/fourierkan_optimizer_tune.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_b_robust_performance_tuning/xgboost_optimizer_tune.yaml
```

Stage B retrain:

```bash
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/chebykan/stage-b-chebykan-optimizer-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-chebykan-optimizer-shortlist --output-experiment-prefix stage-b-chebykan
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/fourierkan/stage-b-fourierkan-optimizer-tune_candidates.json --top-k 5 --seeds 13 29 47 --selection-name stage-b-fourierkan-optimizer-shortlist --output-experiment-prefix stage-b-fourierkan
uv run python main.py --stage retrain --candidate-manifest sweeps/stage_b/xgboost/stage-b-xgboost-optimizer-tune_candidates.json --top-k 1 --seeds 13 29 47 --selection-name stage-b-xgboost-optimizer-shortlist --output-experiment-prefix stage-b-xgboost
```

Stage C:

```bash
uv run python main.py --stage tune --config configs/experiment_stages/stage_c_explanation_package/cheby/all_features/tune/chebykan_pareto_sparsity.yaml
uv run python main.py --stage tune --config configs/experiment_stages/stage_c_explanation_package/fourier/all_features/tune/fourierkan_pareto_sparsity.yaml
uv run python main.py --stage train --config configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgb_best.yaml
```

Interpret stage:

```bash
uv run python main.py --stage interpret --interpret-config configs/interpretability_stage/stage_c_best/chebykan.yaml
uv run python main.py --stage interpret --interpret-config configs/interpretability_stage/stage_c_best/fourierkan.yaml
uv run python main.py --stage interpret --interpret-config configs/interpretability_stage/stage_c_best/xgboost.yaml
```

Cross-model paper comparison:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m src.interpretability.paper_comparison
```
