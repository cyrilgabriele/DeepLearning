# Interpretability Handoff

Date: 2026-04-19

This file is a self-contained handoff for the next agent working on the KAN interpretability stage. It summarizes the current state, the user-aligned recommendation, the main technical constraints, and the exact files and artifact paths to use.

## User Goal

The user wants two deliverables for the trained KAN model:

1. An overview of learned edge functions.
   - Each edge should show the learned function and, when possible, its symbolic fitted form.
   - For edges from the input layer to the first hidden layer, the x-axis should be interpretable for a non-technical user.
   - The user specifically asked for examples like "BMI = 25 -> what function output goes into hidden node X?"

2. A plot illustrating the trained model's resulting function that outputs the risk prediction.
   - The user said both a direct model-output illustration and a symbolic-style view would be nice.

The user also explicitly asked for a root-level `.md` handoff so the next agent does not need the whole prior context again.

## Final Recommendation

Use the sparse Pareto ChebyKAN run as the primary interpretability target, not the dense best-top20 run.

Primary recommendation:

- Build a new layer-0 edge atlas for the sparse Pareto model.
- Build a separate faithful model-output plot based on prediction profiling / partial dependence.
- Do not present the current composed symbolic formula as the exact final risk function for this architecture.

Reason:

- The sparse Pareto model is much easier to inspect and explain.
- The dense model remains too large even with feature restriction.
- The current symbolic composition code is not an exact end-to-end formula for the current `TabKAN` architecture.

## Key Facts Already Verified

### 1. `--max-features 20` helps, but is not enough on the dense model

Relevant code:

- `src/interpretability/pipeline.py`
- `main.py`

The interpret stage already supports `--max-features` and applies it in:

- `src/interpretability/pipeline.py`

It zeroes non-top features in the first layer before symbolic fitting.

However, for the dense run this is still too large for a clean "overview of all edges":

- Dense config: `configs/experiment_stages/stage_c_explanation_package/chebykan_best_top20.yaml`
- Hidden widths: `[128, 64]`
- First hidden width: `128`
- With true top-20 input features, the first layer can still have up to `20 * 128 = 2560` layer-0 edges.

Current dense interpret artifacts confirm the run is still very large:

- `outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/data/chebykan_symbolic_fits.csv`
- Total symbolic edges: `26048`
- Layer-0 symbolic edges: `17856`
- Layer-1 symbolic edges: `8192`

That is still too much for a human-facing edge atlas.

### 2. The sparse Pareto model is the right interpretability target

Primary sparse config:

- `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml`

Primary sparse outputs:

- `outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/`
- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/`

Sparse pruning summary:

- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/reports/chebykan_pruning_summary.json`

Verified values:

- `edges_before = 26112`
- `edges_after = 666`
- `sparsity_ratio = 0.9745`
- `qwk_before = 0.582132`
- `qwk_after = 0.575298`
- `qwk_drop = 0.006833`

Sparse symbolic fits:

- `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/data/chebykan_symbolic_fits.csv`

Verified values:

- Total symbolic edges: `323`
- Layer-0 symbolic edges: `52`
- Layer-1 symbolic edges: `271`

This is the best base for an edge-by-edge human-readable figure.

### 3. The current "raw" BMI is not a real-world BMI like `25`

This matters for the user's request about an axis like `BMI = 25`.

Raw eval rows are reconstructed in:

- `src/training/trainer.py`

Specifically:

- `Trainer._load_raw_eval_features(...)` reads from `config.trainer.train_csv`
- It selects the original eval rows from `data/prudential-life-insurance-assessment/train.csv`

Therefore:

- `outputs/eval/.../X_eval_raw.parquet` really is the raw CSV view
- but the raw CSV itself stores `BMI`, `Wt`, `Ht`, and `Ins_Age` on normalized `0..1` scales

Verified from the actual CSV:

- `BMI`: min `0.0`, median about `0.4513`, max `1.0`
- `Wt`: min `0.0`, median about `0.2887`, max `1.0`
- `Ht`: min `0.0`, max `1.0`
- `Ins_Age`: min `0.0`, median about `0.4030`, max `1.0`

This means:

- We cannot honestly relabel the axis as physical BMI units like `25` using only the current dataset.
- Mean/std or variance are not enough to recover a missing external physical-unit mapping.
- For the current project data, the most honest labels are:
  - dataset raw value
  - cohort percentile
  - possibly low / median / high annotations

If the user still wants real BMI units, the next agent must ask for or locate an external mapping source. It cannot be inferred uniquely from the current data alone.

### 4. The user's intuition about deeper layers was correct

Deeper edges are not directly interpretable on raw feature axes.

Relevant code:

- `src/models/tabkan.py`
- `src/models/kan_layers.py`

`TabKAN` is built as:

- `KAN layer`
- `LayerNorm`
- `KAN layer`
- `LayerNorm`
- final `head = nn.Linear(widths[-1], 1)`

So for layer 1 and beyond:

- the input to an edge is not a raw dataset feature
- it is a hidden activation after aggregation
- and in this architecture also after `LayerNorm`

Therefore:

- layer-0 edge plots are the clean and human-meaningful ones
- deeper-layer edge plots can be shown only as latent / hidden-input functions
- they should not be labeled as direct raw-feature functions

### 5. The current symbolic composition is not an exact final risk formula

Relevant files:

- `src/interpretability/formula_composition.py`
- `src/models/tabkan.py`

The problem:

- `formula_composition.py` composes symbolic expressions across KAN edges only
- it does not include interleaved `LayerNorm`
- it does not include the final linear `head`

Therefore:

- the current outputs
  - `outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/reports/chebykan_symbolic_formulas.json`
  - `outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/reports/chebykan_symbolic_formulas.md`
- should not be treated as the exact trained model function

The absurd end-to-end R² values already visible in the markdown report are a symptom of this mismatch.

## Existing Interpretability Pieces Already in the Repo

These are the main existing components to inspect before editing:

### Pipeline / orchestration

- `main.py`
- `src/interpretability/pipeline.py`

### KAN symbolic fitting and layer-0 activation plots

- `src/interpretability/kan_symbolic.py`

Important:

- `_plot_activation_grid(...)` already maps layer-0 x-axes back through `encode_to_raw_lookup(...)`
- it currently shows top features, overlays all layer-0 edges for a feature, and highlights the best symbolic fit
- this is close to the user's request but not the final artifact they want

### Existing KAN network diagram

- `src/interpretability/kan_network_diagram.py`

Important:

- It draws a KAN-style network diagram with mini-plots on first-layer edges.
- It is useful context, but it is not the requested full edge overview.
- It only treats first-layer edges with mini-plots and shows later layers in simplified form.

### Existing model-output visualization

- `src/interpretability/partial_dependence.py`

Important:

- This already sweeps a feature across its range and plots average model output.
- It is the best base for the second user deliverable.
- It currently plots only the raw model output curve and should be upgraded.

### Style / x-axis mapping utilities

- `src/interpretability/utils/style.py`

Important utility:

- `encode_to_raw_lookup(...)`

This interpolates between encoded values and raw eval values.

### Preprocessing confirmation

- `src/preprocessing/preprocess_kan_paper.py`
- `src/preprocessing/preprocess_paper_base.py`
- `src/training/trainer.py`

Important:

- `kan_paper` does not standardize BMI into z-scores.
- The CSV itself is already in normalized raw space for several features.

## Recommended Implementation Plan

### Task 1. Add a new layer-0 edge atlas module

Create a new file:

- `src/interpretability/layer0_edge_atlas.py`

Purpose:

- Produce a human-readable overview of first-layer edges only.
- This is the main answer to the user's "overview of edge functions" request.

Recommended behavior:

- Input:
  - trained `TabKAN` module
  - symbolic fits dataframe
  - `X_eval`
  - `X_raw`
  - `feat_types`
  - `output_dir`
  - `flavor`
- Filter to layer `0` only
- Focus on active edges only
- Prefer the sparse Pareto run first

Recommended output artifact:

- `outputs/interpretability/kan_paper/<experiment>/figures/chebykan_layer0_edge_atlas.pdf`

Recommended figure design:

- Multi-page PDF
- Group panels by feature
- Within each feature, show one panel per active edge `feature -> hidden node`
- For each panel:
  - solid line = learned edge function
  - dashed line = symbolic fit only if `quality_tier` is `acceptable` or `clean`
  - title / subtitle should include:
    - hidden node id
    - formula string if shown
    - `R²`
    - edge importance or rank
- X-axis:
  - continuous / ordinal: dataset raw scale using `encode_to_raw_lookup(...)`
  - binary / missing-indicator: encoded `-1/+1` positions with labels
  - categorical-like encoded columns: keep encoded / code labels unless a real mapping exists
- Add a top secondary x-axis for percentile if feasible
  - this makes the plot more interpretable for non-technical readers
  - especially important because BMI is not in physical units here

Recommended filtering / ordering:

- order features by first-layer importance
- within each feature, order edges by edge importance descending
- for the sparse Pareto model, show all layer-0 edges
- for dense models, require pagination and/or feature subsetting

Do not:

- pretend dataset raw BMI equals physical BMI
- label a point as `BMI = 25`

### Task 2. Build a faithful model-output figure

Either extend:

- `src/interpretability/partial_dependence.py`

or create:

- `src/interpretability/prediction_profiles.py`

Recommendation:

- Create a new module if the upgraded figure becomes meaningfully more complex than the current PDP grid.

Purpose:

- Answer the user's second request: show how the trained model behaves at the prediction level.

Recommended artifact:

- `outputs/interpretability/kan_paper/<experiment>/figures/chebykan_prediction_profiles.pdf`

Recommended content per feature:

- Top panel:
  - mean raw regression output from the model
- Bottom panel:
  - mean final rounded risk class `1..8`

Optional additions:

- ICE quantile band or shaded variability band
- rug plot
- percentile top axis
- observed response trend for reference if useful

Important:

- The current `TabKANClassifier.predict(...)` rounds to integer classes
- but the model itself outputs a continuous score
- both are useful

So the plot should ideally show both:

- continuous model score
- discrete rounded risk class

### Task 3. Keep symbolic output honest

Current status:

- `src/interpretability/formula_composition.py` is not exact for the current architecture

Recommendation:

- Do not remove it blindly
- either disable it for `TabKAN` with `LayerNorm`
- or relabel it clearly as an internal approximate composition, not the final predictor

Minimum acceptable change:

- prevent the pipeline from presenting this as the exact final risk formula

Better change:

- in `src/interpretability/pipeline.py`, gate the call to `run_formula_composition(...)`
- if the model contains `LayerNorm` in `module.kan_layers` or a nontrivial final head, emit a report explaining why exact symbolic end-to-end composition is not available

Optional future work:

- fit an explicit surrogate symbolic model to the final prediction output
- save it as something like:
  - `reports/chebykan_surrogate_output_formula.md`
- label it clearly as approximate

### Task 4. Wire the new artifacts into the interpret pipeline

Edit:

- `src/interpretability/pipeline.py`

Recommended changes:

- load the same `pruned_module`, `fits_df`, `X_eval`, `X_raw`, and `feat_types`
- call the new layer-0 atlas function
- call the upgraded prediction-profile function
- include the new artifact paths in the final `result["artifacts"]`

Keep existing useful artifacts:

- pruning summary
- symbolic fits CSV
- feature ranking
- current KAN network diagram
- current PDP if still useful

But the new atlas and prediction-profile outputs should become the main user-facing deliverables.

## Exact Files Likely To Edit

Primary edits:

- `src/interpretability/pipeline.py`
- `src/interpretability/partial_dependence.py` or new `src/interpretability/prediction_profiles.py`
- new `src/interpretability/layer0_edge_atlas.py`

Possible cleanup edits:

- `src/interpretability/formula_composition.py`
- `src/interpretability/kan_symbolic.py`

Reference-only files:

- `src/models/tabkan.py`
- `src/models/kan_layers.py`
- `src/interpretability/utils/style.py`
- `src/training/trainer.py`

## Existing Artifact Paths To Use

Primary sparse run:

- Config:
  - `configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml`
- Eval data:
  - `outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/X_eval.parquet`
  - `outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/X_eval_raw.parquet`
  - `outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/y_eval.parquet`
  - `outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/feature_names.json`
  - `outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20/feature_types.json`
- Interpret outputs:
  - `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/models/chebykan_pruned_module.pt`
  - `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/reports/chebykan_pruning_summary.json`
  - `outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20/data/chebykan_symbolic_fits.csv`

Dense comparison run:

- Config:
  - `configs/experiment_stages/stage_c_explanation_package/chebykan_best_top20.yaml`
- Interpret outputs:
  - `outputs/interpretability/kan_paper/stage-c-chebykan-best-top20/`

## Suggested CLI For Re-running Interpretability

If a full rerun is needed for the sparse interpretability target, use:

```bash
uv run python main.py \
  --stage interpret \
  --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml \
  --pruning-threshold 0.001 \
  --qwk-tolerance 0.01 \
  --candidate-library scipy \
  --max-features 20
```

Notes:

- `--pruning-threshold 0.001` matches the existing sparse Pareto interpretability output.
- `--max-features 20` is acceptable here because the sparse model is already very small.
- In sandboxed Codex runs, `uv` may need `UV_CACHE_DIR=/tmp/uv-cache` to avoid cache permission issues.

Example sandbox-friendly variant:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python main.py \
  --stage interpret \
  --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20.yaml \
  --pruning-threshold 0.001 \
  --qwk-tolerance 0.01 \
  --candidate-library scipy \
  --max-features 20
```

## Testing Guidance

Relevant existing tests:

- `tests/interpretability/test_pipeline.py`
- `tests/interpretability/test_formula_composition.py`
- `tests/interpretability/test_kan_network_diagram.py`
- `tests/interpretability/test_style.py`

Recommended new or updated tests:

- add tests for the new atlas module
  - empty dataframe behavior
  - layer-0-only filtering
  - raw-axis mapping behavior
  - symbolic overlay only for acceptable / clean fits
- add tests for the upgraded prediction-profile figure
  - output file is created
  - both raw-score and rounded-risk views are computed
- update pipeline tests if new artifacts are added to `result["artifacts"]`
- update or gate formula-composition tests if behavior changes for architectures with `LayerNorm`

Suggested new test files:

- `tests/interpretability/test_layer0_edge_atlas.py`
- `tests/interpretability/test_prediction_profiles.py`

## Acceptance Criteria

The work is done when all of the following are true:

1. The sparse Pareto ChebyKAN interpretability pipeline produces a dedicated layer-0 edge atlas PDF.
2. The atlas uses honest x-axis labeling.
   - No fake physical BMI units.
   - Dataset raw scale and/or percentile are shown.
3. The interpret pipeline produces a dedicated model-output plot that is understandable to a non-technical user.
4. The repo no longer implies that the current `formula_composition.py` output is the exact final `TabKAN` predictor.
5. Tests pass for the newly added behavior.

## Non-Negotiable Constraints

- Do not claim `BMI = 25` from the current dataset alone.
- Do not describe deeper KAN-layer edge inputs as raw dataset feature values.
- Do not present the current symbolic composition as the exact final risk function for `TabKAN` with `LayerNorm` and final head.

## Short Implementation Order

If another agent wants the fastest correct path:

1. Implement `src/interpretability/layer0_edge_atlas.py`
2. Implement or upgrade the prediction-level profile figure
3. Wire both into `src/interpretability/pipeline.py`
4. Add / update tests
5. Gate or relabel `formula_composition.py` in the pipeline

