# Selection Pipeline Findings

Date: 2026-04-09

## Finding 1

Severity: Medium

The selector stage still ignores post-pruning predictive quality, even though the audit explicitly calls for it.

Affected code paths:

- `src/selection/pipeline.py:140` loads `qwk_after_pruning`
- `src/selection/pipeline.py:150` never summarizes it
- `src/selection/pipeline.py:204` never ranks by it

Reproduction summary:

- `run_select()` chose a smaller candidate with `qwk_after_pruning = 0.2`
- it preferred that candidate over another with `qwk_after_pruning = 0.8`
- the choice happened purely because the smaller candidate had fewer edges after pruning

Practical implication:

- the new `select` stage is only partially aligned with the documented rule
- post-pruning predictive quality is gathered but not actually used in candidate selection

## Finding 2

Severity: Low

`run_select()` writes the selection manifest to the wrong root when `interpretability_root` is provided.

Affected code path:

- `src/selection/pipeline.py:32` switches the output path to `output_root` instead of honoring `selection_output_root`

Reproduction summary:

- `selection_output_root/chebykan_selection.json` was not created
- `output_root/chebykan_selection.json` was created instead

Practical implication:

- callers cannot redirect interpretability reads and selection artifact writes independently
- the function violates its own output-root contract in this branch
