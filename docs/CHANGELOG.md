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

### [2026-03-09] - Gian Seifert

#### What
- Refined feature categorization: moved low-cardinality "continuous" features (e.g., `Product_Info_3`, `Employment_Info_2`) to the `ordinal` group.
- Switched ordinal scaling from `QuantileTransformer` to `MinMaxScaler` mapped to `[-1, 1]`.
- Enabled automated missingness indicators (`missing_` flags) in `SOTAPreprocessor`.
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
