# Getting Into Interpretability With Our Models

This note is a practical guide for working yourself into the interpretability topic with the models in this repository.

It is intentionally simpler than the papers. The goal is not to understand every theoretical detail up front. The goal is to build a usable mental model, know what the current code can and cannot tell you, and follow a sensible order when you start analyzing our models.

For the deeper code-vs-paper audit, see `docs/interpretability/kan_interpretability_review.md`.

## 1. The Simplest Useful Mental Model

For this project, think about interpretability in four layers:

1. Which features matter most?
2. What shape did the model learn for one feature?
3. How much of the learned structure can be pruned without hurting performance too much?
4. Can some learned 1D edge functions be approximated by simple formulas?

That is already enough to start making progress.

You do not need to begin with full symbolic recovery of an entire KAN.

## 2. How To Think About KAN Interpretability

A useful starting point is:

- A KAN does not just learn weights. It learns functions on edges.
- For tabular data, that means you can inspect how one input feature is transformed before it is composed with later layers.
- In principle, this can be more interpretable than a standard MLP because the learned 1D edge functions can be plotted and sometimes approximated symbolically.

However, for this repository specifically, there is an important limitation:

- Most of the current tooling inspects the first KAN layer only.
- So for now, treat the current pipeline primarily as feature-function inspection, not as full compositional KAN interpretability.

That limitation is important because it changes how you should read the outputs. If you see a clean curve for one feature, that is useful. But it is not yet the same thing as having explained the full model.

## 3. What To Start With First

If the goal is to learn the topic using our models, start in this order:

1. Start with `ChebyKAN`, not `FourierKAN`.
2. Start with the top 5 to 10 features, not the whole model.
3. Start with feature ranking and function plots, not symbolic regression.
4. Only after that look at pruning.
5. Only after that look at symbolic fits and R² tiers.
6. Compare against XGBoost and GLM only after you understand one KAN on its own.

Why this order:

- `ChebyKAN` is usually easier to read visually.
- Top-feature inspection gives intuition quickly.
- Pruning and symbolic regression make more sense once you already know what the model is doing on important features.
- Cross-model comparison is more useful after you have a stable intuition for one model family.

## 4. The Questions You Should Ask While Learning

When you inspect a model, use questions like these:

- Which input features dominate the first-layer coefficient ranking?
- For an important feature, is the learned function roughly linear, monotonic, saturating, threshold-like, or oscillatory?
- Does the learned shape agree with domain intuition?
- After pruning, do the important features remain important?
- Which edges are simple enough to be described by a compact formula?
- Which edges are not simple, even after pruning?

These questions are much better than asking, "Do I understand the whole paper now?"

## 5. Recommended Learning Workflow In This Repo

Once the code issues are fixed, the best onboarding workflow is:

1. Export the evaluation split into `outputs/data/`.
2. Run one `ChebyKAN` model through pruning.
3. Inspect coefficient importance and learned feature functions.
4. Run symbolic fitting on the pruned model.
5. Read the symbolic fit quality report.
6. Only then move to `FourierKAN`.
7. Only after both of those, run side-by-side comparisons with GLM and XGBoost.

This is the practical meaning of the workflow:

- `Feature ranking` tells you where to look.
- `Function plots` tell you what shape was learned.
- `Pruning` tells you what structure seems unnecessary.
- `Symbolic fits` tell you which edge functions are simple enough to describe compactly.

## 6. What The Current Code Can Already Teach You

Even in its current state, the repository can already teach you several useful things:

- Paper-native coefficient importance for first-layer KAN features
- Feature-function plots on important features
- Post-hoc pruning based on activation magnitude
- Symbolic fitting quality summaries for active edges

That is enough to learn a lot about how the model behaves locally and which features it relies on.

## 7. What The Current Code Does Not Yet Fully Give You

The current code should not yet be treated as a complete KAN interpretability implementation.

Main limitations:

- Most analyses are first-layer only, so the compositional structure is not fully inspected.
- Symbolic fitting exists, but symbolic lock-in is not implemented.
- Some visualizations still have correctness issues noted in the review.
- Cross-model plots need care because raw importance scales are not always directly comparable.

This means:

- Use the outputs to build intuition.
- Do not overclaim what they prove.

## 8. Repo-Specific Practical Findings

These are the main practical findings from inspecting the current codebase.

### 8.1 Raw model location

The raw trained models do not have to live in one single hardcoded folder for the interpretability scripts.

- The KAN scripts accept explicit checkpoint paths.
- The pipeline's own derived artifacts are written under `outputs/`, especially `outputs/models/`, `outputs/data/`, `outputs/reports/`, and `outputs/figures/`.

So `models/` is a fine storage location for the original checkpoints.

### 8.2 KAN checkpoint compatibility

The bigger issue is not folder location but architecture reconstruction.

The current interpretability loaders rebuild KAN architectures from config files before loading weights. Your saved artifacts in `models/` contain architecture details such as:

- exact widths
- Cheby degree
- Fourier grid size

At the time of writing, the current interpretability loaders rebuild uniform-width models from config, which does not reliably match the saved checkpoints.

So before running the KAN interpretability pipeline on the saved models, the code should be fixed so that the rebuilt architecture matches the artifact metadata exactly.

### 8.3 XGBoost checkpoint compatibility

The current SHAP pipeline expects a `joblib`-serialized wrapper model, not just a raw XGBoost `.json` booster file.

So the KAN checkpoints in `models/` are close to usable after loader fixes, but the current XGBoost artifacts are not yet in the format the SHAP script expects.

### 8.4 Missing eval artifacts

The interpretability scripts expect exported evaluation artifacts such as:

- `outputs/data/X_eval.parquet`
- `outputs/data/y_eval.parquet`
- `outputs/data/X_eval_raw.parquet`
- `outputs/reports/feature_names.json`
- `outputs/reports/feature_types.json`

Those outputs are the bridge between model checkpoints and interpretability scripts. Without them, the pipeline cannot run end to end.

## 9. What I Would Personally Do First After The Code Fixes

I would start with exactly this sequence:

1. Fix the loader and eval-data export issues.
2. Run only `ChebyKAN`.
3. Look at coefficient importance.
4. Look at the top 5 learned feature functions.
5. Prune the model and check whether QWK stays close.
6. Run symbolic fitting on the pruned model.
7. Read the R² report and separate `clean`, `acceptable`, and `flagged` edges.

Only after that would I:

1. run `FourierKAN`
2. compare `ChebyKAN` and `FourierKAN`
3. compare KAN against XGBoost SHAP and GLM coefficients

## 10. How To Read The Papers Without Getting Lost

For `2404.19756v5`, I would not try to read it as one continuous block.

Instead, read it in this order:

1. The basic idea of KANs as learned edge functions
2. Why sparsity matters
3. Why pruning matters
4. What symbolification is trying to achieve

Then map each of those ideas back to one concrete thing in our repo:

- "learned edge functions" -> feature-function plots
- "sparsity" -> regularization and pruning
- "pruning" -> active-edge selection
- "symbolification" -> symbolic fits and R² tiers

This mapping is much more useful than trying to memorize every equation immediately.

## 11. Short Version

If you want the shortest possible guide:

- Start with `ChebyKAN`.
- First learn feature ranking and feature-function plots.
- Then learn pruning.
- Then learn symbolic fitting.
- Treat the current repo as first-layer KAN interpretability, not full compositional interpretability.
- Fix the loader and eval-data issues before trying to run the full pipeline on the saved models.

## 12. Related Document

For the deeper repository audit, limitations, and fix priorities, see:

- `docs/interpretability/kan_interpretability_review.md`
