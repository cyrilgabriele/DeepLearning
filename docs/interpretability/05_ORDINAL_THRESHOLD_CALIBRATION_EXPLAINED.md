# 05 Ordinal Threshold Calibration Explained

This note explains the threshold-adjustment change we introduced for the ordinal Prudential models, especially the TabKAN path.

## Short Version

The model still learns a continuous score.

What changed is how that score is converted into the final ordinal class from `1` to `8`.

Before, we used naive rounding:

$$
\hat{c} = \mathrm{clip}(\mathrm{round}(s), 1, 8)
$$

where:

- $s$ is the model's continuous score
- $\hat{c}$ is the predicted class

Now, we learn 7 score thresholds after fitting the model:

$$
t_1 < t_2 < \dots < t_7
$$

and then map the score to classes using those learned cutpoints instead of fixed `.5` boundaries.

## Why We Changed It

The training target is ordinal, and the project metric is Quadratic Weighted Kappa (QWK).

Naive rounding assumes the right class boundaries are always:

$$
1.5,\ 2.5,\ 3.5,\ 4.5,\ 5.5,\ 6.5,\ 7.5
$$

That assumption is arbitrary. The model's score scale does not have to align perfectly with those boundaries.

If the model tends to place useful separation points at slightly different values, fixed rounding throws away performance and creates a weaker class-definition contract.

The threshold-calibration step fixes that by learning the boundaries that best convert model scores into ordinal classes for the chosen calibration split.

## Old Rule vs New Rule

### Old Rule

The old class mapping was:

$$
\hat{c} = \mathrm{clip}(\mathrm{round}(s), 1, 8)
$$

This means:

- scores below `1.5` map to class `1`
- scores from `1.5` up to `2.5` map to class `2`
- scores from `2.5` up to `3.5` map to class `3`
- and so on

### New Rule

The new class mapping learns thresholds $t_1, \dots, t_7$ and applies:

$$
\hat{c} =
\begin{cases}
1 & \text{if } s < t_1 \\
2 & \text{if } t_1 \le s < t_2 \\
3 & \text{if } t_2 \le s < t_3 \\
\vdots \\
8 & \text{if } s \ge t_7
\end{cases}
$$

In code, this is implemented with `np.digitize(...) + 1` in [src/metrics/qwk.py](/Users/cyrilgabriele/Documents/School/00_Courses/01_DL/03_Project/DeepLearning/src/metrics/qwk.py:9).

## What Is Actually Being Optimized

After the model is trained, we take:

- the true ordinal labels $y$
- the model's continuous scores $s$

and solve:

$$
\hat{t} = \arg\max_t \mathrm{QWK}(y, g(s; t))
$$

where:

- $t = (t_1, \dots, t_7)$ is the threshold vector
- $g(s; t)$ is the score-to-class mapping induced by those thresholds

So we are not retraining the network weights in this step.

We are only calibrating the boundary rule that turns the already-trained continuous score into ordinal classes.

The implementation lives in [src/metrics/qwk.py](/Users/cyrilgabriele/Documents/School/00_Courses/01_DL/03_Project/DeepLearning/src/metrics/qwk.py:14).

## How The Optimizer Works

The threshold optimizer:

1. starts from the naive thresholds
2. evaluates QWK after applying those thresholds
3. moves the thresholds to improve QWK
4. returns the best sorted threshold vector it finds

The initial guess is:

$$
(1.5,\ 2.5,\ 3.5,\ 4.5,\ 5.5,\ 6.5,\ 7.5)
$$

The current implementation uses SciPy's Nelder-Mead optimizer in [src/metrics/qwk.py](/Users/cyrilgabriele/Documents/School/00_Courses/01_DL/03_Project/DeepLearning/src/metrics/qwk.py:23).

## Where The Calibration Data Comes From

For `TabKANClassifier`, the threshold source is:

- the inner validation split when that split is available
- otherwise the training split

That logic is in [src/models/tabkan.py](/Users/cyrilgabriele/Documents/School/00_Courses/01_DL/03_Project/DeepLearning/src/models/tabkan.py:309).

This matters because calibrating thresholds on a held-out validation split is less biased than calibrating them on the same data used to fit the model weights.

The recorded source is stored as:

- `inner_validation`, or
- `training`

## What Changed In The TabKAN Model Contract

The model now exposes two distinct concepts:

1. continuous score prediction
2. ordinal class prediction from stored thresholds

The continuous score path remains available through `predict_scores(...)` in [src/models/tabkan.py](/Users/cyrilgabriele/Documents/School/00_Courses/01_DL/03_Project/DeepLearning/src/models/tabkan.py:316).

The ordinal prediction path now does:

- apply stored optimized thresholds when available
- otherwise fall back to rounded scores

That inference rule is implemented in [src/models/tabkan.py](/Users/cyrilgabriele/Documents/School/00_Courses/01_DL/03_Project/DeepLearning/src/models/tabkan.py:340).

So the fallback still exists, but it is now explicitly the fallback path rather than the primary contract for newly trained calibrated runs.

## Why The Continuous Score Still Matters

Threshold calibration does not replace the score model.

The model still learns a continuous function:

$$
s = f(x)
$$

where $x$ is the processed feature vector.

Threshold calibration only adds the ordinal decision layer on top:

$$
\hat{c} = g(f(x); t)
$$

This distinction matters for interpretability work:

- sensitivities and symbolic explanations act on the score function $f(x)$
- reported business classes act on the thresholded output $g(f(x); t)$

## Persistence Contract

The calibration metadata is persisted as run artifacts, not as preprocessing state.

That is the right design because thresholds are:

- learned after fit
- derived from model scores
- part of the class-definition contract for a trained run

The trainer writes the threshold payload to:

- `artifacts/<experiment>/run-summary-<timestamp>.json`
- `checkpoints/<experiment>/model-<timestamp>.manifest.json`
- `outputs/eval/<recipe>/<experiment>/ordinal_thresholds.json`

The export logic is in [src/training/trainer.py](/Users/cyrilgabriele/Documents/School/00_Courses/01_DL/03_Project/DeepLearning/src/training/trainer.py:478).

For the underwriting artifact, the canonical generator input should be:

- `outputs/eval/<recipe>/<experiment>/ordinal_thresholds.json`

The other two copies are best treated as audit mirrors.

## Payload Shape

The stored threshold payload looks like this:

```json
{
  "method": "optimized_thresholds",
  "num_classes": 8,
  "optimized_qwk_on_source_split": 0.5559111206230054,
  "source_split": "inner_validation",
  "thresholds": [
    1.4928179230983378,
    2.6395737176803262,
    3.8438726438502964,
    4.855444742584975,
    5.556068462787066,
    6.064501321515802,
    6.6009262102599715
  ]
}
```

Example real artifact:

- [ordinal_thresholds.json](/Users/cyrilgabriele/Documents/School/00_Courses/01_DL/03_Project/DeepLearning/outputs/eval/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/ordinal_thresholds.json:1)

## Concrete Example

Suppose the learned thresholds are:

$$
(1.49,\ 2.64,\ 3.84,\ 4.86,\ 5.56,\ 6.06,\ 6.60)
$$

Then:

- if `score = 5.50`, the prediction is class `5`
- if `score = 5.60`, the prediction is class `6`
- if `score = 6.20`, the prediction is class `7`

This already shows why naive rounding is not equivalent.

Under naive rounding, `5.50` would sit exactly at the generic `5.5` boundary. Under learned thresholds, the actual `5` to `6` boundary for this run is approximately `5.56`.

## Why This Matters For Underwriter-Facing Outputs

The underwriting artifact needs more than just a class label. It also needs:

- a stable class-definition rule
- distance to neighboring class thresholds
- consistent interpretation across training, pruning, and case-level reporting

Without persisted thresholds, two different artifacts could talk about the same model score using different class rules:

- one artifact might use calibrated thresholds
- another might silently use `round(score)`

That would make class labels and class-margin explanations inconsistent.

## Relation To Pruning-Stage Reporting

After threshold calibration was introduced in training, pruning-stage reporting also had to be aligned with the same stored thresholds.

Otherwise the comparison would be unfair:

- training QWK would be threshold-calibrated
- pruning QWK would still be computed from rounded scores

That would create an artificial metric gap caused by different class-mapping rules rather than by pruning itself.

## Fallback Behavior

The system still supports rounded-score fallback when calibrated thresholds are unavailable:

$$
\hat{c}_{\mathrm{fallback}} = \mathrm{clip}(\mathrm{round}(s), 1, 8)
$$

But that fallback should now be understood as:

- acceptable for older historical runs that do not have persisted threshold metadata
- not the preferred rule for newly trained calibrated runs

## Main Takeaway

The core model still predicts a continuous score.

The threshold-adjustment change added a learned ordinal calibration layer on top of that score so that class assignment:

- matches the ordinal evaluation metric better
- uses run-specific learned boundaries instead of fixed `.5` rounding boundaries
- can be persisted and reused consistently across downstream artifacts

In short:

$$
\text{old: } x \rightarrow s \rightarrow \mathrm{round}(s)
$$

$$
\text{new: } x \rightarrow s \rightarrow \text{optimized thresholds} \rightarrow \hat{c}
$$
