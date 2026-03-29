# KAN Interpretability Pipeline — Design Decisions

> For each adjustable decision in the pruning, symbolic regression, and R² pipeline, this document
> records the current implementation choice, what the literature says, and — where no paper applies
> — what needs to be tested empirically.
>
> **Notation:**
> - 📄 = backed by a published paper (citation included)
> - 🧪 = no clear paper evidence; requires empirical testing in this project

---

## 1. Pruning Criterion

### Current implementation
Edge output variance over a uniform grid (`kan_pruning.py:29–43`).

### What the literature says

📄 **Liu et al. (2024) — "KAN: Kolmogorov-Arnold Networks"**
arXiv:2404.19756, ICLR 2025.

The original KAN paper uses the **L1 norm of the learned activation function** as the pruning
signal, not output variance. Specifically, it defines:

- *Incoming score* for node `(l, i)`:
  `I_{l,i} = max_k || φ_{l-1,k,i} ||_1`
- *Outgoing score*:
  `O_{l,i} = max_j || φ_{l+1,j,i} ||_1`

A node is retained only if both scores exceed a threshold `θ` (default `θ = 10⁻²`). Pruning is
**node-level**, not edge-level — all edges incident to a pruned node are removed as a consequence.

Crucially, sparsity is not applied post-hoc: the training objective includes an L1 + entropy
regularisation term that drives activations toward zero *before* pruning:

```
ℓ_total = ℓ_pred + λ · (μ₁ Σ ||Φ_l||_1 + μ₂ Σ S(Φ_l))
```

where `S(Φ_l)` is the entropy of activation magnitudes within each layer. Default `μ₁ = μ₂ = 1`.

📄 **Molchanov et al. (2019) — "Importance Estimation for Neural Network Pruning"**
arXiv:1906.10771, CVPR 2019.

For CNN filters, the first-order Taylor expansion of the loss w.r.t. a filter's output
(`score = weight × gradient`) achieves >93% correlation with the true importance on ImageNet.
This is the primary justification for gradient-based pruning in MLPs/CNNs. However, the criterion
requires a scalar "weight" — which KAN edges do not have (they have full univariate functions).
Liu et al. explicitly designed around this by using L1 norm of the activation function instead.

📄 **Hayou et al. (2021) — "Robust Pruning at Initialization"**
arXiv:2002.08797, ICLR 2021.

Gradient-based pruning at initialization can cause **entire layers to be pruned** (structural
collapse). Magnitude-based pruning is structurally more robust when the network is not yet fully
trained. This justifies Liu et al.'s choice of L1 magnitude over gradient for KAN.

📄 **Cheng et al. (2023) — "A Survey on Deep Neural Network Pruning"**
arXiv:2308.06767.

No single criterion universally dominates across architectures. Magnitude-based criteria are the
most stable and widely used baseline. Gradient-based criteria require calibration data and can
degrade if the calibration distribution differs from training.

### Recommendation
**Switch from output variance to L1 norm of activation coefficients**, matching Liu et al.'s
criterion. For ChebyKAN this is `||cheby_coeffs[out, in, :]||_1 + |base_weight[out, in]|`.
This is both theoretically grounded and matches the reference implementation.

### Empirical testing needed 🧪
- Compare sparsity achieved (% edges zeroed) at the same threshold between variance and L1
  criterion on the Prudential dataset — they may differ significantly because variance penalises
  near-constant functions while L1 penalises small-magnitude functions regardless of constancy.

---

## 2. Training-time vs. Post-hoc Pruning

### Current implementation
Post-hoc: train to convergence, then apply variance threshold. No regularisation during training.

### What the literature says

📄 **Liu et al. (2024)** arXiv:2404.19756

Sparsity regularisation (`L1 + entropy`) is applied **during training**, not post-hoc. This drives
the network toward a sparse structure before any threshold is applied. The paper shows that without
regularisation, post-hoc thresholding is less clean (more edges hover near the threshold).

📄 **Liu et al. (2024b) — "KAN 2.0: Kolmogorov-Arnold Networks Meet Science"**
arXiv:2408.10205.

KAN 2.0 confirms the train-time regularisation approach and extends it with sparsity-2 (higher-
order norm) for even cleaner symbolic identification. Post-hoc pruning is only presented as a
diagnostic tool.

### Recommendation
Add the `L1 + entropy` regularisation term to `TabKAN`'s training loss. This is a significant
change that requires a new training run.

### Empirical testing needed 🧪
- Retrain ChebyKAN with `λ=1e-3, μ₁=μ₂=1` and compare: (a) sparsity ratio after pruning,
  (b) mean R² of symbolic fits, (c) QWK vs the current unregularised model.
- Sweep `λ` in `{1e-4, 1e-3, 1e-2}` — too large a penalty will hurt QWK.

---

## 3. Pruning Threshold Value

### Current implementation
Fixed absolute threshold `threshold = 0.01` (CLI default, `kan_pruning.py:144`).

### What the literature says

📄 **Liu et al. (2024)** arXiv:2404.19756

Uses `θ = 10⁻²` as the default node-pruning threshold — identical to the current value.
No principled derivation is given; it is a convention chosen empirically for physics examples.

📄 **Akazan & Mbingui (2025) — "Splines-Based Feature Importance in KANs"**
arXiv:2509.23366.

L1-based feature selectors (`KAN-L1`) are described as *too aggressive in regression* —
pruning useful features. `KAN-SI` and `KAN-KO` (which use a softer, relative threshold)
perform more robustly on noisy tabular data. This supports using a relative threshold.

### Recommendation
**Test a relative threshold**: `threshold = c × mean_L1_across_all_edges`. Setting `c = 0.1`
means "prune edges with L1 < 10% of the average". This scales automatically with model size and
data and avoids needing a new threshold when architecture changes.

### Empirical testing needed 🧪
- Sweep absolute threshold `{0.001, 0.005, 0.01, 0.05}` and record sparsity vs QWK drop for
  both ChebyKAN and FourierKAN.
- Sweep relative threshold multiplier `c ∈ {0.05, 0.1, 0.2}` and compare.
- Target: find the threshold that achieves ≥80% sparsity with QWK drop ≤0.01.

---

## 4. QWK Tolerance for Pruning Validation

### Current implementation
`qwk_tolerance = 0.01` — if QWK drops more than this after pruning, retry with a tighter
threshold (`kan_pruning.py:145, 187–201`).

### What the literature says
No published paper addresses a "QWK drop tolerance" specifically. This is a project-specific
design decision tied to the choice of evaluation metric.

### Empirical testing needed 🧪
- If interpretability is the primary deliverable (over predictive accuracy), raising tolerance
  to `0.03` allows more aggressive pruning at the cost of 3 QWK points.
- Test: what is the maximum sparsity achievable at tolerance `{0.005, 0.01, 0.02, 0.03}`?
  Plot the sparsity–accuracy Pareto curve for ChebyKAN.

---

## 5. Threshold Sweep Schedule

### Current implementation
If initial threshold fails: try `[t/2, t/5, t/10, 0.001]` in order (`kan_pruning.py:189`).
Takes the first that passes QWK tolerance.

### What the literature says
No paper addresses this specific sweep schedule. Binary search would be more principled but this
is a minor engineering choice.

### Empirical testing needed 🧪
- The current coarse sweep may jump over the optimal threshold (e.g., the true best threshold
  might be between `t/2` and `t/5`). Replace with a binary search between the last failing and
  first passing threshold, with a stopping tolerance of `0.0001`. This guarantees finding the
  maximally sparse threshold that meets QWK tolerance.

---

## 6. Sampling Domain for Edge Activation

### Current implementation
`x = linspace(-3, 3, 1000)` → `x_norm = tanh(x)` → range ≈ `[-0.995, +0.995]`
(`kan_symbolic.py:36–37`).

### What the literature says

📄 **Liu et al. (2024)** arXiv:2404.19756 — pykan `API_5_grid` documentation.

Describes `grid_eps` parameter:
- `grid_eps = 1.0`: uniform grid (default)
- `grid_eps = 0.0`: adaptive grid based on data percentiles
- Intermediate values interpolate between the two

The adaptive grid allocates resolution where data is actually distributed, which is directly
relevant to how the sampling domain should be chosen for symbolic regression.

📄 **Anonymous (2025) — "Automatic Grid Updates for KANs using Layer Histograms"**
arXiv:2511.08570.

Proposes AdaptKAN: moving histograms of layer inputs automatically update grids during training.
Identifies the uniform grid as a "critical limitation" of the default KAN. Demonstrates improved
performance on 4 tasks.

📄 **Anonymous (2026) — "A Dynamic Framework for Grid Adaptation in KANs"**
arXiv:2601.18672.

Finds that even quantile-based (data-distribution) grids are suboptimal if the function is
locally linear in data-dense regions. Proposes curvature-based knot allocation.

### Recommendation
For symbolic regression specifically (not training), **sample using the actual distribution of
each feature in `X_eval`**: set `x_min = X_eval[feature].quantile(0.01)` and
`x_max = X_eval[feature].quantile(0.99)` and sample uniformly within that range (in encoded
space). This ensures the formula is fit where data actually lives and the R² reflects real
in-distribution behaviour.

### Empirical testing needed 🧪
- For continuous features: compare R² when sampling over `tanh([-3,3])` vs over the actual
  data range `[q1, q99]` of each feature in encoded space.
- Expected result: similar R² for smooth features; better R² for features where the network
  has learned complex structure only in the data-dense region.

---

## 7. Number of Samples for Edge Activation

### Current implementation
`n_samples = 1000` (`kan_symbolic.py:347`).

### What the literature says
No paper specifies an optimal sample count for KAN edge activation. The choice is an
engineering trade-off.

### Empirical testing needed 🧪
- For ChebyKAN (smooth polynomial edges): 200 samples likely sufficient for R² convergence.
  Measure `|R²(200) - R²(1000)|` across all active edges — if `< 0.001`, reduce to 200 for a 5×
  speedup on the symbolic fitting step.
- For FourierKAN (oscillatory edges): 1000 may be too few. Test 2000 and 5000 to see if the
  flag rate (R² < 0.90) decreases.

---

## 8. Symbolic Regression Candidate Library

### Current implementation
11 fixed scipy candidates: linear, quadratic, cubic, `|x|`, `√|x|`, `log`, `exp`, `sin(x)`,
`sin(2x)`, `cos(x)`, constant (`kan_symbolic.py:94–106`). Selection by best R².

### What the literature says

📄 **Cranmer (2023) — "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"**
arXiv:2305.01582.

PySR uses an open-ended multi-population evolutionary search (not a fixed library). It produces
a **Pareto front** of candidate expressions trading off accuracy vs. complexity. On SRBench
benchmarks, PySR alternates first/third place with Operon, both outperforming fixed-template
methods. Fixed libraries are competitive *only when the true functional form is known or strongly
anticipated*. PySR is 5–30× slower than Operon on high-dimensional datasets.

📄 **Liu et al. (2024)** arXiv:2404.19756

The `suggest_symbolic` function in pykan tries: `x, x², x³, x⁴, 1/x, 1/x², √x, 1/√x, exp,
log, sin, cos, abs, sign`. This is a fixed library of 14 candidates — richer than our current
11 but still bounded.

📄 **Smits & Kotanchek (2005) — "Pareto-Front Exploitation in Symbolic Regression"**
In: *Genetic Programming Theory and Practice II*, Springer, 2005.

Introduced ParetoGP — explicitly treating expression complexity and accuracy as two objectives.
Shows that requiring the user to pre-specify a penalty weight (as in AIC/BIC) is inferior to
presenting the full Pareto front. PySR inherits this approach.

### Recommendation
**For FourierKAN:** extend the fixed library with Fourier-specific candidates:
`a·sin(kx) + b·cos(kx)` for `k = 1, 2, 3, 4`. These match the natural output of the Fourier
basis and would likely recover the ~86% flagged-edge rate.

**For ChebyKAN:** the current 11 candidates already achieve mean R² = 0.9936. No change needed.

**For unknown functional forms:** use PySR selectively on flagged edges only (R² < 0.90 from
scipy), rather than running it on all edges. This saves time by using the cheap scipy pass first.

### Empirical testing needed 🧪
- Add Fourier candidates to the library and re-run on FourierKAN. Measure new mean R² and
  flag rate. Expected improvement: mean R² from 0.56 → >0.85.
- Test PySR on the 86% flagged FourierKAN edges with `maxsize=10` (simpler than current 15)
  and record if R² and formula complexity improve over scipy + Fourier candidates.

---

## 9. Initial Parameter Guesses for `curve_fit`

### Current implementation
Every candidate starts from `p0 = [1., 0., 0., ...]` (`kan_symbolic.py:95–105`).

### What the literature says
No KAN-specific paper addresses `p0` choice. This is a scipy numerical optimisation detail.

### Empirical testing needed 🧪
- `scipy.optimize.curve_fit` uses Levenberg-Marquardt (LM) internally, which is sensitive to
  starting points for non-linear candidates (exp, sin, cos).
- Test multiple random restarts (5×) for the non-linear candidates and take the best R². Cost
  is low for 1000-point curves.
- For the linear candidate, initialise `a ≈ (y_max - y_min) / (x_max - x_min)` — this is
  almost always better than `a=1` and converges in 1–2 iterations.

---

## 10. R² Flag Threshold

### Current implementation
`"flagged": r2 < 0.90` (`kan_symbolic.py:404`, `r2_pipeline.py:97`).

### What the literature says

📄 **Liu et al. (2024)** arXiv:2404.19756

The pykan `auto_symbolic()` function has `r2_threshold` (default `0.0`, i.e., disabled). In
tutorial examples, reliable symbolic identification is demonstrated at R² ≈ 0.9999. The paper
does not state a formal threshold for "trustworthy" symbolic fits.

📄 **Liu et al. (2024b)** arXiv:2408.10205

KAN 2.0 continues the same interactive paradigm. No published R² cutoff. Qualitative standard
in the KAN community (based on public tutorials and issue discussions) is R² > 0.99 before
locking in a formula.

### Recommendation
Consider **two tiers**:
- **Flagged (needs review):** R² < 0.90 — formula is unreliable, do not use for interpretation
- **Soft flag:** 0.90 ≤ R² < 0.99 — formula captures shape but with residuals; annotate with
  caution
- **Clean:** R² ≥ 0.99 — formula is reliable for interpretation

This matches the informal 0.99 standard from the KAN community while preserving the current
0.90 gate.

### Empirical testing needed 🧪
- Review the 2 edges currently below R²=0.95 in ChebyKAN: are they binary features (where
  high R² is trivially guaranteed) or genuine polynomial fits? If binary, exclude them from the
  aggregate metric computation.
- For FourierKAN: after adding Fourier candidates (#8 above), recount edges at each tier
  (< 0.90, 0.90–0.99, ≥ 0.99) to see if the 0.99 bar is achievable.

---

## 11. Formula Complexity / Model Selection

### Current implementation
Selection = highest R² among all candidates. No complexity penalty.

### What the literature says

📄 **Smits & Kotanchek (2005)** Springer GPTP-II.

Pareto-front approach (implemented in PySR) is superior to AIC/BIC because the accuracy–
complexity trade-off does not need to be specified in advance. Presenting the full Pareto
frontier lets the practitioner choose.

📄 **Cranmer (2023)** arXiv:2305.01582.

PySR uses the Pareto front by default, reporting all non-dominated (accuracy, complexity)
solutions. The final selection is made by the user or via a custom `score` function.

### Recommendation
When using the fixed scipy library (no Pareto front), apply **BIC as a tiebreaker** between
candidates with similar R²:

```
score = R² - (k · log(n)) / n
```

where `k` = number of free parameters, `n` = 1000 samples. This will prefer `a·x + b` (k=2)
over `a·x³ + b·x² + c·x + d` (k=4) when the R² gain from cubic over linear is small (< ~0.003
for n=1000). Makes the output more interpretable.

### Empirical testing needed 🧪
- Apply BIC scoring to the current ChebyKAN fits. Count how many edges change their selected
  formula (e.g., from cubic to linear) without R² dropping below 0.95. If > 20% of edges
  simplify, this is a worthwhile change.

---

## 12. Selective PySR on Flagged Edges

### Current implementation
`--use-pysr` is an all-or-nothing flag (`kan_symbolic.py:153`). Either scipy on all edges,
or PySR on all edges.

### What the literature says

📄 **Cranmer (2023)** arXiv:2305.01582.

PySR is 5–30× slower than Operon and orders of magnitude slower than scipy. Running it
on all edges is not cost-efficient if most edges already get R² > 0.99 from scipy.

### Recommendation
Implement **two-stage fitting**: run scipy first; apply PySR only to edges where scipy gives
R² < 0.90. This is the most natural use of the existing `flagged` column.

### Empirical testing needed 🧪
- Measure: for the 86% of flagged FourierKAN edges, how much does PySR improve R² vs scipy?
  Run on a random 10% sample of flagged edges (for speed) and extrapolate.
- Time budget: if PySR takes > 60 s/edge and there are 500 flagged edges, the total cost is
  > 8 hours — may not be feasible. In that case, stick to the extended fixed library (#8).

---

## 13. Aggregate R² Metrics

### Current implementation
Reports flat mean R², median R², count below 0.90 and 0.95 across all active edges
(`r2_pipeline.py:113–116`).

### What the literature says
No paper specifies how to aggregate symbolic fit quality. This is a reporting choice.

### Recommendation
The current mean R² is **inflated by binary features** (linear fits trivially achieve R²=1.0
for 2-point functions) and by **constant edges** (R²=1.0 by the `ss_tot < 1e-12` guard).

Report separately:
- Mean R² for **continuous and ordinal features only**
- Mean R² for **binary and missing-indicator features** (expect near 1.0 trivially)
- Mean R² for **deeper layers** (layer 1+) separately from layer 0

### Empirical testing needed 🧪
- Recompute aggregate R² excluding binary-typed features and constant-output edges. Check if
  the reported mean R²=0.9936 for ChebyKAN holds or drops — if it drops significantly, the
  current number is misleading.

---

## 14. Soft vs. Structural Pruning

### Current implementation
Soft pruning: zeroes coefficient tensors in-place. Architecture unchanged, parameter count
unchanged (`kan_pruning.py:108–116`).

### What the literature says

📄 **Hou et al. (2024) — "KANs: A Critical Assessment"**
arXiv:2407.11075.

KANs already have 1.36–100× higher FLOP cost than MLPs. Soft pruning does not reduce inference
cost — only structural pruning (removing neurons) does. For a project whose goal is
interpretability (not deployment efficiency), soft pruning is adequate.

📄 **Liu et al. (2024)** arXiv:2404.19756

Also uses soft pruning (node masking), but the pykan library offers `prune_node()` which
removes the node from the architecture after marking it inactive.

### Recommendation
For interpretability purposes only, soft pruning is fine. If inference speed matters in the
future, structural pruning (rebuilding the weight matrices) can be done after symbolic fits
are established.

### Empirical testing needed 🧪
- No immediate testing needed. Flag as future work if model is deployed in a scoring pipeline.

---

## 15. Feature Importance Metric (Ranking)

### Current implementation
Sum of first-layer edge output variances per input feature: `importance(xᵢ) = Σⱼ Var(φᵢⱼ)`
(`kan_symbolic.py:169–179`).

### What the literature says

📄 **Akazan & Mbingui (2025)** arXiv:2509.23366.

Proposes `KAN-L1` (L1 norm of activation functions) as the feature importance measure —
the same signal used for pruning. Shows this is interpretable as "how strongly the KAN
relies on each input" and works as a feature selector. L1 is more aligned with Liu et al.'s
pruning criterion than variance.

📄 **Liu et al. (2024)** arXiv:2404.19756

Uses `max_j ||φ_{0,j,i}||_1` (max over outgoing edges) as the importance of input `i` for
node pruning. Summing over all outgoing edges (`Σⱼ ||φ_{0,j,i}||_1`) is a reasonable extension.

### Recommendation
**Switch feature ranking from variance to L1 norm** of activation coefficients, consistent
with the pruning criterion. For ChebyKAN: `importance(xᵢ) = Σⱼ ||cheby_coeffs[j, i, :]||_1`.
This makes pruning and ranking use the same underlying signal.

### Empirical testing needed 🧪
- Compute both variance-based and L1-based rankings on ChebyKAN.
- Check rank correlation (Spearman) between the two methods — if > 0.95, the difference is
  negligible. If lower, compare which ranking better aligns with GLM coefficients and SHAP
  values.

---

## Summary

| # | Decision | Status | Change Recommended | Testing Needed |
|---|---|---|---|---|
| 1 | Pruning criterion | ⚠️ Suboptimal | Switch to L1 norm (Liu et al. 2024) | Compare sparsity & QWK vs variance |
| 2 | Training-time regularisation | ⚠️ Missing | Add L1+entropy loss term (Liu et al. 2024) | Sweep λ ∈ {1e-4, 1e-3, 1e-2} |
| 3 | Pruning threshold | ✅ Matches Liu et al. | Consider relative threshold | Sweep absolute & relative |
| 4 | QWK tolerance | ✅ Reasonable | Raise to 0.03 for interpretability focus | Sparsity–accuracy Pareto curve |
| 5 | Threshold sweep schedule | ⚠️ Coarse | Replace with binary search | Minor engineering |
| 6 | Sampling domain | ⚠️ Uniform only | Use data-distribution domain for SR | Compare R² uniform vs q1–q99 |
| 7 | n_samples | ✅ Fine for Cheby | Reduce to 200 for Cheby; increase for Fourier | Test R² convergence |
| 8 | Candidate library | ⚠️ Missing Fourier | Add Fourier harmonics k=1..4 | Re-run FourierKAN; expect R² >> 0.56 |
| 9 | p0 initialisation | ⚠️ Naive | Multi-restart for non-linear candidates | Test on non-linear edges |
| 10 | R² flag threshold | ⚠️ Too loose | Add 3-tier: < 0.90 / 0.90–0.99 / ≥ 0.99 | Recount by tier after library fix |
| 11 | Formula complexity penalty | ⚠️ Missing | Apply BIC as tiebreaker (Smits & Kotanchek 2005) | Count how many formulas simplify |
| 12 | Selective PySR | ⚠️ All-or-nothing | Two-stage: scipy first, PySR on flagged only | Cost–benefit on flagged edges |
| 13 | Aggregate R² reporting | ⚠️ Inflated | Report by feature type and layer | Recompute excluding binary/constant |
| 14 | Soft vs structural pruning | ✅ Adequate | No change for interpretability scope | — |
| 15 | Feature importance metric | ⚠️ Inconsistent | Switch to L1 norm (Akazan 2025) | Spearman rank vs variance-based |

---

## References

1. Liu, Z., Wang, Y., Vaidya, S. et al. **"KAN: Kolmogorov-Arnold Networks."** arXiv:2404.19756, ICLR 2025. https://arxiv.org/abs/2404.19756

2. Liu, Z., Ma, P., Wang, Y., Matusik, W., Tegmark, M. **"KAN 2.0: Kolmogorov-Arnold Networks Meet Science."** arXiv:2408.10205, 2024. https://arxiv.org/abs/2408.10205

3. Molchanov, P., Mallya, A., Tyree, S., Frosio, I., Kautz, J. **"Importance Estimation for Neural Network Pruning."** arXiv:1906.10771, CVPR 2019. https://arxiv.org/abs/1906.10771

4. Hayou, S., Ton, J-F., Doucet, A., Teh, Y.W. **"Robust Pruning at Initialization."** arXiv:2002.08797, ICLR 2021. https://arxiv.org/abs/2002.08797

5. Cheng, H., Zhang, M., Shi, J.Q. **"A Survey on Deep Neural Network Pruning: Taxonomy, Comparison, Analysis, and Recommendations."** arXiv:2308.06767, 2023. https://arxiv.org/abs/2308.06767

6. Cranmer, M. **"Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl."** arXiv:2305.01582, 2023. https://arxiv.org/abs/2305.01582

7. Poeta, E. et al. **"A Benchmarking Study of Kolmogorov-Arnold Networks on Tabular Data."** arXiv:2406.14529, 2024. https://arxiv.org/abs/2406.14529

8. Eslamian, A., Afzal Aghaei, A., Cheng, Q. **"TabKAN: Advancing Tabular Data Analysis using Kolmogorov-Arnold Network."** arXiv:2504.06559, 2025. https://arxiv.org/abs/2504.06559

9. Akazan, A-C., Mbingui, V.R. **"Splines-Based Feature Importance in Kolmogorov-Arnold Networks."** arXiv:2509.23366, 2025. https://arxiv.org/abs/2509.23366

10. Hou, Y., Ji, T., Zhang, D., Stefanidis, A. **"Kolmogorov-Arnold Networks: A Critical Assessment."** arXiv:2407.11075, 2024. https://arxiv.org/abs/2407.11075

11. Smits, G., Kotanchek, M. **"Pareto-Front Exploitation in Symbolic Regression."** In: *Genetic Programming Theory and Practice II*, Springer, 2005. https://link.springer.com/chapter/10.1007/0-387-23254-0_17

12. Anonymous. **"Automatic Grid Updates for Kolmogorov-Arnold Networks using Layer Histograms (AdaptKAN)."** arXiv:2511.08570, 2025. https://arxiv.org/abs/2511.08570

13. Anonymous. **"A Dynamic Framework for Grid Adaptation in Kolmogorov-Arnold Networks."** arXiv:2601.18672, 2026. https://arxiv.org/abs/2601.18672

14. Das, R.J., Ma, L., Shen, Z. **"Beyond Size: How Gradients Shape Pruning Decisions in Large Language Models."** arXiv:2311.04902, 2023. https://arxiv.org/abs/2311.04902

15. Xu, W. et al. **"FourierKAN-GCF: Fourier Kolmogorov-Arnold Network for Graph Collaborative Filtering."** arXiv:2406.01034, 2024. https://arxiv.org/abs/2406.01034
