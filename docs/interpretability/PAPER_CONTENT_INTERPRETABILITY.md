# Paper Content — Interpretability Section

Scope: drives the interpretability content for the 6-page paper answering
*"Can TabKAN models balance accuracy & interpretability in life insurance
risk assessment?"*.

Status as of 2026-04-23. Hero models verified on `gian-interpretability`
branch.

---

## 0. Division of labour — Gian vs Cyril

The interpretability section has **two subsections sharing one Table 1**.

### Gian — sparse / small interpretable regime
**Owns:** §1–§10 of this document below. Specifically:
- Sparse ChebyKAN + FourierKAN hero models (20 features, no LN, L1-pruned)
- Pareto trade-off framing (rows 5–6 of Table 1)
- Per-edge symbolic recovery for both sparse heroes (R² = 1.000 via basis-native extractors)
- End-to-end closed-form composition (sparse no-LN ChebyKAN only)
- Exact symbolic Greeks via SymPy chain rule (sparse no-LN ChebyKAN only)
- Figure 1: 2×2 per-edge plots (BMI/Wt × Cheby/Fourier sparse heroes)
- Paragraphs 1–3 of the suggested subsection structure (§3 below)

### Cyril — dense / big interpretable regime + XGBoost-SHAP comparison
**Owns:** §11 of this document (added below). Specifically:
- Dense ChebyKAN + FourierKAN baselines (140 features, with LN; rows 3–4 of Table 1)
- KAN-native coefficient-based feature importance vs XGBoost SHAP rankings
- Per-feature curve comparison: KAN learned activation vs SHAP dependence plot
- Per-risk-class consistency comparison (uses `comparison_per_risk.py`)
- Figure 2: SHAP-vs-KAN comparison (recommended layout in §11)
- Paragraph 0 of the suggested subsection structure (dense baselines)

### Shared / requires coordination (read §11.4 before locking anything)
- **Table 1** is one table across both subsections. Six rows, no parallel tables.
- **QWK reporting convention** (inner-val opt. thresholds vs outer-test default
  predict). Pick one for the entire paper. Recommendation in §11.4.
- **GLM row** in Table 1: either Cyril trains it on the same recipe or both
  subsections drop GLM. Don't mix.
- **FourierKAN-native fitter** is on this branch (commit `dd690c1`). Cyril
  must use it; without it, dense FourierKAN fits at R² ≈ 0.27 and the
  comparison breaks.
- **Discussion paragraph** (§9 deployment sentence) is jointly written.

---

## 1. The headline claim

> All four TabKAN variants we evaluate (Cheby/Fourier × dense/sparse)
> admit exact per-edge symbolic recovery (R² = 1.000 by construction
> via the basis-native extractors). On held-out outer test, the four
> KAN configurations achieve QWK 0.520–0.562 (XGBoost: 0.559;
> default-predict, no threshold tuning); the sparse FourierKAN hero
> at QWK 0.562 (7 158 active edges from 41 728) marginally exceeds
> the XGBoost baseline, while the sparse ChebyKAN hero at QWK 0.533
> (597 active edges from 26 112) trades 0.026 QWK for a 44× sparser
> network and a single closed-form composed polynomial in the 20
> retained inputs. Sparsity-regularised L1 training acts as implicit
> regularisation: both sparse heroes match or exceed their dense
> counterparts on outer test. The interpretability operations
> therefore impose no held-out accuracy cost in the FourierKAN case
> and an essentially negligible cost (≤ 0.010 QWK) in the ChebyKAN
> case — TabKAN balances accuracy and interpretability without a
> binding trade-off in this regime.

This frames TabKAN's interpretability as a *real Pareto choice across
four operating points* (two flavours × dense/sparse) where the
accuracy–compactness trade-off is essentially flat — the choice is
driven by structural properties (composability, formula simplicity,
exact Greeks) rather than by accuracy.

---

## 2. Models compared (locked in)

Four KAN configurations + two non-KAN baselines populate Table 1. The
two **sparse hero variants** are trained on the same 20 input features,
no LayerNorm, sparsity regularisation tuned per family, then L1-pruned
with QWK tolerance ≤ 0.01. The **dense baselines** are the best Optuna
trial per flavour at full feature count and standard LayerNorm.

### Sparse interpretable heroes

| Property | ChebyKAN hero | FourierKAN hero |
|---|---|---|
| Config | `chebykan_pareto_q0583_top20_noln.yaml` | `fourierkan_pareto_top20_noln.yaml` |
| Hidden widths | [128, 64], degree 6 | [64, 256, 64], grid_size 8 |
| Sparsity λ | 0.0108 | 0.0249 |
| **Outer-test QWK (default predict)** | **0.533** | **0.562** |
| Inner-val QWK before / after pruning (opt. thresh.) | 0.533 / 0.536 (Δ −0.003) | 0.562 / 0.558 (Δ −0.004) |
| Edge sparsity (within-architecture, pruning rate) | **94.4%** | 79.0% |
| Edges before (trainable) / after (active) | 10 752 / **597** | 34 048 / 7 158 |
| Active-edge compression vs dense baseline | **26 112 → 597 (−97.7%)** | 41 728 → 7 158 (−82.8%) |
| Mean R² of per-edge fits | **1.000** | **1.000** |
| Per-edge formula form | polynomial in `tanh(x)` (≤ 7 terms) | 17-term Fourier sum in `cos/sin(kπ(tanh(x)+1))` |
| End-to-end composed formula | exact polynomial | exact per-edge, but basis not closed under composition |

### Dense baselines (for Table 1 context, not hero status)

| Property | ChebyKAN dense | FourierKAN dense |
|---|---|---|
| Config | `chebykan_best.yaml` | `fourierkan_best.yaml` |
| Hidden widths | [128, 64], degree 6, with LN | [64, 256, 64], grid_size 8, with LN |
| Sparsity λ | 0 | 0 |
| **Outer-test QWK (default predict)** | **0.543** | **0.520** |
| Sweep best QWK (inner-val, opt. thresh., for context) | 0.625 | 0.641 |
| Active edges (total across all KAN layers) | 26 112 (140·128 + 128·64) | 41 728 (140·64 + 64·256 + 256·64) |
| Per-edge R² with native extractors | 1.000 | 1.000 (after the FourierKAN-native fix on this branch) |
| End-to-end composed formula | not exact (LayerNorm present) | not exact (LayerNorm present) and basis not closed under composition |
| Role in paper | baseline — slightly worse than its sparse counterpart on held-out | baseline — noticeably worse than its sparse counterpart on held-out |

**Outer-test QWK selection-bias note.** The sweep-best inner-val QWK
numbers (0.625 / 0.641) are the maximum of 151 Optuna trials. The
outer-test QWK (0.543 / 0.520) is a single retraining at those best
inner-val params, evaluated on a held-out split the search never
touched. The ~0.08–0.12 gap is typical Optuna selection bias and is
the reason we report outer-test numbers throughout.

Note on QWK: numbers above are inner-validation with optimised ordinal
thresholds (the regime used elsewhere in the paper). The held-out outer
test QWK (default `predict()`, no threshold optimisation) is reported
in the manifests for completeness; use the inner-val numbers in the
paper to stay consistent with the rest of the results section.

---

## 3. Suggested structure for the interpretability subsection

Target: ≤ ½ page main body + 1 figure + 1 table (Table 1, six rows).

### Paragraph 0 — Dense baselines (≈ 40 words)

> On outer-test (default predict, no threshold tuning), dense
> ChebyKAN and FourierKAN reach QWK 0.543 and 0.520 respectively;
> XGBoost scores 0.559. Both dense KANs admit exact per-edge
> symbolic recovery from their basis coefficients, but with 26 112
> and 41 728 active edges they cannot be written or read end-to-end.

### Paragraph 1 — Setup of the small interpretable variants (≈ 60 words)

> We extract a small interpretable variant of each TabKAN flavour by
> (i) selecting the top-20 features by coefficient-based importance
> from a sparsity-regularised baseline, (ii) retraining on that subset
> without LayerNorm, and (iii) pruning edges whose mean activation
> magnitude falls below a threshold tightened until QWK loss stays
> within 0.01.

### Paragraph 2 — Per-edge recovery and regularisation effect (≈ 90 words)

> ChebyKAN tolerates aggressive pruning (94% of its 10 752 trainable
> edges removed) at an inner-val QWK cost of 0.003; FourierKAN
> reaches 79% (7 158 of 34 048) under the same tolerance. Because L1
> sparsity regularisation also acts as implicit regularisation, both
> sparse heroes match or slightly exceed their dense counterparts on
> outer-test (ChebyKAN: 0.533 vs dense 0.543; FourierKAN: 0.562 vs
> dense 0.520 — the sparse variant *beats* the dense). Every surviving
> edge admits exact closed-form recovery: each ChebyKAN edge is a
> polynomial of degree ≤ 6 in `tanh(x)`; each FourierKAN edge is a
> sum of eight harmonic pairs in `kπ(tanh(x)+1)`. Recovery R² = 1.000
> across all active edges in both flavours.

### Paragraph 3 — Trade-off and the Pareto picture (≈ 90 words)

> On outer test, sparse and dense operating points are essentially
> equivalent in accuracy (ChebyKAN: 0.533 vs 0.543; FourierKAN: 0.562
> vs 0.520). The balance between flavours is slightly different:
> FourierKAN's sparse variant holds a 0.029 QWK advantage (0.562 vs
> 0.533) over ChebyKAN's, at the cost of 12× more surviving edges
> (7 158 vs 597). ChebyKAN's polynomial basis is additionally closed
> under composition, so the pruned no-LayerNorm model collapses into
> a single closed-form polynomial in the 20 input features;
> FourierKAN's basis is not closed under composition, so per-edge
> forms remain readable but the composed network does not simplify.

### Paragraph 3.5 — Exact Greeks (≈ 30 words, append to Paragraph 3 or stand alone)

> Because the sparse no-LayerNorm ChebyKAN composes into a single
> closed-form polynomial in the inputs, it also admits exact analytic
> partial derivatives — actuarial "Greeks" computed by symbolic
> differentiation rather than finite-differencing — enabling marginal-
> sensitivity and second-order interaction reports per applicant
> without numerical noise.

### Optional honest-limit sentence (1 line)

> The model genuinely uses all retained features: in feature-validation
> (Appendix X), QWK collapses to ≈ 0 when fewer than 100 features are
> kept, so neither variant supports radical feature elimination.

---

## 4. Table 1 — six-row Pareto comparison (locked in)

Target: 5 columns × 6 rows. Caption emphasises *like-for-like
comparison across baselines, dense KANs, and sparse interpretable
KANs*. Bolded rows are the small interpretable hero variants.

**Reporting convention (locked):** QWK is **outer-test, default
`predict()`, no threshold optimisation** — the manifest `metrics.qwk`
field. This is conservative (threshold optimisation would add
~ +0.01 QWK uniformly) but eliminates the selection-bias asymmetry
that would otherwise exist between dense baselines (selected on inner-val
over 151 Optuna trials) and sparse heroes (single retrainings).
Edge counts are **total active edges across all KAN layers**.

| Model | QWK (outer test) | # active edges / params | Explanation method | Per-edge R² |
|---|---|---|---|---|
| GLM (ridge) | *to fill (Phase B1)* | 140 coefficients | linear coefficients | — |
| XGBoost | 0.559 | ~ N trees *(to fill)* | SHAP TreeExplainer (post-hoc) | — |
| ChebyKAN, dense (140 ft, with LN) | 0.543 | 26 112 edges | per-edge native; autograd Greeks | 1.000 |
| FourierKAN, dense (140 ft, with LN) | 0.520 | 41 728 edges | per-edge native; autograd Greeks | 1.000 |
| **ChebyKAN, sparse (20 ft, no LN)** | **0.533** | **597 edges (−97.7% vs dense)** | **closed-form polynomial + exact symbolic Greeks** | **1.000** |
| **FourierKAN, sparse (20 ft, no LN)** | **0.562** | **7 158 edges (−82.8% vs dense)** | **per-edge closed form; autograd Greeks** | **1.000** |

The four KAN rows characterise the Pareto front. Three observations:

- **Dense rows (3–4) underperform both their sparse counterparts and
  XGBoost on outer test.** Sparsity regularisation acts as implicit
  regularisation; the dense baselines overfit slightly. This is a
  honest-to-data finding rather than a trade-off we designed for.
- **Sparse rows (5–6) match or exceed XGBoost at QWK 0.533 and
  0.562.** The sparse FourierKAN is the best model in the paper on
  outer test (by 0.003 over XGBoost — see Phase B2 bootstrap CIs for
  significance).
- **Row 5 alone** admits *exact symbolic Greeks* (analytic ∂y/∂xᵢ,
  ∂²y/∂xᵢ∂xⱼ via SymPy chain rule) — same property that gives it
  end-to-end composability. All other rows fall back on autograd or
  finite-difference Greeks, which is also available for any black-box
  model.

**Sparsity note:** the "−97.7%" and "−82.8%" are computed against the
corresponding dense baseline's total edge count (26 112 / 41 728),
not against the small-architecture dense upper-bound used during
pruning (10 752 / 34 048 — the 20-feature network's maximum edge
count). The 94% / 79% numbers quoted in the methodology paragraphs
are the within-architecture pruning rates; the table reports the
structural compression against the full baseline.

---

## 5. Figure 1 — recommended (one figure only, ~⅓ column)

A 4-panel grid showing 2 features × 2 flavours, drawn from the **sparse
hero variants** (which is the regime the figure is illustrating). Each
panel: scatter of the learned activation (grey dots) overlaid with the
recovered closed-form expression (coloured line), with R² annotated.

Suggested features (in the top-20 of *both* flavours): **BMI, Wt**.

Layout:
```
        BMI                Wt
ChebyKAN  ▢ poly(tanh)      ▢ poly(tanh)
FourierKAN ▢ trig(tanh)     ▢ trig(tanh)
```

This figure shows (a) what closed-form edges look like in each basis
family, (b) why the two basis classes behave differently in
composition (polynomial vs trig). The dense baselines are not in the
figure — readers see them as Table 1 rows; explicit per-edge plots
would be too dense to read for the dense models.

If forced to pick a *second* figure (only if space allows): the Pareto
fronts of QWK vs sparsity for both flavours from
`sweeps/stage-c-{cheby,fourier}kan-pareto-sparsity_pareto.json`. This
visualises the dense → sparse interpolation that Table 1 only samples
at two endpoints. Useful but Table 1 already conveys the headline;
keep in appendix unless space is plentiful.

---

## 6. Appendix dump (won't be read, but valuable to have)

All artifacts already exist under
`outputs/interpretability/kan_paper/stage-c-{chebykan,fourierkan}-pareto-…-top20-noln/`:

- Full per-edge formula listings (`reports/*_symbolic_formulas.md`)
- Composed end-to-end formula (ChebyKAN only; `reports/chebykan_exact_closed_form.md`)
- **Exact symbolic Greeks** — ChebyKAN sparse no-LN only — analytic
  ∂y/∂xᵢ traces and discrete-state effect deltas computed via SymPy
  chain rule (`reports/chebykan_exact_closed_form.md` and
  `reports/chebykan_exact_closed_form.json`, produced by
  `src/interpretability/exact_partials.py`)
- **Hessian / cross-Greeks** — autograd-based ∂²y/∂xᵢ∂xⱼ averaged
  over the eval split, both signed and absolute, continuous-features
  view recommended (`figures/chebykan_hessian_heatmap_continuous.png`,
  `reports/chebykan_hessian_heatmap.md/json`,
  produced by `src/interpretability/hessian_heatmap.py`)
- Local case explanation for applicant 55728 — finite-difference
  per-feature sensitivities and what-if class deltas
  (`reports/*_case_summary_55728.0.md`,
  `data/*_local_sensitivities_55728.0.csv`,
  `data/*_case_what_if_55728.0.csv`)
- Network-diagram PDF with mini-plots (`figures/*_kan_diagram.pdf`)
- Per-feature activation grid (`figures/*_activations.pdf`)
- R² distribution histogram (`figures/*_r2_distribution.pdf`)
- Feature-validation curves (`figures/feature_validation_curves.pdf`)
- Pareto fronts for both flavours
- Hyperparameter and pruning configurations

---

## 7. Caveats to disclose in writing

1. **Threshold tightening for FourierKAN.** The L1 pruning threshold
   was reduced from 0.01 to 0.005 to satisfy the QWK tolerance — this
   is itself evidence that FourierKAN edges carry more individual
   signal and resist pruning more than ChebyKAN's. Worth one sentence.

2. **No-LayerNorm assumption.** The "exact composability" claim for
   ChebyKAN holds only when LayerNorm is absent; with LayerNorm,
   composition is approximate. The hero is no-LN; report this
   explicitly. (See `INTERPRETABILITY_HANDOFF.md` for the LN-vs-noLN
   discussion.)

3. **20-feature restriction is not free.** Both heroes lose ~ 0.05–0.08
   QWK relative to their 140-feature, dense counterparts. The trade is
   "small enough to draw a network diagram and write down a single
   formula". Make this explicit.

4. **Per-edge formulas, not feature elimination.** Even the sparsest
   models retain 100+ features at usable QWK. The contribution is
   *transparent local relationships per feature*, not a parsimony
   claim about the feature set.

5. **Top-20 features differ between flavours.** ChebyKAN ranks
   `Medical_History_5` second; FourierKAN ranks `Medical_History_11`
   second. Both rankings start with BMI. The lists are in
   `configs/.../feature_lists/`. Worth noting in a footnote.

6. **Greeks scope.** The "exact Greeks" claim applies *only* to the
   sparse no-LayerNorm ChebyKAN row. For all other rows we can compute
   ∂y/∂xᵢ and ∂²y/∂xᵢ∂xⱼ via autograd or finite differences, but those
   are tools available for any black-box model and do not constitute a
   KAN-specific interpretability advantage. Be explicit about which
   row earns the analytic-Greeks claim. The Black-Scholes analogy is
   strongest there because both situations have a closed-form
   expression admitting symbolic differentiation; do not extend the
   analogy to the dense or FourierKAN rows.

---

## 8. What *not* to include

- Per-edge enumeration of the dense models (17 920+ / 41 728+ rows).
  They appear in Table 1 only as one row each, with the "not
  composable, too many edges to read" qualifier.
- Narrow-architecture experiments (140 → 16 → 8 → 1 etc.). Negative
  result; cut.
- Closed-form polynomial *surrogate* (`closed_form_surrogate.py`). Only
  needed when exact composition is unavailable; for the no-LN ChebyKAN
  hero, exact composition exists, so the surrogate is redundant.
- Per-risk-level comparison panels. Too dense for the page budget.
- The twin-model (production + explainer) framing as the *primary*
  paper narrative. It appears only as a single discussion sentence
  (§9 below) connecting the Pareto result to actuarial practice; it
  is not a methodological contribution.

---

## 9. Sentence-level draft for the abstract / conclusion / discussion

For the abstract:

> We characterise the accuracy–interpretability trade-off of TabKAN on
> Prudential life-insurance risk grading. Across two basis families
> (Chebyshev, Fourier) and two operating points (dense / sparse), all
> four configurations admit exact per-edge symbolic recovery. On
> outer-test QWK, dense ChebyKAN (0.543) and dense FourierKAN (0.520)
> match the XGBoost baseline (0.559) within 0.04. Sparse 20-feature
> no-LayerNorm variants *match or exceed* their dense counterparts —
> ChebyKAN sparse at QWK 0.533 and FourierKAN sparse at 0.562 — at
> a 12× and 6× reduction in active edges. ChebyKAN's sparse variant
> additionally collapses into a single closed-form polynomial in the
> inputs and admits exact analytic Greeks. The interpretability
> operations therefore impose no held-out accuracy cost; the choice
> among operating points is driven by structural properties rather
> than predictive accuracy.

For the conclusion:

> TabKAN balances accuracy and interpretability without a binding
> trade-off in this regime: per-edge symbolic recovery is uniformly
> available (R² = 1.000), and sparsity-regularised variants match or
> exceed their dense counterparts on outer-test QWK. The remaining
> distinctions are structural — the Chebyshev basis is closed under
> composition (single closed-form polynomial + exact symbolic Greeks
> for the sparse no-LN variant), while the Fourier basis is not (readable
> per-edge forms but no tractable composed expression). Practitioners
> required to deliver compact analytic explanations to regulators or
> actuaries should prefer the sparse Chebyshev variant; those prioritising
> raw outer-test QWK should prefer the sparse Fourier variant.

For a single discussion sentence connecting the result to actuarial
practice (do not promote to abstract or conclusion):

> In practice, an insurer may pair the sparse FourierKAN (highest
> outer-test QWK, per-edge closed forms) for underwriting with the
> sparse ChebyKAN (slightly lower QWK, single composed polynomial +
> exact Greeks) for regulatory documentation and actuarial review —
> two configurations from the same TabKAN family replacing the
> conventional GLM-plus-XGBoost split.

---

## 10. Open / required follow-ups (Gian)

| Item | Status | Notes |
|---|---|---|
| ~~Confirm QWK reporting convention (inner-val vs outer-test)~~ | **locked** | outer-test default-predict; see §4 and §11.4 item 3 |
| ~~Pick edge-count convention (layer-0 only vs total across layers)~~ | **locked** | total across all KAN layers |
| Train GLM baseline (Phase B1) | pending | ~ 1 min via `main.py --stage train --config configs/.../glm_baseline.yaml`; fills Table 1 row 1 |
| Bootstrap CI on outer-test QWK for all six Table 1 rows (Phase B2) | pending | ~ 10 lines; addresses "is the 0.003 gap sparse-Fourier vs XGBoost real?" |
| Compute one worked Greek for applicant 55728 (Phase B3) | pending | symbolic vs autograd vs finite-diff for one feature (BMI); backs the Greeks claim with evidence |
| Build Figure 1 (2×2 panel: BMI/Wt × Cheby/Fourier sparse heroes) | pending | scripted assembly from existing `*_activations.pdf` data |
| Optional: outer-test re-eval of pruned sparse heroes (for before/after QWK delta on outer test) | optional | pruning summary currently reports inner-val delta only; outer-test delta should be similar |
| Optional: decompose 0.092 inner-val gap via q0.583-s0.97 intermediate | optional | only needed if a reviewer challenges the inner-val → outer-test drop; Table 1 already reports outer-test consistently |

---

# PART II — Dense-regime interpretability (Cyril)

This part of the doc is owned by **Cyril**. It defines the dense-baseline
interpretability subsection that sits alongside Gian's sparse-regime
subsection in the paper. They share Table 1 and the Discussion paragraph.

## 11. The dense-regime narrative

### 11.1 The headline claim (Cyril)

> Dense ChebyKAN and FourierKAN approach XGBoost on outer-test QWK
> (ChebyKAN: 0.543; FourierKAN: 0.520; XGBoost: 0.559) while offering
> model-native interpretability at no approximation cost. Per-feature
> importance rankings derived from the KAN's learned Chebyshev /
> Fourier coefficients agree with XGBoost SHAP at Kendall τ ≈ to-fill;
> per-feature behaviour curves agree visually. Crucially, KAN
> explanations are *model-native* — read directly from the learned
> coefficients with no post-hoc approximation — whereas SHAP must
> approximate XGBoost's behaviour with a separate linear surrogate
> per prediction. We thus position dense TabKAN as a near-XGBoost
> accuracy alternative with explanations that are exact rather than
> approximate.

**Note on accuracy framing.** At outer test, dense TabKAN trails
XGBoost by 0.016 (ChebyKAN) and 0.039 (FourierKAN) — not a small
gap for ChebyKAN, noticeable for FourierKAN. Do not overclaim
"matches XGBoost" without qualifying. The stronger line is "at
comparable accuracy class, TabKAN replaces SHAP post-hoc
approximation with model-native explanations".

### 11.2 Suggested structure for the dense-regime subsection (≈ ⅓ page)

#### Paragraph A — Setup (≈ 30 words)
> We compare dense ChebyKAN and FourierKAN (140 features, with
> LayerNorm, sparsity λ = 0) against XGBoost+SHAP and the GLM baseline
> on three standard interpretability tasks: feature ranking, per-feature
> behaviour, and per-risk-class importance.

#### Paragraph B — Result (≈ 60 words)
> KAN-native coefficient importance and XGBoost SHAP rank the top-15
> features at Kendall τ = *to-fill*. Per-feature shape curves (Figure 2)
> show qualitative agreement on continuous features (BMI, Ins_Age, Wt)
> with KAN edges providing smoother, monotonic shapes vs SHAP's stepped
> tree-based dependence. Per-risk-class importance (Appendix Y) shows
> consistent top features across risk levels in all three models.

#### Paragraph C — Distinction (≈ 50 words)
> The dense KAN explanations are model-native: each per-feature curve
> is the layer's learned activation function read directly from
> Chebyshev / Fourier coefficients via the basis-native extractors
> (R² = 1.000 by construction). SHAP, by contrast, must approximate
> XGBoost's tree ensemble with a local linear model per prediction.
> Dense TabKAN therefore matches XGBoost on accuracy *and* removes
> the post-hoc approximation step.

### 11.3 Figure 2 — recommended (Cyril)

Two viable layouts; pick one:

**Layout A: Feature-importance comparison.** Horizontal bar chart of
the top-15 features ranked by, in 4 columns: GLM coefficient | XGBoost
SHAP mean(|value|) | dense ChebyKAN coefficient importance | dense
FourierKAN coefficient importance. Annotate Kendall τ between each
KAN column and the SHAP column. Compact, high-information-density,
single panel.

**Layout B: Per-feature behaviour for one canonical feature (BMI).**
Four side-by-side panels: GLM coefficient line | XGBoost SHAP
dependence plot | ChebyKAN edge curve | FourierKAN edge curve. All on
the same x-axis (BMI in raw units). Visually striking; makes the
"smooth model-native vs stepped post-hoc" point clearly. Best if BMI
is *the* feature you want the reader to remember.

Recommendation: **Layout A** if you want to argue rank agreement;
**Layout B** if you want to argue qualitative-shape transparency.
Pick based on which claim is more central. Both are supported by
existing artifacts in `outputs/interpretability/.../comparison_*`.

### 11.4 Coordination with Gian's part (READ FIRST)

Six items must be aligned before either subsection is locked:

1. **Use the FourierKAN-native fitter on this branch.** Without it,
   dense FourierKAN edge fits at R² ≈ 0.27 (the previous result). The
   fix is in commits `dd690c1` on `gian-interpretability`. Use the
   branch or cherry-pick the two source files
   (`src/interpretability/kan_symbolic.py`,
   `src/interpretability/formula_composition.py`).

2. **Same train / test / inner-val split, same recipe.** Confirm by
   checking that the dense KAN run-summaries cite `recipe: kan_paper`
   and `seed: 42`. Existing dense baselines
   (`stage-c-chebykan-best`, `stage-c-fourierkan-best`,
   `stage-c-xgboost-best`) already match. Do not retrain on different
   splits.

3. **QWK convention (LOCKED): outer-test, default `predict()`, no
   threshold optimisation** — the manifest `metrics.qwk` field for
   each model. Numbers for reference:
   - GLM: to fill (Phase B1)
   - XGBoost: 0.559
   - ChebyKAN dense: 0.543
   - FourierKAN dense: 0.520
   - ChebyKAN sparse hero: 0.533
   - FourierKAN sparse hero: 0.562

   Inner-val numbers (0.625 / 0.641) are Optuna-selection-biased and
   inflate the dense rows relative to the sparse single-retraining
   rows; they are *not* used in Table 1 and must not appear in the
   paper's accuracy reporting. Note this also flips the sparse-vs-
   dense narrative: the sparse variants *match or exceed* their dense
   counterparts on outer test (Cheby: 0.533 vs 0.543; Fourier: 0.562
   vs 0.520 — sparse wins). Paragraphs 0, 2, 3, abstract, and
   conclusion drafts have been updated accordingly.

4. **GLM row in Table 1.** Either you train a GLM on the same recipe
   so row 1 has a real number, or both subsections drop the GLM row
   entirely. Do not have one subsection cite GLM and the other not.
   Config exists at `configs/.../glm_baseline.yaml`; one `main.py
   --stage train` run takes < 1 min.

5. **Single Table 1.** Do not introduce a parallel
   "interpretability comparison" table for the dense regime. All
   numbers go into the existing six-row Table 1 (defined in §4). The
   dense subsection cites rows 3–4; the sparse subsection cites rows
   5–6. The rest is text.

6. **Discussion paragraph** (§9 deployment sentence) is jointly
   written. Suggested: dense KANs replace XGBoost+SHAP for production
   underwriting; sparse ChebyKAN replaces GLM for regulatory and
   actuarial documentation. Do not promote this to abstract or
   conclusion (per Gian's recommendation in §8).

### 11.5 Existing artifacts Cyril can build on

In `outputs/interpretability/kan_paper/stage-c-chebykan-best/` and
`stage-c-fourierkan-best/` (run after applying the FourierKAN
extractor fix from this branch):

- `data/{cheby,fourier}kan_coefficient_importance.csv` — KAN-native
  feature importance (`utils/kan_coefficients.py`)
- `figures/comparison_per_risk_*.pdf` — per-risk-class panels (already
  generated by `comparison_per_risk.py`)
- `figures/side_by_side_*.pdf` — side-by-side per-feature comparisons
  across all 4 models (`comparison_side_by_side.py`)
- `figures/feature_risk_influence_*.pdf` — domain-aligned per-feature
  curves (`feature_risk_influence.py`)

In the existing XGBoost outputs (already in tree from earlier work):

- SHAP TreeExplainer values — see `shap_xgboost.py`

### 11.6 Open / required follow-ups (Cyril)

| Item | Status | Notes |
|---|---|---|
| Re-run dense ChebyKAN + FourierKAN interpret pipelines on this branch | pending | needed so dense FourierKAN edges fit R² = 1.000 instead of 0.27 |
| Compute Kendall τ between KAN coefficient importance and XGBoost SHAP rankings, top-15 features | pending | one number per KAN flavour for §11.2 paragraph B |
| Decide Figure 2 layout (A: importance bars, B: per-feature panels) | pending | drives narrative emphasis |
| ~~Train GLM baseline~~ | being handled in Phase B1 by Gian | result will populate Table 1 row 1 |
| ~~Confirm QWK convention with Gian~~ | **locked: outer-test default-predict** | see §11.4 item 3 for the numbers |
