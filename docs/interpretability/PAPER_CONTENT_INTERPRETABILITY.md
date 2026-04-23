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
> via the basis-native extractors). The accuracy–interpretability
> trade-off is therefore not about *whether* edges are recoverable but
> about *how many edges remain* and whether the composed model can be
> written as a single expression. Dense ChebyKAN and FourierKAN reach
> QWK 0.625 and 0.641 (within 0.03 of the XGBoost baseline at 0.655)
> but produce networks of 17 920+ and 41 728+ active edges. The
> sparse 20-feature, no-LayerNorm heroes trade ~0.08–0.09 QWK for
> 12–30× fewer edges; ChebyKAN's polynomial basis additionally
> collapses the entire pruned model into a single closed-form
> polynomial in the 20 inputs. The Pareto trade-off is thus
> well-characterised — TabKAN balances accuracy and interpretability,
> with the choice of operating point (and basis) driven by deployment
> needs rather than by any single dominating configuration.

This frames TabKAN's interpretability as a *real Pareto choice across
four operating points* (two flavours × dense/sparse), which is the
strongest available answer to the research question.

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
| Inner-val QWK before pruning (opt. thresh.) | 0.533 | **0.562** |
| Inner-val QWK after pruning | 0.536 (Δ −0.003) | 0.558 (Δ −0.004) |
| Edge sparsity | **94.4%** | 79.0% |
| Edges before / after | 10 752 / **597** | 34 048 / 7 158 |
| Mean R² of per-edge fits | **1.000** | **1.000** |
| Per-edge formula form | polynomial in `tanh(x)` (≤ 7 terms) | 17-term Fourier sum in `cos/sin(kπ(tanh(x)+1))` |
| End-to-end composed formula | exact polynomial | exact, but expansion of `sin/cos ∘ tanh ∘ sin/cos` is heavy |

### Dense baselines (for Table 1 context, not hero status)

| Property | ChebyKAN dense | FourierKAN dense |
|---|---|---|
| Config | `chebykan_best.yaml` | `fourierkan_best.yaml` |
| Hidden widths | [128, 64], degree 6, with LN | [64, 256, 64], grid_size 8, with LN |
| Sparsity λ | 0 | 0 |
| Sweep best QWK (inner-val, opt. thresh.) | 0.625 | 0.641 |
| Active edges (across all KAN layers) | 17 920+ (layer-0) / 26 112 (total) | 41 728+ (sum across layers) |
| Per-edge R² with native extractors | 1.000 | 1.000 (after the FourierKAN-native fix on this branch) |
| End-to-end composed formula | not exact (LayerNorm present) | not exact (LayerNorm present) and basis not closed under composition |
| Headline role | strong-accuracy KAN baseline | strongest KAN baseline (within 0.014 of XGBoost) |

Note on QWK: numbers above are inner-validation with optimised ordinal
thresholds (the regime used elsewhere in the paper). The held-out outer
test QWK (default `predict()`, no threshold optimisation) is reported
in the manifests for completeness; use the inner-val numbers in the
paper to stay consistent with the rest of the results section.

---

## 3. Suggested structure for the interpretability subsection

Target: ≤ ½ page main body + 1 figure + 1 table (Table 1, six rows).

### Paragraph 0 — Dense baselines (≈ 40 words)

> Dense ChebyKAN and FourierKAN reach QWK 0.625 and 0.641 respectively
> — within 0.03 of the XGBoost baseline. Both admit exact per-edge
> symbolic recovery from their basis coefficients; however, with
> 17 920+ and 41 728+ active edges they cannot be written or read
> end-to-end.

### Paragraph 1 — Setup of the small interpretable variants (≈ 60 words)

> We extract a small interpretable variant of each TabKAN flavour by
> (i) selecting the top-20 features by coefficient-based importance
> from a sparsity-regularised baseline, (ii) retraining on that subset
> without LayerNorm, and (iii) pruning edges whose mean activation
> magnitude falls below a threshold tightened until QWK loss stays
> within 0.01.

### Paragraph 2 — Per-edge recovery (≈ 80 words)

> ChebyKAN tolerates aggressive pruning (94% of edges removed) at a
> cost of 0.003 QWK; FourierKAN reaches a maximum sparsity of 79% under
> the same tolerance. Every surviving edge in either model admits an
> exact closed-form recovery: each ChebyKAN edge is a polynomial of
> degree ≤ 6 in `tanh(x)`, and each FourierKAN edge is a sum of eight
> harmonic pairs in `kπ(tanh(x)+1)`. Mean recovery R² = 1.000 across
> all 597 / 7 158 active edges respectively.

### Paragraph 3 — Trade-off and the Pareto picture (≈ 90 words)

> The four TabKAN configurations occupy distinct points on the
> accuracy–compactness frontier (Table 1). Going from dense to sparse
> costs ~0.08–0.09 QWK in either flavour; within the sparse end,
> FourierKAN keeps a 0.029 QWK advantage at the cost of 12× more
> surviving edges. ChebyKAN's polynomial basis is additionally closed
> under composition, so the pruned no-LayerNorm model collapses into a
> single closed-form polynomial in the 20 input features; FourierKAN's
> basis does not have this property, so per-edge forms remain readable
> but the composed network does not simplify.

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

## 4. Table 1 — six-row Pareto comparison (lock this in)

Target: 5 columns × 6 rows. Caption emphasises *like-for-like
comparison across baselines, dense KANs, and sparse interpretable
KANs*. Bolded rows are the small interpretable hero variants.

| Model | QWK | # active edges / params | Explanation method | Per-edge R² |
|---|---|---|---|---|
| GLM (ridge) | *to fill* | 140 coefficients | linear coefficients | — |
| XGBoost | 0.655 | ~ N trees | SHAP TreeExplainer (post-hoc) | — |
| ChebyKAN, dense (140 ft, with LN) | 0.625 | 17 920+ edges | per-edge native; autograd Greeks | 1.000 |
| FourierKAN, dense (140 ft, with LN) | 0.641 | 41 728+ edges | per-edge native; autograd Greeks | 1.000 |
| **ChebyKAN, sparse (20 ft, no LN)** | **0.533** | **597 edges (−97%)** | **closed-form polynomial + exact symbolic Greeks** | **1.000** |
| **FourierKAN, sparse (20 ft, no LN)** | **0.562** | **7 158 edges (−83%)** | **per-edge closed form; autograd Greeks** | **1.000** |

The four KAN rows are the Pareto front the paper characterises:
- **Dense rows (3–4)** show TabKAN is competitive with XGBoost and that
  per-edge symbolic recovery already works at full scale — but the
  network is too large to read.
- **Sparse rows (5–6)** show what interpretability actually costs:
  ~0.08–0.09 QWK in both flavours, in exchange for 12–30× fewer edges.
- The **gap between rows 5 and 6** (0.029 QWK, 12× edge ratio)
  characterises the basis-family choice within the interpretable end.
- **Row 5 alone** has the additional property of admitting *exact
  symbolic Greeks* (analytic ∂y/∂xᵢ, ∂²y/∂xᵢ∂xⱼ via SymPy chain rule)
  — the same property that gives it end-to-end composability also
  gives it actuarial-style sensitivities without finite-differencing.
  All other rows must rely on autograd or finite-difference Greeks,
  the same tools available for any black-box model.

**Open item**: the GLM row needs a real number. No GLM checkpoint
exists in `checkpoints/`; the only number anywhere is the stale 0.568
(validation, March) in `outputs/reports/final_comparison_matrix.md`.
Either retrain on the same split or drop the GLM row.

**Open item**: clarify per-paper convention whether QWK is inner-val
(opt. thresholds) or outer-test (default `predict()`). Numbers above
are inner-val. Outer-test is systematically lower (e.g., ChebyKAN-best
manifest reports 0.543 vs 0.625 sweep).

**Note on edge counts**: the dense edge counts are layer-0 only
(in × hidden_widths[0]). Total active edges across all KAN layers are
larger; if reporting full counts, recompute from the model architecture
(ChebyKAN dense: 140·128 + 128·64 = 26 112; FourierKAN dense:
140·64 + 64·256 + 256·64 = 41 984). Pick one convention and apply
uniformly.

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
> four configurations admit exact per-edge symbolic recovery. Dense
> ChebyKAN and FourierKAN reach QWK 0.625 and 0.641 respectively
> (XGBoost: 0.655); sparse 20-feature, no-LayerNorm variants give up
> ~0.08–0.09 QWK in exchange for 12–30× fewer active edges. ChebyKAN's
> sparse variant additionally collapses into a single closed-form
> polynomial in the inputs. The choice of operating point and basis is
> therefore application-driven.

For the conclusion:

> TabKAN can balance accuracy and interpretability: per-edge symbolic
> recovery is available across the entire Pareto front. The cost of
> compactness is a uniform ~0.08–0.09 QWK in either flavour; the
> additional cost of end-to-end composability (achievable only with the
> Chebyshev basis) is a further ~0.03 QWK. Practitioners with a hard
> accuracy floor should prefer the Fourier basis at the dense end;
> those required to deliver compact closed-form explanations to
> regulators or actuaries should prefer the sparse Chebyshev variant.

For a single discussion sentence connecting the result to actuarial
practice (do not promote to abstract or conclusion):

> In practice, an insurer may deploy the dense FourierKAN for
> accuracy-critical underwriting and use the sparse ChebyKAN as a
> transparent companion model for regulatory documentation and
> actuarial review — pairing two configurations from the same TabKAN
> family rather than the conventional GLM-plus-XGBoost split.

---

## 10. Open / required follow-ups (Gian)

| Item | Owner | Notes |
|---|---|---|
| Confirm QWK reporting convention (inner-val vs outer-test) | Gian + Cyril | affects all six numbers in Table 1 |
| Pick edge-count convention (layer-0 only vs total across layers) | Gian + Cyril | affects rows 3–6 |
| Build Figure 1 (2×2 panel: BMI/Wt × Cheby/Fourier sparse heroes) | Gian | scripted assembly from existing `*_activations.pdf` data |
| Compute one worked Greek for applicant 55728 (symbolic vs autograd vs finite-diff) | Gian | one-row example to back the "exact Greeks" claim with evidence |
| Decompose 0.092 QWK gap into "sparsity cost" + "feature/LN cost" via the q0.583-s0.97 intermediate | Gian | reviewer-defence against selection-bias critique |
| Bootstrap CI on QWK for the four KAN rows | Gian + Cyril | ~10 lines of code; addresses "is the gap real?" |

---

# PART II — Dense-regime interpretability (Cyril)

This part of the doc is owned by **Cyril**. It defines the dense-baseline
interpretability subsection that sits alongside Gian's sparse-regime
subsection in the paper. They share Table 1 and the Discussion paragraph.

## 11. The dense-regime narrative

### 11.1 The headline claim (Cyril)

> Dense ChebyKAN and FourierKAN match XGBoost on the standard
> interpretability tasks performed via SHAP. Per-feature importance
> rankings agree with XGBoost SHAP (Kendall τ ≈ to-fill); per-feature
> behaviour curves agree visually. Crucially, KAN explanations are
> *model-native* — read directly from the learned Chebyshev / Fourier
> coefficients with no post-hoc approximation — whereas SHAP must
> approximate XGBoost's behaviour with a separate linear surrogate
> per prediction. We thus position dense TabKAN as a drop-in
> replacement for XGBoost+SHAP in the standard insurer interpretability
> stack: same accuracy class, same kinds of explanations, no
> approximation gap.

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

3. **One QWK reporting convention across the paper.**
   - Inner-validation with optimised ordinal thresholds (the Optuna
     sweep number, e.g. ChebyKAN dense = 0.625).
   - Held-out outer test with default `predict()` (the manifest
     `metrics.qwk` field, e.g. ChebyKAN dense = 0.543).
   These differ by ~0.05–0.08. **Recommendation: outer test.**
   Reasoning: dense baselines were *selected* on inner-val (best of
   151 Optuna trials); the sparse heroes are single retrainings.
   Reporting inner-val for both creates an asymmetric selection-bias
   that systematically over-states the dense-row QWK. Outer test
   eliminates this asymmetry.

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

| Item | Owner | Notes |
|---|---|---|
| Re-run dense ChebyKAN + FourierKAN interpret pipelines on this branch | Cyril | needed so dense FourierKAN edges fit R² = 1.000 instead of 0.27 |
| Compute Kendall τ between KAN coefficient importance and XGBoost SHAP rankings, top-15 features | Cyril | one number per KAN flavour for §11.2 paragraph B |
| Decide Figure 2 layout (A: importance bars, B: per-feature panels) | Cyril | drives narrative emphasis |
| Train GLM baseline if Table 1 row 1 is to be filled | Cyril (or Gian) | ~ 1 min via existing config |
| Confirm QWK convention with Gian | Cyril + Gian | recommendation: outer test (see §11.4 item 3) |
