# Paper Content — Interpretability Section

Scope: drives the interpretability content for the 6-page paper answering
*"Can TabKAN models balance accuracy & interpretability in life insurance
risk assessment?"*. Targets ≤ ½ page main body + table + figure, with
appendix dump.

Status as of 2026-04-23. Hero models verified on `gian-interpretability`
branch.

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

## 10. Open / required follow-ups

| Item | Owner | Notes |
|---|---|---|
| Train GLM baseline on the same split | — | needed for Table 1 row 1; otherwise drop it |
| Confirm QWK reporting convention (inner-val vs outer-test) | — | affects all six numbers in Table 1 |
| Pick edge-count convention (layer-0 only vs total across layers) | — | affects rows 3–6 |
| Build Figure 1 (2×2 panel: BMI/Wt × Cheby/Fourier sparse heroes) | — | scripted assembly from existing `*_activations.pdf` data |
| Optional: compute prediction-agreement (Pearson r) between dense and sparse hero of each flavour | — | only needed if a reviewer asks "does the small model still match the big one?"; supports the §9 deployment sentence if included |
