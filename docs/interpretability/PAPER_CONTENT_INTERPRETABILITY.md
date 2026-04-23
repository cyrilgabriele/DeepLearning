# Paper Content — Interpretability Section

Scope: drives the interpretability content for the 6-page paper answering
*"Can TabKAN models balance accuracy & interpretability in life insurance
risk assessment?"*. Targets ≤ ½ page main body + table + figure, with
appendix dump.

Status as of 2026-04-23. Hero models verified on `gian-interpretability`
branch.

---

## 1. The headline claim

> Both ChebyKAN and FourierKAN admit exact per-edge symbolic recovery
> after sparsity-regularised training and L1 pruning. They occupy
> distinct points on the accuracy/interpretability Pareto front:
> FourierKAN keeps higher QWK (0.562) but produces a 12× larger and
> structurally heavier explainable model; ChebyKAN trades 0.029 QWK
> (0.533) for a network that is 12× sparser and admits a single
> end-to-end closed-form polynomial. Neither dominates — the choice
> depends on whether the use case prioritises predictive accuracy or
> structural compactness.

This frames TabKAN's interpretability as a *real Pareto choice*, which
is the strongest available answer to the research question.

---

## 2. Hero models (locked in)

Both trained on the same 20 input features, no LayerNorm, sparsity
regularisation tuned per family, then L1-pruned with QWK tolerance ≤ 0.01.

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

Note on QWK: numbers above are inner-validation with optimised ordinal
thresholds (the regime used elsewhere in the paper). The held-out outer
test QWK (default `predict()`, no threshold optimisation) is reported
in the manifests for completeness; use the inner-val numbers in the
paper to stay consistent with the rest of the results section.

---

## 3. Suggested structure for the interpretability subsection

Target: ≤ ½ page main body + 1 figure + 1 table.

### Paragraph 1 — Setup (≈ 60 words)

> We extract a small interpretable variant of each TabKAN flavour by
> (i) selecting the top-20 features by coefficient-based importance
> from a sparsity-regularised baseline, (ii) retraining on that subset
> without LayerNorm, and (iii) pruning edges whose mean activation
> magnitude falls below a threshold tightened until QWK loss stays
> within 0.01.

### Paragraph 2 — Result (≈ 80 words)

> ChebyKAN tolerates aggressive pruning (94% of edges removed) at a
> cost of 0.003 QWK; FourierKAN reaches a maximum sparsity of 79% under
> the same tolerance. Every surviving edge in either model admits an
> exact closed-form recovery: each ChebyKAN edge is a polynomial of
> degree ≤ 6 in `tanh(x)`, and each FourierKAN edge is a sum of eight
> harmonic pairs in `kπ(tanh(x)+1)`. Mean recovery R² = 1.000 across
> all 597 / 7 158 active edges respectively.

### Paragraph 3 — Trade-off (≈ 80 words)

> ChebyKAN's polynomial basis is closed under composition, so the entire
> pruned model collapses into a single closed-form polynomial in the
> 20 input features. FourierKAN's basis is not closed under composition;
> per-edge forms remain readable, but the composed network does not
> simplify. The quantitative trade-off is therefore: FourierKAN keeps a
> 0.029 QWK advantage at the cost of 12× more surviving edges and a
> structurally heavier explanation. We position ChebyKAN as the
> compact-explanation choice and FourierKAN as the high-accuracy choice;
> both fall on the same Pareto front of accuracy vs. structural complexity.

### Optional honest-limit sentence (1 line)

> The model genuinely uses all retained features: in feature-validation
> (Appendix X), QWK collapses to ≈ 0 when fewer than 100 features are
> kept, so neither variant supports radical feature elimination.

---

## 4. Table 1 — recommended (lock this in)

Target: 5 columns × 5 rows (compact). Caption emphasises *like-for-like
comparison*.

| Model | QWK | # active edges / params | Explanation method | Per-edge R² |
|---|---|---|---|---|
| GLM (ridge) | *to fill* | 140 coefficients | linear coefficients | — |
| XGBoost | 0.655 | ~ N trees | SHAP TreeExplainer (post-hoc) | — |
| ChebyKAN, full (140 features, dense) | 0.625 | 17 920 edges | per-edge native | — |
| **ChebyKAN, sparse (20 features, no LN)** | **0.533** | **597 edges (−94%)** | **closed form** | **1.000** |
| FourierKAN, full (140 features, dense) | 0.641 | ≈ 41 728 edges | per-edge native | — |
| **FourierKAN, sparse (20 features, no LN)** | **0.562** | **7 158 edges (−79%)** | **closed form** | **1.000** |

Two bolded rows = the hero variants. Their relationship to the dense
baselines (rows 3 and 5) is the entire interpretability story — visible
in one glance.

**Open item**: the GLM row needs a real number. No GLM checkpoint
exists in `checkpoints/`; the only number anywhere is the stale 0.568
(validation, March) in `outputs/reports/final_comparison_matrix.md`.
Either retrain on the same split or drop the GLM row.

**Open item**: clarify per-paper convention whether QWK is inner-val
(opt. thresholds) or outer-test (default `predict()`). Numbers above
are inner-val. Outer-test is systematically lower (e.g., ChebyKAN-best
manifest reports 0.543 vs 0.625 sweep).

---

## 5. Figure 1 — recommended (one figure only, ~⅓ column)

A 4-panel grid showing 2 features × 2 flavours. Each panel: scatter of
the learned activation (grey dots) overlaid with the recovered
closed-form expression (coloured line), with R² annotated.

Suggested features (in the top-20 of *both* flavours): **BMI, Wt**.

Layout:
```
        BMI                Wt
ChebyKAN  ▢ poly(tanh)      ▢ poly(tanh)
FourierKAN ▢ trig(tanh)     ▢ trig(tanh)
```

This single figure simultaneously shows (a) both flavours admit
recovery, (b) what the two formula classes *look like*, (c) why they
behave differently in composition (polynomial vs trig).

If forced to pick a *second* figure (only if space allows): the Pareto
fronts of QWK vs sparsity for both flavours from
`sweeps/stage-c-{cheby,fourier}kan-pareto-sparsity_pareto.json` —
proves FourierKAN collapses faster under aggressive sparsity. But
Table 1 already conveys the headline; the figure is optional.

---

## 6. Appendix dump (won't be read, but valuable to have)

All artifacts already exist under
`outputs/interpretability/kan_paper/stage-c-{chebykan,fourierkan}-pareto-…-top20-noln/`:

- Full per-edge formula listings (`reports/*_symbolic_formulas.md`)
- Composed end-to-end formula (ChebyKAN only; `reports/*_exact_closed_form.md`)
- Network-diagram PDF with mini-plots (`figures/*_kan_diagram.pdf`)
- Per-feature activation grid (`figures/*_activations.pdf`)
- R² distribution histogram (`figures/*_r2_distribution.pdf`)
- Hessian heatmap, continuous features only (`figures/chebykan_hessian_heatmap_continuous.png`)
- Feature-validation curves (`figures/feature_validation_curves.pdf`)
- Local case explanation for applicant 55728 (`reports/*_case_summary_55728.0.md`)
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

---

## 8. What *not* to include

- Dense / full-feature interpretability runs (too many edges to draw or
  list). One sentence in setup is enough.
- Narrow-architecture experiments (140 → 16 → 8 → 1 etc.). Negative
  result; cut.
- Closed-form polynomial *surrogate* (`closed_form_surrogate.py`). Only
  needed when exact composition is unavailable; for the no-LN ChebyKAN
  hero, exact composition exists, so the surrogate is redundant.
- Per-risk-level comparison panels. Too dense for the page budget.
- The twin-model (production + explainer) framing. Was tempting from
  industry practice but adds an "agreement" experiment we don't have
  data for. The Pareto framing is cleaner and supported by the data.

---

## 9. Sentence-level draft for the abstract / conclusion

For the abstract:

> We show that two TabKAN variants — Chebyshev and Fourier — achieve
> exact per-edge symbolic recovery after sparsity-regularised training
> and pruning. The variants occupy distinct points on the accuracy–
> compactness frontier: FourierKAN reaches QWK 0.562 with 7 158 active
> edges; ChebyKAN reaches QWK 0.533 with 597 active edges and a single
> closed-form composed expression. The choice between flavours is
> therefore application-driven, not architecture-dominated.

For the conclusion:

> TabKAN can balance accuracy and interpretability, but the choice of
> basis dictates the *shape* of the trade-off rather than its
> existence. Practitioners with a hard accuracy floor should prefer
> the Fourier basis; those required to deliver compact closed-form
> explanations to regulators or actuaries should prefer the Chebyshev
> basis.

---

## 10. Open / required follow-ups

| Item | Owner | Notes |
|---|---|---|
| Train GLM baseline on the same split | — | needed for Table 1 row |
| Confirm QWK reporting convention (inner-val vs outer-test) | — | affects all numbers |
| Build Figure 1 (2×2 panel) from existing `*_activations.pdf` data | — | scripted assembly |
| Optional: compute prediction-agreement (Pearson r) between full and sparse hero of each flavour, in case a reviewer asks "does the small model still match the big one?" | — | one number per flavour |
