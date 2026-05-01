# 20-Feature Model Rerun: Design Spec

**Date:** 2026-05-01
**Author:** Gian (with Claude assistance)
**Scope:** Re-derive top-20 feature lists from Cyril's new full-feature big models, run a fresh per-flavor sparsity Pareto on those 20-feature inputs, run 50-trial Optuna at the 20-feature budget, train winners, and refresh the paper's §3.2 numbers and figures.
**Methodology:** Same as the existing paper. Only the *inputs* (big-model checkpoints, feature lists, baseline architectures, fixed λ) change.

---

## Background

Cyril recently re-tuned the dense full-feature ChebyKAN, FourierKAN, and XGBoost models through a three-stage chain (Stage A architecture → Stage B optimizer + multi-seed retrain → Stage C dense Pareto sparsity). The resulting tuned hyperparameters and architectures differ from the previous paper iteration, and the paper's `--stage interpret` driver now consumes a YAML config (`--interpret-config`) rather than CLI flags.

Gian's previous 20-feature tuning (Optuna sweeps under `sweeps/stage-c-{model}-top20-tune.db`) was done against the *old* full-feature models. The 20-feature top lists, baseline architectures, and fixed sparsity λ were derived from the old big models. None of those inputs match the new full-feature setup. The numbers in `local_files/main (1).tex` Table 2 sparse-regime rows are therefore stale.

Cyril's new full-feature artifacts (sweep DBs and pareto JSONs) are on disk; the *checkpoints* and *interpret outputs* are not. They need to be regenerated locally before the 20-feature rerun can begin.

---

## Goals

- Re-derive each model's top-20 feature list from its **new** full-feature interpret outputs.
- Re-run a fresh sparsity Pareto on the 20-feature input space (KANs only) and pick a strong sparse-reference λ per flavor.
- Re-run 50-trial Optuna at the 20-feature budget, with the new top-20 features and the new baseline architectures.
- Train the winners.
- Update `local_files/main (1).tex` Table 2 sparse-regime rows + §3.2.2 inline numbers; regenerate Figure 1 and the closed-forms table; refresh bootstrap CIs.

## Non-goals

- Re-running Cyril's full-feature Stage A, B, or C Pareto sweeps. The new full-feature numbers in §3.2.1 are kept as-is.
- Changing the methodology (search spaces, sampler, seeds, recipe choices, LayerNorm convention).
- Adding new features, models, or paper sections.
- Cross-model preprocessing alignment. Each model uses Cyril's per-model recipe (`kan_paper` for KANs, `xgboost_paper` for XGB), not a shared recipe across model families.

---

## Architecture

The rerun is a single linear pipeline, executed as one overnight session, with **per-model independence** for failure recovery: if any model's branch fails at any step, the other two models still proceed end to end.

```
                         ┌──────────────────────────────────┐
                         │ §1: Stage C dense full-feature   │
                         │     train (3 train cmds)         │
                         └──────────────┬───────────────────┘
                                        │
                         ┌──────────────▼───────────────────┐
                         │ §2: Patch interpret wrappers,    │
                         │     run --stage interpret ×3     │
                         └──────────────┬───────────────────┘
                                        │
                         ┌──────────────▼───────────────────┐
                         │ §3: Derive new top-20 JSONs      │
                         └──────────────┬───────────────────┘
                                        │
                         ┌──────────────▼───────────────────┐
                         │ §3.5: 20-feature sparsity Pareto │
                         │       (KANs only, 33-trial grid) │
                         │       Pick strong sparse λ       │
                         └──────────────┬───────────────────┘
                                        │
                         ┌──────────────▼───────────────────┐
                         │ §4: Update 3 tune YAMLs          │
                         └──────────────┬───────────────────┘
                                        │
                         ┌──────────────▼───────────────────┐
                         │ §5: Run 50-trial Optuna ×3,      │
                         │     train winners                │
                         └──────────────┬───────────────────┘
                                        │
                         ┌──────────────▼───────────────────┐
                         │ §6: Update main (1).tex          │
                         │     Regenerate Fig 1, closed-    │
                         │     forms, bootstrap CIs         │
                         └──────────────────────────────────┘
```

---

## §1 Stage C dense full-feature train

Three train commands using Cyril's existing tuned configs, no edits:

```bash
uv run python main.py --stage train \
  --config configs/experiment_stages/stage_c_explanation_package/cheby/all_features/train/chebykan_best.yaml

uv run python main.py --stage train \
  --config configs/experiment_stages/stage_c_explanation_package/fourier/all_features/train/fourierkan_best.yaml

uv run python main.py --stage train \
  --config configs/experiment_stages/stage_c_explanation_package/xgboost/all_features/train/xgb_best.yaml
```

**Outputs:**
- `checkpoints/stage-c-chebykan-best/model-<ts>.pt`
- `checkpoints/stage-c-fourierkan-best/model-<ts>.pt`
- `checkpoints/stage-c-xgb-best/model-<ts>.joblib`

**Hyperparameters used (from Cyril's configs, unchanged):**
- ChebyKAN: `[128]`, `degree=5`, `lr=0.000881…`, `wd=9.47e-5`, `bs=256`, `λ=0.0`, `use_layernorm=true` (default), recipe `kan_paper`.
- FourierKAN: `[128]`, `grid_size=5`, `lr=0.001120…`, `wd=1.10e-6`, `bs=256`, `λ=0.0`, `use_layernorm=true`, recipe `kan_paper`.
- XGBoost: `n_estimators=950`, `max_depth=3`, `lr=0.0837`, `reg_lambda=7.26`, recipe `xgboost_paper`.

Estimated wall-clock: ~30–45 min total on MPS (32 GB Apple Silicon).

---

## §2 Run interpret

**Patch in place:** `configs/interpretability_stage/stage_c_best/chebykan.yaml` and `fourierkan.yaml` currently point at the `*_trial009.yaml` configs and `stage-c-{flavor}-pareto-sparsity-trial-009/...pt` checkpoints (artifacts from Cyril's full-feature Pareto sweep, which we are *not* re-running). Repoint to the train-only artifacts produced in §1:

| Wrapper | `config:` | `checkpoint:` |
|---|---|---|
| `chebykan.yaml` | `.../cheby/all_features/train/chebykan_best.yaml` | `checkpoints/stage-c-chebykan-best/model-<ts>.pt` |
| `fourierkan.yaml` | `.../fourier/all_features/train/fourierkan_best.yaml` | `checkpoints/stage-c-fourierkan-best/model-<ts>.pt` |
| `xgboost.yaml` | already correct (points at `xgb_best.yaml` and `stage-c-xgb-best/`) | — |

Then run:

```bash
uv run python main.py --stage interpret \
  --interpret-config configs/interpretability_stage/stage_c_best/chebykan.yaml
uv run python main.py --stage interpret \
  --interpret-config configs/interpretability_stage/stage_c_best/fourierkan.yaml
uv run python main.py --stage interpret \
  --interpret-config configs/interpretability_stage/stage_c_best/xgboost.yaml
```

**Outputs:**
- KANs: `outputs/interpretability/kan_paper/stage-c-{chebykan,fourierkan}-best/data/{flavor}_feature_ranking.csv` (`feature,importance` columns).
- XGB: `outputs/interpretability/xgboost_paper/stage-c-xgb-best/data/shap_xgb_values.parquet` (per-applicant SHAP matrix).

Estimated wall-clock: ~1 h total.

---

## §3 Derive top-20 feature lists

A small one-shot script reads each interpret output and overwrites the *existing* feature-list JSONs (keeping their filenames so no other config edits are needed):

| Source | Sort key | Output JSON |
|---|---|---|
| `stage-c-chebykan-best/data/chebykan_feature_ranking.csv` | `importance` desc | `configs/.../feature_lists/chebykan_pareto_q0583_top20_features.json` |
| `stage-c-fourierkan-best/data/fourierkan_feature_ranking.csv` | `importance` desc | `configs/.../feature_lists/fourierkan_tuned_top20_features.json` |
| `stage-c-xgb-best/data/shap_xgb_values.parquet` | `mean(\|SHAP\|)` per feature, desc | `configs/.../feature_lists/xgb_tuned_top20_features.json` |

JSON shape preserved (top-level array of feature-name strings, top-20). Old contents stay in git history.

Script: new file `scripts/derive_top20_from_interpret.py`. Pure pandas + json. Idempotent.

---

## §3.5 20-feature sparsity Pareto (KANs only)

Two new tune configs, mirroring Cyril's full-feature `*_pareto_sparsity.yaml` exactly except for: feature subset, architecture/HP baseline (Cyril's full-feature tuned values), `use_layernorm: false`, sweep storage path.

**Files to create:**
- `configs/.../cheby/20_features/tune/chebykan_top20_pareto_sparsity.yaml`
- `configs/.../fourier/20_features/tune/fourierkan_top20_pareto_sparsity.yaml`

**Common config shape (per flavor):**
```yaml
trainer:
  experiment_name: stage-c-{flavor}-top20-pareto-sparsity
  seed: 42
preprocessing:
  recipe: kan_paper
  selected_features_path: configs/.../feature_lists/{old-filename}.json
model:
  flavor: {chebykan|fourierkan}
  hidden_widths: [128]
  {degree: 5 | grid_size: 5}
  use_layernorm: false   # ← deviation from Cyril's full-feature
  params:
    {lr, wd, bs from Cyril's full-feature tuned}
    sparsity_lambda: 0.0   # overwritten per trial
tune:
  storage: sweeps/stage_c/{flavor}/20_features/stage-c-{flavor}-top20-pareto-sparsity.db
  n_trials: 33
  directions: [maximize, maximize]
  sampler: grid
  top_k_candidates: 33
  search_space:
    sparsity_lambda:
      type: grid
      values: [33-value grid identical to Cyril's full-feature config]
```

**Run:**
```bash
uv run python main.py --stage tune --config configs/.../cheby/20_features/tune/chebykan_top20_pareto_sparsity.yaml
uv run python main.py --stage tune --config configs/.../fourier/20_features/tune/fourierkan_top20_pareto_sparsity.yaml
```

**Pareto λ selection** (one per flavor): from each `*_pareto.json`, choose the highest-QWK Pareto point with `sparsity_ratio ≥ 0.5`. This matches Cyril's "strong sparse reference" rule (high sparsity, negligible QWK loss). Selection is fully scripted; no manual choice.

Estimated wall-clock: ~3 h total (66 trials at ~3 min each on MPS, sequential).

---

## §4 Update three 20-feature tune configs

In place at `configs/.../{cheby,fourier,xgboost}/20_features/tune/*.yaml`. Per file:

| Field | Cheby | Fourier | XGB |
|---|---|---|---|
| `preprocessing.recipe` | `kan_paper` | `kan_paper` | **`xgboost_paper`** ← changed from `kan_paper` |
| `preprocessing.selected_features_path` | unchanged path, new contents | unchanged path, new contents | unchanged path, new contents |
| `model.hidden_widths` / `degree` / `grid_size` | `[128]`, `degree=5` | `[128]`, `grid_size=5` | `n_estimators=950`, `max_depth=3` |
| `model.use_layernorm` | **`false`** | **`false`** | n/a |
| `model.params.lr` | Cyril's full-feature tuned | Cyril's full-feature tuned | `0.0837` |
| `model.params.weight_decay` / `reg_lambda` | Cyril's | Cyril's | `7.26` |
| `model.params.batch_size` | `256` | `256` | n/a |
| `model.params.sparsity_lambda` | from §3.5 Pareto pick | from §3.5 Pareto pick | n/a |
| `tune.storage` | `sweeps/stage_c/chebykan/20_features/stage-c-chebykan-top20-tune.db` | `sweeps/stage_c/fourierkan/20_features/stage-c-fourierkan-top20-tune.db` | `sweeps/stage_c/xgboost/20_features/stage-c-xgb-top20-tune.db` |
| `tune.search_space` | unchanged from old config | unchanged from old config | unchanged from old config |

Old DBs at `sweeps/stage-c-{model}-top20-tune.db` remain on disk as historical reference; nothing references them after this edit.

**Fallback path:** if §3.5 fails to complete for one or both KANs, the affected tune YAMLs are populated with Cyril's full-feature Pareto pick instead (`λ=0.0015351` Cheby, `λ=0.0005` Fourier). One-line edit per affected file.

---

## §5 Run sweeps + train winners

**Sweeps:**
```bash
uv run python main.py --stage tune --config configs/.../cheby/20_features/tune/chebykan_top20_tune.yaml
uv run python main.py --stage tune --config configs/.../fourier/20_features/tune/fourierkan_top20_tune.yaml
uv run python main.py --stage tune --config configs/.../xgboost/20_features/tune/xgb_top20_tune.yaml
```

Each sweep auto-emits `<storage>_best.yaml` next to the DB. **Winner train:**
```bash
uv run python main.py --stage train --config sweeps/stage_c/chebykan/20_features/stage-c-chebykan-top20-tune_best.yaml
# same shape for fourier, xgb
```

After each winner trains, the existing retrain config files at:
- `cheby/20_features/train/chebykan_pareto_q0583_top20.yaml`
- `fourier/20_features/train/fourierkan_pareto_top20_noln.yaml`
- `xgboost/20_features/train/xgboost_top20_retuned.yaml`

are overwritten with the winner's hyperparameters (architecture, optimizer, fixed λ for KANs). Filenames stay identical so paper-reproduction recipes don't shift.

**Interpret on the new ChebyKAN winner** is also run, to produce the symbolic/closed-form artifacts §6 needs:
```bash
uv run python main.py --stage interpret --config <new chebykan winner config> \
  --pruning-threshold 0.001 --qwk-tolerance 0.01 --candidate-library scipy --max-features 20
```
(FourierKAN and XGB winners do not feed §6 figures, so their interpret runs are skipped to save time. If desired they can be added later.)

Estimated wall-clock: ~5 h total (3 × 50-trial sweeps + 3 retrains + 1 ChebyKAN interpret).

---

## §6 Paper update

**Paper file:** `local_files/main (1).tex`. Updates are scoped to §3.2.2 (sparse regime) plus the sparse-regime rows of Table 2.

### Numeric edits in `main (1).tex`

| Line(s) | What | Source |
|---|---|---|
| 124 (Table 2, XGB tuned) | QWK + 95% bootstrap CI + tree count | new XGB-top20 winner + `bootstrap_qwk_table1.py` |
| 125 (Table 2, ChebyKAN sparse) | QWK + 95% bootstrap CI + active-edge count | new ChebyKAN-top20 winner pruned module |
| 126 (Table 2, FourierKAN tuned) | QWK + 95% bootstrap CI + active-edge count | new FourierKAN-top20 winner |
| 162 (inline) | "XGBoost (X.XXX), tuned FourierKAN (X.XXX), and the sparse ChebyKAN configuration (X.XXX)" | new winner QWKs |
| 162 (inline) | sparse-config description: hidden widths, degree, λ, pruning threshold, "X of Y KAN-layer edges, corresponding to ZZ% pruning" | new ChebyKAN winner config + interpret pruning summary |
| 164 (inline) | example BMI closed-form edge | regenerated from `simplified_closed_forms_table.py` against the new winner |
| 173 (inline) | applicant 55728 reference score, predicted score, predicted class | regenerated from `build_figure3_waterfall.py` against the new winner |

The qualitative prose ("largest positive contributions come from medical history and biometric variables", etc.) is **not** auto-edited. If the new winner changes the qualitative direction, that is flagged in the run log for manual review.

### Regenerated artifacts

- `outputs/figures/fig1_interpretability.pdf` — regenerated by `scripts/build_figure3_waterfall.py` after repointing its hardcoded `EXP`, `CKPT`, `CONFIG`, `EVAL` paths to the new ChebyKAN winner's interpret outputs. The PDF is then copied to `local_files/paper_graphics/fig1_interpretability.pdf` and `local_files/fig1_interpretability.pdf` so the paper figure path resolves.
- `outputs/reports/table_closed_forms_latex.tex` — regenerated by `scripts/simplified_closed_forms_table.py` after repointing its hardcoded `CONFIG` and `CKPT` to the new ChebyKAN winner. The five representative features may change; the script handles that automatically.
- `outputs/reports/table1_bootstrap_qwk.json` — regenerated by `scripts/bootstrap_qwk_table1.py` after appending three new entries to its `MODELS` list (new XGB-top20, ChebyKAN-top20, FourierKAN-top20 with their `(exp_name, ckpt_stem, recipe, config_yaml)` tuples). Bootstrap CIs from this JSON populate Table 2.

### Scripts that need their hardcoded paths repointed

- `scripts/build_figure1_interpretability.py` — currently hardcodes the old hero. Repoint to new ChebyKAN winner.
- `scripts/build_figure3_waterfall.py` — same.
- `scripts/simplified_closed_forms_table.py` — same.
- `scripts/bootstrap_qwk_table1.py` — append entries; do not delete existing entries.

Edits are inline at the top of each script (constants block). No structural code changes.

---

## Failure handling

**Per-model independence.** The pipeline is structured so each model's branch is self-contained from §1 through §5. A failure in any step for one model logs to `runs/2026-05-01-rerun/<model>/<step>.log`, marks that branch failed, and skips the remaining steps for that model only. The other two models proceed.

§6 paper updates are also per-model: only update the rows / inline numbers / artifacts for models that completed successfully. Failed-model rows in `main (1).tex` are left untouched, with a note appended to the run summary.

**No partial fakes.** No fallback to "approximate" numbers. If a model fails, its row in Table 2 stays with the *old* numbers and the run summary flags it explicitly for manual follow-up.

**§3.5 fallback** (described in §4): if the new 20-feature Pareto fails for a flavor, that flavor's 20-feature tune config is populated with Cyril's full-feature Pareto λ instead, and the run continues. This is a documented degradation, not a silent fallback.

---

## Run-log layout

```
runs/2026-05-01-rerun/
├── master.log              # high-level step boundaries + status
├── 01_train_full/
│   ├── chebykan.log
│   ├── fourierkan.log
│   └── xgb.log
├── 02_interpret_full/
├── 03_derive_top20/
├── 04_pareto_20feat/
├── 05_tune_20feat/
├── 06_train_winners/
└── 07_paper_update/
    ├── summary.md          # per-model status + flagged manual-review items
    └── ...
```

Each step's command is printed to `master.log` before execution, and to its own log file. On failure, the failing log file is the source of truth.

---

## Out of scope

- New feature engineering or preprocessing recipes.
- New model families.
- Updating §3.2.1 (full-feature regime) — Cyril's numbers are kept as-is.
- Updating §3.1 single-objective Optuna table — unchanged.
- Updating Cyril's full-feature Pareto sweep results — unchanged.
- Bootstrap CIs for any non-20-feature row in Table 2 (full-feature CIs are already populated).

---

## Estimated wall-clock

| Step | Time |
|---|---|
| §1 train full (3) | 30–45 min |
| §2 interpret full (3) | ~1 h |
| §3 derive top-20 | < 1 min |
| §3.5 Pareto KAN ×2 (66 trials) | ~3 h |
| §4 update tune YAMLs | < 1 min |
| §5 tune ×3 + winner train ×3 + cheby winner interpret | ~5 h |
| §6 paper update + figure regen + bootstrap | ~30 min |
| **Total** | **~10 h sequential** |

Hardware: Apple Silicon, MPS, 32 GB. Sequential. Mac on AC. No parallelization across models (hardware-bound, single MPS device).

---

## Open items

None. All design decisions are locked.

---
