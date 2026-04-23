# KAN vs GAM Grid Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grid search over 1-layer (GAM) and narrow 2-layer (KAN) ChebyKAN configurations, extract three levels of closed-form formulas (per-edge exact, per-feature conditional, full model composition), and produce actuary-facing comparison outputs.

**Architecture:** Single script `scripts/kan_vs_gam_search.py` runs the grid search, trains each config, extracts all three formula levels, and saves results. A second script `scripts/plot_kan_vs_gam.py` reads the saved results and produces comparison figures and tables. Both scripts reuse existing infrastructure (`TabKAN`, `sample_edge`, `fit_symbolic_edge`, `_compute_edge_l1`, `_quality_tier`).

**Tech Stack:** PyTorch, Lightning, numpy, scipy, matplotlib, pandas. No new dependencies.

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/kan_vs_gam_search.py` (create) | Grid search: train models, extract 3-level formulas, save results |
| `scripts/plot_kan_vs_gam.py` (create) | Read results, produce actuary-facing figures and tables |
| `outputs/kan_vs_gam/` (created at runtime) | All outputs: CSVs, JSONs, PDFs |

No existing files are modified.

---

### Task 1: Grid Search Script — Data Loading & Config Grid

**Files:**
- Create: `scripts/kan_vs_gam_search.py`

- [ ] **Step 1: Create the script with imports, constants, and config grid**

```python
#!/usr/bin/env python3
"""KAN vs GAM grid search: 1-layer (GAM) vs narrow 2-layer (KAN).

Trains ChebyKAN models across architectures, extracts three levels of
closed-form formulas, and saves results for comparison plotting.

Usage:
    uv run python scripts/kan_vs_gam_search.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import lightning as L

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocess_kan_paper import KANPreprocessor
from src.models.tabkan import TabKAN
from src.models.kan_layers import ChebyKANLayer
from src.interpretability.kan_pruning import _compute_edge_l1
from src.interpretability.kan_symbolic import (
    sample_edge,
    fit_symbolic_edge,
    _quality_tier,
)
from src.metrics.qwk import quadratic_weighted_kappa


TOP_FEATURES_RANKED = [
    "BMI", "Medical_Keyword_3", "Product_Info_4", "Wt",
    "Medical_History_20", "Medical_Keyword_38", "Employment_Info_6",
    "Medical_History_4", "Medical_History_30", "Medical_Keyword_29",
    "Medical_Keyword_13", "Medical_Keyword_47", "Medical_Keyword_46",
    "Medical_Keyword_31", "Medical_Keyword_35", "Ins_Age",
    "Medical_Keyword_12", "Medical_Keyword_14", "Medical_Keyword_43",
    "Medical_Keyword_5",
]

OUTPUT_DIR = Path("outputs/kan_vs_gam")


@dataclass
class SearchConfig:
    name: str
    n_features: int
    hidden_widths: list[int]
    degree: int = 3
    sparsity_lambda: float = 0.0
    lr: float = 5e-3
    weight_decay: float = 5e-4
    max_epochs: int = 150
    batch_size: int = 2048
    pruning_threshold: float = 0.01

    @property
    def n_layers(self) -> int:
        return len(self.hidden_widths)

    @property
    def model_type(self) -> str:
        return "GAM (1-layer)" if self.n_layers == 1 else "KAN (2-layer)"


def build_search_grid() -> list[SearchConfig]:
    """Build the grid of 1-layer (GAM) and 2-layer (KAN) configs."""
    configs = []

    for n_feat in [10, 15, 20]:
        for sp in [0.0, 0.003, 0.005]:
            sp_str = f"_sp{sp}" if sp > 0 else ""

            # 1-layer = GAM baselines
            for w in [4, 8, 16]:
                configs.append(SearchConfig(
                    name=f"GAM_f{n_feat}_w{w}{sp_str}",
                    n_features=n_feat,
                    hidden_widths=[w],
                    sparsity_lambda=sp,
                ))

            # 2-layer = KAN (narrow second layer)
            for w1, w2 in [(8, 1), (16, 1), (8, 2), (16, 2)]:
                configs.append(SearchConfig(
                    name=f"KAN_f{n_feat}_w{w1}x{w2}{sp_str}",
                    n_features=n_feat,
                    hidden_widths=[w1, w2],
                    sparsity_lambda=sp,
                ))

    return configs
```

- [ ] **Step 2: Verify the file is syntactically correct**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python -c "import ast; ast.parse(open('scripts/kan_vs_gam_search.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/kan_vs_gam_search.py
git commit -m "feat: kan vs gam search scaffold with config grid"
```

---

### Task 2: Grid Search Script — Training & QWK Evaluation

**Files:**
- Modify: `scripts/kan_vs_gam_search.py`

- [ ] **Step 1: Add data loading and feature selection (reuse from existing scripts)**

```python
def load_data(seed: int = 42) -> dict:
    """Load and preprocess data using kan_paper pipeline."""
    csv_path = Path("data/prudential-life-insurance-assessment/train.csv")
    preprocessor = KANPreprocessor()
    return preprocessor.run_pipeline(csv_path, random_seed=seed)


def select_features(
    data: dict, n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Select top-N features from preprocessed data."""
    all_names = data["feature_names"]
    keep = TOP_FEATURES_RANKED[:n_features]
    keep_indices = []
    kept_names = []
    for feat in keep:
        if feat in all_names:
            keep_indices.append(all_names.index(feat))
            kept_names.append(feat)
    idx = np.array(keep_indices)
    return (
        data["X_train_outer"][:, idx],
        data["X_test_outer"][:, idx],
        data["y_train_outer"],
        data["y_test_outer"],
        kept_names,
    )
```

- [ ] **Step 2: Add model training function**

```python
def train_model(
    cfg: SearchConfig,
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> tuple[TabKAN, float, float]:
    """Train a TabKAN model. Returns (module, qwk, training_time_s)."""
    L.seed_everything(42)

    module = TabKAN(
        in_features=X_train.shape[1],
        widths=cfg.hidden_widths,
        kan_type="chebykan",
        degree=cfg.degree,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        sparsity_lambda=cfg.sparsity_lambda,
        l1_weight=1.0,
        entropy_weight=1.0,
        use_layer_norm=False,  # always off for formula extraction
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t.unsqueeze(1)),
        batch_size=cfg.batch_size, shuffle=True,
    )
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.float32)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t.unsqueeze(1)),
        batch_size=cfg.batch_size, shuffle=False,
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs, accelerator="auto",
        enable_progress_bar=False, enable_model_summary=False, logger=False,
    )
    t0 = time.time()
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_time = time.time() - t0

    module.eval()
    with torch.no_grad():
        preds = module(X_val_t).cpu().numpy().flatten()
    preds_r = np.clip(np.round(preds), 1, 8).astype(int)
    y_true = np.clip(np.round(y_test), 1, 8).astype(int)
    qwk = quadratic_weighted_kappa(y_true, preds_r)

    return module, qwk, train_time
```

- [ ] **Step 3: Verify syntax**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python -c "import ast; ast.parse(open('scripts/kan_vs_gam_search.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/kan_vs_gam_search.py
git commit -m "feat: add training and evaluation to kan vs gam search"
```

---

### Task 3: Level 1 — Per-Edge Exact Closed Forms

**Files:**
- Modify: `scripts/kan_vs_gam_search.py`

- [ ] **Step 1: Add per-edge symbolic analysis for all layers**

This reuses `sample_edge`, `fit_symbolic_edge`, `_compute_edge_l1`, and `_quality_tier` from the existing interpretability pipeline. It extracts per-edge formulas for BOTH layer 1 and layer 2 (for 2-layer models).

```python
def extract_edge_formulas(
    module: TabKAN,
    feature_names: list[str],
    pruning_threshold: float = 0.01,
) -> pd.DataFrame:
    """Level 1: Extract exact symbolic formula for every active edge in all layers."""
    records = []
    layer_idx = 0

    for layer in module.kan_layers:
        if not isinstance(layer, ChebyKANLayer):
            continue

        l1_scores = _compute_edge_l1(layer)
        layer_degree = getattr(layer, "degree", 3)

        for out_i in range(layer.out_features):
            for in_i in range(layer.in_features):
                if l1_scores[out_i, in_i].item() < pruning_threshold:
                    continue

                x_vals, y_vals = sample_edge(layer, out_i, in_i, n=1000)
                formula, r2 = fit_symbolic_edge(
                    x_vals, y_vals, max_poly_degree=layer_degree,
                )

                if layer_idx == 0 and in_i < len(feature_names):
                    feat_name = feature_names[in_i]
                else:
                    feat_name = f"h{in_i}"

                records.append({
                    "layer": layer_idx,
                    "edge_in": in_i,
                    "edge_out": out_i,
                    "input_feature": feat_name,
                    "formula": formula,
                    "r_squared": round(r2, 6),
                    "quality_tier": _quality_tier(r2),
                    "l1_norm": round(l1_scores[out_i, in_i].item(), 6),
                })

        layer_idx += 1

    return pd.DataFrame(records)
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python -c "import ast; ast.parse(open('scripts/kan_vs_gam_search.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/kan_vs_gam_search.py
git commit -m "feat: level 1 per-edge exact formula extraction"
```

---

### Task 4: Level 2 — Per-Feature Conditional Response Formulas

**Files:**
- Modify: `scripts/kan_vs_gam_search.py`

- [ ] **Step 1: Add per-feature conditional response extraction**

For 1-layer models this is exact (same as `compose_input_to_output`). For 2-layer models, it holds all other features at their training mean and traces the exact model forward pass, capturing the interaction-modified marginal effect.

```python
def extract_feature_formulas(
    module: TabKAN,
    feature_names: list[str],
    X_train: np.ndarray,
) -> tuple[list[dict], float]:
    """Level 2: Per-feature conditional response holding others at training mean.

    For 1-layer models, this is mathematically exact (pure additive).
    For 2-layer models, this captures the interaction-modified marginal effect
    for a typical (mean) policyholder.

    Returns (feature_formulas, head_bias).
    """
    n_features = len(feature_names)
    feature_means = X_train.mean(axis=0)  # shape (n_features,)

    # Build the mean-centered input: all features at their training mean
    mean_input = torch.tensor(feature_means, dtype=torch.float32).unsqueeze(0)

    # Get baseline prediction at the mean
    module.eval()
    with torch.no_grad():
        baseline_pred = module(mean_input).item()

    n_points = 1000
    results = []

    for feat_idx, feat_name in enumerate(feature_names):
        # Create batch: 1000 copies of mean input, vary only this feature
        x_grid = torch.linspace(-3.0, 3.0, n_points)  # pre-tanh range
        batch = mean_input.repeat(n_points, 1)  # (1000, n_features)
        batch[:, feat_idx] = x_grid

        with torch.no_grad():
            preds = module(batch).cpu().numpy().flatten()

        x_norm = torch.tanh(x_grid).numpy()  # the effective input domain
        # Center on the mean response to show marginal effect
        y_effect = preds - baseline_pred

        # Check if feature has any effect
        y_range = float(y_effect.max() - y_effect.min())
        if y_range < 1e-6:
            results.append({
                "feature": feat_name,
                "formula": "0 (no effect)",
                "r_squared": 1.0,
                "quality_tier": "clean",
                "y_range": 0.0,
                "is_active": False,
            })
            continue

        # Fit symbolic formula to the conditional response curve
        formula, r2 = fit_symbolic_edge(x_norm, y_effect, max_poly_degree=3)

        results.append({
            "feature": feat_name,
            "formula": formula,
            "r_squared": round(r2, 6),
            "quality_tier": _quality_tier(r2),
            "y_range": round(y_range, 4),
            "is_active": True,
        })

    return results, baseline_pred
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python -c "import ast; ast.parse(open('scripts/kan_vs_gam_search.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/kan_vs_gam_search.py
git commit -m "feat: level 2 per-feature conditional response formulas"
```

---

### Task 5: Level 3 — Full Model Exact Composition

**Files:**
- Modify: `scripts/kan_vs_gam_search.py`

- [ ] **Step 1: Add full model formula extraction**

Extracts ALL Chebyshev coefficients and head weights into a JSON-serializable structure that exactly specifies the complete model as a nested composition. An actuary can implement this in any language.

```python
def extract_full_model_formula(
    module: TabKAN,
    feature_names: list[str],
) -> dict:
    """Level 3: Extract the exact full model as a nested composition.

    Returns a dict specifying:
    - layers[i].edges[j][k] = {cheby_coeffs: [...], base_weight: float}
    - head = {weights: [...], bias: float}

    The exact model is:
      y = head.bias + sum_k head.weights[k] *
          sum_j layer1_edge[k][j](tanh(
              sum_i layer0_edge[j][i](x_i)
          ))

    where each edge(x) = sum_d coeffs[d]*T_d(tanh(x)) + base_weight*x
    """
    layers_data = []
    for layer in module.kan_layers:
        if not isinstance(layer, ChebyKANLayer):
            continue

        edges = {}
        l1_scores = _compute_edge_l1(layer)

        for out_i in range(layer.out_features):
            for in_i in range(layer.in_features):
                coeffs = layer.cheby_coeffs[out_i, in_i, :].detach().tolist()
                base_w = layer.base_weight[out_i, in_i].detach().item()
                l1 = l1_scores[out_i, in_i].item()
                edges[f"{out_i},{in_i}"] = {
                    "cheby_coeffs": [round(c, 8) for c in coeffs],
                    "base_weight": round(base_w, 8),
                    "l1_norm": round(l1, 6),
                }

        layers_data.append({
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "degree": layer.degree,
            "edges": edges,
        })

    head_w = module.head.weight.detach().squeeze().tolist()
    if isinstance(head_w, float):
        head_w = [head_w]
    head_b = module.head.bias.detach().item()

    return {
        "feature_names": feature_names,
        "n_layers": len(layers_data),
        "layers": layers_data,
        "head": {
            "weights": [round(w, 8) for w in head_w],
            "bias": round(head_b, 8),
        },
        "formula_description": (
            "y = head.bias + sum_k(head.weights[k] * "
            "layer[-1]_edge[k][j](tanh(... layer[0]_edge[j][i](x_i) ...))). "
            "Each edge(x) = sum_d(coeffs[d] * T_d(tanh(x))) + base_weight * x, "
            "where T_d is the d-th Chebyshev polynomial."
        ),
    }
```

- [ ] **Step 2: Verify syntax**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python -c "import ast; ast.parse(open('scripts/kan_vs_gam_search.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/kan_vs_gam_search.py
git commit -m "feat: level 3 full model exact formula extraction"
```

---

### Task 6: Grid Search Script — Main Loop & Results Saving

**Files:**
- Modify: `scripts/kan_vs_gam_search.py`

- [ ] **Step 1: Add the experiment runner that calls all three levels**

```python
@dataclass
class SearchResult:
    name: str
    model_type: str
    n_features: int
    hidden_widths: list[int]
    degree: int
    sparsity_lambda: float
    qwk: float
    # Level 1: per-edge
    n_active_edges: int
    n_total_edges: int
    n_clean: int
    n_acceptable: int
    n_flagged: int
    edge_mean_r2: float
    edge_pct_clean: float
    # Level 2: per-feature
    n_active_features: int
    feature_mean_r2: float
    feature_pct_clean: float
    # Composite score
    score: float
    training_time_s: float
    # Raw data (not serialized to summary CSV)
    edge_formulas: Optional[pd.DataFrame] = field(default=None, repr=False)
    feature_formulas: Optional[list[dict]] = field(default=None, repr=False)
    full_model: Optional[dict] = field(default=None, repr=False)


def run_single_experiment(cfg: SearchConfig, data: dict) -> SearchResult:
    """Run a single experiment: train + extract all 3 formula levels."""
    print(f"\n{'='*70}")
    print(f"{cfg.name} | {cfg.model_type} | features={cfg.n_features} "
          f"widths={cfg.hidden_widths} sp={cfg.sparsity_lambda}")
    print(f"{'='*70}")

    X_train, X_test, y_train, y_test, feat_names = select_features(
        data, cfg.n_features,
    )

    # Train
    module, qwk, train_time = train_model(cfg, X_train, X_test, y_train, y_test)
    print(f"  QWK = {qwk:.4f} ({train_time:.1f}s)")

    # Level 1: per-edge
    edge_df = extract_edge_formulas(module, feat_names, cfg.pruning_threshold)
    if edge_df.empty:
        n_active, n_total = 0, 0
        n_clean = n_acceptable = n_flagged = 0
        edge_mean_r2, edge_pct_clean = 0.0, 0.0
    else:
        n_active = len(edge_df)
        n_total = sum(
            l.out_features * l.in_features
            for l in module.kan_layers if isinstance(l, ChebyKANLayer)
        )
        n_clean = int((edge_df["quality_tier"] == "clean").sum())
        n_acceptable = int((edge_df["quality_tier"] == "acceptable").sum())
        n_flagged = int((edge_df["quality_tier"] == "flagged").sum())
        edge_mean_r2 = float(edge_df["r_squared"].mean())
        edge_pct_clean = n_clean / n_active * 100 if n_active > 0 else 0.0

    print(f"  L1 edges: {n_active}/{n_total} active, "
          f"{n_clean} clean, {n_flagged} flagged, mean R²={edge_mean_r2:.4f}")

    # Level 2: per-feature conditional
    feat_formulas, baseline_pred = extract_feature_formulas(
        module, feat_names, X_train,
    )
    active_feats = [f for f in feat_formulas if f["is_active"]]
    n_active_feats = len(active_feats)
    feat_r2s = [f["r_squared"] for f in active_feats]
    feat_mean_r2 = float(np.mean(feat_r2s)) if feat_r2s else 0.0
    feat_n_clean = sum(1 for f in active_feats if f["quality_tier"] == "clean")
    feat_pct_clean = feat_n_clean / n_active_feats * 100 if n_active_feats > 0 else 0.0

    print(f"  L2 features: {n_active_feats}/{cfg.n_features} active, "
          f"mean R²={feat_mean_r2:.4f}, {feat_pct_clean:.0f}% clean")

    # Level 3: full model
    full_model = extract_full_model_formula(module, feat_names)

    # Composite score: QWK * pct_interpretable_features / 100
    feat_pct_interp = (
        sum(1 for f in active_feats if f["quality_tier"] in ("clean", "acceptable"))
        / max(n_active_feats, 1) * 100
    )
    score = qwk * feat_pct_interp / 100

    print(f"  Score (QWK * %interpretable) = {score:.4f}")

    return SearchResult(
        name=cfg.name,
        model_type=cfg.model_type,
        n_features=cfg.n_features,
        hidden_widths=cfg.hidden_widths,
        degree=cfg.degree,
        sparsity_lambda=cfg.sparsity_lambda,
        qwk=qwk,
        n_active_edges=n_active,
        n_total_edges=n_total,
        n_clean=n_clean,
        n_acceptable=n_acceptable,
        n_flagged=n_flagged,
        edge_mean_r2=edge_mean_r2,
        edge_pct_clean=edge_pct_clean,
        n_active_features=n_active_feats,
        feature_mean_r2=feat_mean_r2,
        feature_pct_clean=feat_pct_clean,
        score=score,
        training_time_s=train_time,
        edge_formulas=edge_df,
        feature_formulas=feat_formulas,
        full_model=full_model,
    )
```

- [ ] **Step 2: Add results saving and main function**

```python
def save_results(results: list[SearchResult], output_dir: Path) -> None:
    """Save summary CSV + per-model detail JSONs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    rows = []
    for r in results:
        rows.append({
            "name": r.name,
            "model_type": r.model_type,
            "n_features": r.n_features,
            "hidden_widths": str(r.hidden_widths),
            "degree": r.degree,
            "sparsity_lambda": r.sparsity_lambda,
            "qwk": r.qwk,
            "n_active_edges": r.n_active_edges,
            "n_total_edges": r.n_total_edges,
            "n_clean_edges": r.n_clean,
            "n_flagged_edges": r.n_flagged,
            "edge_mean_r2": r.edge_mean_r2,
            "edge_pct_clean": r.edge_pct_clean,
            "n_active_features": r.n_active_features,
            "feature_mean_r2": r.feature_mean_r2,
            "feature_pct_clean": r.feature_pct_clean,
            "score": r.score,
            "training_time_s": r.training_time_s,
        })
    df = pd.DataFrame(rows)
    csv_path = output_dir / "search_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV → {csv_path}")

    # Per-model detail JSONs (edge formulas + feature formulas + full model)
    details_dir = output_dir / "models"
    details_dir.mkdir(exist_ok=True)
    for r in results:
        model_data = {
            "name": r.name,
            "model_type": r.model_type,
            "qwk": r.qwk,
            "hidden_widths": r.hidden_widths,
            "n_features": r.n_features,
            "feature_formulas": r.feature_formulas,
            "full_model": r.full_model,
        }
        out_path = details_dir / f"{r.name}.json"
        out_path.write_text(json.dumps(model_data, indent=2, default=str))

        # Save edge formulas as CSV
        if r.edge_formulas is not None and not r.edge_formulas.empty:
            edge_path = details_dir / f"{r.name}_edges.csv"
            r.edge_formulas.to_csv(edge_path, index=False)


def main():
    print("Loading data...")
    data = load_data(seed=42)
    print(f"Loaded: {data['X_train_outer'].shape[0]} train, "
          f"{len(data['feature_names'])} features")

    configs = build_search_grid()
    print(f"\nRunning {len(configs)} experiments...\n")

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]", end="")
        try:
            result = run_single_experiment(cfg, data)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Print summary table
    print(f"\n{'='*120}")
    print("RESULTS SUMMARY (sorted by score = QWK * %interpretable features)")
    print(f"{'='*120}")
    print(f"{'Name':35s} {'Type':>12s} {'Feat':>4s} {'Widths':>8s} "
          f"{'QWK':>6s} {'EdgeR²':>6s} {'FeatR²':>6s} {'F%Cln':>5s} {'Score':>6s}")
    print("-" * 120)

    scored = sorted(results, key=lambda r: r.score, reverse=True)
    for r in scored:
        w_str = "x".join(str(w) for w in r.hidden_widths)
        print(f"{r.name:35s} {r.model_type:>12s} {r.n_features:>4d} {w_str:>8s} "
              f"{r.qwk:>6.4f} {r.edge_mean_r2:>6.4f} {r.feature_mean_r2:>6.4f} "
              f"{r.feature_pct_clean:>4.0f}% {r.score:>6.4f}")

    save_results(results, OUTPUT_DIR)

    # Highlight best per type
    gam_results = [r for r in scored if "GAM" in r.model_type]
    kan_results = [r for r in scored if "KAN" in r.model_type]

    if gam_results:
        best_gam = gam_results[0]
        print(f"\nBEST GAM: {best_gam.name} — QWK={best_gam.qwk:.4f}, "
              f"Feature R²={best_gam.feature_mean_r2:.4f}")
    if kan_results:
        best_kan = kan_results[0]
        print(f"BEST KAN: {best_kan.name} — QWK={best_kan.qwk:.4f}, "
              f"Feature R²={best_kan.feature_mean_r2:.4f}")
    if gam_results and kan_results:
        delta = best_kan.qwk - best_gam.qwk
        print(f"KAN advantage: {delta:+.4f} QWK ({delta/best_gam.qwk*100:+.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify the complete script is syntactically correct**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python -c "import ast; ast.parse(open('scripts/kan_vs_gam_search.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/kan_vs_gam_search.py
git commit -m "feat: complete kan vs gam grid search with main loop and results saving"
```

---

### Task 7: Run the Grid Search

**Files:**
- No files modified; this runs the script.

- [ ] **Step 1: Run the grid search**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python scripts/kan_vs_gam_search.py`

Expected: ~63 experiments training sequentially (~35-45 minutes total). Each prints QWK, edge stats, and feature stats. Final summary table shows all results sorted by composite score.

- [ ] **Step 2: Verify outputs exist**

Run: `ls -la outputs/kan_vs_gam/search_results.csv outputs/kan_vs_gam/models/`

Expected: `search_results.csv` plus one `.json` and one `_edges.csv` per model.

- [ ] **Step 3: Commit outputs**

```bash
git add outputs/kan_vs_gam/
git commit -m "data: kan vs gam grid search results (63 configs)"
```

---

### Task 8: Plotting Script — Actuary-Facing Comparison Figures

**Files:**
- Create: `scripts/plot_kan_vs_gam.py`

- [ ] **Step 1: Create the plotting script with Figure 1 — Model Comparison Table**

```python
#!/usr/bin/env python3
"""Plot KAN vs GAM comparison figures for actuary audience.

Reads results from outputs/kan_vs_gam/ and produces:
  Fig 1: Model comparison table (GAM vs KAN best models)
  Fig 2: QWK vs Feature R² scatter (all configs, colored by type)
  Fig 3: Per-feature risk curves side-by-side (best GAM vs best KAN)
  Fig 4: Per-edge formula catalog for best KAN (layer 1 + layer 2)

Usage:
    uv run python scripts/plot_kan_vs_gam.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

INPUT_DIR = Path("outputs/kan_vs_gam")
OUTPUT_DIR = INPUT_DIR / "figures"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.interpretability.utils.style import apply_paper_style


def load_results() -> pd.DataFrame:
    return pd.read_csv(INPUT_DIR / "search_results.csv")


def load_model_detail(name: str) -> dict:
    path = INPUT_DIR / "models" / f"{name}.json"
    return json.loads(path.read_text())


def find_best(df: pd.DataFrame, model_type_prefix: str) -> pd.Series:
    """Find the best model of a given type by composite score."""
    mask = df["model_type"].str.startswith(model_type_prefix)
    subset = df[mask].sort_values("score", ascending=False)
    return subset.iloc[0]


# ── Figure 1: Comparison table as image ──────────────────────────────────────

def fig1_comparison_table(df: pd.DataFrame) -> Path:
    """Render a GAM vs KAN comparison table as a publication-quality image."""
    apply_paper_style()

    best_gam = find_best(df, "GAM")
    best_kan = find_best(df, "KAN")

    metrics = [
        ("Architecture", best_gam["hidden_widths"], best_kan["hidden_widths"]),
        ("Features", int(best_gam["n_features"]), int(best_kan["n_features"])),
        ("QWK", f"{best_gam['qwk']:.4f}", f"{best_kan['qwk']:.4f}"),
        ("Active edges", int(best_gam["n_active_edges"]), int(best_kan["n_active_edges"])),
        ("Edge mean R²", f"{best_gam['edge_mean_r2']:.4f}", f"{best_kan['edge_mean_r2']:.4f}"),
        ("Edge % clean", f"{best_gam['edge_pct_clean']:.0f}%", f"{best_kan['edge_pct_clean']:.0f}%"),
        ("Feature mean R²", f"{best_gam['feature_mean_r2']:.4f}", f"{best_kan['feature_mean_r2']:.4f}"),
        ("Feature % clean", f"{best_gam['feature_pct_clean']:.0f}%", f"{best_kan['feature_pct_clean']:.0f}%"),
        ("Captures interactions", "No", "Yes"),
        ("Per-feature formula", "Exact", "Conditional"),
        ("Full model formula", "Exact", "Exact"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    col_labels = ["Metric", "GAM (1-layer)", "KAN (2-layer)"]
    cell_text = [[m[0], str(m[1]), str(m[2])] for m in metrics]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header
    for j in range(3):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight QWK row
    for j in range(3):
        table[3, j].set_facecolor("#E8F5E9")

    plt.title("KAN vs GAM — Best Model Comparison", fontsize=13,
              fontweight="bold", pad=20)
    plt.tight_layout()

    out = OUTPUT_DIR / "fig1_comparison_table.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Fig 1 → {out}")
    return out
```

- [ ] **Step 2: Add Figure 2 — QWK vs Feature R² scatter plot**

```python
def fig2_accuracy_vs_interpretability(df: pd.DataFrame) -> Path:
    """Scatter: QWK (y) vs feature mean R² (x), colored by model type."""
    apply_paper_style()

    fig, ax = plt.subplots(figsize=(8, 6))

    gam_mask = df["model_type"].str.startswith("GAM")
    kan_mask = df["model_type"].str.startswith("KAN")

    ax.scatter(
        df.loc[gam_mask, "feature_mean_r2"],
        df.loc[gam_mask, "qwk"],
        c="#4C72B0", label="GAM (1-layer)", s=60, alpha=0.7, edgecolors="white",
    )
    ax.scatter(
        df.loc[kan_mask, "feature_mean_r2"],
        df.loc[kan_mask, "qwk"],
        c="#55A868", label="KAN (2-layer)", s=60, alpha=0.7, marker="D",
        edgecolors="white",
    )

    # Annotate best of each type
    best_gam = find_best(df, "GAM")
    best_kan = find_best(df, "KAN")
    for best, color in [(best_gam, "#4C72B0"), (best_kan, "#55A868")]:
        ax.annotate(
            best["name"],
            (best["feature_mean_r2"], best["qwk"]),
            textcoords="offset points", xytext=(10, 5),
            fontsize=7, color=color, fontweight="bold",
        )

    ax.set_xlabel("Per-Feature Formula R² (interpretability)", fontsize=10)
    ax.set_ylabel("QWK (accuracy)", fontsize=10)
    ax.set_title("Accuracy vs Interpretability — All Configurations", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()

    out = OUTPUT_DIR / "fig2_accuracy_vs_interpretability.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Fig 2 → {out}")
    return out
```

- [ ] **Step 3: Add Figure 3 — Per-feature risk curves side-by-side**

```python
def fig3_feature_risk_curves(df: pd.DataFrame) -> Path:
    """Side-by-side per-feature risk curves: best GAM vs best KAN."""
    apply_paper_style()

    best_gam_row = find_best(df, "GAM")
    best_kan_row = find_best(df, "KAN")

    gam_detail = load_model_detail(best_gam_row["name"])
    kan_detail = load_model_detail(best_kan_row["name"])

    gam_feats = {f["feature"]: f for f in gam_detail["feature_formulas"]}
    kan_feats = {f["feature"]: f for f in kan_detail["feature_formulas"]}

    # Use features present in both, sorted by KAN y_range (importance)
    common_feats = sorted(
        [f for f in gam_feats if f in kan_feats and gam_feats[f]["is_active"]],
        key=lambda f: kan_feats[f].get("y_range", 0),
        reverse=True,
    )[:12]  # top 12 for a 3x4 grid

    ncols = 4
    nrows = (len(common_feats) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    axes_flat = axes.flatten() if nrows > 1 else (axes if ncols > 1 else [axes])

    for i, feat in enumerate(common_feats):
        ax = axes_flat[i]
        gam_f = gam_feats[feat]
        kan_f = kan_feats[feat]

        gam_tier = gam_f["quality_tier"]
        kan_tier = kan_f["quality_tier"]

        ax.text(
            0.02, 0.98,
            f"GAM: {gam_f['formula']}\n  R²={gam_f['r_squared']:.4f} [{gam_tier}]"
            f"\nKAN: {kan_f['formula']}\n  R²={kan_f['r_squared']:.4f} [{kan_tier}]",
            transform=ax.transAxes, fontsize=6, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        ax.set_title(feat, fontsize=9, fontweight="bold")
        ax.set_xlabel("tanh(x)", fontsize=7)
        ax.set_ylabel("Risk contribution", fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Per-Feature Risk Formulas — GAM ({best_gam_row['name']}) vs KAN ({best_kan_row['name']})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    out = OUTPUT_DIR / "fig3_feature_risk_curves.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Fig 3 → {out}")
    return out
```

- [ ] **Step 4: Add Figure 4 — Per-edge formula catalog + main function**

```python
def fig4_edge_catalog(df: pd.DataFrame) -> Path:
    """Show all per-edge formulas for the best KAN model in a compact table."""
    apply_paper_style()

    best_kan_row = find_best(df, "KAN")
    edges_path = INPUT_DIR / "models" / f"{best_kan_row['name']}_edges.csv"
    if not edges_path.exists():
        print(f"  Skipping Fig 4: no edges CSV for {best_kan_row['name']}")
        return Path()

    edges_df = pd.read_csv(edges_path)

    # Separate layer 0 (input features) and layer 1 (hidden→hidden2)
    l0 = edges_df[edges_df["layer"] == 0].sort_values("l1_norm", ascending=False)
    l1 = edges_df[edges_df["layer"] == 1].sort_values("l1_norm", ascending=False)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, max(6, len(l0) * 0.3)))

    for ax, layer_df, title in [
        (ax0, l0, "Layer 0: Input → Hidden"),
        (ax1, l1, "Layer 1: Hidden → Output"),
    ]:
        ax.axis("off")
        if layer_df.empty:
            ax.text(0.5, 0.5, "No active edges", ha="center", va="center")
            ax.set_title(title)
            continue

        cell_text = []
        for _, row in layer_df.iterrows():
            tier_sym = {"clean": "+", "acceptable": "~", "flagged": "X"}[row["quality_tier"]]
            cell_text.append([
                row["input_feature"],
                f"→ h{int(row['edge_out'])}",
                row["formula"],
                f"{row['r_squared']:.4f}",
                tier_sym,
                f"{row['l1_norm']:.4f}",
            ])

        table = ax.table(
            cellText=cell_text,
            colLabels=["Input", "Output", "Formula", "R²", "Q", "L1"],
            loc="center", cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)
        for j in range(6):
            table[0, j].set_facecolor("#2C3E50")
            table[0, j].set_text_props(color="white", fontweight="bold")
        ax.set_title(title, fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Edge Formula Catalog — {best_kan_row['name']} (QWK={best_kan_row['qwk']:.4f})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    out = OUTPUT_DIR / "fig4_edge_catalog.pdf"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Fig 4 → {out}")
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    print(f"Loaded {len(df)} results from {INPUT_DIR / 'search_results.csv'}")

    fig1_comparison_table(df)
    fig2_accuracy_vs_interpretability(df)
    fig3_feature_risk_curves(df)
    fig4_edge_catalog(df)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Verify the complete script is syntactically correct**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python -c "import ast; ast.parse(open('scripts/plot_kan_vs_gam.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/plot_kan_vs_gam.py
git commit -m "feat: actuary-facing KAN vs GAM comparison plots (4 figures)"
```

---

### Task 9: Run Plotting & Verify Outputs

**Files:**
- No files modified; runs the plotting script.

- [ ] **Step 1: Run the plotting script**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python scripts/plot_kan_vs_gam.py`

Expected: 4 PDF figures generated in `outputs/kan_vs_gam/figures/`.

- [ ] **Step 2: Verify all outputs exist**

Run: `ls -la outputs/kan_vs_gam/figures/`

Expected:
```
fig1_comparison_table.pdf
fig2_accuracy_vs_interpretability.pdf
fig3_feature_risk_curves.pdf
fig4_edge_catalog.pdf
```

- [ ] **Step 3: Commit figures**

```bash
git add outputs/kan_vs_gam/figures/
git commit -m "data: kan vs gam comparison figures"
```

---

### Task 10: Review & Iterate

- [ ] **Step 1: Read the search results CSV and evaluate**

Run: `cd /Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning && uv run python -c "
import pandas as pd
df = pd.read_csv('outputs/kan_vs_gam/search_results.csv')
print('=== TOP 5 GAM ===')
gam = df[df['model_type'].str.startswith('GAM')].sort_values('score', ascending=False)
print(gam[['name','qwk','feature_mean_r2','feature_pct_clean','score']].head().to_string())
print('\n=== TOP 5 KAN ===')
kan = df[df['model_type'].str.startswith('KAN')].sort_values('score', ascending=False)
print(kan[['name','qwk','feature_mean_r2','feature_pct_clean','score']].head().to_string())
print(f'\nBest GAM QWK: {gam.iloc[0][\"qwk\"]:.4f}')
print(f'Best KAN QWK: {kan.iloc[0][\"qwk\"]:.4f}')
print(f'KAN advantage: {kan.iloc[0][\"qwk\"] - gam.iloc[0][\"qwk\"]:+.4f}')
"`

Expected: Summary of top models from each type, with QWK delta showing KAN advantage.

- [ ] **Step 2: Open figures and verify visual quality**

Visually inspect the 4 PDFs in `outputs/kan_vs_gam/figures/`. Check that:
- Fig 1 table renders with clear GAM vs KAN comparison
- Fig 2 scatter shows separation between GAM (blue) and KAN (green)
- Fig 3 shows per-feature formulas with readable annotations
- Fig 4 shows edge catalog with quality tiers

- [ ] **Step 3: If figures need adjustment, iterate on `scripts/plot_kan_vs_gam.py`**

Common adjustments: font sizes, layout spacing, color choices, annotation positions.
