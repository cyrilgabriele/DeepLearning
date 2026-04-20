#!/usr/bin/env python3
"""Train interpretable KAN without LayerNorm for exact input→output formulas.

Without LayerNorm the model is a pure additive model:
    prediction = Σᵢ fᵢ(xᵢ) + bias
where fᵢ(xᵢ) = Σⱼ wⱼ · φⱼᵢ(xᵢ)  (linear head weight × edge function).

This gives one closed-form formula per input feature, composable end-to-end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.preprocess_kan_paper import KANPreprocessor
from src.models.tabkan import TabKAN
from src.models.kan_layers import ChebyKANLayer
from src.interpretability.kan_pruning import _compute_edge_l1
from src.interpretability.kan_symbolic import sample_edge, fit_symbolic_edge, _quality_tier
from src.metrics.qwk import quadratic_weighted_kappa


TOP_FEATURES = [
    "BMI", "Medical_Keyword_3", "Product_Info_4", "Wt",
    "Medical_History_20", "Medical_Keyword_38", "Employment_Info_6",
    "Medical_History_4", "Medical_History_30", "Medical_Keyword_29",
    "Medical_Keyword_13", "Medical_Keyword_47", "Medical_Keyword_46",
    "Medical_Keyword_31", "Medical_Keyword_35", "Ins_Age",
    "Medical_Keyword_12", "Medical_Keyword_14", "Medical_Keyword_43",
    "Medical_Keyword_5",
]


def load_data(seed: int = 42) -> dict:
    csv_path = Path("data/prudential-life-insurance-assessment/train.csv")
    preprocessor = KANPreprocessor()
    return preprocessor.run_pipeline(csv_path, random_seed=seed)


def select_features(data, n_features, top_features=TOP_FEATURES):
    all_names = data["feature_names"]
    keep = top_features[:n_features]
    keep_indices = [all_names.index(f) for f in keep if f in all_names]
    kept_names = [all_names[i] for i in keep_indices]
    idx = np.array(keep_indices)
    return (
        data["X_train_outer"][:, idx],
        data["X_test_outer"][:, idx],
        data["y_train_outer"],
        data["y_test_outer"],
        kept_names,
    )


def train_no_layernorm(
    X_train, X_test, y_train, y_test,
    widths, degree=3, sparsity_lambda=0.005,
    lr=5e-3, weight_decay=5e-4, max_epochs=150,
):
    L.seed_everything(42)
    in_features = X_train.shape[1]

    module = TabKAN(
        in_features=in_features,
        widths=widths,
        kan_type="chebykan",
        degree=degree,
        lr=lr,
        weight_decay=weight_decay,
        sparsity_lambda=sparsity_lambda,
        l1_weight=1.0,
        entropy_weight=1.0,
        use_layer_norm=False,
    )

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_t, y_t.unsqueeze(1)),
        batch_size=2048, shuffle=True,
    )
    X_val_t = torch.tensor(X_test, dtype=torch.float32)
    y_val_t = torch.tensor(y_test, dtype=torch.float32)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t.unsqueeze(1)),
        batch_size=2048, shuffle=False,
    )

    trainer = L.Trainer(
        max_epochs=max_epochs, accelerator="auto",
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


def compose_input_to_output(module, feature_names, pruning_threshold=0.005):
    """Compose exact input→output formulas through the linear head.

    output = Σⱼ w_j · (Σᵢ φ_{j,i}(xᵢ)) + bias
           = Σᵢ (Σⱼ w_j · φ_{j,i}(xᵢ)) + bias
           = Σᵢ fᵢ(xᵢ) + bias

    For each feature i, sample the weighted sum fᵢ(x) = Σⱼ w_j · φ_{j,i}(x)
    and fit a symbolic formula.
    """
    first_layer = next(l for l in module.kan_layers if isinstance(l, ChebyKANLayer))
    head_weight = module.head.weight.detach().squeeze()  # shape (hidden,)
    head_bias = module.head.bias.detach().item()

    l1_scores = _compute_edge_l1(first_layer)
    n_samples = 1000

    results = []
    for in_i, feat in enumerate(feature_names):
        # Compute the weighted composite: fᵢ(x) = Σⱼ wⱼ · φⱼᵢ(x)
        x_ref = None
        composite_y = None

        active_edges = 0
        total_l1 = 0.0
        for out_j in range(first_layer.out_features):
            if l1_scores[out_j, in_i].item() < pruning_threshold:
                continue
            active_edges += 1
            total_l1 += l1_scores[out_j, in_i].item()

            x_vals, y_vals = sample_edge(first_layer, out_j, in_i, n=n_samples)
            w_j = head_weight[out_j].item()

            if composite_y is None:
                x_ref = x_vals
                composite_y = w_j * y_vals
            else:
                composite_y += w_j * y_vals

        if composite_y is None:
            results.append({
                "feature": feat,
                "active_edges": 0,
                "total_l1": 0.0,
                "formula": "0 (pruned)",
                "r_squared": 1.0,
                "quality_tier": "clean",
                "params": None,
                "composite_y_range": 0.0,
            })
            continue

        # Fit symbolic formula to the composite function
        formula, r2 = fit_symbolic_edge(x_ref, composite_y)

        # Refit to get actual parameters
        candidates = {
            "a*x + b": (lambda x, a, b: a*x + b, ["a", "b"]),
            "a*x^2 + b*x + c": (lambda x, a, b, c: a*x**2 + b*x + c, ["a", "b", "c"]),
            "a*x^3 + b*x^2 + c*x + d": (lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, ["a", "b", "c", "d"]),
            "a*|x| + b": (lambda x, a, b: a*np.abs(x) + b, ["a", "b"]),
            "a*cos(x) + b": (lambda x, a, b: a*np.cos(x) + b, ["a", "b"]),
            "a*sin(x) + b": (lambda x, a, b: a*np.sin(x) + b, ["a", "b"]),
            "a*sin(2*x) + b": (lambda x, a, b: a*np.sin(2*x) + b, ["a", "b"]),
            "a*sin(2*x) + b*cos(2*x)": (lambda x, a, b: a*np.sin(2*x) + b*np.cos(2*x), ["a", "b"]),
            "a*sin(x) + b*cos(x)": (lambda x, a, b: a*np.sin(x) + b*np.cos(x), ["a", "b"]),
            "a*exp(x) + b": (lambda x, a, b: a*np.exp(np.clip(x, -5, 5)) + b, ["a", "b"]),
            "a*log(|x|+1) + b": (lambda x, a, b: a*np.log(np.abs(x)+1) + b, ["a", "b"]),
            "a*sqrt(|x|) + b": (lambda x, a, b: a*np.sqrt(np.abs(x)) + b, ["a", "b"]),
            "a (constant)": (lambda x, a: np.full_like(x, a), ["a"]),
        }

        params = None
        if formula in candidates:
            func, pnames = candidates[formula]
            try:
                popt, _ = curve_fit(func, x_ref, composite_y, p0=[1.0]*len(pnames), maxfev=3000)
                params = {n: round(float(v), 6) for n, v in zip(pnames, popt)}
            except Exception:
                pass

        results.append({
            "feature": feat,
            "active_edges": active_edges,
            "total_l1": round(total_l1, 4),
            "formula": formula,
            "r_squared": round(r2, 6),
            "quality_tier": _quality_tier(r2),
            "params": params,
            "composite_y_range": round(float(composite_y.max() - composite_y.min()), 4),
        })

    return results, head_bias


def format_formula(result):
    """Format a result into a human-readable formula string."""
    f = result["formula"]
    p = result["params"]
    if p is None:
        return f

    if f == "a*x^3 + b*x^2 + c*x + d":
        return f"{p['a']:.4f}·x³ + {p['b']:.4f}·x² + {p['c']:.4f}·x + {p['d']:.4f}"
    elif f == "a*x^2 + b*x + c":
        return f"{p['a']:.4f}·x² + {p['b']:.4f}·x + {p['c']:.4f}"
    elif f == "a*x + b":
        return f"{p['a']:.4f}·x + {p['b']:.4f}"
    elif f == "a*cos(x) + b":
        return f"{p['a']:.4f}·cos(x) + {p['b']:.4f}"
    elif f == "a*sin(2*x) + b*cos(2*x)":
        return f"{p['a']:.4f}·sin(2x) + {p['b']:.4f}·cos(2x)"
    elif f == "a*sin(2*x) + b":
        return f"{p['a']:.4f}·sin(2x) + {p['b']:.4f}"
    elif f == "a*sin(x) + b*cos(x)":
        return f"{p['a']:.4f}·sin(x) + {p['b']:.4f}·cos(x)"
    elif f == "a*sin(x) + b":
        return f"{p['a']:.4f}·sin(x) + {p['b']:.4f}"
    elif f == "a*exp(x) + b":
        return f"{p['a']:.4f}·exp(x) + {p['b']:.4f}"
    elif f == "a*log(|x|+1) + b":
        return f"{p['a']:.4f}·log(|x|+1) + {p['b']:.4f}"
    elif f == "a*sqrt(|x|) + b":
        return f"{p['a']:.4f}·sqrt(|x|) + {p['b']:.4f}"
    elif f == "a (constant)":
        return f"{p['a']:.4f}"
    elif f == "a*|x| + b":
        return f"{p['a']:.4f}·|x| + {p['b']:.4f}"
    return f


def main():
    print("Loading data...")
    data = load_data(seed=42)

    # Test configurations: vary width and features
    configs = [
        (20, [16], 3, 0.005, 150),
        (20, [8],  3, 0.005, 150),
        (15, [16], 3, 0.005, 150),
        (15, [8],  3, 0.005, 150),
        (10, [16], 3, 0.005, 150),
        (10, [8],  3, 0.005, 150),
        (20, [16], 3, 0.01,  150),  # stronger sparsity
        (20, [16], 3, 0.001, 150),  # weaker sparsity
    ]

    best_result = None
    best_score = -1

    for n_feat, widths, degree, sp, epochs in configs:
        w_str = "x".join(str(w) for w in widths)
        name = f"f{n_feat}_w{w_str}_d{degree}_sp{sp}_noLN"
        print(f"\n{'='*80}")
        print(f"Experiment: {name}")
        print(f"{'='*80}")

        X_train, X_test, y_train, y_test, feat_names = select_features(data, n_feat)
        module, qwk, t = train_no_layernorm(
            X_train, X_test, y_train, y_test,
            widths=widths, degree=degree, sparsity_lambda=sp, max_epochs=epochs,
        )
        print(f"  QWK = {qwk:.4f}  (train time: {t:.1f}s)")

        formulas, bias = compose_input_to_output(module, feat_names)

        n_clean = sum(1 for r in formulas if r["quality_tier"] == "clean")
        n_accept = sum(1 for r in formulas if r["quality_tier"] == "acceptable")
        n_flag = sum(1 for r in formulas if r["quality_tier"] == "flagged")
        n_active = sum(1 for r in formulas if r["active_edges"] > 0)
        mean_r2 = np.mean([r["r_squared"] for r in formulas if r["active_edges"] > 0])

        print(f"  Active features: {n_active}/{n_feat}")
        print(f"  Clean: {n_clean}, Acceptable: {n_accept}, Flagged: {n_flag}")
        print(f"  Mean R²: {mean_r2:.4f}")
        print(f"  Bias: {bias:.4f}")

        # Score: QWK × fraction interpretable (clean+acceptable)
        pct_interp = (n_clean + n_accept) / max(n_active, 1) * 100
        score = qwk * pct_interp / 100

        print(f"\n  FULL MODEL (input → output):")
        print(f"  prediction = bias + Σ fᵢ(xᵢ)")
        print(f"  bias = {bias:.4f}")
        print()

        for r in sorted(formulas, key=lambda r: r["composite_y_range"], reverse=True):
            if r["active_edges"] == 0:
                continue
            tier_sym = {"clean": "✓", "acceptable": "~", "flagged": "✗"}[r["quality_tier"]]
            readable = format_formula(r)
            print(f"  [{tier_sym}] f({r['feature']:25s}) = {readable:45s}  "
                  f"R²={r['r_squared']:.4f}  range={r['composite_y_range']:.3f}  "
                  f"edges={r['active_edges']}")

        pruned = [r for r in formulas if r["active_edges"] == 0]
        if pruned:
            print(f"\n  Pruned (zero contribution): {', '.join(r['feature'] for r in pruned)}")

        if score > best_score:
            best_score = score
            best_result = {
                "name": name, "qwk": qwk, "formulas": formulas, "bias": bias,
                "feat_names": feat_names, "n_feat": n_feat, "widths": widths,
                "pct_interp": pct_interp, "mean_r2": mean_r2,
            }

    # Final summary
    print(f"\n{'='*80}")
    print(f"BEST: {best_result['name']}")
    print(f"  QWK={best_result['qwk']:.4f}, Interpretable={best_result['pct_interp']:.1f}%")
    print(f"  Mean R²={best_result['mean_r2']:.4f}")
    print(f"{'='*80}")

    # Save best formulas
    output_dir = Path("outputs/interpretable_kan_no_layernorm")
    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    formulas_out = []
    for r in best_result["formulas"]:
        formulas_out.append({
            "feature": r["feature"],
            "formula_type": r["formula"],
            "formula_readable": format_formula(r),
            "r_squared": r["r_squared"],
            "quality_tier": r["quality_tier"],
            "params": r["params"],
            "active_edges": r["active_edges"],
            "total_l1": r["total_l1"],
            "output_range": r["composite_y_range"],
        })
    payload = {
        "model": best_result["name"],
        "qwk": best_result["qwk"],
        "bias": best_result["bias"],
        "n_features": best_result["n_feat"],
        "hidden_widths": best_result["widths"],
        "feature_formulas": formulas_out,
    }
    out_path = output_dir / "best_input_to_output_formulas.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
