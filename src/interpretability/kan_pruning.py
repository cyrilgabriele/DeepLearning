"""Issues 04 & 05 — Edge pruning for ChebyKAN and FourierKAN.

Removes mathematically insignificant edges (those whose mean absolute
activation over the input domain is below a threshold) and verifies the
QWK drop stays within tolerance.

Usage:
    uv run python -m src.interpretability.kan_pruning \
        --checkpoint checkpoints/stage-b-chebykan-.../model-<timestamp>.pt \
        --config    configs/experiment_stages/stage_c_explanation_package/materialized/chebykan_best_interpretable.yaml \
        --flavor    chebykan
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from src.config import ExperimentConfig


# ── Edge magnitude ────────────────────────────────────────────────────────────

def _edge_l1_chebykan(layer, n_samples: int = 1000) -> torch.Tensor:
    """Mean absolute output (L1 norm) for each (out, in) edge of a ChebyKANLayer.

    Uses the criterion from Liu et al. (2024) arXiv:2404.19756: edges are ranked
    by ||φ||_1 = (1/n) Σ |φ(xᵢ)|, which measures the functional magnitude of the
    activation rather than its variance.
    """
    x = torch.linspace(-1.0, 1.0, n_samples)
    x_norm = torch.tanh(x)

    cheby = [torch.ones(n_samples), x_norm]
    for _ in range(2, layer.degree + 1):
        cheby.append(2 * x_norm * cheby[-1] - cheby[-2])
    basis = torch.stack(cheby, dim=-1)  # (n_samples, degree+1)

    # coeffs: (out, in, degree+1)  →  edge output: (n_samples, out, in)
    edge_out = torch.einsum("sd,oid->soi", basis, layer.cheby_coeffs.detach())
    base_contrib = layer.base_weight.detach().unsqueeze(0) * x_norm[:, None, None]
    edge_out = edge_out + base_contrib
    return edge_out.abs().mean(dim=0)  # (out, in) — L1 norm per edge


def _edge_l1_fourierkan(layer, n_samples: int = 1000) -> torch.Tensor:
    """Mean absolute output (L1 norm) for each (out, in) edge of a FourierKANLayer.

    Uses the criterion from Liu et al. (2024) arXiv:2404.19756.
    """
    import math
    x = torch.linspace(-1.0, 1.0, n_samples)
    x_scaled = (torch.tanh(x) + 1) * math.pi  # (n_samples,)

    k = torch.arange(1, layer.grid_size + 1, dtype=torch.float32)
    x_k = x_scaled.unsqueeze(-1) * k  # (n_samples, grid_size)
    cos_b = torch.cos(x_k)
    sin_b = torch.sin(x_k)

    # fourier_a/b: (out, in, grid_size)
    cos_out = torch.einsum("sg,oig->soi", cos_b, layer.fourier_a.detach())
    sin_out = torch.einsum("sg,oig->soi", sin_b, layer.fourier_b.detach())
    base_contrib = layer.base_weight.detach().unsqueeze(0) * x.tanh()[:, None, None]
    edge_out = cos_out + sin_out + base_contrib
    return edge_out.abs().mean(dim=0)  # (out, in) — L1 norm per edge


def _compute_edge_l1(layer) -> torch.Tensor:
    """Return per-edge L1 norm tensor (out, in) for any supported KAN layer.

    Implements the activation magnitude criterion from Liu et al. (2024)
    arXiv:2404.19756 (§2.5): ||φ||_1 = (1/n) Σ|φ(xᵢ)|.
    """
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    if isinstance(layer, ChebyKANLayer):
        return _edge_l1_chebykan(layer)
    if isinstance(layer, FourierKANLayer):
        return _edge_l1_fourierkan(layer)
    raise TypeError(f"Unsupported layer type: {type(layer)}")


# ── Pruning ───────────────────────────────────────────────────────────────────

class PruningStats(NamedTuple):
    threshold: float
    edges_before: int
    edges_after: int
    sparsity_ratio: float


def prune_kan(model, threshold: float = 0.01) -> tuple:
    """Zero out edges whose activation L1 norm is below *threshold*.

    Returns (pruned_model, stats, edge_masks).
    edge_masks is a list of boolean tensors (out, in) — True = edge is active.
    """
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer

    pruned = copy.deepcopy(model)
    total_before = 0
    total_after = 0
    masks = []

    for layer in pruned.kan_layers:
        if not isinstance(layer, (ChebyKANLayer, FourierKANLayer)):
            masks.append(None)
            continue

        l1_scores = _compute_edge_l1(layer)  # (out, in) — L1 norm per edge
        mask = l1_scores >= threshold  # True = keep

        total_before += mask.numel()
        total_after += int(mask.sum().item())
        masks.append(mask)

        with torch.no_grad():
            if isinstance(layer, ChebyKANLayer):
                # cheby_coeffs: (out, in, degree+1)
                layer.cheby_coeffs *= mask.unsqueeze(-1)
            else:
                # fourier_a/b: (out, in, grid_size)
                layer.fourier_a *= mask.unsqueeze(-1)
                layer.fourier_b *= mask.unsqueeze(-1)
            layer.base_weight *= mask

    sparsity = 1.0 - (total_after / total_before) if total_before > 0 else 0.0
    stats = PruningStats(
        threshold=threshold,
        edges_before=total_before,
        edges_after=total_after,
        sparsity_ratio=round(sparsity, 4),
    )
    return pruned, stats, masks


# ── QWK evaluation ────────────────────────────────────────────────────────────

def _evaluate_qwk(model_wrapper, X_eval: pd.DataFrame, y_eval: pd.Series) -> float:
    preds = model_wrapper.predict(X_eval)
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_eval, preds, weights="quadratic"))


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    checkpoint_path: Path,
    config: ExperimentConfig,
    flavor: str,
    eval_features_path: Path = Path("outputs/X_eval.parquet"),
    eval_labels_path: Path = Path("outputs/y_eval.parquet"),
    threshold: float = 0.01,
    qwk_tolerance: float = 0.01,
    output_dir: Path = Path("outputs"),
) -> dict:
    from src.models.tabkan import TabKANClassifier, TabKAN

    X_eval = pd.read_parquet(eval_features_path)
    y_eval = pd.read_parquet(eval_labels_path)["Response"]

    # Reconstruct the wrapper and load weights
    in_features = X_eval.shape[1]
    wrapper = TabKANClassifier(
        preset=config.model.name,
        flavor=flavor,
        hidden_widths=config.model.resolved_hidden_widths(),
        depth=config.model.depth,
        width=config.model.width,
        degree=config.model.degree or 3,
    )
    widths = wrapper.widths
    wrapper.module = TabKAN(
        in_features=in_features,
        widths=widths,
        kan_type=flavor,
        degree=wrapper.degree,
        grid_size=wrapper.grid_size,
    )
    wrapper.module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    wrapper.module.eval()

    qwk_before = _evaluate_qwk(wrapper, X_eval, y_eval)
    print(f"QWK before pruning: {qwk_before:.4f}")

    # Sweep threshold until QWK drop ≤ tolerance
    best_threshold = threshold
    pruned_module, stats, masks = prune_kan(wrapper.module, threshold)
    wrapper_pruned = copy.deepcopy(wrapper)
    wrapper_pruned.module = pruned_module

    qwk_after = _evaluate_qwk(wrapper_pruned, X_eval, y_eval)
    qwk_drop = qwk_before - qwk_after

    if qwk_drop > qwk_tolerance:
        print(f"QWK drop {qwk_drop:.4f} exceeds tolerance {qwk_tolerance}. Tightening threshold…")
        for t in [threshold / 2, threshold / 5, threshold / 10, 0.001]:
            pm, st, mk = prune_kan(wrapper.module, t)
            wt = copy.deepcopy(wrapper)
            wt.module = pm
            q = _evaluate_qwk(wt, X_eval, y_eval)
            drop = qwk_before - q
            print(f"  threshold={t:.5f}  edges_after={st.edges_after}  qwk={q:.4f}  drop={drop:.4f}")
            if drop <= qwk_tolerance:
                best_threshold, pruned_module, stats, masks = t, pm, st, mk
                wrapper_pruned.module = pm
                qwk_after = q
                qwk_drop = drop
                break

    print(f"QWK after pruning:  {qwk_after:.4f}  (drop={qwk_drop:.4f})")
    print(f"Edges: {stats.edges_before} → {stats.edges_after}  sparsity={stats.sparsity_ratio:.2%}")

    result = {
        "flavor": flavor,
        "threshold": best_threshold,
        "edges_before": stats.edges_before,
        "edges_after": stats.edges_after,
        "sparsity_ratio": stats.sparsity_ratio,
        "qwk_before": round(qwk_before, 6),
        "qwk_after": round(qwk_after, 6),
        "qwk_drop": round(qwk_drop, 6),
    }

    from src.interpretability.utils.paths import reports as rep_dir, models as mod_dir

    out_path = rep_dir(output_dir) / f"{flavor}_pruning_summary.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved → {out_path}")

    # Save pruned checkpoint for symbolic regression
    pruned_ckpt = mod_dir(output_dir) / f"{flavor}_pruned_module.pt"
    torch.save(pruned_module.state_dict(), pruned_ckpt)
    print(f"Saved pruned module → {pruned_ckpt}")

    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KAN edge pruning")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--flavor", choices=["chebykan", "fourierkan"], required=True)
    p.add_argument("--eval-features", type=Path, default=Path("outputs/data/X_eval.parquet"))
    p.add_argument("--eval-labels", type=Path, default=Path("outputs/data/y_eval.parquet"))
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    from src.config import load_experiment_config

    run(
        args.checkpoint,
        load_experiment_config(args.config),
        args.flavor,
        args.eval_features,
        args.eval_labels,
        args.threshold,
        output_dir=args.output_dir,
    )
