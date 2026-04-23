"""Autograd-based Hessian heatmap for the no-LayerNorm ChebyKAN.

Computes per-row Hessians of the scalar model output with respect to the input
features via torch.autograd.functional.hessian, averages them across a sample
of X_eval rows, and emits:

- Signed-mean Hessian heatmap PNG over all features.
- Absolute-mean Hessian heatmap PNG over all features.
- Signed-mean Hessian heatmap PNG over continuous features only (clean story).
- JSON dump of the aggregated matrices.
- Markdown summary with the top-K pairwise interactions.

Only semantically meaningful for no-LayerNorm ChebyKAN runs; differentiation
is valid for any TabKAN module but discrete-feature entries should be
interpreted as smooth-extension sensitivities rather than true derivatives.

Usage:
    uv run python -m src.interpretability.hessian_heatmap \
        --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import ExperimentConfig, load_experiment_config
from src.interpretability.utils.paths import (
    eval_run_dir,
    interpret_run_dir,
    reports as reports_dir,
)
from src.models.tabkan import TabKAN


_DEFAULT_SAMPLE_SIZE = 1000
_DEFAULT_TOP_K = 15
_ANNOTATION_THRESHOLD = 22


def _load_pruned_module(
    config: ExperimentConfig,
    feature_names: list[str],
    checkpoint_path: Path,
) -> TabKAN:
    module = TabKAN(
        in_features=len(feature_names),
        widths=config.model.resolved_hidden_widths(),
        kan_type=config.model.flavor or "chebykan",
        degree=config.model.degree or 3,
        grid_size=config.model.params.get("grid_size", 4),
        use_layernorm=config.model.use_layernorm,
    )
    module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def _compute_mean_hessian(
    module: TabKAN,
    X_sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_signed, mean_abs, std) Hessian matrices averaged over rows."""
    n_rows, n_features = X_sample.shape
    X_tensor = torch.tensor(X_sample, dtype=torch.float32)

    def scalar_forward(x: torch.Tensor) -> torch.Tensor:
        return module(x.unsqueeze(0)).reshape(())

    sum_signed = np.zeros((n_features, n_features), dtype=np.float64)
    sum_abs = np.zeros((n_features, n_features), dtype=np.float64)
    sum_sq = np.zeros((n_features, n_features), dtype=np.float64)

    for row_idx in range(n_rows):
        row = X_tensor[row_idx].clone().requires_grad_(True)
        hess = torch.autograd.functional.hessian(
            scalar_forward, row, create_graph=False, vectorize=True
        )
        hess_np = hess.detach().cpu().numpy().astype(np.float64)
        # Enforce exact symmetry (autograd produces near-symmetric numerical noise).
        hess_np = 0.5 * (hess_np + hess_np.T)
        sum_signed += hess_np
        sum_abs += np.abs(hess_np)
        sum_sq += hess_np * hess_np

    mean_signed = sum_signed / n_rows
    mean_abs = sum_abs / n_rows
    # Row-level standard deviation of each Hessian entry.
    var = sum_sq / n_rows - mean_signed ** 2
    std = np.sqrt(np.clip(var, 0.0, None))
    return mean_signed, mean_abs, std


def _plot_heatmap(
    matrix: np.ndarray,
    feature_names: list[str],
    title: str,
    output_path: Path,
    *,
    symmetric_colors: bool,
    cmap: str,
) -> None:
    n = len(feature_names)
    fig_size = max(6.0, 0.45 * n + 3.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if symmetric_colors:
        vmax = float(np.max(np.abs(matrix))) if matrix.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        vmin = -vmax
    else:
        vmax = float(np.max(matrix)) if matrix.size else 1.0
        vmin = 0.0

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)

    if n <= _ANNOTATION_THRESHOLD:
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                ref = vmax if vmax > 0 else 1.0
                txt_color = "white" if abs(value) > 0.6 * ref else "black"
                ax.text(
                    j,
                    i,
                    f"{value:+.2f}" if symmetric_colors else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=txt_color,
                )

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _top_k_interactions(
    matrix: np.ndarray,
    feature_names: list[str],
    k: int,
) -> list[dict[str, object]]:
    n = len(feature_names)
    entries: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            entries.append((feature_names[i], feature_names[j], float(matrix[i, j])))
    entries.sort(key=lambda item: abs(item[2]), reverse=True)
    return [
        {"feature_a": a, "feature_b": b, "value": v, "abs_value": abs(v)}
        for a, b, v in entries[:k]
    ]


def _first_order_importance(
    module: TabKAN,
    X_sample: np.ndarray,
) -> np.ndarray:
    """Mean absolute first-order gradient per feature, used for feature ordering."""
    X_tensor = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True)
    y = module(X_tensor).reshape(-1)
    grads_sum = torch.autograd.grad(y.sum(), X_tensor, create_graph=False)[0]
    return grads_sum.abs().mean(dim=0).detach().cpu().numpy()


def _build_markdown(
    *,
    experiment_name: str,
    sample_size: int,
    feature_names: list[str],
    continuous_features: list[str],
    discrete_features: list[str],
    top_k_all: list[dict[str, object]],
    top_k_cont: list[dict[str, object]],
    diagonal: np.ndarray,
    fig_all: str,
    fig_abs: str,
    fig_cont: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# ChebyKAN Hessian heatmap — {experiment_name}\n")
    lines.append(
        "Autograd-based Hessian of the scalar model output with respect to its "
        f"20 input features, averaged across {sample_size} randomly sampled "
        "rows from the evaluation set.\n"
    )
    lines.append(
        "Diagonal entries are second-order self-curvature (gamma). "
        "Off-diagonal entries are pairwise interactions — the sign indicates "
        "whether two features amplify (+) or dampen (-) each other's effect on "
        "the predicted risk score.\n"
    )
    lines.append("## Feature inventory\n")
    lines.append(f"- **Continuous ({len(continuous_features)})**: {', '.join(continuous_features)}")
    lines.append(f"- **Discrete ({len(discrete_features)})**: {len(discrete_features)} categorical/binary features.")
    lines.append(
        "  Discrete-feature derivatives are smooth-extension quantities of the "
        "ChebyKAN formula; they are informative about model reliance but are "
        "not true derivatives in the Black–Scholes sense.\n"
    )

    lines.append("## Figures\n")
    lines.append(f"- Signed mean Hessian, all features: `{fig_all}`")
    lines.append(f"- Absolute mean Hessian, all features: `{fig_abs}`")
    lines.append(f"- Signed mean Hessian, continuous features only: `{fig_cont}`\n")

    lines.append("## Self-curvature (diagonal, gamma)\n")
    lines.append("| Feature | Mean ∂²y/∂xᵢ² |")
    lines.append("|---|---|")
    for name, value in zip(feature_names, diagonal, strict=True):
        lines.append(f"| {name} | {value:+.4f} |")
    lines.append("")

    lines.append(f"## Top-{len(top_k_cont)} pairwise interactions (continuous features only)\n")
    if top_k_cont:
        lines.append("| Rank | Feature pair | Signed mean | \\|value\\| |")
        lines.append("|---|---|---|---|")
        for rank, entry in enumerate(top_k_cont, start=1):
            lines.append(
                f"| {rank} | {entry['feature_a']} × {entry['feature_b']} | "
                f"{entry['value']:+.4f} | {entry['abs_value']:.4f} |"
            )
    else:
        lines.append("_No continuous-continuous feature pairs available._")
    lines.append("")

    lines.append(f"## Top-{len(top_k_all)} pairwise interactions (all features)\n")
    lines.append("| Rank | Feature pair | Signed mean | \\|value\\| |")
    lines.append("|---|---|---|---|")
    for rank, entry in enumerate(top_k_all, start=1):
        lines.append(
            f"| {rank} | {entry['feature_a']} × {entry['feature_b']} | "
            f"{entry['value']:+.4f} | {entry['abs_value']:.4f} |"
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def run_from_config(
    *,
    config_path: Path,
    output_root: Path = Path("outputs"),
    checkpoint_path: Path | None = None,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    top_k: int = _DEFAULT_TOP_K,
    seed: int = 42,
) -> dict[str, object]:
    config = load_experiment_config(config_path)
    recipe = config.preprocessing.recipe
    experiment_name = config.trainer.experiment_name
    flavor = config.model.flavor or "chebykan"

    if config.model.use_layernorm:
        raise ValueError(
            "Hessian heatmap pipeline currently targets no-LayerNorm ChebyKAN "
            "variants only. Re-run with a config that sets model.use_layernorm=false."
        )

    eval_dir = eval_run_dir(output_root, recipe, experiment_name, create=False)
    interpret_dir = interpret_run_dir(output_root, recipe, experiment_name, create=True)

    feature_names = json.loads((eval_dir / "feature_names.json").read_text())
    if not isinstance(feature_names, list):
        raise TypeError("feature_names.json must contain a list.")
    feature_names = [str(item) for item in feature_names]

    feature_types = json.loads((eval_dir / "feature_types.json").read_text())
    if not isinstance(feature_types, dict):
        raise TypeError("feature_types.json must contain an object.")
    feature_types = {str(key): str(value) for key, value in feature_types.items()}

    continuous_features = [f for f in feature_names if feature_types.get(f) == "continuous"]
    discrete_features = [f for f in feature_names if feature_types.get(f) != "continuous"]

    X_eval = pd.read_parquet(eval_dir / "X_eval.parquet").loc[:, feature_names]

    resolved_checkpoint = checkpoint_path or (interpret_dir / "models" / f"{flavor}_pruned_module.pt")
    module = _load_pruned_module(config, feature_names, resolved_checkpoint)

    rng = np.random.default_rng(seed)
    n_available = len(X_eval)
    effective_sample = min(sample_size, n_available)
    sample_idx = rng.choice(n_available, size=effective_sample, replace=False)
    X_sample = X_eval.iloc[sample_idx].to_numpy(dtype=np.float32, copy=True)

    print(
        f"Computing autograd Hessian over {effective_sample} sampled rows "
        f"({len(feature_names)} features)…"
    )
    mean_signed, mean_abs, std = _compute_mean_hessian(module, X_sample)
    first_order = _first_order_importance(module, X_sample)

    ordering = np.argsort(-first_order)
    ordered_names = [feature_names[i] for i in ordering]
    mean_signed_ord = mean_signed[np.ix_(ordering, ordering)]
    mean_abs_ord = mean_abs[np.ix_(ordering, ordering)]

    cont_idx = [feature_names.index(f) for f in continuous_features]
    cont_order = [i for i in ordering if i in set(cont_idx)]
    cont_names_ord = [feature_names[i] for i in cont_order]
    mean_signed_cont = mean_signed[np.ix_(cont_order, cont_order)] if cont_order else np.zeros((0, 0))

    report_dir = reports_dir(interpret_dir)
    fig_dir = interpret_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig_all = fig_dir / f"{flavor}_hessian_heatmap_all.png"
    fig_abs = fig_dir / f"{flavor}_hessian_heatmap_abs.png"
    fig_cont = fig_dir / f"{flavor}_hessian_heatmap_continuous.png"

    _plot_heatmap(
        mean_signed_ord,
        ordered_names,
        title=f"Mean Hessian (signed) — {experiment_name}",
        output_path=fig_all,
        symmetric_colors=True,
        cmap="RdBu_r",
    )
    _plot_heatmap(
        mean_abs_ord,
        ordered_names,
        title=f"Mean |Hessian| — {experiment_name}",
        output_path=fig_abs,
        symmetric_colors=False,
        cmap="magma",
    )
    if cont_order:
        _plot_heatmap(
            mean_signed_cont,
            cont_names_ord,
            title=f"Mean Hessian (signed) — continuous only",
            output_path=fig_cont,
            symmetric_colors=True,
            cmap="RdBu_r",
        )

    top_k_all = _top_k_interactions(mean_signed, feature_names, top_k)
    top_k_cont = (
        _top_k_interactions(mean_signed[np.ix_(cont_idx, cont_idx)], continuous_features, top_k)
        if cont_idx
        else []
    )
    diagonal_ord = np.diag(mean_signed_ord)

    json_payload = {
        "experiment_name": experiment_name,
        "flavor": flavor,
        "sample_size": effective_sample,
        "seed": seed,
        "feature_names": feature_names,
        "feature_types": feature_types,
        "continuous_features": continuous_features,
        "discrete_features": discrete_features,
        "feature_order_by_first_order_importance": ordered_names,
        "mean_first_order_abs_gradient": {
            name: float(value) for name, value in zip(feature_names, first_order, strict=True)
        },
        "mean_signed_hessian": {
            name_i: {
                name_j: float(mean_signed[i, j])
                for j, name_j in enumerate(feature_names)
            }
            for i, name_i in enumerate(feature_names)
        },
        "mean_abs_hessian": {
            name_i: {
                name_j: float(mean_abs[i, j])
                for j, name_j in enumerate(feature_names)
            }
            for i, name_i in enumerate(feature_names)
        },
        "std_hessian": {
            name_i: {
                name_j: float(std[i, j])
                for j, name_j in enumerate(feature_names)
            }
            for i, name_i in enumerate(feature_names)
        },
        "top_k_interactions_all_features": top_k_all,
        "top_k_interactions_continuous_only": top_k_cont,
        "figures": {
            "signed_all": str(fig_all.relative_to(interpret_dir)),
            "abs_all": str(fig_abs.relative_to(interpret_dir)),
            "signed_continuous": str(fig_cont.relative_to(interpret_dir)) if cont_order else None,
        },
    }

    json_path = report_dir / f"{flavor}_hessian_heatmap.json"
    md_path = report_dir / f"{flavor}_hessian_heatmap.md"
    json_path.write_text(json.dumps(json_payload, indent=2))
    md_path.write_text(
        _build_markdown(
            experiment_name=experiment_name,
            sample_size=effective_sample,
            feature_names=ordered_names,
            continuous_features=continuous_features,
            discrete_features=discrete_features,
            top_k_all=top_k_all,
            top_k_cont=top_k_cont,
            diagonal=diagonal_ord,
            fig_all=str(fig_all.relative_to(interpret_dir)),
            fig_abs=str(fig_abs.relative_to(interpret_dir)),
            fig_cont=str(fig_cont.relative_to(interpret_dir)) if cont_order else "(n/a)",
        )
    )

    print(f"Saved Hessian report -> {json_path}")
    print(f"Saved Hessian markdown -> {md_path}")
    print(f"Saved figure -> {fig_all}")
    return json_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autograd-based Hessian heatmap generator for no-LayerNorm ChebyKAN.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Experiment config for the target no-LayerNorm ChebyKAN run.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("outputs"),
        type=Path,
        help="Root directory containing eval/ and interpretability/ artifacts.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="Optional explicit pruned checkpoint path. Defaults to the canonical run-scoped models/ path.",
    )
    parser.add_argument(
        "--sample-size",
        default=_DEFAULT_SAMPLE_SIZE,
        type=int,
        help="Number of eval rows to sample for Hessian averaging.",
    )
    parser.add_argument(
        "--top-k",
        default=_DEFAULT_TOP_K,
        type=int,
        help="Number of pairwise interactions to surface.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for row sampling.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_from_config(
        config_path=args.config,
        output_root=args.output_root,
        checkpoint_path=args.checkpoint,
        sample_size=args.sample_size,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
