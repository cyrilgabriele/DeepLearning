"""One worked Greek for applicant 55728, sparse ChebyKAN hero.

Computes ∂score/∂BMI three ways:
  (1) Symbolic — exact SymPy chain rule via `exact_partials.py`
      (compact per-layer graph, no combinatorial expansion)
  (2) Autograd — PyTorch backward
  (3) Finite difference — (y(x + ε e_BMI) - y(x - ε e_BMI)) / (2ε)

Agreement of (1) ≈ (2) ≈ (3) validates the 'exact analytic Greeks'
claim for the sparse no-LayerNorm ChebyKAN hero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# PyTorch 2.6 weights_only default fix
_orig_load = torch.load
def _patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
torch.load = _patched

from src.config import load_experiment_config
from src.interpretability.exact_partials import (
    compose_exact_chebykan_symbolic_graph,
    build_continuous_partial_trace,
    evaluate_continuous_partial_trace_row,
    evaluate_symbolic_graph_row,
)
from src.models.tabkan import TabKAN


REPO = Path("/Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning")
EXP = "chebykan-lambda-0.100"
CKPT = REPO / "outputs" / "interpretability" / "kan_paper" / EXP / "models" / "chebykan_pruned_module.pt"
EVAL = REPO / "outputs" / "eval" / "kan_paper" / EXP
CONFIG = REPO / "configs" / "experiment_stages" / "stage_c_explanation_package" / "chebykan_tuned_sparse_hero_final.yaml"

APPLICANT_ID = 55728.0
FEATURE = "BMI"


def _load_pruned_module() -> TabKAN:
    cfg = load_experiment_config(CONFIG)
    state = torch.load(CKPT, map_location="cpu")
    first_key = next(k for k in state.keys() if "cheby_coeffs" in k)
    in_features = state[first_key].shape[1]
    widths = cfg.model.resolved_hidden_widths()
    module = TabKAN(
        in_features=in_features,
        widths=widths,
        kan_type=cfg.model.flavor,
        use_layernorm=cfg.model.use_layernorm,
        degree=cfg.model.degree or 3,
    )
    module.load_state_dict(state)
    module.eval()
    return module


def _load_applicant_row() -> tuple[pd.Series, np.ndarray, list[str]]:
    X = pd.read_parquet(EVAL / "X_eval.parquet")
    X_raw = pd.read_parquet(EVAL / "X_eval_raw.parquet")
    feature_names = list(X.columns)

    if "Id" in X_raw.columns:
        mask = X_raw["Id"] == APPLICANT_ID
        if not mask.any():
            mask = X_raw["Id"].astype(float) == float(APPLICANT_ID)
        if not mask.any():
            raise ValueError(f"Applicant {APPLICANT_ID} not in X_eval_raw (Id column)")
        idx = int(np.flatnonzero(mask.to_numpy())[0])
    else:
        idx = 0
        print("Warning: no Id column in X_eval_raw; using row 0")

    row_series = X.iloc[idx]
    return row_series, row_series.to_numpy(dtype=np.float32), feature_names


def autograd_derivative(module: TabKAN, feature_names: list[str],
                        applicant_row: np.ndarray, feature: str) -> float:
    feat_idx = feature_names.index(feature)
    x = torch.tensor(applicant_row, dtype=torch.float32).unsqueeze(0)
    x.requires_grad_(True)
    y = module(x).squeeze()
    y.backward()
    return float(x.grad[0, feat_idx])


def finite_difference(module: TabKAN, feature_names: list[str],
                      applicant_row: np.ndarray, feature: str,
                      eps: float = 1e-3) -> float:
    feat_idx = feature_names.index(feature)
    x_plus = applicant_row.copy()
    x_minus = applicant_row.copy()
    x_plus[feat_idx] += eps
    x_minus[feat_idx] -= eps
    with torch.no_grad():
        y_plus = module(torch.tensor(x_plus, dtype=torch.float32).unsqueeze(0))
        y_minus = module(torch.tensor(x_minus, dtype=torch.float32).unsqueeze(0))
    return float((y_plus - y_minus) / (2 * eps))


def main() -> None:
    print(f"Loading sparse ChebyKAN hero for applicant {APPLICANT_ID} ...")
    module = _load_pruned_module()
    row_series, applicant_row, feature_names = _load_applicant_row()
    print(f"Feature names: {feature_names}")
    print(f"{FEATURE} value: {row_series[FEATURE]:.6f}")

    with torch.no_grad():
        base_score = float(module(torch.tensor(applicant_row, dtype=torch.float32).unsqueeze(0)))
    print(f"Base predicted score: {base_score:.6f}")

    print("\nBuilding exact symbolic graph via exact_partials.py ...")
    graph = compose_exact_chebykan_symbolic_graph(module, feature_names)
    print(f"  Graph: {len(graph.layers)} KAN layers, {sum(len(l) for l in graph.layers)} hidden nodes")

    print(f"Building continuous partial trace for ∂y/∂{FEATURE} ...")
    trace = build_continuous_partial_trace(graph, FEATURE)
    print(f"  Trace: {len(trace.layers)} derivative layers")

    print(f"\nComputing ∂score/∂{FEATURE} three ways ...")

    print("  (1) Symbolic chain rule via SymPy (exact_partials) ...")
    d_symbolic = evaluate_continuous_partial_trace_row(graph, trace, row_series)
    print(f"      = {d_symbolic:.6f}")

    print("  (2) Autograd (PyTorch backward) ...")
    d_autograd = autograd_derivative(module, feature_names, applicant_row, FEATURE)
    print(f"      = {d_autograd:.6f}")

    print("  (3) Finite difference (ε = 1e-3, central) ...")
    d_fd = finite_difference(module, feature_names, applicant_row, FEATURE)
    print(f"      = {d_fd:.6f}")

    # Also sanity-check: symbolic graph evaluation should match model output
    y_symbolic = evaluate_symbolic_graph_row(graph, row_series)
    print(f"\nSanity check:")
    print(f"  Model output: {base_score:.6f}")
    print(f"  Symbolic eval: {y_symbolic:.6f}  (diff {abs(y_symbolic - base_score):.3e})")

    print(f"\nDerivative agreement:")
    print(f"  |symbolic - autograd|    = {abs(d_symbolic - d_autograd):.3e}")
    print(f"  |symbolic - finite-diff| = {abs(d_symbolic - d_fd):.3e}")
    print(f"  |autograd - finite-diff| = {abs(d_autograd - d_fd):.3e}")

    out = {
        "applicant_id": APPLICANT_ID,
        "feature": FEATURE,
        "feature_value_at_applicant": float(row_series[FEATURE]),
        "base_score": base_score,
        "symbolic_graph_output": float(y_symbolic),
        "partial_symbolic": float(d_symbolic),
        "partial_autograd": d_autograd,
        "partial_finite_difference": d_fd,
        "max_abs_disagreement_between_methods": float(max(
            abs(d_symbolic - d_autograd),
            abs(d_symbolic - d_fd),
            abs(d_autograd - d_fd),
        )),
        "method_note": (
            "Sparse ChebyKAN hero, no-LayerNorm, 20 features. "
            "Symbolic path uses compact per-layer graph from exact_partials.py "
            "(no combinatorial expansion). The closed-form expression is exact "
            "for this model variant; ∂y/∂x_BMI is computed analytically via "
            "SymPy chain rule on that graph."
        ),
    }
    out_path = REPO / "outputs" / "reports" / "worked_greek_applicant_55728.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
