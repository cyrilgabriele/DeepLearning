"""Produce a simplified LaTeX table of 5 representative sparse ChebyKAN
layer-0 edges (one per named feature) for the interpretability section.

For each chosen feature (BMI, Wt, Ins_Age, Product_Info_4, Medical_Keyword_3):
  1. Find the highest-L1 output of layer 0 fed by that feature.
  2. Read the native Chebyshev coefficients + base weight.
  3. Drop terms with |coef| < 5e-3; round the rest to 3 decimals.
  4. Render as a clean LaTeX expression in T_k(tanh(x)) basis notation.

The point of the table is to let the reader SEE what a sparse-ChebyKAN
layer-0 edge actually looks like when you strip the bookkeeping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# PyTorch 2.6 weights_only fix
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})

from src.config import load_experiment_config
from src.interpretability.kan_pruning import _compute_edge_l1
from src.models.kan_layers import ChebyKANLayer
from src.models.tabkan import TabKAN


REPO = Path("/Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning")
CONFIG = REPO / "configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml"
CKPT = REPO / "outputs/interpretability/kan_paper/stage-c-chebykan-pareto-q0583-top20-noln/models/chebykan_pruned_module.pt"

REPRESENTATIVE_FEATURES = [
    "BMI",
    "Wt",
    "Ins_Age",
    "Product_Info_4",
    "Medical_Keyword_3",
]
COEF_THRESHOLD = 5e-3  # drop smaller terms
ROUND_DIGITS = 3


def _load_module() -> TabKAN:
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


def _first_layer(module: TabKAN) -> ChebyKANLayer:
    for layer in module.kan_layers:
        if isinstance(layer, ChebyKANLayer):
            return layer
    raise RuntimeError("No ChebyKANLayer found")


def _feature_names() -> list[str]:
    import json
    path = REPO / "configs/experiment_stages/stage_c_explanation_package/feature_lists/chebykan_pareto_q0583_top20_features.json"
    return json.loads(path.read_text())


def _render_term(coef: float, basis_label: str) -> str:
    """Format a term as LaTeX. basis_label is e.g. 'x', 'T_1(\\tanh x)', ..."""
    sign = "+" if coef >= 0 else "-"
    mag = abs(round(coef, ROUND_DIGITS))
    if basis_label == "1":
        return f"{sign}\\,{mag:.3f}"
    return f"{sign}\\,{mag:.3f}\\,{basis_label}"


def _render_edge_formula(coefs: np.ndarray, base_weight: float,
                         threshold: float = COEF_THRESHOLD,
                         top_k: int = 3) -> str:
    """Return simplified LaTeX formula for base_w * x + Σ_k c_k T_k(tanh x),
    showing only the top-`top_k` terms by absolute coefficient magnitude and
    collapsing the remainder into a trailing `+ ...` marker."""
    # Gather all nonzero terms as (abs_coef, signed_coef, label)
    candidates: list[tuple[float, float, str]] = []
    if abs(base_weight) >= threshold:
        candidates.append((abs(base_weight), float(base_weight), "x"))
    for k, c in enumerate(coefs):
        if abs(c) < threshold:
            continue
        if k == 0:
            label = "1"
        elif k == 1:
            label = "\\tanh x"
        else:
            label = f"T_{{{k}}}(\\tanh x)"
        candidates.append((abs(float(c)), float(c), label))

    if not candidates:
        return "0"

    # Sort by absolute magnitude, take top-k
    candidates.sort(key=lambda t: t[0], reverse=True)
    kept = candidates[:top_k]
    dropped = candidates[top_k:]

    # Render in descending-importance order (keeps the visually dominant term first)
    parts = [_render_term(c[1], c[2]) for c in kept]
    text = " ".join(parts)
    if text.startswith("+"):
        text = text.lstrip("+ \\,")
    if dropped:
        text = text + " +\\,\\dots"
    return text


def main() -> None:
    module = _load_module()
    layer = _first_layer(module)
    feature_names = _feature_names()

    l1 = _compute_edge_l1(layer)  # shape (out, in)
    coeffs = layer.cheby_coeffs.detach().cpu().numpy()
    base_weights = layer.base_weight.detach().cpu().numpy()

    rows: list[tuple[str, int, int, float, str, int]] = []
    for feat in REPRESENTATIVE_FEATURES:
        if feat not in feature_names:
            print(f"Warning: {feat} not in feature list; skipping.")
            continue
        in_idx = feature_names.index(feat)
        out_idx = int(l1[:, in_idx].argmax().item())
        edge_l1 = float(l1[out_idx, in_idx].item())
        c = coeffs[out_idx, in_idx, :]
        bw = float(base_weights[out_idx, in_idx])

        formula = _render_edge_formula(c, bw)
        nonzero_terms = int(np.sum(np.abs(c) >= COEF_THRESHOLD)) + int(abs(bw) >= COEF_THRESHOLD)
        rows.append((feat, in_idx, out_idx, edge_l1, formula, nonzero_terms))

    # Emit LaTeX table (spconf style, fits single column)
    out_tex = REPO / "outputs" / "reports" / "table_closed_forms_latex.tex"
    lines = []
    lines.append("% Simplified closed-form edges from the sparse ChebyKAN hero.")
    lines.append("% Generated by scripts/simplified_closed_forms_table.py")
    lines.append("% Each row: the most important layer-0 edge from that input feature")
    lines.append("% (highest L1 activation norm).")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Five representative closed-form layer-0 edges from the sparse ChebyKAN hero (out of 597 active edges). Each row shows the edge from the named input feature to its highest-L1 hidden output, with coefficients rounded to three decimals and terms $< 5 \times 10^{-3}$ dropped. All edges are exact (R$^2 = 1.000$) by construction via the basis-native extractor.}")
    lines.append(r"\label{tab:closed_forms}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\hline")
    lines.append(r"Feature & Closed-form edge $\phi(x)$ \\")
    lines.append(r"\hline")
    for feat, in_idx, out_idx, edge_l1, formula, nz in rows:
        feat_latex = feat.replace("_", r"\_")
        lines.append(f"\\texttt{{{feat_latex}}} & $\\displaystyle {formula}$ \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines))
    print(f"Saved → {out_tex}")

    print()
    print("=== Preview ===")
    for feat, _, out_idx, edge_l1, formula, nz in rows:
        print(f"{feat:20s} → h{out_idx}  (L1={edge_l1:.3f}, nonzero={nz}):")
        print(f"  {formula}")
        print()


if __name__ == "__main__":
    main()
