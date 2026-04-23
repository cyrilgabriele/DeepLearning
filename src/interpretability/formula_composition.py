"""Compose per-edge symbolic fits into end-to-end closed-form expressions.

After symbolic regression assigns a formula to each active edge, this module
traverses the pruned KAN graph and composes edge formulas through layers
using SymPy, following Liu et al. (2024) arXiv:2404.19756 Section 2.5.

For a 2-layer KAN [n_in, n_hidden, n_out]:
    output_j = sum_h phi_{1,h,j}( sum_i phi_{0,i,h}(x_i) )

Each phi is the symbolic formula fitted to that edge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit


# ── Formula string -> SymPy expression ───────────────────────────────────────

_SYMPY_TEMPLATES: dict[str, sp.Expr] = {}


def _build_templates() -> dict[str, sp.Expr]:
    """Build SymPy expression templates for each formula in the scipy library."""
    if _SYMPY_TEMPLATES:
        return _SYMPY_TEMPLATES

    x = sp.Symbol("x")
    a, b, c, d = sp.symbols("a b c d")

    templates = {
        "a*x + b": a * x + b,
        "a*x^2 + b*x + c": a * x**2 + b * x + c,
        "a*x^3 + b*x^2 + c*x + d": a * x**3 + b * x**2 + c * x + d,
        "a*|x| + b": a * sp.Abs(x) + b,
        "a*sqrt(|x|) + b": a * sp.sqrt(sp.Abs(x)) + b,
        "a*log(|x|+1) + b": a * sp.log(sp.Abs(x) + 1) + b,
        "a*exp(x) + b": a * sp.exp(x) + b,
        "a*sin(x) + b": a * sp.sin(x) + b,
        "a*sin(2*x) + b": a * sp.sin(2 * x) + b,
        "a*cos(x) + b": a * sp.cos(x) + b,
        "a (constant)": a,
        "a*sin(x) + b*cos(x)": a * sp.sin(x) + b * sp.cos(x),
        "a*sin(2*x) + b*cos(2*x)": a * sp.sin(2 * x) + b * sp.cos(2 * x),
        "a*sin(3*x) + b*cos(3*x)": a * sp.sin(3 * x) + b * sp.cos(3 * x),
        "a*sin(4*x) + b*cos(4*x)": a * sp.sin(4 * x) + b * sp.cos(4 * x),
    }
    _SYMPY_TEMPLATES.update(templates)
    return _SYMPY_TEMPLATES


# Map from formula name -> (callable, n_params) for refitting
_REFIT_FUNCTIONS: dict[str, tuple] = {
    "a*x + b": (lambda x, a, b: a * x + b, 2),
    "a*x^2 + b*x + c": (lambda x, a, b, c: a * x**2 + b * x + c, 3),
    "a*x^3 + b*x^2 + c*x + d": (lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, 4),
    "a*|x| + b": (lambda x, a, b: a * np.abs(x) + b, 2),
    "a*sqrt(|x|) + b": (lambda x, a, b: a * np.sqrt(np.abs(x)) + b, 2),
    "a*log(|x|+1) + b": (lambda x, a, b: a * np.log(np.abs(x) + 1) + b, 2),
    "a*exp(x) + b": (lambda x, a, b: a * np.exp(np.clip(x, -5, 5)) + b, 2),
    "a*sin(x) + b": (lambda x, a, b: a * np.sin(x) + b, 2),
    "a*sin(2*x) + b": (lambda x, a, b: a * np.sin(2 * x) + b, 2),
    "a*cos(x) + b": (lambda x, a, b: a * np.cos(x) + b, 2),
    "a (constant)": (lambda x, a: np.full_like(x, float(a)), 1),
    "a*sin(x) + b*cos(x)": (lambda x, a, b: a * np.sin(x) + b * np.cos(x), 2),
    "a*sin(2*x) + b*cos(2*x)": (lambda x, a, b: a * np.sin(2 * x) + b * np.cos(2 * x), 2),
    "a*sin(3*x) + b*cos(3*x)": (lambda x, a, b: a * np.sin(3 * x) + b * np.cos(3 * x), 2),
    "a*sin(4*x) + b*cos(4*x)": (lambda x, a, b: a * np.sin(4 * x) + b * np.cos(4 * x), 2),
}

# Parameter symbols for substitution
_PARAM_SYMBOLS = {
    "a": sp.Symbol("a"),
    "b": sp.Symbol("b"),
    "c": sp.Symbol("c"),
    "d": sp.Symbol("d"),
}


def _fit_params(formula_name: str, x: np.ndarray, y: np.ndarray) -> dict[str, float] | None:
    """Re-fit a formula to (x, y) data and return named parameter values."""
    if formula_name not in _REFIT_FUNCTIONS:
        return None
    func, n_params = _REFIT_FUNCTIONS[formula_name]
    try:
        popt, _ = curve_fit(func, x, y, p0=[1.0] * n_params, maxfev=3000)
    except Exception:
        return None

    param_names = list("abcd")[:n_params]
    return dict(zip(param_names, popt))


def formula_to_sympy(
    formula_name: str,
    params: dict[str, float],
    input_symbol: sp.Symbol,
) -> sp.Expr | None:
    """Convert a named formula + fitted params into a SymPy expression.

    Substitutes the parameter values and replaces the template variable ``x``
    with ``input_symbol`` (which may be a feature symbol or a composed expr).
    """
    templates = _build_templates()
    if formula_name not in templates:
        return None

    template = templates[formula_name]
    x = sp.Symbol("x")

    # Substitute parameter values (round to 4 decimal places for readability)
    subs = {}
    for name, sym in _PARAM_SYMBOLS.items():
        if name in params:
            subs[sym] = round(params[name], 4)

    expr = template.subs(subs)
    # Replace template x with the actual input symbol
    expr = expr.subs(x, input_symbol)
    return expr


# ── Graph composition ────────────────────────────────────────────────────────


def compose_symbolic_model(
    fits_df: pd.DataFrame,
    module,
    feature_names: list[str],
    *,
    min_r2: float = 0.90,
    n_samples: int = 1000,
    max_edges_per_node: int = 10,
) -> dict:
    """Compose per-edge symbolic fits into end-to-end formulas per output node.

    Parameters
    ----------
    fits_df : DataFrame
        Output of ``kan_symbolic.run()`` with columns:
        layer, edge_in, edge_out, input_feature, formula, r_squared, quality_tier.
    module : TabKAN
        The trained (or pruned) model, used to sample edges for parameter refitting.
    feature_names : list[str]
        Names of input features for layer 0.
    min_r2 : float
        Only include edges with R² >= min_r2 in the composition.
    n_samples : int
        Number of sample points for parameter refitting.
    max_edges_per_node : int
        Maximum incoming edges per node (top-K by R²).  Prevents expression
        blowup in dense networks.

    Returns
    -------
    dict with keys:
        formulas: dict[int, str]  — output_node -> simplified SymPy formula string
        sympy_exprs: dict[int, sp.Expr]  — output_node -> SymPy expression
        coverage: dict  — fraction of edges included, total edges, included edges
        feature_symbols: dict[str, sp.Symbol]  — name -> symbol mapping
    """
    from src.interpretability.kan_symbolic import sample_edge
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer

    kan_layers = [l for l in module.kan_layers if isinstance(l, (ChebyKANLayer, FourierKANLayer))]
    if not kan_layers:
        return {"formulas": {}, "sympy_exprs": {}, "coverage": {}, "feature_symbols": {}}

    # Filter to usable edges
    usable = fits_df[fits_df["r_squared"] >= min_r2].copy()

    # Create feature symbols
    feat_syms = {name: sp.Symbol(name.replace(" ", "_")) for name in feature_names}

    # Build layer-by-layer composition
    # Layer 0: input features -> hidden nodes
    # Layer 1+: hidden nodes -> next layer / output

    # For each layer, build a dict: (layer, edge_out) -> SymPy expression
    # This represents what each node computes as a sum of incoming edge functions.

    n_layers = len(kan_layers)
    # node_exprs[layer_idx][node_idx] = SymPy expression for that node's output
    node_exprs: list[dict[int, sp.Expr]] = [{} for _ in range(n_layers + 1)]

    # Layer 0 inputs are the feature symbols
    for i, name in enumerate(feature_names):
        node_exprs[0][i] = feat_syms[name]

    for layer_idx, layer in enumerate(kan_layers):
        layer_fits = usable[usable["layer"] == layer_idx]

        for out_i in range(layer.out_features):
            # Take only the top-K incoming edges by R² to keep expressions tractable
            node_edges = layer_fits[layer_fits["edge_out"] == out_i]
            if len(node_edges) > max_edges_per_node:
                node_edges = node_edges.nlargest(max_edges_per_node, "r_squared")

            node_sum = sp.Integer(0)
            has_any_edge = False

            for _, row in node_edges.iterrows():
                in_i = int(row["edge_in"])
                formula_name = str(row["formula"])

                # Get the input to this edge
                input_expr = node_exprs[layer_idx].get(in_i)
                if input_expr is None:
                    # Input node was pruned or not reachable
                    continue

                # Re-fit to get actual parameter values
                x_vals, y_vals = sample_edge(layer, out_i, in_i, n=n_samples)
                params = _fit_params(formula_name, x_vals, y_vals)
                if params is None:
                    continue

                # Convert to SymPy with the input expression
                edge_expr = formula_to_sympy(formula_name, params, input_expr)
                if edge_expr is None:
                    continue

                node_sum = node_sum + edge_expr
                has_any_edge = True

            if has_any_edge:
                node_exprs[layer_idx + 1][out_i] = node_sum

    # The final layer's node_exprs are the output formulas
    output_layer_idx = n_layers
    output_exprs = node_exprs[output_layer_idx]

    # Simplify each output expression (skip for very large expressions)
    _MAX_OPS_FOR_SIMPLIFY = 500
    formulas = {}
    sympy_exprs = {}
    for out_i, expr in output_exprs.items():
        n_ops = expr.count_ops()
        if n_ops <= _MAX_OPS_FOR_SIMPLIFY:
            try:
                simplified = sp.nsimplify(expr, tolerance=1e-3, rational=False)
                simplified = sp.simplify(simplified)
            except Exception:
                simplified = expr
        else:
            print(
                f"  Skipping simplify for output node {out_i} "
                f"({n_ops} ops > {_MAX_OPS_FOR_SIMPLIFY} limit)"
            )
            simplified = expr
        sympy_exprs[out_i] = simplified
        formulas[out_i] = str(simplified)

    total_edges = len(fits_df)
    included_edges = len(usable)

    return {
        "formulas": formulas,
        "sympy_exprs": sympy_exprs,
        "coverage": {
            "total_edges": total_edges,
            "included_edges": included_edges,
            "fraction": round(included_edges / total_edges, 4) if total_edges > 0 else 0.0,
            "min_r2_threshold": min_r2,
        },
        "feature_symbols": feat_syms,
    }


def _compute_end_to_end_r2(
    sympy_exprs: dict[int, sp.Expr],
    feature_symbols: dict[str, sp.Symbol],
    module,
    X_eval: pd.DataFrame,
) -> dict[int, float]:
    """Compute R² between the symbolic formula and the actual model output.

    Evaluates the composed SymPy expression on the eval data and compares
    to the model's forward pass.
    """
    import torch

    feature_names = list(X_eval.columns)
    sym_list = [feature_symbols.get(n) for n in feature_names]

    X_tensor = torch.tensor(X_eval.values, dtype=torch.float32)
    with torch.no_grad():
        model_out = module(X_tensor).cpu().numpy().flatten()

    r2_per_output: dict[int, float] = {}

    for out_i, expr in sympy_exprs.items():
        try:
            # Lambdify the expression for fast evaluation
            valid_syms = [s for s in sym_list if s is not None and s in expr.free_symbols]
            if not valid_syms:
                continue
            f = sp.lambdify(valid_syms, expr, modules=["numpy"])

            # Build argument arrays in the order lambdify expects
            sym_to_col = {feature_symbols[n]: n for n in feature_names if feature_symbols.get(n) is not None}
            args = [X_eval[sym_to_col[s]].values for s in valid_syms]
            y_symbolic = np.array(f(*args), dtype=float).flatten()

            ss_res = np.sum((model_out - y_symbolic) ** 2)
            ss_tot = np.sum((model_out - np.mean(model_out)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
            r2_per_output[out_i] = round(r2, 6)
        except Exception:
            continue

    return r2_per_output


_EXACT_MD_FULL_FORMULA_LIMIT = 12000
_EXACT_MD_PREVIEW_LIMIT = 4000
_EXACT_USABLE_MAX_OPS = 3000
_EXACT_USABLE_MAX_CHARS = 25000
_ZERO_TOL = 1e-10


def _has_layernorm(module) -> bool:
    import torch.nn as nn

    return any(isinstance(layer, nn.LayerNorm) for layer in module.kan_layers)


def _float_expr(value: float) -> sp.Expr:
    return sp.Float(float(value))


def _build_cheby_basis_exprs(input_expr: sp.Expr, degree: int) -> list[sp.Expr]:
    x_norm = sp.tanh(input_expr)
    basis = [sp.Integer(1)]
    if degree >= 1:
        basis.append(x_norm)
    for _ in range(2, degree + 1):
        basis.append(2 * x_norm * basis[-1] - basis[-2])
    return basis


def _compose_exact_chebykan_edge(layer, *, out_idx: int, in_idx: int, input_expr: sp.Expr) -> sp.Expr:
    coeffs = layer.cheby_coeffs.detach().cpu().numpy()[out_idx, in_idx]
    base_weight = float(layer.base_weight.detach().cpu().numpy()[out_idx, in_idx])
    basis = _build_cheby_basis_exprs(input_expr, int(layer.degree))

    edge_expr = sp.Integer(0)
    if abs(base_weight) > _ZERO_TOL:
        edge_expr += _float_expr(base_weight) * input_expr

    for coeff, basis_expr in zip(coeffs, basis):
        coeff_value = float(coeff)
        if abs(coeff_value) <= _ZERO_TOL:
            continue
        edge_expr += _float_expr(coeff_value) * basis_expr
    return edge_expr


def _compose_exact_fourierkan_edge(layer, *, out_idx: int, in_idx: int, input_expr: sp.Expr) -> sp.Expr:
    """Build the exact symbolic edge for FourierKAN.

    Mirrors the runtime forward pass in `_sample_fourierkan_edge`:
        x_scaled = (tanh(x) + 1) * pi
        y(x) = base_weight * x + Σₖ aₖ cos(k·x_scaled) + bₖ sin(k·x_scaled)
    """
    a_coeffs = layer.fourier_a.detach().cpu().numpy()[out_idx, in_idx]
    b_coeffs = layer.fourier_b.detach().cpu().numpy()[out_idx, in_idx]
    base_weight = float(layer.base_weight.detach().cpu().numpy()[out_idx, in_idx])
    grid_size = int(layer.grid_size)

    x_scaled = (sp.tanh(input_expr) + sp.Integer(1)) * sp.pi

    edge_expr = sp.Integer(0)
    if abs(base_weight) > _ZERO_TOL:
        edge_expr += _float_expr(base_weight) * input_expr

    for k_idx in range(grid_size):
        k = k_idx + 1
        a_value = float(a_coeffs[k_idx])
        b_value = float(b_coeffs[k_idx])
        if abs(a_value) > _ZERO_TOL:
            edge_expr += _float_expr(a_value) * sp.cos(k * x_scaled)
        if abs(b_value) > _ZERO_TOL:
            edge_expr += _float_expr(b_value) * sp.sin(k * x_scaled)
    return edge_expr


def compose_exact_chebykan_model(
    module,
    feature_names: list[str],
) -> dict[str, object]:
    from src.models.kan_layers import ChebyKANLayer

    if _has_layernorm(module):
        return {
            "exact_available": False,
            "reason": "layernorm_present",
            "sympy_expr": None,
            "formula": None,
            "feature_symbols": {},
        }

    cheby_layers = [layer for layer in module.kan_layers if isinstance(layer, ChebyKANLayer)]
    if not cheby_layers:
        return {
            "exact_available": False,
            "reason": "unsupported_layer_stack",
            "sympy_expr": None,
            "formula": None,
            "feature_symbols": {},
        }

    feature_symbols = {name: sp.Symbol(name.replace(" ", "_")) for name in feature_names}
    node_exprs: list[dict[int, sp.Expr]] = [{i: feature_symbols[name] for i, name in enumerate(feature_names)}]

    for layer in cheby_layers:
        prev_nodes = node_exprs[-1]
        current_nodes: dict[int, sp.Expr] = {}
        for out_idx in range(layer.out_features):
            node_expr = sp.Integer(0)
            for in_idx in range(layer.in_features):
                input_expr = prev_nodes.get(in_idx)
                if input_expr is None:
                    continue
                edge_expr = _compose_exact_chebykan_edge(
                    layer,
                    out_idx=out_idx,
                    in_idx=in_idx,
                    input_expr=input_expr,
                )
                if edge_expr == 0:
                    continue
                node_expr += edge_expr
            current_nodes[out_idx] = node_expr
        node_exprs.append(current_nodes)

    final_hidden = node_exprs[-1]
    head_weight = module.head.weight.detach().cpu().numpy()[0]
    head_bias = float(module.head.bias.detach().cpu().numpy()[0])

    output_expr = _float_expr(head_bias) if abs(head_bias) > _ZERO_TOL else sp.Integer(0)
    for hidden_idx, weight in enumerate(head_weight):
        weight_value = float(weight)
        if abs(weight_value) <= _ZERO_TOL:
            continue
        hidden_expr = final_hidden.get(hidden_idx)
        if hidden_expr is None or hidden_expr == 0:
            continue
        output_expr += _float_expr(weight_value) * hidden_expr

    return {
        "exact_available": True,
        "reason": None,
        "sympy_expr": output_expr,
        "formula": str(output_expr),
        "feature_symbols": feature_symbols,
    }


def _build_exact_closed_form_report(
    *,
    module,
    feature_names: list[str],
    flavor: str,
    X_eval: pd.DataFrame | None,
) -> dict[str, object]:
    base_report: dict[str, object] = {
        "flavor": flavor,
        "mode": "exact_closed_form",
        "exact_available": False,
        "exact_label": None,
        "reason": None,
        "has_layernorm": _has_layernorm(module),
        "formula": None,
        "formula_preview": None,
        "operation_count": None,
        "expression_length": None,
        "usable": False,
        "usability_reason": None,
        "end_to_end_r2": None,
        "sympy_derivable": False,
    }

    if flavor != "chebykan":
        base_report["reason"] = "unsupported_flavor"
        return base_report

    exact_result = compose_exact_chebykan_model(module, feature_names)
    if not exact_result["exact_available"]:
        base_report["reason"] = exact_result["reason"]
        return base_report

    expr = exact_result["sympy_expr"]
    formula = exact_result["formula"]
    op_count = int(expr.count_ops())
    expr_len = len(formula)
    usable = op_count <= _EXACT_USABLE_MAX_OPS and expr_len <= _EXACT_USABLE_MAX_CHARS
    end_to_end_r2 = None
    if X_eval is not None and op_count <= _EXACT_USABLE_MAX_OPS:
        r2_report = _compute_end_to_end_r2(
            {0: expr},
            exact_result["feature_symbols"],
            module,
            X_eval,
        )
        end_to_end_r2 = r2_report.get(0)

    usability_reason = None
    if not usable:
        parts = []
        if op_count > _EXACT_USABLE_MAX_OPS:
            parts.append(f"operation_count>{_EXACT_USABLE_MAX_OPS}")
        if expr_len > _EXACT_USABLE_MAX_CHARS:
            parts.append(f"expression_length>{_EXACT_USABLE_MAX_CHARS}")
        usability_reason = ",".join(parts)

    base_report.update(
        {
            "exact_available": True,
            "exact_label": "exact for the deployed no-LayerNorm ChebyKAN variant",
            "reason": None,
            "formula": formula,
            "formula_preview": formula[:_EXACT_MD_PREVIEW_LIMIT],
            "operation_count": op_count,
            "expression_length": expr_len,
            "usable": usable,
            "usability_reason": usability_reason,
            "end_to_end_r2": end_to_end_r2,
            "sympy_derivable": True,
        }
    )
    return base_report


def _write_exact_closed_form_reports(report: dict[str, object], output_dir: Path, flavor: str) -> None:
    from src.interpretability.utils.paths import reports as rep_dir

    report_dir = rep_dir(output_dir)
    exact_json_path = report_dir / f"{flavor}_exact_closed_form.json"
    exact_md_path = report_dir / f"{flavor}_exact_closed_form.md"
    legacy_json_path = report_dir / f"{flavor}_symbolic_formulas.json"
    legacy_md_path = report_dir / f"{flavor}_symbolic_formulas.md"

    json_payload = json.dumps(report, indent=2)
    exact_json_path.write_text(json_payload)
    legacy_json_path.write_text(json_payload)

    md_lines = [f"# {flavor} — Exact Closed Form\n"]
    if report["exact_available"]:
        md_lines.extend(
            [
                f"- Label: {report['exact_label']}",
                f"- Operation count: {report['operation_count']}",
                f"- Expression length: {report['expression_length']}",
                f"- Usable exact expression: {report['usable']}",
                f"- SymPy-derivable: {report['sympy_derivable']}",
            ]
        )
        if report["end_to_end_r2"] is not None:
            md_lines.append(f"- End-to-end R^2 on eval: {report['end_to_end_r2']}")
        if report["usability_reason"]:
            md_lines.append(f"- Usability note: {report['usability_reason']}")
        md_lines.append("")
        formula = str(report["formula"] or "")
        if len(formula) <= _EXACT_MD_FULL_FORMULA_LIMIT:
            md_lines.extend(["## Formula", "", "```text", formula, "```"])
        else:
            md_lines.extend(
                [
                    "## Formula Preview",
                    "",
                    "```text",
                    formula[:_EXACT_MD_PREVIEW_LIMIT],
                    "...",
                    "```",
                    "",
                    f"Full formula stored in `{exact_json_path.name}`.",
                ]
            )
    else:
        md_lines.extend(
            [
                f"- Exact closed form available: {report['exact_available']}",
                f"- Reason: {report['reason']}",
            ]
        )

    markdown = "\n".join(md_lines) + "\n"
    exact_md_path.write_text(markdown)
    legacy_md_path.write_text(markdown)


def run(
    fits_csv_path: Path,
    module,
    feature_names: list[str],
    output_dir: Path,
    flavor: str,
    *,
    min_r2: float = 0.90,
    X_eval: pd.DataFrame | None = None,
) -> dict:
    """Run formula composition and save results."""
    if fits_csv_path.exists():
        # Preserve the read for compatibility with the existing interpret stage.
        pd.read_csv(fits_csv_path)

    report = _build_exact_closed_form_report(
        module=module,
        feature_names=feature_names,
        flavor=flavor,
        X_eval=X_eval,
    )
    _write_exact_closed_form_reports(report, output_dir, flavor)
    print(f"Saved exact closed-form report -> {Path(output_dir) / 'reports' / f'{flavor}_exact_closed_form.json'}")
    return report
