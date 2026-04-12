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
    from src.interpretability.utils.paths import data as data_dir, reports as rep_dir

    fits_df = pd.read_csv(fits_csv_path)
    result = compose_symbolic_model(fits_df, module, feature_names, min_r2=min_r2)

    # Compute end-to-end R² if eval data is available
    e2e_r2 = {}
    if X_eval is not None and result["sympy_exprs"]:
        e2e_r2 = _compute_end_to_end_r2(
            result["sympy_exprs"], result["feature_symbols"], module, X_eval
        )

    # Save formulas to JSON
    report = {
        "flavor": flavor,
        "min_r2_threshold": min_r2,
        "coverage": result["coverage"],
        "formulas": result["formulas"],
        "end_to_end_r2": e2e_r2,
    }
    report_path = rep_dir(output_dir) / f"{flavor}_symbolic_formulas.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Saved symbolic formulas -> {report_path}")

    # Save human-readable markdown
    md_lines = [
        f"# {flavor} — Composed Symbolic Formulas\n",
        f"Coverage: {result['coverage']['included_edges']}/{result['coverage']['total_edges']} "
        f"edges ({result['coverage']['fraction']:.1%}) with R² >= {min_r2}\n",
    ]
    for out_i, formula in sorted(result["formulas"].items()):
        r2_str = f" (end-to-end R² = {e2e_r2[out_i]:.4f})" if out_i in e2e_r2 else ""
        md_lines.append(f"## Output node {out_i}{r2_str}\n")
        md_lines.append(f"```\n{formula}\n```\n")

    md_path = rep_dir(output_dir) / f"{flavor}_symbolic_formulas.md"
    md_path.write_text("\n".join(md_lines))
    print(f"Saved formula report -> {md_path}")

    return report
