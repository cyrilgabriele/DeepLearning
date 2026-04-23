"""Standalone exact partials and discrete observed-state effects for ChebyKAN.

Usage:
    uv run python -m src.interpretability.exact_partials \
        --config configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sympy as sp
import torch

from src.config import ExperimentConfig, load_experiment_config
from src.interpretability.formula_composition import (
    _compose_exact_chebykan_edge,
    _has_layernorm,
)
from src.interpretability.utils.paths import eval_run_dir, interpret_run_dir, reports as reports_dir
from src.models.kan_layers import ChebyKANLayer
from src.models.tabkan import TabKAN
from src.preprocessing.preprocess_kan_paper import KANPreprocessor

_ZERO_TOL = 1e-10
_RAW_IDENTITY_ATOL = 1e-6
_MD_EXPR_FULL_LIMIT = 3000
_MD_EXPR_PREVIEW_LIMIT = 1200


@dataclass(frozen=True)
class EdgeTerm:
    """One exact edge function entering a hidden node."""

    layer_index: int
    source_index: int
    target_index: int
    parent_symbol: sp.Symbol
    expr: sp.Expr
    local_derivative_expr: sp.Expr


@dataclass(frozen=True)
class NodeDefinition:
    """Exact symbolic definition of one hidden node."""

    layer_index: int
    node_index: int
    symbol: sp.Symbol
    expr: sp.Expr
    edge_terms: tuple[EdgeTerm, ...]


@dataclass(frozen=True)
class SymbolicModelGraph:
    """Nested symbolic representation of the exact no-LayerNorm ChebyKAN."""

    feature_names: tuple[str, ...]
    feature_symbols: dict[str, sp.Symbol]
    layers: tuple[tuple[NodeDefinition, ...], ...]
    output_symbol: sp.Symbol
    output_expr: sp.Expr


@dataclass(frozen=True)
class DerivativeDefinition:
    """One chain-rule derivative node definition."""

    layer_index: int
    node_index: int
    symbol: sp.Symbol
    expr: sp.Expr


@dataclass(frozen=True)
class ContinuousPartialTrace:
    """Nested exact partial derivative trace for one continuous feature."""

    feature_name: str
    derivative_symbol: sp.Symbol
    derivative_expr: sp.Expr
    layers: tuple[tuple[DerivativeDefinition, ...], ...]


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.replace(" ", "_"))


def _expr_text(expr: sp.Expr) -> str:
    return sp.sstr(expr, full_prec=True)


def _json_scalar(value: float | int) -> int | float:
    as_float = float(value)
    rounded = round(as_float)
    if abs(as_float - rounded) <= 1e-9:
        return int(rounded)
    return as_float


def _state_token(value: float | int) -> str:
    scalar = _json_scalar(value)
    token = str(scalar).replace("-", "neg").replace(".", "p")
    return token


def _render_expr_block(expr_text: str) -> list[str]:
    if len(expr_text) <= _MD_EXPR_FULL_LIMIT:
        return ["```text", expr_text, "```"]
    return ["```text", expr_text[:_MD_EXPR_PREVIEW_LIMIT], "...", "```"]


def _evaluate_sympy_expr(expr: sp.Expr, values: dict[str, float]) -> float:
    symbols = sorted(expr.free_symbols, key=lambda item: item.name)
    if not symbols:
        return float(expr)
    func = sp.lambdify(symbols, expr, modules=["numpy"])
    args = [values[symbol.name] for symbol in symbols]
    result = func(*args)
    return float(np.asarray(result, dtype=float))


def compose_exact_chebykan_symbolic_graph(
    module,
    feature_names: list[str],
) -> SymbolicModelGraph:
    """Build an exact nested symbolic graph for a no-LayerNorm ChebyKAN."""

    if _has_layernorm(module):
        raise ValueError("Exact nested symbolic graph is unavailable when LayerNorm is present.")

    cheby_layers = [layer for layer in module.kan_layers if isinstance(layer, ChebyKANLayer)]
    if not cheby_layers:
        raise ValueError("Exact nested symbolic graph currently supports only ChebyKAN layers.")

    feature_symbols = {name: sp.Symbol(_safe_name(name)) for name in feature_names}
    previous_symbols = {index: feature_symbols[name] for index, name in enumerate(feature_names)}
    layer_definitions: list[tuple[NodeDefinition, ...]] = []

    for layer_index, layer in enumerate(cheby_layers):
        current_layer_nodes: list[NodeDefinition] = []
        for node_index in range(layer.out_features):
            node_symbol = sp.Symbol(f"h_{layer_index}_{node_index}")
            node_expr = sp.Integer(0)
            edge_terms: list[EdgeTerm] = []
            for source_index in range(layer.in_features):
                parent_symbol = previous_symbols[source_index]
                edge_expr = _compose_exact_chebykan_edge(
                    layer,
                    out_idx=node_index,
                    in_idx=source_index,
                    input_expr=parent_symbol,
                )
                if edge_expr == 0:
                    continue
                local_derivative_expr = sp.diff(edge_expr, parent_symbol)
                edge_terms.append(
                    EdgeTerm(
                        layer_index=layer_index,
                        source_index=source_index,
                        target_index=node_index,
                        parent_symbol=parent_symbol,
                        expr=edge_expr,
                        local_derivative_expr=local_derivative_expr,
                    )
                )
                node_expr += edge_expr
            current_layer_nodes.append(
                NodeDefinition(
                    layer_index=layer_index,
                    node_index=node_index,
                    symbol=node_symbol,
                    expr=node_expr,
                    edge_terms=tuple(edge_terms),
                )
            )
        layer_definitions.append(tuple(current_layer_nodes))
        previous_symbols = {node.node_index: node.symbol for node in current_layer_nodes}

    output_symbol = sp.Symbol("y")
    head_weight = module.head.weight.detach().cpu().numpy()[0]
    head_bias = float(module.head.bias.detach().cpu().numpy()[0])
    output_expr = sp.Float(head_bias) if abs(head_bias) > _ZERO_TOL else sp.Integer(0)
    for hidden_index, weight in enumerate(head_weight):
        weight_value = float(weight)
        if abs(weight_value) <= _ZERO_TOL:
            continue
        output_expr += sp.Float(weight_value) * previous_symbols[hidden_index]

    return SymbolicModelGraph(
        feature_names=tuple(feature_names),
        feature_symbols=feature_symbols,
        layers=tuple(layer_definitions),
        output_symbol=output_symbol,
        output_expr=output_expr,
    )


def build_continuous_partial_trace(
    graph: SymbolicModelGraph,
    feature_name: str,
) -> ContinuousPartialTrace:
    """Construct the exact nested chain-rule partial for one continuous input feature."""

    if feature_name not in graph.feature_symbols:
        raise KeyError(f"Unknown feature for partial trace: {feature_name}")

    feature_index = list(graph.feature_names).index(feature_name)
    feature_token = _safe_name(feature_name)
    derivative_layers: list[tuple[DerivativeDefinition, ...]] = []
    previous_derivative_symbols: dict[int, sp.Symbol] = {}

    for layer_index, layer in enumerate(graph.layers):
        current_layer_derivatives: list[DerivativeDefinition] = []
        current_derivative_symbols: dict[int, sp.Symbol] = {}
        for node in layer:
            derivative_symbol = sp.Symbol(f"d_{node.symbol.name}_d_{feature_token}")
            derivative_expr = sp.Integer(0)
            for edge in node.edge_terms:
                if layer_index == 0:
                    if edge.source_index != feature_index:
                        continue
                    derivative_expr += edge.local_derivative_expr
                    continue

                upstream_derivative_symbol = previous_derivative_symbols[edge.source_index]
                derivative_expr += edge.local_derivative_expr * upstream_derivative_symbol

            current_layer_derivatives.append(
                DerivativeDefinition(
                    layer_index=layer_index,
                    node_index=node.node_index,
                    symbol=derivative_symbol,
                    expr=derivative_expr,
                )
            )
            current_derivative_symbols[node.node_index] = derivative_symbol

        derivative_layers.append(tuple(current_layer_derivatives))
        previous_derivative_symbols = current_derivative_symbols

    derivative_symbol = sp.Symbol(f"d_y_d_{feature_token}")
    derivative_expr = sp.Integer(0)
    output_terms = graph.output_expr.as_ordered_terms()
    for term in output_terms:
        free_symbols = term.free_symbols
        matched_hidden = [
            symbol
            for symbol in free_symbols
            if symbol.name.startswith("h_")
        ]
        if not matched_hidden:
            continue
        if len(matched_hidden) != 1:
            raise ValueError("Output expression contained an unexpected non-linear hidden-node term.")
        hidden_symbol = matched_hidden[0]
        hidden_index = int(hidden_symbol.name.split("_")[-1])
        coeff = term / hidden_symbol
        derivative_expr += coeff * previous_derivative_symbols[hidden_index]

    return ContinuousPartialTrace(
        feature_name=feature_name,
        derivative_symbol=derivative_symbol,
        derivative_expr=derivative_expr,
        layers=tuple(derivative_layers),
    )


def evaluate_symbolic_graph_row(
    graph: SymbolicModelGraph,
    row: dict[str, float] | pd.Series,
) -> float:
    """Evaluate the nested symbolic model on one row."""

    if isinstance(row, pd.Series):
        row_mapping = {name: float(row[name]) for name in graph.feature_names}
    else:
        row_mapping = {name: float(row[name]) for name in graph.feature_names}

    values = {
        graph.feature_symbols[name].name: row_mapping[name]
        for name in graph.feature_names
    }
    for layer in graph.layers:
        for node in layer:
            values[node.symbol.name] = _evaluate_sympy_expr(node.expr, values)
    values[graph.output_symbol.name] = _evaluate_sympy_expr(graph.output_expr, values)
    return values[graph.output_symbol.name]


def evaluate_continuous_partial_trace_row(
    graph: SymbolicModelGraph,
    trace: ContinuousPartialTrace,
    row: dict[str, float] | pd.Series,
) -> float:
    """Evaluate one continuous partial trace numerically on one row."""

    if isinstance(row, pd.Series):
        row_mapping = {name: float(row[name]) for name in graph.feature_names}
    else:
        row_mapping = {name: float(row[name]) for name in graph.feature_names}

    values = {
        graph.feature_symbols[name].name: row_mapping[name]
        for name in graph.feature_names
    }
    for layer in graph.layers:
        for node in layer:
            values[node.symbol.name] = _evaluate_sympy_expr(node.expr, values)

    derivative_values: dict[str, float] = {}
    for layer in trace.layers:
        for definition in layer:
            eval_values = {**values, **derivative_values}
            derivative_values[definition.symbol.name] = _evaluate_sympy_expr(definition.expr, eval_values)

    final_values = {**values, **derivative_values}
    return _evaluate_sympy_expr(trace.derivative_expr, final_values)


def evaluate_discrete_effect_contract_row(
    graph: SymbolicModelGraph,
    feature_name: str,
    reference_state: float | int,
    target_state: float | int,
    row: dict[str, float] | pd.Series,
) -> float:
    """Evaluate one discrete substitution-difference contract numerically on one row."""

    if isinstance(row, pd.Series):
        row_mapping = {name: float(row[name]) for name in graph.feature_names}
    else:
        row_mapping = {name: float(row[name]) for name in graph.feature_names}

    target_row = dict(row_mapping)
    target_row[feature_name] = float(target_state)
    reference_row = dict(row_mapping)
    reference_row[feature_name] = float(reference_state)
    return evaluate_symbolic_graph_row(graph, target_row) - evaluate_symbolic_graph_row(graph, reference_row)


def _load_selected_features(selected_path: Path) -> list[str]:
    raw_text = selected_path.read_text()
    if selected_path.suffix.lower() == ".json":
        payload = json.loads(raw_text)
        if isinstance(payload, list):
            selected = [str(item) for item in payload]
        elif isinstance(payload, dict):
            for key in ("features", "selected_features", "feature_names"):
                if key in payload:
                    candidate = payload[key]
                    if not isinstance(candidate, list):
                        raise TypeError(f"Selected feature payload key '{key}' must contain a list.")
                    selected = [str(item) for item in candidate]
                    break
            else:
                raise ValueError("Selected feature JSON did not contain a supported feature list key.")
        else:
            raise TypeError("Selected feature JSON must be a list or object.")
        return list(dict.fromkeys(selected))
    return list(dict.fromkeys(line.strip() for line in raw_text.splitlines() if line.strip()))


def _reconstruct_outer_train_selected(
    config: ExperimentConfig,
    eval_feature_names: list[str],
) -> pd.DataFrame:
    if config.preprocessing.recipe != "kan_paper":
        raise ValueError("Standalone exact partial reconstruction currently expects the kan_paper recipe.")

    preprocessor = KANPreprocessor()
    outputs = preprocessor.run_pipeline(config.trainer.train_csv, random_seed=config.trainer.seed)
    X_train_outer = pd.DataFrame(outputs["X_train_outer"], columns=outputs["feature_names"])

    selected_features = eval_feature_names
    selected_path = config.preprocessing.selected_features_path
    if selected_path is not None:
        configured_features = _load_selected_features(selected_path)
        if set(configured_features) != set(eval_feature_names):
            missing = sorted(set(configured_features) - set(eval_feature_names))
            extra = sorted(set(eval_feature_names) - set(configured_features))
            raise ValueError(
                "Configured selected features do not match eval artifact features. "
                f"Missing from eval: {missing[:5]} | Missing from config: {extra[:5]}"
            )
        selected_features = eval_feature_names

    return X_train_outer.loc[:, selected_features].copy()


def _raw_identity_note(
    feature_name: str,
    X_eval: pd.DataFrame,
    X_eval_raw: pd.DataFrame | None,
) -> str:
    if X_eval_raw is None or feature_name not in X_eval_raw.columns:
        return (
            "Under kan_paper this feature is treated as raw-scale at model input in this target run, "
            "but X_eval_raw was unavailable so the exported eval identity check could not be recomputed here."
        )

    processed = X_eval[feature_name].to_numpy(dtype=np.float64, copy=False)
    raw = X_eval_raw[feature_name].to_numpy(dtype=np.float64, copy=False)
    max_abs_diff = float(np.max(np.abs(processed - raw))) if len(processed) else 0.0
    identity_holds = bool(np.allclose(processed, raw, rtol=0.0, atol=_RAW_IDENTITY_ATOL))
    if identity_holds:
        return (
            "Raw-space and transformed-space derivatives are exactly the same quantity for this target run. "
            f"Under kan_paper there is no nontrivial affine rescaling for `{feature_name}`, and the exported "
            f"eval processed values match raw values up to float32 rounding (max abs diff {max_abs_diff:.3e})."
        )
    return (
        "The exported eval raw/processed identity check exceeded the expected float32 tolerance for this feature "
        f"(max abs diff {max_abs_diff:.3e}). The target spec still declares the raw-space and transformed-space "
        "derivatives identical for this run, so this should be reviewed before treating the raw-space note as verified."
    )


def _serialize_state_counts(series: pd.Series) -> tuple[list[int | float], list[dict[str, int | float]], int | float]:
    counts: dict[int | float, int] = {}
    for value in series.to_numpy(dtype=np.float64, copy=False):
        state = _json_scalar(value)
        counts[state] = counts.get(state, 0) + 1

    observed_states = sorted(counts, key=lambda item: float(item))
    modal_count = max(counts.values()) if counts else 0
    reference_state = min(
        (state for state, count in counts.items() if count == modal_count),
        key=lambda item: float(item),
    )
    serialized_counts = [
        {"state": state, "count": counts[state]}
        for state in observed_states
    ]
    return observed_states, serialized_counts, reference_state


def _serialize_model_graph(graph: SymbolicModelGraph) -> dict[str, object]:
    hidden_nodes: list[dict[str, object]] = []
    for layer in graph.layers:
        for node in layer:
            hidden_nodes.append(
                {
                    "node_id": node.symbol.name,
                    "layer_index": node.layer_index,
                    "node_index": node.node_index,
                    "symbol": node.symbol.name,
                    "expression": _expr_text(node.expr),
                    "incoming_term_count": len(node.edge_terms),
                }
            )

    return {
        "representation": "nested_symbolic_graph",
        "input_symbols": [
            {
                "feature": feature_name,
                "symbol": graph.feature_symbols[feature_name].name,
            }
            for feature_name in graph.feature_names
        ],
        "hidden_node_count": len(hidden_nodes),
        "hidden_node_definitions": hidden_nodes,
        "output_definition": {
            "symbol": graph.output_symbol.name,
            "expression": _expr_text(graph.output_expr),
        },
    }


def _serialize_continuous_trace(
    trace: ContinuousPartialTrace,
) -> dict[str, object]:
    definitions: list[dict[str, object]] = []
    for layer in trace.layers:
        for definition in layer:
            definitions.append(
                {
                    "node_id": definition.symbol.name,
                    "layer_index": definition.layer_index,
                    "node_index": definition.node_index,
                    "symbol": definition.symbol.name,
                    "expression": _expr_text(definition.expr),
                }
            )
    return {
        "symbol": trace.derivative_symbol.name,
        "expression": _expr_text(trace.derivative_expr),
        "trace_definition_count": len(definitions),
        "trace_definitions": definitions,
    }


def build_exact_partials_report(
    *,
    module,
    feature_names: list[str],
    feature_types: dict[str, str],
    X_eval: pd.DataFrame,
    X_train_outer_selected: pd.DataFrame,
    flavor: str,
    X_eval_raw: pd.DataFrame | None = None,
    preprocessing_recipe: str | None = None,
    experiment_name: str | None = None,
    selected_features_path: str | None = None,
) -> dict[str, object]:
    """Build the exact partial/discrete-effect report as a pure Python payload."""

    report: dict[str, object] = {
        "flavor": flavor,
        "mode": "exact_partials",
        "exact_available": False,
        "reason": None,
        "has_layernorm": _has_layernorm(module),
        "preprocessing_recipe": preprocessing_recipe,
        "experiment_name": experiment_name,
        "selected_features_path": selected_features_path,
        "symbolic_model": None,
        "continuous_selected_features": [],
        "discrete_selected_features": [],
        "features": [],
        "numeric_example": None,
    }

    if flavor != "chebykan":
        report["reason"] = "unsupported_flavor"
        return report
    if _has_layernorm(module):
        report["reason"] = "layernorm_present"
        return report

    graph = compose_exact_chebykan_symbolic_graph(module, feature_names)
    report["exact_available"] = True
    report["symbolic_model"] = _serialize_model_graph(graph)

    feature_records: list[dict[str, object]] = []
    continuous_features: list[str] = []
    discrete_features: list[str] = []

    for feature_name in feature_names:
        feature_type = feature_types.get(feature_name, "unknown")
        if feature_type == "continuous":
            trace = build_continuous_partial_trace(graph, feature_name)
            transformed_payload = _serialize_continuous_trace(trace)
            raw_payload = {
                "symbol": f"{trace.derivative_symbol.name}_raw",
                "expression": transformed_payload["expression"],
                "trace_definition_count": 0,
                "trace_definitions": [],
            }
            feature_records.append(
                {
                    "feature_name": feature_name,
                    "feature_type": feature_type,
                    "feature_symbol": graph.feature_symbols[feature_name].name,
                    "representation_type": "continuous_exact_partial",
                    "derivative_with_respect_to_transformed_feature": transformed_payload,
                    "derivative_with_respect_to_raw_feature": raw_payload,
                    "raw_space_validity_note": _raw_identity_note(feature_name, X_eval, X_eval_raw),
                    "observed_state_set": [],
                    "observed_state_counts": [],
                    "reference_state": None,
                    "reference_state_reason": None,
                    "discrete_effects": [],
                    "pairwise_contrast_note": None,
                    "numeric_example": None,
                }
            )
            continuous_features.append(feature_name)
            continue

        if feature_type not in {"binary", "categorical", "ordinal", "missing_indicator"}:
            raise ValueError(
                f"Unsupported feature type `{feature_type}` for exact partial/discrete-effect generation "
                f"on feature `{feature_name}`."
            )

        observed_states, observed_state_counts, reference_state = _serialize_state_counts(
            X_train_outer_selected[feature_name]
        )
        effects: list[dict[str, object]] = []
        for state in observed_states:
            if state == reference_state:
                continue
            effect_symbol = (
                f"delta_{_safe_name(feature_name)}_"
                f"{_state_token(reference_state)}_to_{_state_token(state)}"
            )
            effects.append(
                {
                    "effect_symbol": effect_symbol,
                    "target_state": state,
                    "expression_type": "substitution_difference_on_shared_nested_model",
                    "expression": (
                        f"y[{feature_name}:={state}] - y[{feature_name}:={reference_state}]"
                    ),
                    "reference_substitution": {feature_name: reference_state},
                    "target_substitution": {feature_name: state},
                    "reconstruction_note": (
                        "Apply the two substitutions to the shared nested symbolic model stored in `symbolic_model` "
                        "and subtract the reference output from the target output."
                    ),
                }
            )

        feature_records.append(
            {
                "feature_name": feature_name,
                "feature_type": feature_type,
                "feature_symbol": graph.feature_symbols[feature_name].name,
                "representation_type": "reference_based_discrete_effect",
                "derivative_with_respect_to_transformed_feature": None,
                "derivative_with_respect_to_raw_feature": None,
                "raw_space_validity_note": None,
                "observed_state_set": observed_states,
                "observed_state_counts": observed_state_counts,
                "reference_state": reference_state,
                "reference_state_reason": (
                    "Modal observed state from the reconstructed run-specific preprocessed outer training split "
                    "after feature subsetting."
                ),
                "discrete_effects": effects,
                "pairwise_contrast_note": (
                    "Any pairwise contrast is derivable from the stored reference-based effects via "
                    "delta_{a->b} = delta_{r->b} - delta_{r->a}."
                ),
                "numeric_example": None,
            }
        )
        discrete_features.append(feature_name)

    report["continuous_selected_features"] = continuous_features
    report["discrete_selected_features"] = discrete_features
    report["features"] = feature_records
    return report


def _build_markdown_report(report: dict[str, object]) -> str:
    lines = ["# chebykan — Exact Partials And Discrete Effects", ""]
    if not report["exact_available"]:
        lines.extend(
            [
                f"- Exact artifact available: {report['exact_available']}",
                f"- Reason: {report['reason']}",
                "",
            ]
        )
        return "\n".join(lines)

    symbolic_model = report["symbolic_model"]
    assert isinstance(symbolic_model, dict)
    output_definition = symbolic_model["output_definition"]
    assert isinstance(output_definition, dict)

    lines.extend(
        [
            f"- Experiment: {report['experiment_name']}",
            f"- Preprocessing: {report['preprocessing_recipe']}",
            f"- Hidden nodes stored in shared nested model: {symbolic_model['hidden_node_count']}",
            (
                "- Continuous exact partials: "
                + ", ".join(report["continuous_selected_features"])
            ),
            (
                "- Discrete observed-state features: "
                + ", ".join(report["discrete_selected_features"])
            ),
            "",
            "## Shared Nested Model",
            "",
            "- The JSON artifact stores the full hidden-node graph once and all per-feature records refer back to it.",
            f"- Output symbol: `{output_definition['symbol']}`",
            "",
            "### Output Definition",
            "",
            *_render_expr_block(str(output_definition["expression"])),
        ]
    )

    for feature_record in report["features"]:
        assert isinstance(feature_record, dict)
        lines.extend(["", f"## {feature_record['feature_name']}", ""])
        lines.append(f"- Representation: `{feature_record['representation_type']}`")
        lines.append(f"- Feature type: `{feature_record['feature_type']}`")

        if feature_record["representation_type"] == "continuous_exact_partial":
            transformed = feature_record["derivative_with_respect_to_transformed_feature"]
            raw = feature_record["derivative_with_respect_to_raw_feature"]
            assert isinstance(transformed, dict)
            assert isinstance(raw, dict)
            lines.extend(
                [
                    f"- Transformed-space derivative symbol: `{transformed['symbol']}`",
                    f"- Raw-space derivative symbol: `{raw['symbol']}`",
                    f"- Stored chain-rule trace definitions: {transformed['trace_definition_count']}",
                    f"- Raw-space note: {feature_record['raw_space_validity_note']}",
                    "",
                    "### Final Exact Partial",
                    "",
                    *_render_expr_block(str(transformed["expression"])),
                ]
            )
            continue

        observed_counts = ", ".join(
            f"{item['state']}: {item['count']}"
            for item in feature_record["observed_state_counts"]
        )
        lines.extend(
            [
                "- Observed states derived from reconstructed outer training split after feature subsetting: "
                + str(feature_record["observed_state_set"]),
                f"- Observed state counts: {observed_counts}",
                f"- Reference state: {feature_record['reference_state']}",
                f"- Reference rule: {feature_record['reference_state_reason']}",
                "",
                "### Stored Reference-Based Effects",
                "",
            ]
        )
        for effect in feature_record["discrete_effects"]:
            assert isinstance(effect, dict)
            lines.append(
                f"- `{effect['effect_symbol']}` = `{effect['expression']}`"
            )
        lines.extend(["", f"- {feature_record['pairwise_contrast_note']}"])

    return "\n".join(lines) + "\n"


def write_exact_partials_reports(
    report: dict[str, object],
    output_dir: Path,
    flavor: str,
) -> tuple[Path, Path]:
    """Write the JSON and Markdown report files."""

    report_dir = reports_dir(output_dir)
    json_path = report_dir / f"{flavor}_exact_partials.json"
    markdown_path = report_dir / f"{flavor}_exact_partials.md"
    json_path.write_text(json.dumps(report, indent=2))
    markdown_path.write_text(_build_markdown_report(report))
    return json_path, markdown_path


def run(
    *,
    module,
    feature_names: list[str],
    feature_types: dict[str, str],
    X_eval: pd.DataFrame,
    X_train_outer_selected: pd.DataFrame,
    output_dir: Path,
    flavor: str,
    X_eval_raw: pd.DataFrame | None = None,
    preprocessing_recipe: str | None = None,
    experiment_name: str | None = None,
    selected_features_path: str | None = None,
) -> dict[str, object]:
    """Build and persist the standalone exact-partials artifact."""

    report = build_exact_partials_report(
        module=module,
        feature_names=feature_names,
        feature_types=feature_types,
        X_eval=X_eval,
        X_eval_raw=X_eval_raw,
        X_train_outer_selected=X_train_outer_selected,
        flavor=flavor,
        preprocessing_recipe=preprocessing_recipe,
        experiment_name=experiment_name,
        selected_features_path=selected_features_path,
    )
    json_path, _ = write_exact_partials_reports(report, output_dir, flavor)
    print(f"Saved exact partials report -> {json_path}")
    return report


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
    return module


def run_from_config(
    *,
    config_path: Path,
    output_root: Path = Path("outputs"),
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Standalone entry point that reconstructs the target run from config + artifacts."""

    config = load_experiment_config(config_path)
    recipe = config.preprocessing.recipe
    experiment_name = config.trainer.experiment_name
    flavor = config.model.flavor or "chebykan"

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

    X_eval = pd.read_parquet(eval_dir / "X_eval.parquet")
    X_eval_raw = pd.read_parquet(eval_dir / "X_eval_raw.parquet") if (eval_dir / "X_eval_raw.parquet").exists() else None
    X_train_outer_selected = _reconstruct_outer_train_selected(config, feature_names)

    resolved_checkpoint = checkpoint_path
    if resolved_checkpoint is None:
        resolved_checkpoint = interpret_dir / "models" / f"{flavor}_pruned_module.pt"
    module = _load_pruned_module(config, feature_names, resolved_checkpoint)

    selected_features_path = (
        str(config.preprocessing.selected_features_path)
        if config.preprocessing.selected_features_path is not None
        else None
    )
    return run(
        module=module,
        feature_names=feature_names,
        feature_types=feature_types,
        X_eval=X_eval,
        X_eval_raw=X_eval_raw,
        X_train_outer_selected=X_train_outer_selected,
        output_dir=interpret_dir,
        flavor=flavor,
        preprocessing_recipe=recipe,
        experiment_name=experiment_name,
        selected_features_path=selected_features_path,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone exact partials / discrete effects report generator.")
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
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_from_config(
        config_path=args.config,
        output_root=args.output_root,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
