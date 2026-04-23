import pandas as pd
import pytest
import torch

from src.interpretability.exact_partials import (
    build_continuous_partial_trace,
    build_exact_partials_report,
    compose_exact_chebykan_symbolic_graph,
    evaluate_continuous_partial_trace_row,
    evaluate_discrete_effect_contract_row,
    evaluate_symbolic_graph_row,
    run,
)
from src.models.tabkan import TabKAN


@pytest.fixture
def tiny_exact_module():
    module = TabKAN(in_features=3, widths=[2, 2], kan_type="chebykan", degree=2, use_layernorm=False)
    with torch.no_grad():
        for layer in module.kan_layers:
            if hasattr(layer, "cheby_coeffs"):
                layer.cheby_coeffs.zero_()
                layer.base_weight.zero_()

        first = module.kan_layers[0]
        second = module.kan_layers[1]

        first.base_weight[0, 0] = 0.2
        first.cheby_coeffs[0, 1, 1] = 0.4
        first.cheby_coeffs[0, 2, 2] = -0.15

        first.base_weight[1, 1] = -0.3
        first.cheby_coeffs[1, 0, 1] = 0.25
        first.cheby_coeffs[1, 2, 1] = 0.1

        second.base_weight[0, 0] = 0.5
        second.cheby_coeffs[0, 1, 1] = 0.35

        second.base_weight[1, 1] = -0.4
        second.cheby_coeffs[1, 0, 2] = 0.2

        module.head.weight.zero_()
        module.head.weight[0, 0] = 1.3
        module.head.weight[0, 1] = -0.6
        module.head.bias[0] = 0.05

    module.eval()
    return module


@pytest.fixture
def feature_names():
    return ["feat_cont_a", "feat_cont_b", "feat_cat"]


@pytest.fixture
def feature_types():
    return {
        "feat_cont_a": "continuous",
        "feat_cont_b": "continuous",
        "feat_cat": "categorical",
    }


@pytest.fixture
def x_eval(feature_names):
    return pd.DataFrame(
        [
            {"feat_cont_a": -0.2, "feat_cont_b": 0.4, "feat_cat": 0.0},
            {"feat_cont_a": 0.7, "feat_cont_b": -0.1, "feat_cat": 2.0},
        ],
        columns=feature_names,
    )


@pytest.fixture
def x_eval_raw(x_eval):
    return x_eval.copy()


@pytest.fixture
def x_train_outer_selected(feature_names):
    return pd.DataFrame(
        [
            {"feat_cont_a": -0.8, "feat_cont_b": 0.0, "feat_cat": 1.0},
            {"feat_cont_a": -0.1, "feat_cont_b": 0.5, "feat_cat": 1.0},
            {"feat_cont_a": 0.2, "feat_cont_b": 0.3, "feat_cat": 0.0},
            {"feat_cont_a": 0.9, "feat_cont_b": -0.6, "feat_cat": 2.0},
            {"feat_cont_a": 0.6, "feat_cont_b": 0.1, "feat_cat": 1.0},
        ],
        columns=feature_names,
    )


def _autograd_partial(module, row: pd.Series, feature_names: list[str], feature_name: str) -> float:
    feature_index = feature_names.index(feature_name)
    x_tensor = torch.tensor(
        row[feature_names].to_numpy(dtype="float32", copy=False)[None, :],
        dtype=torch.float32,
        requires_grad=True,
    )
    output = module(x_tensor)
    output.backward()
    return float(x_tensor.grad[0, feature_index].item())


def test_symbolic_graph_matches_module_forward(tiny_exact_module, feature_names, x_eval):
    graph = compose_exact_chebykan_symbolic_graph(tiny_exact_module, feature_names)
    for _, row in x_eval.iterrows():
        symbolic = evaluate_symbolic_graph_row(graph, row)
        with torch.no_grad():
            tensor = torch.tensor(row[feature_names].to_numpy(dtype="float32")[None, :], dtype=torch.float32)
            direct = float(tiny_exact_module(tensor).item())
        assert symbolic == pytest.approx(direct, abs=1e-6)


def test_continuous_partial_trace_matches_autograd(tiny_exact_module, feature_names, x_eval):
    graph = compose_exact_chebykan_symbolic_graph(tiny_exact_module, feature_names)
    trace = build_continuous_partial_trace(graph, "feat_cont_a")
    for _, row in x_eval.iterrows():
        symbolic = evaluate_continuous_partial_trace_row(graph, trace, row)
        autograd = _autograd_partial(tiny_exact_module, row, feature_names, "feat_cont_a")
        assert symbolic == pytest.approx(autograd, abs=1e-5)


def test_exact_partials_report_includes_continuous_and_discrete_records(
    tiny_exact_module,
    feature_names,
    feature_types,
    x_eval,
    x_eval_raw,
    x_train_outer_selected,
):
    report = build_exact_partials_report(
        module=tiny_exact_module,
        feature_names=feature_names,
        feature_types=feature_types,
        X_eval=x_eval,
        X_eval_raw=x_eval_raw,
        X_train_outer_selected=x_train_outer_selected,
        flavor="chebykan",
        preprocessing_recipe="kan_paper",
        experiment_name="tiny-exact",
        selected_features_path="configs/example.json",
    )

    assert report["exact_available"] is True
    assert report["continuous_selected_features"] == ["feat_cont_a", "feat_cont_b"]
    assert report["discrete_selected_features"] == ["feat_cat"]
    assert report["symbolic_model"]["hidden_node_count"] == 4

    continuous = next(item for item in report["features"] if item["feature_name"] == "feat_cont_a")
    assert continuous["representation_type"] == "continuous_exact_partial"
    assert continuous["derivative_with_respect_to_transformed_feature"]["trace_definition_count"] == 4
    assert "Raw-space and transformed-space derivatives are exactly the same quantity" in continuous["raw_space_validity_note"]

    discrete = next(item for item in report["features"] if item["feature_name"] == "feat_cat")
    assert discrete["representation_type"] == "reference_based_discrete_effect"
    assert discrete["observed_state_set"] == [0, 1, 2]
    assert discrete["reference_state"] == 1
    assert len(discrete["discrete_effects"]) == 2


def test_discrete_effect_contract_evaluates_to_model_difference(
    tiny_exact_module,
    feature_names,
    feature_types,
    x_eval,
    x_eval_raw,
    x_train_outer_selected,
):
    report = build_exact_partials_report(
        module=tiny_exact_module,
        feature_names=feature_names,
        feature_types=feature_types,
        X_eval=x_eval,
        X_eval_raw=x_eval_raw,
        X_train_outer_selected=x_train_outer_selected,
        flavor="chebykan",
    )
    graph = compose_exact_chebykan_symbolic_graph(tiny_exact_module, feature_names)
    discrete = next(item for item in report["features"] if item["feature_name"] == "feat_cat")
    first_effect = discrete["discrete_effects"][0]

    row = x_eval.iloc[0]
    symbolic = evaluate_discrete_effect_contract_row(
        graph,
        feature_name="feat_cat",
        reference_state=discrete["reference_state"],
        target_state=first_effect["target_state"],
        row=row,
    )

    target = row.copy()
    target["feat_cat"] = first_effect["target_state"]
    reference = row.copy()
    reference["feat_cat"] = discrete["reference_state"]
    with torch.no_grad():
        target_tensor = torch.tensor(target[feature_names].to_numpy(dtype="float32")[None, :], dtype=torch.float32)
        reference_tensor = torch.tensor(reference[feature_names].to_numpy(dtype="float32")[None, :], dtype=torch.float32)
        direct = float(tiny_exact_module(target_tensor).item() - tiny_exact_module(reference_tensor).item())

    assert symbolic == pytest.approx(direct, abs=1e-6)


def test_run_writes_json_and_markdown(
    tmp_path,
    tiny_exact_module,
    feature_names,
    feature_types,
    x_eval,
    x_eval_raw,
    x_train_outer_selected,
):
    output_dir = tmp_path / "interpretability"
    report = run(
        module=tiny_exact_module,
        feature_names=feature_names,
        feature_types=feature_types,
        X_eval=x_eval,
        X_eval_raw=x_eval_raw,
        X_train_outer_selected=x_train_outer_selected,
        output_dir=output_dir,
        flavor="chebykan",
        preprocessing_recipe="kan_paper",
        experiment_name="tiny-exact",
    )
    assert report["exact_available"] is True
    assert (output_dir / "reports" / "chebykan_exact_partials.json").exists()
    assert (output_dir / "reports" / "chebykan_exact_partials.md").exists()


def test_report_refuses_layernorm(feature_names, feature_types, x_eval, x_eval_raw, x_train_outer_selected):
    module = TabKAN(in_features=3, widths=[2, 1], kan_type="chebykan", degree=2, use_layernorm=True)
    report = build_exact_partials_report(
        module=module,
        feature_names=feature_names,
        feature_types=feature_types,
        X_eval=x_eval,
        X_eval_raw=x_eval_raw,
        X_train_outer_selected=x_train_outer_selected,
        flavor="chebykan",
    )
    assert report["exact_available"] is False
    assert report["reason"] == "layernorm_present"
