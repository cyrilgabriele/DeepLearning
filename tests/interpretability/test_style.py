import numpy as np
import pandas as pd
import pytest


def test_encode_to_raw_lookup_monotone():
    """Sorted lookup table should be monotone — interp requires this."""
    from src.interpretability.utils.style import encode_to_raw_lookup
    X_eval = pd.DataFrame({"BMI": [-0.8, -0.3, 0.0, 0.4, 0.9]})
    X_raw = pd.DataFrame({"BMI": [18.5, 22.0, 25.0, 30.0, 40.0]})
    result = encode_to_raw_lookup("BMI", X_eval, X_raw)
    assert result[0] < result[2] < result[4]


def test_encode_to_raw_lookup_boundary_clamp():
    """Out-of-range encoded values clamp to boundary raw values (np.interp default)."""
    from src.interpretability.utils.style import encode_to_raw_lookup
    X_eval = pd.DataFrame({"BMI": [-0.5, 0.0, 0.5]})
    X_raw = pd.DataFrame({"BMI": [20.0, 25.0, 30.0]})
    result = encode_to_raw_lookup("BMI", X_eval, X_raw, x_norm=np.array([-2.0, 0.0, 2.0]))
    assert result[0] == pytest.approx(20.0)
    assert result[2] == pytest.approx(30.0)


def test_encode_to_raw_lookup_identity_when_linear():
    from src.interpretability.utils.style import encode_to_raw_lookup
    encoded = np.linspace(-1, 1, 10)
    raw = np.linspace(10.0, 50.0, 10)
    X_eval = pd.DataFrame({"F": encoded})
    X_raw = pd.DataFrame({"F": raw})
    result = encode_to_raw_lookup("F", X_eval, X_raw, x_norm=encoded)
    np.testing.assert_allclose(result, raw, atol=1e-6)


def test_feature_type_label_appends_marker():
    from src.interpretability.utils.style import feature_type_label
    assert feature_type_label("BMI", {"BMI": "continuous"}) == "BMI [C]"
    assert feature_type_label("Medical_Keyword_3", {"Medical_Keyword_3": "binary"}) == "Medical_Keyword_3 [B]"
    assert feature_type_label("Unknown", {}) == "Unknown"


def test_resolve_feature_display_spec_marks_identity_binary_as_discrete():
    from src.interpretability.utils.style import resolve_feature_display_spec

    spec = resolve_feature_display_spec(
        "Medical_Keyword_3",
        feat_types={"Medical_Keyword_3": "binary"},
        preprocessing_recipe="kan_paper",
    )

    assert spec.raw_feature == "Medical_Keyword_3"
    assert spec.transform == "identity"
    assert spec.model_input_kind == "discrete"


def test_resolve_feature_display_spec_marks_catboost_features_as_continuous():
    from src.interpretability.utils.style import resolve_feature_display_spec

    spec = resolve_feature_display_spec(
        "cb_Product_Info_2",
        feat_types={"cb_Product_Info_2": "categorical"},
        preprocessing_recipe="kan_sota",
    )

    assert spec.raw_feature == "Product_Info_2"
    assert spec.transform == "catboost_encoded"
    assert spec.model_input_kind == "continuous"


def test_discrete_feature_ticks_use_raw_labels_for_label_encoded_categories():
    from src.interpretability.utils.style import discrete_feature_ticks, resolve_feature_display_spec

    X_eval = pd.DataFrame({"Product_Info_2": [0.0, 1.0, 0.0]})
    X_raw = pd.DataFrame({"Product_Info_2": ["D3", "E1", "D3"]})
    spec = resolve_feature_display_spec(
        "Product_Info_2",
        feat_types={"Product_Info_2": "categorical"},
        preprocessing_recipe="kan_paper",
    )

    positions, labels = discrete_feature_ticks(spec, X_eval, X_raw)

    np.testing.assert_allclose(positions, np.array([0.0, 1.0]))
    assert labels == ["D3", "E1"]


def test_savefig_pdf_creates_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import savefig_pdf
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    out = tmp_path / "test.pdf"
    savefig_pdf(fig, out)
    assert out.exists()
    assert out.stat().st_size > 100
    plt.close()


def test_apply_paper_style_sets_serif():
    import matplotlib as mpl
    from src.interpretability.utils.style import apply_paper_style
    apply_paper_style()
    assert "serif" in mpl.rcParams["font.family"]
