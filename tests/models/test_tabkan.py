import pytest

from src.models.tabkan import build_tabkan_model, TabKANFlavor


def test_build_tabkan_model_respects_width_depth_overrides():
    model = build_tabkan_model(
        "tabkan-base",
        random_state=0,
        depth=3,
        width=42,
    )
    assert model.estimator.hidden_layer_sizes == (42, 42, 42)


def test_chebykan_degree_override_ignored_for_other_flavors():
    cheby = build_tabkan_model(
        "tabkan-small",
        random_state=0,
        flavor=TabKANFlavor.CHEBYKAN,
        degree=5,
    )
    assert cheby.model_params["degree"] == 5

    fourier = build_tabkan_model(
        "tabkan-small",
        random_state=0,
        flavor=TabKANFlavor.FOURIERKAN,
        degree=7,
    )
    assert fourier.model_params["degree"] is None


def test_invalid_overrides_raise_errors():
    with pytest.raises(ValueError):
        build_tabkan_model("tabkan-base", random_state=0, depth=0, width=32)
    with pytest.raises(ValueError):
        build_tabkan_model("tabkan-base", random_state=0, depth=2, width=0)
