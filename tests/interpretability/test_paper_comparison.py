from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.interpretability.paper_comparison.feature_effects import (
    KanArtifacts,
    TabPFNArtifacts,
    build_ranking_comparison,
    plot_feature_effect_comparison,
    run,
    select_features_for_effect_plot,
    _load_tabpfn_ranking,
)


def test_ranking_comparison_reports_overlap_and_scores():
    xgb = pd.Series([4.0, 3.0, 2.0, 1.0], index=["BMI", "Wt", "Age", "Noise"])
    tabpfn = pd.Series([5.0, 4.0, 3.0, 2.0], index=["BMI", "Wt", "Other", "Age"])
    cheby = pd.Series([5.0, 4.0, 3.0, 2.0], index=["BMI", "Age", "Wt", "Other"])
    fourier = pd.Series([6.0, 5.0, 4.0, 3.0], index=["BMI", "Wt", "Other", "Age"])

    table, summary = build_ranking_comparison(
        xgb_ranking=xgb,
        tabpfn_ranking=tabpfn,
        cheby_ranking=cheby,
        fourier_ranking=fourier,
        feature_types={"BMI": "continuous", "Wt": "continuous", "Age": "continuous"},
        top_n=4,
    )

    assert summary["shared_all_models_count"] == 3
    assert summary["shared_all_three_count"] == 3
    assert summary["tabpfn_vs_xgboost"]["shared_count"] == 3
    assert summary["chebykan_vs_xgboost"]["shared_count"] == 3
    assert summary["fourierkan_vs_xgboost"]["shared_count"] == 3
    assert "tabpfn_rank" in table.columns
    assert set(table["feature"]) >= {"BMI", "Wt", "Age"}


def test_tabpfn_ranking_loader_reads_ablation_csv(tmp_path):
    ranking_dir = tmp_path / "tabpfn" / "data"
    ranking_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "feature": ["BMI", "Wt"],
            "importance": [0.2, 0.4],
            "importance_std": [0.0, 0.0],
        }
    ).to_csv(ranking_dir / "tabpfn_feature_ranking.csv", index=False)

    ranking = _load_tabpfn_ranking(tmp_path / "tabpfn")

    assert ranking.index.tolist() == ["Wt", "BMI"]
    assert ranking.loc["Wt"] == 0.4


def test_missing_tabpfn_ranking_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="TabPFN feature ranking"):
        _load_tabpfn_ranking(tmp_path / "tabpfn")


def test_select_features_prefers_overlap_with_type_diversity():
    xgb = pd.Series(
        range(8, 0, -1),
        index=["BMI", "Product_Info_4", "Wt", "Medical_Keyword_3", "Noise", "A", "B", "C"],
    )
    cheby = pd.Series(
        range(8, 0, -1),
        index=["BMI", "Wt", "Product_Info_4", "Medical_Keyword_3", "D", "E", "F", "G"],
    )
    fourier = pd.Series(
        range(8, 0, -1),
        index=["BMI", "Product_Info_4", "Medical_Keyword_3", "Wt", "H", "I", "J", "K"],
    )
    tabpfn = pd.Series(
        range(8, 0, -1),
        index=["BMI", "Product_Info_4", "Wt", "Medical_Keyword_3", "L", "M", "N", "O"],
    )

    selected = select_features_for_effect_plot(
        xgb_ranking=xgb,
        tabpfn_ranking=tabpfn,
        cheby_ranking=cheby,
        fourier_ranking=fourier,
        feature_types={
            "BMI": "continuous",
            "Product_Info_4": "continuous",
            "Wt": "continuous",
            "Medical_Keyword_3": "binary",
        },
        available_features=set(xgb.index) & set(cheby.index) & set(fourier.index),
        n_features=4,
        pool_n=8,
    )

    assert selected[:3] == ["BMI", "Product_Info_4", "Wt"]
    assert "Medical_Keyword_3" in selected


def test_plot_feature_effect_comparison_renders_four_rows(tmp_path, monkeypatch):
    class FakeTabPFN:
        def predict_continuous(self, X):
            return X["BMI"].to_numpy(dtype=float) * 0.1 + X["Wt"].to_numpy(dtype=float) * 0.01

    class FakeKan(torch.nn.Module):
        def forward(self, X):
            return X[:, 0:1] * 0.1 + X[:, 1:2] * 0.01

    X_eval = pd.DataFrame(
        {
            "BMI": np.linspace(20.0, 35.0, 12),
            "Wt": np.linspace(60.0, 95.0, 12),
        }
    )
    shap_df = pd.DataFrame(
        {
            "BMI": np.linspace(-0.2, 0.2, 12),
            "Wt": np.linspace(0.1, -0.1, 12),
        }
    )
    feature_types = {"BMI": "continuous", "Wt": "continuous"}
    run_summary = {
        "metrics": {"qwk": 0.6},
        "preprocessing": {"feature_count": 2, "recipe": "tabpfn_paper"},
        "random_seed": 42,
    }
    tabpfn = TabPFNArtifacts(
        interpret_dir=tmp_path / "tabpfn",
        eval_dir=tmp_path / "eval-tabpfn",
        checkpoint=tmp_path / "model.joblib",
        model=FakeTabPFN(),
        X_eval=X_eval,
        X_raw=X_eval,
        feature_types=feature_types,
        ranking=pd.Series([1.0, 0.5], index=["BMI", "Wt"]),
        run_summary=run_summary,
        pdp_subsample_size=5,
    )

    def fake_kan(flavor: str) -> KanArtifacts:
        return KanArtifacts(
            flavor=flavor,
            interpret_dir=tmp_path / flavor,
            eval_dir=tmp_path / f"eval-{flavor}",
            module=FakeKan(),
            X_eval=X_eval,
            X_raw=X_eval,
            feature_types=feature_types,
            ranking=pd.Series([1.0, 0.5], index=["BMI", "Wt"]),
            pruning_summary={"qwk_after": 0.6, "edges_after": 2},
            r2_report={"symbolic_fits": [{"r_squared": 1.0}]},
            run_summary=run_summary,
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_subplots = plt.subplots
    captured = {}

    def capture_subplots(*args, **kwargs):
        captured["shape"] = args[:2]
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", capture_subplots)

    figure = plot_feature_effect_comparison(
        features=["BMI", "Wt"],
        shap_df=shap_df,
        xgb_eval=X_eval,
        xgb_raw=X_eval,
        feature_types=feature_types,
        tabpfn=tabpfn,
        cheby=fake_kan("chebykan"),
        fourier=fake_kan("fourierkan"),
        output_dir=tmp_path / "comparison",
    )

    assert captured["shape"] == (4, 2)
    assert figure.exists()
    assert figure.with_suffix(".png").exists()


def test_run_writes_comparison_artifacts(tmp_path, monkeypatch):
    xgb_dir = tmp_path / "outputs" / "interpretability" / "xgboost_paper" / "stage-c-xgboost-best"
    tabpfn_dir = tmp_path / "outputs" / "interpretability" / "tabpfn_paper" / "tabpfn-run"
    cheby_dir = tmp_path / "outputs" / "interpretability" / "kan_paper" / "cheby-run"
    fourier_dir = tmp_path / "outputs" / "interpretability" / "kan_paper" / "fourier-run"
    xgb_eval_dir = tmp_path / "outputs" / "eval" / "xgboost_paper" / "stage-c-xgboost-best"
    tabpfn_eval_dir = tmp_path / "outputs" / "eval" / "tabpfn_paper" / "tabpfn-run"
    cheby_eval_dir = tmp_path / "outputs" / "eval" / "kan_paper" / "cheby-run"
    fourier_eval_dir = tmp_path / "outputs" / "eval" / "kan_paper" / "fourier-run"
    for path in [
        xgb_dir / "data",
        tabpfn_dir / "data",
        cheby_dir / "data",
        fourier_dir / "data",
        xgb_eval_dir,
        tabpfn_eval_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    shap = pd.DataFrame(
        {
            "BMI": [0.1, -0.3, 0.2],
            "Wt": [0.2, 0.1, -0.1],
            "Medical_Keyword_3": [0.0, 0.4, -0.2],
        }
    )
    shap.to_parquet(xgb_dir / "data" / "shap_xgb_values.parquet")
    X_eval = pd.DataFrame(
        {
            "BMI": [20.0, 25.0, 30.0],
            "Wt": [60.0, 70.0, 80.0],
            "Medical_Keyword_3": [0.0, 1.0, 0.0],
        }
    )
    X_eval.to_parquet(xgb_eval_dir / "X_eval.parquet")
    X_eval.to_parquet(xgb_eval_dir / "X_eval_raw.parquet")
    (xgb_eval_dir / "feature_types.json").write_text(
        '{"BMI": "continuous", "Wt": "continuous", "Medical_Keyword_3": "binary"}'
    )
    X_eval.to_parquet(tabpfn_eval_dir / "X_eval.parquet")
    X_eval.to_parquet(tabpfn_eval_dir / "X_eval_raw.parquet")
    (tabpfn_eval_dir / "feature_types.json").write_text(
        '{"BMI": "continuous", "Wt": "continuous", "Medical_Keyword_3": "binary"}'
    )
    pd.DataFrame(
        {
            "feature": ["BMI", "Wt", "Medical_Keyword_3"],
            "importance": [3.0, 2.0, 1.0],
        }
    ).to_csv(tabpfn_dir / "data" / "tabpfn_feature_ranking.csv", index=False)

    for flavor, directory in [("chebykan", cheby_dir), ("fourierkan", fourier_dir)]:
        pd.DataFrame(
            {
                "feature": ["BMI", "Wt", "Medical_Keyword_3"],
                "importance": [3.0, 2.0, 1.0],
            }
        ).to_csv(directory / "data" / f"{flavor}_feature_ranking.csv", index=False)

    def fake_load_pruned_kan(*, interpret_dir: Path, eval_dir: Path, flavor: str):
        return KanArtifacts(
            flavor=flavor,
            interpret_dir=interpret_dir,
            eval_dir=eval_dir,
            module=None,
            X_eval=X_eval,
            X_raw=X_eval,
            feature_types={
                "BMI": "continuous",
                "Wt": "continuous",
                "Medical_Keyword_3": "binary",
            },
            ranking=pd.Series([3.0, 2.0, 1.0], index=["BMI", "Wt", "Medical_Keyword_3"]),
            pruning_summary={"qwk_after": 0.6, "edges_after": 12},
            r2_report={"symbolic_fits": [{"r_squared": 1.0}]},
            run_summary={"metrics": {"qwk": 0.6}, "preprocessing": {"feature_count": 3, "recipe": "kan_paper"}, "random_seed": 42},
        )

    def fake_load_tabpfn(*, interpret_dir: Path, eval_dir: Path, checkpoint: Path | None, pdp_subsample_size: int):
        return TabPFNArtifacts(
            interpret_dir=interpret_dir,
            eval_dir=eval_dir,
            checkpoint=Path("checkpoints/tabpfn/model.joblib"),
            model=None,
            X_eval=X_eval,
            X_raw=X_eval,
            feature_types={
                "BMI": "continuous",
                "Wt": "continuous",
                "Medical_Keyword_3": "binary",
            },
            ranking=pd.Series([3.0, 2.0, 1.0], index=["BMI", "Wt", "Medical_Keyword_3"]),
            run_summary={"metrics": {"qwk": 0.62}, "preprocessing": {"feature_count": 3, "recipe": "tabpfn_paper"}, "random_seed": 42},
            pdp_subsample_size=pdp_subsample_size,
        )

    def fake_plot(**kwargs):
        out = kwargs["output_dir"] / "figures" / "feature_effect_comparison.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("stub\n")
        return out

    import src.interpretability.paper_comparison.feature_effects as module

    monkeypatch.setattr(module, "_load_pruned_kan", fake_load_pruned_kan)
    monkeypatch.setattr(module, "_load_tabpfn", fake_load_tabpfn)
    monkeypatch.setattr(module, "plot_feature_effect_comparison", fake_plot)
    monkeypatch.setattr(
        module,
        "_latest_run_summary",
        lambda experiment_name: (
            Path("run-summary.json"),
            {"metrics": {"qwk": 0.55}, "preprocessing": {"feature_count": 3, "recipe": "xgboost_paper"}, "random_seed": 42},
        ),
    )

    artifacts = run(
        xgb_dir=xgb_dir,
        tabpfn_dir=tabpfn_dir,
        cheby_dir=cheby_dir,
        fourier_dir=fourier_dir,
        xgb_eval_dir=xgb_eval_dir,
        tabpfn_eval_dir=tabpfn_eval_dir,
        cheby_eval_dir=cheby_eval_dir,
        fourier_eval_dir=fourier_eval_dir,
        output_dir=tmp_path / "comparison",
        features=["BMI", "Wt"],
        top_n=2,
    )

    assert artifacts["ranking_comparison"].exists()
    assert artifacts["overlap_summary"].exists()
    assert artifacts["selected_features"].exists()
    assert artifacts["model_summary"].exists()
    assert artifacts["feature_effect_figure"].exists()
    assert artifacts["report"].exists()
