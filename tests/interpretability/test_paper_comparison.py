from pathlib import Path

import pandas as pd

from src.interpretability.paper_comparison.feature_effects import (
    KanArtifacts,
    build_ranking_comparison,
    run,
    select_features_for_effect_plot,
)


def test_ranking_comparison_reports_overlap_and_scores():
    xgb = pd.Series([4.0, 3.0, 2.0, 1.0], index=["BMI", "Wt", "Age", "Noise"])
    cheby = pd.Series([5.0, 4.0, 3.0, 2.0], index=["BMI", "Age", "Wt", "Other"])
    fourier = pd.Series([6.0, 5.0, 4.0, 3.0], index=["BMI", "Wt", "Other", "Age"])

    table, summary = build_ranking_comparison(
        xgb_ranking=xgb,
        cheby_ranking=cheby,
        fourier_ranking=fourier,
        feature_types={"BMI": "continuous", "Wt": "continuous", "Age": "continuous"},
        top_n=4,
    )

    assert summary["shared_all_three_count"] == 3
    assert summary["chebykan_vs_xgboost"]["shared_count"] == 3
    assert summary["fourierkan_vs_xgboost"]["shared_count"] == 3
    assert set(table["feature"]) >= {"BMI", "Wt", "Age"}


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

    selected = select_features_for_effect_plot(
        xgb_ranking=xgb,
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


def test_run_writes_comparison_artifacts(tmp_path, monkeypatch):
    xgb_dir = tmp_path / "outputs" / "interpretability" / "xgboost_paper" / "stage-c-xgboost-best"
    cheby_dir = tmp_path / "outputs" / "interpretability" / "kan_paper" / "cheby-run"
    fourier_dir = tmp_path / "outputs" / "interpretability" / "kan_paper" / "fourier-run"
    xgb_eval_dir = tmp_path / "outputs" / "eval" / "xgboost_paper" / "stage-c-xgboost-best"
    cheby_eval_dir = tmp_path / "outputs" / "eval" / "kan_paper" / "cheby-run"
    fourier_eval_dir = tmp_path / "outputs" / "eval" / "kan_paper" / "fourier-run"
    for path in [xgb_dir / "data", cheby_dir / "data", fourier_dir / "data", xgb_eval_dir]:
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

    def fake_plot(**kwargs):
        out = kwargs["output_dir"] / "figures" / "feature_effect_comparison.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("stub\n")
        return out

    import src.interpretability.paper_comparison.feature_effects as module

    monkeypatch.setattr(module, "_load_pruned_kan", fake_load_pruned_kan)
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
        cheby_dir=cheby_dir,
        fourier_dir=fourier_dir,
        xgb_eval_dir=xgb_eval_dir,
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
