"""Issue 11 — Final comparison matrix across all 4 models.

Assembles QWK, explanation method, structural compactness, mean R²,
and top-feature agreement vs. GLM into a Markdown + PNG table.

Usage:
    uv run python -m src.interpretability.final_comparison \
        --artifacts-dir artifacts/ \
        --outputs-dir   outputs/
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ── Metric collectors ─────────────────────────────────────────────────────────

def _load_qwk(artifacts_dir: Path, experiment_name: str) -> float | None:
    pattern = str(artifacts_dir / experiment_name / "run-summary-*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    data = json.loads(Path(files[-1]).read_text())
    return data.get("metrics", {}).get("qwk")


def _count_glm_nonzero(coef_path: Path) -> int | None:
    if not coef_path.exists():
        return None
    df = pd.read_csv(coef_path)
    return int((df["coefficient"].abs() > 1e-6).sum())


def _count_xgb_leaves(xgb_checkpoint: Path) -> int | None:
    if not xgb_checkpoint.exists():
        return None
    try:
        import joblib
        wrapper = joblib.load(xgb_checkpoint)
        booster = wrapper.model.get_booster()
        trees_df = booster.trees_to_dataframe()
        return int((trees_df["Feature"] == "Leaf").sum())
    except Exception:
        return None


def _count_kan_active_edges(pruning_summary_path: Path) -> int | None:
    if not pruning_summary_path.exists():
        return None
    data = json.loads(pruning_summary_path.read_text())
    return data.get("edges_after")


def _mean_r2(symbolic_fits_path: Path) -> float | None:
    if not symbolic_fits_path.exists():
        return None
    df = pd.read_csv(symbolic_fits_path)
    if df.empty:
        return None
    return round(float(df["r_squared"].mean()), 4)


def _feature_agreement_tau(
    glm_coef_path: Path,
    other_importance: pd.Series | None,
    top_n: int = 10,
) -> float | None:
    """Kendall's τ between GLM top-n and another method's top-n feature ranking."""
    if other_importance is None or not glm_coef_path.exists():
        return None
    from scipy.stats import kendalltau

    glm_df = pd.read_csv(glm_coef_path)
    glm_ranked = glm_df.nlargest(top_n, "abs_magnitude")["feature"].tolist()

    other_top = other_importance.nlargest(top_n).index.tolist()

    # Build ordinal ranks over the union of features
    all_feats = list(dict.fromkeys(glm_ranked + other_top))
    glm_rank = {f: i for i, f in enumerate(glm_ranked)}
    other_rank = {f: i for i, f in enumerate(other_top)}
    ranks_glm = [glm_rank.get(f, top_n) for f in all_feats]
    ranks_other = [other_rank.get(f, top_n) for f in all_feats]

    tau, _ = kendalltau(ranks_glm, ranks_other)
    return round(float(tau), 4)


# ── Narratives ────────────────────────────────────────────────────────────────

_NARRATIVES = {
    "GLM": (
        "The GLM (Ridge regression) serves as the linear interpretability baseline. "
        "Its coefficients provide a direct measure of each feature's marginal effect on the "
        "predicted risk score, making it trivially interpretable. However, its inability to "
        "capture non-linear interactions limits predictive performance relative to the other models."
    ),
    "XGBoost": (
        "XGBoost is the strongest non-linear baseline, capturing complex feature interactions "
        "via gradient-boosted trees. SHAP TreeExplainer provides exact, fast explanations that "
        "are locally faithful and globally consistent. The main interpretability limitation is "
        "that individual tree structures are difficult to read, making SHAP an indirect proxy "
        "rather than a transparent model explanation."
    ),
    "ChebyKAN": (
        "ChebyKAN represents the learned relationships as Chebyshev polynomial activations on "
        "each edge. After pruning, the surviving edges can be symbolically regressed to yield "
        "closed-form mathematical expressions. This makes ChebyKAN the most structurally "
        "transparent neural model: each surviving edge has a human-readable formula. The trade-off "
        "is a potentially higher number of active parameters than GLM and a more complex training "
        "pipeline than XGBoost."
    ),
    "FourierKAN": (
        "FourierKAN uses Fourier series (cosine + sine) activations, making it naturally suited "
        "to periodic or oscillatory patterns in the data. Like ChebyKAN, edge pruning and symbolic "
        "regression can reduce the model to a sparse set of analytical expressions. The Fourier "
        "basis may yield simpler symbolic fits than Chebyshev for features with periodic structure, "
        "but can be harder to interpret for monotonic relationships."
    ),
}


# ── Table assembly ────────────────────────────────────────────────────────────

def run(
    artifacts_dir: Path = Path("artifacts"),
    outputs_dir: Path = Path("outputs"),
) -> None:
    from src.interpretability.paths import figures as fig_dir, data as data_dir, reports as rep_dir

    glm_coef_path = data_dir(outputs_dir) / "glm_coefficients.csv"
    shap_path = data_dir(outputs_dir) / "shap_xgb_values.parquet"
    chebykan_sym = data_dir(outputs_dir) / "chebykan_symbolic_fits.csv"
    fourierkan_sym = data_dir(outputs_dir) / "fourierkan_symbolic_fits.csv"
    chebykan_prune = rep_dir(outputs_dir) / "chebykan_pruning_summary.json"
    fourierkan_prune = rep_dir(outputs_dir) / "fourierkan_pruning_summary.json"

    # ── XGBoost: find latest checkpoint ──────────────────────────────────────
    xgb_ckpts = sorted(Path("checkpoints/xgb-baseline").glob("*.joblib")) if Path("checkpoints/xgb-baseline").exists() else []
    xgb_ckpt = xgb_ckpts[-1] if xgb_ckpts else Path("")

    # ── SHAP global feature importance for τ calculation ─────────────────────
    xgb_importance = None
    if shap_path.exists():
        shap_df = pd.read_parquet(shap_path)
        xgb_importance = shap_df.abs().mean()

    kan_importance = None
    if chebykan_sym.exists():
        sym_df = pd.read_csv(chebykan_sym)
        layer0 = sym_df[sym_df["layer"] == 0]
        kan_importance = layer0.groupby("input_feature").size().astype(float)

    # ── Build matrix rows ─────────────────────────────────────────────────────
    models = ["GLM", "XGBoost", "ChebyKAN", "FourierKAN"]
    exp_names = {
        "GLM": "glm-baseline",
        "XGBoost": "xgb-baseline",
        "ChebyKAN": "chebykan-base",
        "FourierKAN": "fourierkan-base",
    }

    rows = {}
    for m in models:
        qwk = _load_qwk(artifacts_dir, exp_names[m])
        rows[m] = {"Validation QWK": f"{qwk:.4f}" if qwk is not None else "N/A"}

    rows["GLM"]["Explanation method"] = "Coefficients"
    rows["XGBoost"]["Explanation method"] = "SHAP (TreeExplainer)"
    rows["ChebyKAN"]["Explanation method"] = "Symbolic (scipy/PySR)"
    rows["FourierKAN"]["Explanation method"] = "Symbolic (scipy/PySR)"

    rows["GLM"]["Structural compactness"] = str(_count_glm_nonzero(glm_coef_path) or "N/A") + " non-zero coefs"
    rows["XGBoost"]["Structural compactness"] = str(_count_xgb_leaves(xgb_ckpt) or "N/A") + " leaves"
    rows["ChebyKAN"]["Structural compactness"] = str(_count_kan_active_edges(chebykan_prune) or "N/A") + " active edges"
    rows["FourierKAN"]["Structural compactness"] = str(_count_kan_active_edges(fourierkan_prune) or "N/A") + " active edges"

    rows["GLM"]["Mean R² of explanations"] = "N/A"
    rows["XGBoost"]["Mean R² of explanations"] = "N/A"
    rows["ChebyKAN"]["Mean R² of explanations"] = str(_mean_r2(chebykan_sym) or "N/A")
    rows["FourierKAN"]["Mean R² of explanations"] = str(_mean_r2(fourierkan_sym) or "N/A")

    rows["GLM"]["Top-feature agreement with GLM (τ)"] = "1.000"
    rows["XGBoost"]["Top-feature agreement with GLM (τ)"] = str(
        _feature_agreement_tau(glm_coef_path, xgb_importance) or "N/A"
    )
    rows["ChebyKAN"]["Top-feature agreement with GLM (τ)"] = str(
        _feature_agreement_tau(glm_coef_path, kan_importance) or "N/A"
    )
    rows["FourierKAN"]["Top-feature agreement with GLM (τ)"] = "N/A"

    # ── Markdown table ────────────────────────────────────────────────────────
    dimensions = [
        "Validation QWK",
        "Explanation method",
        "Structural compactness",
        "Mean R² of explanations",
        "Top-feature agreement with GLM (τ)",
    ]

    header = "| Dimension | " + " | ".join(models) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(models)) + " |"
    table_lines = [header, sep]
    for dim in dimensions:
        cells = " | ".join(rows[m].get(dim, "N/A") for m in models)
        table_lines.append(f"| {dim} | {cells} |")

    narratives_md = "\n\n".join(
        f"### {m}\n\n{_NARRATIVES[m]}" for m in models
    )

    md_content = f"""# Final Interpretability Comparison

## Comparison Matrix

{chr(10).join(table_lines)}

## Model Narratives

{narratives_md}

---
*All figures and CSVs are in the `outputs/` directory.*
"""

    md_path = rep_dir(outputs_dir) / "final_comparison_matrix.md"
    md_path.write_text(md_content)
    print(f"Saved Markdown → {md_path}")

    narratives_path = rep_dir(outputs_dir) / "model_narratives.md"
    narratives_path.write_text(f"# Model Narratives\n\n{narratives_md}\n")
    print(f"Saved narratives → {narratives_path}")

    # ── PNG table ─────────────────────────────────────────────────────────────
    _render_table_png(rows, models, dimensions, fig_dir(outputs_dir))


def _render_table_png(rows, models, dimensions, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell_text = [[rows[m].get(d, "N/A") for m in models] for d in dimensions]
    fig, ax = plt.subplots(figsize=(14, len(dimensions) * 0.8 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=dimensions,
        colLabels=models,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)

    # Header styling
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF5FB")

    plt.title("Interpretability Comparison Matrix", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    png_path = output_dir / "final_comparison_matrix.png"  # output_dir is already fig_dir here
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved table PNG → {png_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final interpretability comparison matrix")
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.artifacts_dir, args.outputs_dir)
