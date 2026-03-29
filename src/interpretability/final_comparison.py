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


# ── New helpers ───────────────────────────────────────────────────────────────

def _mask_features(X: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Return copy of X with all columns not in keep set to 0.0."""
    X_masked = X.copy()
    for col in X_masked.columns:
        if col not in keep:
            X_masked[col] = 0.0
    return X_masked


def _top5_overlap(rankings: dict[str, list[str]]) -> int:
    """Count features appearing in all methods' top-5."""
    sets = [set(v[:5]) for v in rankings.values()]
    if not sets:
        return 0
    common = sets[0].intersection(*sets[1:])
    return len(common)


def _compute_qwk_retention_curve(
    glm_coef_path: Path,
    shap_path: Path,
    chebykan_ckpt: Path,
    fourierkan_ckpt: Path,
    chebykan_cfg: Path,
    fourierkan_cfg: Path,
    chebykan_prune: Path,
    fourierkan_prune: Path,
    glm_ckpt: Path,
    xgb_ckpt: Path,
    eval_features_path: Path,
    eval_labels_path: Path,
) -> tuple[dict[str, list[float]], dict[str, float], dict[str, list[str]]]:
    """Evaluate each model at retention levels 10%..100% by feature masking.

    Returns:
        qwk_curves: {model: [qwk_at_10pct, ..., qwk_at_100pct]}
        qwk_at_50: {model: qwk}
        rankings: {model: [feat_ranked_1st, ...]}
    """
    import torch
    import joblib
    from sklearn.metrics import cohen_kappa_score
    from src.interpretability.utils.kan_coefficients import coefficient_importance_from_module
    from src.models.kan_layers import ChebyKANLayer, FourierKANLayer
    from src.configs import load_experiment_config
    from src.models.tabkan import TabKAN

    X_eval = pd.read_parquet(eval_features_path)
    y_eval = pd.read_parquet(eval_labels_path)["Response"]
    feature_names = list(X_eval.columns)
    n_feats = len(feature_names)

    def qwk(y_true, y_pred):
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    # Build importance rankings
    glm_df = pd.read_csv(glm_coef_path)
    glm_ranked = glm_df.sort_values("abs_magnitude", ascending=False)["feature"].tolist()

    shap_df = pd.read_parquet(shap_path)
    xgb_ranked = shap_df.abs().mean().sort_values(ascending=False).index.tolist()

    def _kan_feature_ranking(ckpt, cfg_path, flavor):
        cfg = load_experiment_config(cfg_path)
        module = TabKAN(
            in_features=n_feats,
            widths=[cfg.model.width] * cfg.model.depth,
            kan_type=flavor,
            degree=cfg.model.degree or 3,
        )
        module.load_state_dict(torch.load(ckpt, map_location="cpu"))
        module.eval()
        ranking = coefficient_importance_from_module(module, feature_names)
        if ranking.empty:
            return feature_names, module
        return ranking.index.tolist(), module

    cheby_ranked, cheby_module = _kan_feature_ranking(chebykan_ckpt, chebykan_cfg, "chebykan")
    fourier_ranked, fourier_module = _kan_feature_ranking(fourierkan_ckpt, fourierkan_cfg, "fourierkan")

    rankings = {
        "GLM": glm_ranked,
        "XGBoost": xgb_ranked,
        "ChebyKAN": cheby_ranked,
        "FourierKAN": fourier_ranked,
    }

    # Load inference wrappers
    glm_wrapper = joblib.load(glm_ckpt)
    xgb_wrapper = joblib.load(xgb_ckpt)

    # For KAN inference, use the loaded modules directly via forward pass
    import torch

    def _kan_predict(module, X_df):
        X_t = torch.tensor(X_df.values, dtype=torch.float32)
        with torch.no_grad():
            preds = module(X_t).cpu().numpy().flatten()
        return np.clip(np.round(preds), 1, 8).astype(int)

    model_predict = {
        "GLM": lambda X: glm_wrapper.predict(X),
        "XGBoost": lambda X: xgb_wrapper.predict(X),
        "ChebyKAN": lambda X: _kan_predict(cheby_module, X),
        "FourierKAN": lambda X: _kan_predict(fourier_module, X),
    }

    # Retention sweep
    retention_levels = [r / 10 for r in range(1, 11)]
    qwk_curves: dict[str, list[float]] = {m: [] for m in rankings}
    qwk_at_50: dict[str, float] = {}

    for pct in retention_levels:
        k = max(1, int(pct * n_feats))
        for model_name, ranked in rankings.items():
            keep = ranked[:k]
            X_masked = _mask_features(X_eval, keep)
            preds = model_predict[model_name](X_masked)
            score = qwk(y_eval, preds)
            qwk_curves[model_name].append(score)
            if abs(pct - 0.5) < 1e-9:
                qwk_at_50[model_name] = score

    return qwk_curves, qwk_at_50, rankings


def _plot_qwk_retention(qwk_curves: dict[str, list[float]], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf, MODEL_COLORS
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()
    retention_pcts = list(range(10, 110, 10))
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, scores in qwk_curves.items():
        auc = float(np.trapezoid(scores, dx=0.1) if hasattr(np, "trapezoid") else np.trapz(scores, dx=0.1))
        ax.plot(retention_pcts, scores, color=MODEL_COLORS[model_name],
                lw=2, marker="o", ms=5, label=f"{model_name} (AUC={auc:.3f})")

    ax.set_xlabel("Features retained (%)", fontsize=9)
    ax.set_ylabel("QWK", fontsize=9)
    ax.set_title("QWK vs. Feature Retention (feature masking, no retraining)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(retention_pcts)
    ax.legend(fontsize=8, loc="lower right")
    ax.text(0.5, -0.12,
            "Note: masked features set to 0.0 (≈ median in encoded [-1,1] space), not absent.",
            ha="center", transform=ax.transAxes, fontsize=6, color="gray")
    plt.tight_layout()
    out = fig_dir(output_dir) / "qwk_feature_retention.pdf"
    savefig_pdf(fig, out)
    print(f"Saved → {out}")
    plt.close()


def _render_table_pdf(rows, models, dimensions, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.interpretability.utils.style import apply_paper_style, savefig_pdf
    from src.interpretability.utils.paths import figures as fig_dir

    apply_paper_style()
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
    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EBF5FB")
    plt.title("Interpretability Comparison Matrix", fontsize=12, fontweight="bold", pad=15)
    plt.tight_layout()
    out = fig_dir(output_dir) / "final_comparison_matrix.pdf"
    savefig_pdf(fig, out)
    print(f"Saved table PDF → {out}")
    plt.close()


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
    glm_checkpoint: Path | None = None,
    xgb_checkpoint: Path | None = None,
    chebykan_checkpoint: Path | None = None,
    fourierkan_checkpoint: Path | None = None,
    chebykan_config: Path | None = None,
    fourierkan_config: Path | None = None,
    chebykan_pruning_summary: Path | None = None,
    fourierkan_pruning_summary: Path | None = None,
    eval_features: Path | None = None,
    eval_labels: Path | None = None,
) -> None:
    from src.interpretability.utils.paths import figures as fig_dir, data as data_dir, reports as rep_dir

    glm_coef_path = data_dir(outputs_dir) / "glm_coefficients.csv"
    shap_path = data_dir(outputs_dir) / "shap_xgb_values.parquet"
    chebykan_sym = data_dir(outputs_dir) / "chebykan_symbolic_fits.csv"
    fourierkan_sym = data_dir(outputs_dir) / "fourierkan_symbolic_fits.csv"
    chebykan_prune = chebykan_pruning_summary or (rep_dir(outputs_dir) / "chebykan_pruning_summary.json")
    fourierkan_prune = fourierkan_pruning_summary or (rep_dir(outputs_dir) / "fourierkan_pruning_summary.json")

    # Auto-find checkpoints if not provided
    def _find_latest(pattern: str) -> Path:
        matches = sorted(glob.glob(pattern))
        return Path(matches[-1]) if matches else Path("")

    xgb_ckpt = xgb_checkpoint or _find_latest("checkpoints/xgb-baseline/*.joblib")
    glm_ckpt = glm_checkpoint or _find_latest("checkpoints/glm-baseline/*.joblib")
    cheby_ckpt = chebykan_checkpoint or Path("outputs/models/chebykan_pruned_module.pt")
    fourier_ckpt = fourierkan_checkpoint or Path("outputs/models/fourierkan_pruned_module.pt")
    cheby_cfg = chebykan_config or Path("configs/chebykan_experiment.yaml")
    fourier_cfg = fourierkan_config or Path("configs/fourierkan_experiment.yaml")
    eval_feat_path = eval_features or (data_dir(outputs_dir) / "X_eval.parquet")
    eval_lbl_path = eval_labels or (data_dir(outputs_dir) / "y_eval.parquet")

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

    # ── QWK retention curves (optional — requires all checkpoints) ────────────
    qwk_curves: dict[str, list[float]] = {}
    qwk_at_50: dict[str, float] = {}
    rankings: dict[str, list[str]] = {}
    all_ckpts_present = all(
        p.exists() for p in [glm_ckpt, xgb_ckpt, cheby_ckpt, fourier_ckpt,
                               cheby_cfg, fourier_cfg, eval_feat_path, eval_lbl_path, glm_coef_path]
        if str(p)
    )

    if all_ckpts_present:
        try:
            qwk_curves, qwk_at_50, rankings = _compute_qwk_retention_curve(
                glm_coef_path, shap_path,
                cheby_ckpt, fourier_ckpt,
                cheby_cfg, fourier_cfg,
                chebykan_prune, fourierkan_prune,
                glm_ckpt, xgb_ckpt,
                eval_feat_path, eval_lbl_path,
            )
            _plot_qwk_retention(qwk_curves, outputs_dir)

            for m in models:
                rows[m]["QWK @ 50% features retained"] = (
                    f"{qwk_at_50[m]:.4f}" if m in qwk_at_50 else "N/A"
                )
            overlap = _top5_overlap(rankings)
            for m in models:
                rows[m]["Top-5 feature overlap"] = str(overlap)

        except Exception as e:
            print(f"Warning: QWK retention curve failed: {e}")
            for m in models:
                rows[m]["QWK @ 50% features retained"] = "N/A"
                rows[m]["Top-5 feature overlap"] = "N/A"
    else:
        for m in models:
            rows[m]["QWK @ 50% features retained"] = "N/A"
            rows[m]["Top-5 feature overlap"] = "N/A"

    # ── Key Findings (fixed template) ─────────────────────────────────────────
    best_model = max(
        (m for m in models if rows[m].get("Validation QWK", "N/A") != "N/A"),
        key=lambda m: float(rows[m].get("Validation QWK", "0") or "0"),
        default="N/A",
    )
    best_qwk_val = rows[best_model].get("Validation QWK", "N/A") if best_model != "N/A" else "N/A"
    best_50_model = max(qwk_at_50, key=qwk_at_50.get) if qwk_at_50 else "N/A"
    if best_50_model != "N/A":
        best_full_qwk = float(rows[best_50_model].get("Validation QWK", "0") or "0")
        best_delta = best_full_qwk - qwk_at_50.get(best_50_model, 0.0)
    else:
        best_delta = float("nan")
    universal = sorted(
        set(rankings.get("GLM", [])[:5])
        .intersection(set(rankings.get("XGBoost", [])[:5]))
        .intersection(set(rankings.get("ChebyKAN", [])[:5]))
        .intersection(set(rankings.get("FourierKAN", [])[:5]))
    )
    universal_top3 = ", ".join(universal[:3]) if universal else "none"

    key_findings = (
        f"The best-performing model under full features is {best_model} (QWK={best_qwk_val}). "
        f"Under 50% feature retention, {best_50_model} degrades least "
        f"(ΔQWK={best_delta:.3f}). "
        f"The features {universal_top3} appear in all four methods' top-5, "
        f"indicating the strongest consensus signals."
    )

    # ── Markdown table ────────────────────────────────────────────────────────
    dimensions = [
        "Validation QWK",
        "Explanation method",
        "Structural compactness",
        "Mean R² of explanations",
        "Top-feature agreement with GLM (τ)",
        "QWK @ 50% features retained",
        "Top-5 feature overlap",
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

## Key Findings

{key_findings}

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

    # ── PDF table (replaces PNG) ───────────────────────────────────────────────
    _render_table_pdf(rows, models, dimensions, outputs_dir)



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final interpretability comparison matrix")
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    p.add_argument("--glm-checkpoint", type=Path, default=None)
    p.add_argument("--xgb-checkpoint", type=Path, default=None)
    p.add_argument("--chebykan-checkpoint", type=Path, default=None)
    p.add_argument("--fourierkan-checkpoint", type=Path, default=None)
    p.add_argument("--chebykan-config", type=Path, default=None)
    p.add_argument("--fourierkan-config", type=Path, default=None)
    p.add_argument("--chebykan-pruning-summary", type=Path, default=None)
    p.add_argument("--fourierkan-pruning-summary", type=Path, default=None)
    p.add_argument("--eval-features", type=Path, default=None)
    p.add_argument("--eval-labels", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        artifacts_dir=args.artifacts_dir,
        outputs_dir=args.outputs_dir,
        glm_checkpoint=args.glm_checkpoint,
        xgb_checkpoint=args.xgb_checkpoint,
        chebykan_checkpoint=args.chebykan_checkpoint,
        fourierkan_checkpoint=args.fourierkan_checkpoint,
        chebykan_config=args.chebykan_config,
        fourierkan_config=args.fourierkan_config,
        chebykan_pruning_summary=args.chebykan_pruning_summary,
        fourierkan_pruning_summary=args.fourierkan_pruning_summary,
        eval_features=args.eval_features,
        eval_labels=args.eval_labels,
    )
