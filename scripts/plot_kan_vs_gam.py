#!/usr/bin/env python3
"""Produce four actuary-facing PDF figures from the KAN-vs-GAM grid search.

Reads:
    outputs/kan_vs_gam/kan_vs_gam_summary.csv
    outputs/kan_vs_gam/details/{name}.json

Writes (all PDFs):
    outputs/kan_vs_gam/figures/fig1_comparison_table.pdf
    outputs/kan_vs_gam/figures/fig2_accuracy_vs_interpretability.pdf
    outputs/kan_vs_gam/figures/fig3_feature_risk_curves.pdf
    outputs/kan_vs_gam/figures/fig4_edge_catalog.pdf

Usage:
    uv run python scripts/plot_kan_vs_gam.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.interpretability.utils.style import apply_paper_style, savefig_pdf

# ── Colours ──────────────────────────────────────────────────────────────────
GAM_BLUE = "#4C72B0"
KAN_GREEN = "#55A868"
HEADER_BG = "#2C3E50"
QWK_HIGHLIGHT = "#D5F5E3"  # light green for the QWK row

# ── Formula evaluation library (same candidates as the search script) ────────
FORMULA_FUNCS: dict[str, callable] = {
    "a*x + b": lambda x, p: p["a"] * x + p["b"],
    "a*x^2 + b*x + c": lambda x, p: p["a"] * x**2 + p["b"] * x + p["c"],
    "a*x^3 + b*x^2 + c*x + d": (
        lambda x, p: p["a"] * x**3 + p["b"] * x**2 + p["c"] * x + p["d"]
    ),
    "a*|x| + b": lambda x, p: p["a"] * np.abs(x) + p["b"],
    "a*cos(x) + b": lambda x, p: p["a"] * np.cos(x) + p["b"],
    "a*sin(x) + b": lambda x, p: p["a"] * np.sin(x) + p["b"],
    "a*sin(2*x) + b": lambda x, p: p["a"] * np.sin(2 * x) + p["b"],
    "a*sin(2*x) + b*cos(2*x)": (
        lambda x, p: p["a"] * np.sin(2 * x) + p["b"] * np.cos(2 * x)
    ),
    "a*sin(x) + b*cos(x)": (
        lambda x, p: p["a"] * np.sin(x) + p["b"] * np.cos(x)
    ),
    "a*exp(x) + b": (
        lambda x, p: p["a"] * np.exp(np.clip(x, -5, 5)) + p["b"]
    ),
    "a*log(|x|+1) + b": (
        lambda x, p: p["a"] * np.log(np.abs(x) + 1) + p["b"]
    ),
    "a*sqrt(|x|) + b": lambda x, p: p["a"] * np.sqrt(np.abs(x)) + p["b"],
    "a (constant)": lambda x, p: np.full_like(x, p["a"]),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_widths(val: str) -> list[int]:
    """Safely parse the hidden_widths string, e.g. '[8]' -> [8]."""
    return ast.literal_eval(val)


def _is_gam(widths: list[int]) -> bool:
    """1-layer (GAM) models have a single-element widths list."""
    return len(widths) == 1


def _load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["_widths"] = df["hidden_widths"].apply(_parse_widths)
    df["is_gam"] = df["_widths"].apply(_is_gam)
    return df


def _load_detail(detail_dir: Path, name: str) -> dict:
    path = detail_dir / f"{name}.json"
    return json.loads(path.read_text())


def _eval_formula(formula: str, params: dict | None, x: np.ndarray) -> np.ndarray | None:
    """Evaluate a formula string with fitted params over x.  Returns None on failure."""
    if params is None:
        return None
    func = FORMULA_FUNCS.get(formula)
    if func is None:
        return None
    try:
        return func(x, params)
    except Exception:
        return None


def _format_formula_short(formula: str, params: dict | None) -> str:
    """Return a compact human-readable formula with numbers substituted."""
    if params is None:
        return formula
    mapping = {
        "a*x^3 + b*x^2 + c*x + d":
            lambda p: f"{p['a']:.3f}x\u00b3 + {p['b']:.3f}x\u00b2 + {p['c']:.3f}x + {p['d']:.3f}",
        "a*x^2 + b*x + c":
            lambda p: f"{p['a']:.3f}x\u00b2 + {p['b']:.3f}x + {p['c']:.3f}",
        "a*x + b":
            lambda p: f"{p['a']:.3f}x + {p['b']:.3f}",
        "a*cos(x) + b":
            lambda p: f"{p['a']:.3f}cos(x) + {p['b']:.3f}",
        "a*sin(x) + b":
            lambda p: f"{p['a']:.3f}sin(x) + {p['b']:.3f}",
        "a*sin(2*x) + b":
            lambda p: f"{p['a']:.3f}sin(2x) + {p['b']:.3f}",
        "a*sin(2*x) + b*cos(2*x)":
            lambda p: f"{p['a']:.3f}sin(2x) + {p['b']:.3f}cos(2x)",
        "a*sin(x) + b*cos(x)":
            lambda p: f"{p['a']:.3f}sin(x) + {p['b']:.3f}cos(x)",
        "a*exp(x) + b":
            lambda p: f"{p['a']:.3f}exp(x) + {p['b']:.3f}",
        "a*log(|x|+1) + b":
            lambda p: f"{p['a']:.3f}log(|x|+1) + {p['b']:.3f}",
        "a*sqrt(|x|) + b":
            lambda p: f"{p['a']:.3f}\u221a|x| + {p['b']:.3f}",
        "a*|x| + b":
            lambda p: f"{p['a']:.3f}|x| + {p['b']:.3f}",
        "a (constant)":
            lambda p: f"{p['a']:.3f}",
    }
    fmt = mapping.get(formula)
    if fmt is not None:
        try:
            return fmt(params)
        except KeyError:
            pass
    return formula


# ── Figure 1: Model Comparison Table ─────────────────────────────────────────

def fig1_comparison_table(df: pd.DataFrame, detail_dir: Path, out_path: Path) -> None:
    """Render a side-by-side comparison table of best GAM vs best KAN as a PDF."""
    apply_paper_style()

    best_gam = df.loc[df[df["is_gam"]]["composite_score"].idxmax()]
    best_kan = df.loc[df[~df["is_gam"]]["composite_score"].idxmax()]

    gam_detail = _load_detail(detail_dir, best_gam["name"])
    kan_detail = _load_detail(detail_dir, best_kan["name"])

    # Derive row values
    gam_widths = _parse_widths(best_gam["hidden_widths"])
    kan_widths = _parse_widths(best_kan["hidden_widths"])

    rows = [
        ("Architecture",
         f"1-layer GAM  {best_gam['n_features']}\u2192{gam_widths[0]}\u21921",
         f"2-layer KAN  {best_kan['n_features']}\u2192{'x'.join(str(w) for w in kan_widths)}\u21921"),
        ("Features",
         str(int(best_gam["n_features"])),
         str(int(best_kan["n_features"]))),
        ("QWK",
         f"{best_gam['qwk']:.4f}",
         f"{best_kan['qwk']:.4f}"),
        ("Active edges",
         str(int(best_gam["n_active_edges"])),
         str(int(best_kan["n_active_edges"]))),
        ("Edge mean R\u00b2",
         f"{best_gam['mean_r2_l1']:.4f}",
         f"{best_kan['mean_r2_l1']:.4f}"),
        ("Edge % clean",
         f"{best_gam['pct_clean']:.1f}%",
         f"{best_kan['pct_clean']:.1f}%"),
        ("Feature mean R\u00b2",
         f"{best_gam['mean_r2_l2']:.4f}",
         f"{best_kan['mean_r2_l2']:.4f}"),
        ("Feature % clean",
         f"{best_gam['pct_features_interpretable']:.1f}%",
         f"{best_kan['pct_features_interpretable']:.1f}%"),
        ("Captures interactions",
         "No",
         "Yes"),
        ("Per-feature formula",
         "Exact",
         "Conditional"),
        ("Full model formula",
         "Exact",
         "Exact"),
    ]

    # Build the table
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    col_labels = ["Metric", "Best GAM (1-layer)", "Best KAN (2-layer)"]
    cell_text = [[r[0], r[1], r[2]] for r in rows]

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header row
    for col_idx in range(3):
        cell = table[0, col_idx]
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight QWK row (row index 3 in the table = row index 2 in rows, which is QWK)
    qwk_row_idx = 3  # 0 = header, 1 = Architecture, 2 = Features, 3 = QWK
    for col_idx in range(3):
        table[qwk_row_idx, col_idx].set_facecolor(QWK_HIGHLIGHT)

    # Left-align the metric column
    for row_idx in range(len(rows) + 1):
        table[row_idx, 0].set_text_props(ha="left")
        # Widen metric column
        table[row_idx, 0].set_width(0.35)

    fig.suptitle(
        f"Best GAM vs Best KAN — Model Comparison",
        fontsize=12, fontweight="bold", y=0.95,
    )

    savefig_pdf(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 2: Accuracy vs Interpretability Scatter ───────────────────────────

def fig2_accuracy_vs_interpretability(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of QWK vs per-feature R-squared, coloured by architecture type."""
    apply_paper_style()

    fig, ax = plt.subplots(figsize=(7, 5))

    gam = df[df["is_gam"]]
    kan = df[~df["is_gam"]]

    ax.scatter(
        gam["mean_r2_l2"], gam["qwk"],
        c=GAM_BLUE, marker="o", s=50, alpha=0.7, edgecolors="white",
        linewidths=0.5, label="1-layer (GAM)", zorder=3,
    )
    ax.scatter(
        kan["mean_r2_l2"], kan["qwk"],
        c=KAN_GREEN, marker="D", s=50, alpha=0.7, edgecolors="white",
        linewidths=0.5, label="2-layer (KAN)", zorder=3,
    )

    # Annotate best of each type
    best_gam = gam.loc[gam["composite_score"].idxmax()]
    best_kan = kan.loc[kan["composite_score"].idxmax()]

    for best, color, tag in [
        (best_gam, GAM_BLUE, "best GAM"),
        (best_kan, KAN_GREEN, "best KAN"),
    ]:
        ax.annotate(
            tag,
            (best["mean_r2_l2"], best["qwk"]),
            textcoords="offset points", xytext=(8, 6),
            fontsize=8, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )

    ax.set_xlabel("Per-Feature R\u00b2 (Level 2)")
    ax.set_ylabel("QWK (Accuracy)")
    ax.set_title("Accuracy vs Interpretability")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.15)

    savefig_pdf(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 3: Per-Feature Risk Curves ────────────────────────────────────────

def fig3_feature_risk_curves(
    df: pd.DataFrame, detail_dir: Path, out_path: Path
) -> None:
    """3x4 grid of reconstructed per-feature response curves for best GAM & KAN."""
    apply_paper_style()

    best_gam_row = df.loc[df[df["is_gam"]]["composite_score"].idxmax()]
    best_kan_row = df.loc[df[~df["is_gam"]]["composite_score"].idxmax()]

    gam_detail = _load_detail(detail_dir, best_gam_row["name"])
    kan_detail = _load_detail(detail_dir, best_kan_row["name"])

    # Build feature -> response dicts
    gam_resp = {r["feature"]: r for r in gam_detail["level2_feature_responses"]}
    kan_resp = {r["feature"]: r for r in kan_detail["level2_feature_responses"]}

    # Features present in both, sorted by max response_range descending
    common_features = sorted(
        set(gam_resp.keys()) & set(kan_resp.keys()),
        key=lambda f: max(
            gam_resp[f].get("response_range", 0),
            kan_resp[f].get("response_range", 0),
        ),
        reverse=True,
    )
    top12 = common_features[:12]

    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 10))
    axes_flat = axes.flatten()

    x = np.linspace(-3.0, 3.0, 500)

    for i, feat in enumerate(top12):
        ax = axes_flat[i]

        for resp, color, label in [
            (gam_resp[feat], GAM_BLUE, "GAM"),
            (kan_resp[feat], KAN_GREEN, "KAN"),
        ]:
            y = _eval_formula(resp["formula"], resp.get("params"), x)
            if y is not None:
                ax.plot(x, y, color=color, lw=1.8, label=label)
            else:
                # Cannot reconstruct -- plot placeholder
                ax.text(
                    0.5, 0.5, f"{label}: {resp['formula']}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color=color,
                )

        # Build subtitle with formula info
        gam_r2 = gam_resp[feat]["r_squared"]
        kan_r2 = kan_resp[feat]["r_squared"]
        gam_tier = gam_resp[feat]["quality_tier"]
        kan_tier = kan_resp[feat]["quality_tier"]
        gam_short = _format_formula_short(
            gam_resp[feat]["formula"], gam_resp[feat].get("params"),
        )
        kan_short = _format_formula_short(
            kan_resp[feat]["formula"], kan_resp[feat].get("params"),
        )

        ax.set_title(feat, fontsize=9, fontweight="bold")

        # Annotation box with formula details
        info_text = (
            f"GAM: {gam_short}\n"
            f"  R\u00b2={gam_r2:.4f} [{gam_tier}]\n"
            f"KAN: {kan_short}\n"
            f"  R\u00b2={kan_r2:.4f} [{kan_tier}]"
        )
        ax.text(
            0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel("x (normalised)", fontsize=7)
        ax.set_ylabel("f(x)", fontsize=7)
        ax.grid(True, alpha=0.15)

        if i == 0:
            ax.legend(fontsize=7, loc="lower right")

    # Hide unused axes
    for j in range(len(top12), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "Per-Feature Conditional Risk Curves — GAM (blue) vs KAN (green)\n"
        f"Best GAM: {best_gam_row['name']}  |  Best KAN: {best_kan_row['name']}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    savefig_pdf(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Figure 4: Edge Formula Catalog ───────────────────────────────────────────

def fig4_edge_catalog(df: pd.DataFrame, detail_dir: Path, out_path: Path) -> None:
    """Tables of all edge formulas for the best 2-layer KAN, split by layer."""
    apply_paper_style()

    best_kan_row = df.loc[df[~df["is_gam"]]["composite_score"].idxmax()]
    detail = _load_detail(detail_dir, best_kan_row["name"])
    edges = detail.get("level1_edge_fits", [])

    if not edges:
        print("  WARNING: no edge fits found for best KAN, skipping fig4")
        return

    # Split by layer
    layer0 = [e for e in edges if e["layer"] == 0]
    layer1 = [e for e in edges if e["layer"] == 1]

    # Sort each by l1_norm descending
    layer0.sort(key=lambda e: e.get("l1_norm", 0), reverse=True)
    layer1.sort(key=lambda e: e.get("l1_norm", 0), reverse=True)

    def _render_panel(ax, title: str, layer_edges: list[dict]) -> None:
        ax.axis("off")
        if not layer_edges:
            ax.text(0.5, 0.5, "(no active edges)", ha="center", va="center")
            ax.set_title(title, fontsize=10, fontweight="bold")
            return

        col_labels = ["Input", "Output", "Formula", "R\u00b2", "Quality", "L1"]
        cell_text = []
        for e in layer_edges:
            cell_text.append([
                str(e.get("input_feature", e["edge_in"])),
                str(e["edge_out"]),
                e["formula"],
                f"{e['r_squared']:.4f}",
                e["quality_tier"],
                f"{e.get('l1_norm', 0):.4f}",
            ])

        table = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(list(range(len(col_labels))))
        table.scale(1.0, 1.3)

        # Header style
        for col_idx in range(len(col_labels)):
            cell = table[0, col_idx]
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color="white", fontweight="bold")

        # Colour code quality tiers
        tier_colors = {
            "clean": "#D5F5E3",
            "acceptable": "#FEF9E7",
            "flagged": "#FADBD8",
        }
        for row_idx in range(1, len(cell_text) + 1):
            tier = cell_text[row_idx - 1][4]
            bg = tier_colors.get(tier, "white")
            for col_idx in range(len(col_labels)):
                table[row_idx, col_idx].set_facecolor(bg)

        ax.set_title(title, fontsize=10, fontweight="bold", pad=12)

    # Calculate figure height based on number of edges
    n_rows = max(len(layer0), len(layer1), 1)
    fig_height = max(6, 1.5 + 0.35 * n_rows)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, fig_height * 2),
        gridspec_kw={"height_ratios": [max(len(layer0), 1), max(len(layer1), 1)]},
    )

    _render_panel(
        ax_top,
        f"Layer 0: Input \u2192 Hidden  ({len(layer0)} active edges)",
        layer0,
    )
    _render_panel(
        ax_bot,
        f"Layer 1: Hidden \u2192 Output  ({len(layer1)} active edges)",
        layer1,
    )

    fig.suptitle(
        f"Edge Formula Catalog — Best KAN: {best_kan_row['name']}",
        fontsize=12, fontweight="bold", y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    savefig_pdf(fig, out_path)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    base_dir = Path("outputs/kan_vs_gam")
    csv_path = base_dir / "kan_vs_gam_summary.csv"
    detail_dir = base_dir / "details"
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"ERROR: summary CSV not found at {csv_path}")
        print("Run `uv run python scripts/kan_vs_gam_search.py` first.")
        sys.exit(1)

    print("Loading summary CSV ...")
    df = _load_summary(csv_path)
    print(f"  {len(df)} models loaded  "
          f"({df['is_gam'].sum()} GAM, {(~df['is_gam']).sum()} KAN)")

    print("\nFigure 1: Model Comparison Table")
    fig1_comparison_table(df, detail_dir, fig_dir / "fig1_comparison_table.pdf")

    print("\nFigure 2: Accuracy vs Interpretability Scatter")
    fig2_accuracy_vs_interpretability(df, fig_dir / "fig2_accuracy_vs_interpretability.pdf")

    print("\nFigure 3: Per-Feature Risk Curves")
    fig3_feature_risk_curves(df, detail_dir, fig_dir / "fig3_feature_risk_curves.pdf")

    print("\nFigure 4: Edge Formula Catalog")
    fig4_edge_catalog(df, detail_dir, fig_dir / "fig4_edge_catalog.pdf")

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
