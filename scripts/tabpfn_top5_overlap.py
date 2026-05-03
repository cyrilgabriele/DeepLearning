"""Top-5 feature-overlap report between TabPFN, XGBoost-SHAP, and ChebyKAN.

Reads the three feature-ranking artifacts and writes a JSON summary with
the three top-5 lists, intersection counts, and shared feature names.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd


REPO = _REPO_ROOT
TABPFN_RANKING = REPO / "outputs/interpretability/tabpfn_paper/stage-c-tabpfn-full/data/tabpfn_feature_ranking.csv"
XGB_SHAP = REPO / "outputs/interpretability/xgboost_paper/stage-c-xgb-best/data/shap_xgb_values.parquet"
CHEBYKAN_RANKING = REPO / "outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_feature_ranking.csv"
OUTPUT = REPO / "outputs/interpretability/tabpfn_paper/feature_overlap.json"


def _tabpfn_top5() -> list[str]:
    df = pd.read_csv(TABPFN_RANKING)
    return df["feature"].head(5).tolist()


def _xgb_top5() -> list[str]:
    shap_df = pd.read_parquet(XGB_SHAP)
    if "Id" in shap_df.columns:
        shap_df = shap_df.drop(columns=["Id"])
    abs_mean = shap_df.abs().mean(axis=0).sort_values(ascending=False)
    return abs_mean.head(5).index.tolist()


def _chebykan_top5() -> list[str]:
    df = pd.read_csv(CHEBYKAN_RANKING)
    importance_col = "importance" if "importance" in df.columns else df.columns[1]
    return df.sort_values(importance_col, ascending=False)["feature"].head(5).tolist()


def main() -> None:
    tabpfn = _tabpfn_top5()
    xgb = _xgb_top5()
    chebykan = _chebykan_top5()

    tabpfn_xgb_intersection = sorted(set(tabpfn) & set(xgb))
    tabpfn_chebykan_intersection = sorted(set(tabpfn) & set(chebykan))

    payload = {
        "tabpfn_top5": tabpfn,
        "xgb_top5": xgb,
        "chebykan_top5": chebykan,
        "tabpfn_xgb_intersection_count": len(tabpfn_xgb_intersection),
        "tabpfn_chebykan_intersection_count": len(tabpfn_chebykan_intersection),
        "tabpfn_xgb_intersection": tabpfn_xgb_intersection,
        "tabpfn_chebykan_intersection": tabpfn_chebykan_intersection,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
