"""Derive top-20 feature lists from the new full-feature interpret outputs.

Reads:
  - outputs/interpretability/kan_paper/stage-c-chebykan-best/data/chebykan_feature_ranking.csv
  - outputs/interpretability/kan_paper/stage-c-fourierkan-best/data/fourierkan_feature_ranking.csv
  - outputs/interpretability/xgboost_paper/stage-c-xgb-best/data/shap_xgb_values.parquet

Writes (overwrites in place, keeping legacy filenames):
  - configs/experiment_stages/stage_c_explanation_package/feature_lists/chebykan_pareto_q0583_top20_features.json
  - configs/experiment_stages/stage_c_explanation_package/feature_lists/fourierkan_tuned_top20_features.json
  - configs/experiment_stages/stage_c_explanation_package/feature_lists/xgb_tuned_top20_features.json

Each output JSON: top-level array of 20 feature-name strings, sorted by importance desc.

Per-model failure isolation: if one source is missing, that JSON is skipped with a
warning and the others still update. Exit 0 if at least one succeeded, 1 otherwise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
INTERPRET_ROOT = REPO / "outputs" / "interpretability"
FEATURE_LISTS = REPO / "configs" / "experiment_stages" / "stage_c_explanation_package" / "feature_lists"

JOBS = [
    {
        "name": "chebykan",
        "kind": "kan",
        "source": INTERPRET_ROOT / "kan_paper" / "stage-c-chebykan-best" / "data" / "chebykan_feature_ranking.csv",
        "out": FEATURE_LISTS / "chebykan_pareto_q0583_top20_features.json",
    },
    {
        "name": "fourierkan",
        "kind": "kan",
        "source": INTERPRET_ROOT / "kan_paper" / "stage-c-fourierkan-best" / "data" / "fourierkan_feature_ranking.csv",
        "out": FEATURE_LISTS / "fourierkan_tuned_top20_features.json",
    },
    {
        "name": "xgb",
        "kind": "xgb",
        "source": INTERPRET_ROOT / "xgboost_paper" / "stage-c-xgb-best" / "data" / "shap_xgb_values.parquet",
        "out": FEATURE_LISTS / "xgb_tuned_top20_features.json",
    },
]

TOP_K = 20


def _kan_top20(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns or "importance" not in df.columns:
        raise ValueError(f"Expected columns feature,importance in {csv_path}; got {df.columns.tolist()}")
    df = df.sort_values("importance", ascending=False, kind="stable")
    return df["feature"].head(TOP_K).tolist()


def _xgb_top20(parquet_path: Path) -> list[str]:
    shap = pd.read_parquet(parquet_path)
    if "Id" in shap.columns:
        shap = shap.drop(columns=["Id"])
    mean_abs = shap.abs().mean(axis=0).sort_values(ascending=False, kind="stable")
    return mean_abs.head(TOP_K).index.tolist()


def main() -> int:
    successes = 0
    for job in JOBS:
        name = job["name"]
        src = job["source"]
        out = job["out"]
        try:
            if not src.exists():
                print(f"[SKIP {name}] source missing: {src}", file=sys.stderr)
                continue
            features = _kan_top20(src) if job["kind"] == "kan" else _xgb_top20(src)
            if len(features) < TOP_K:
                print(f"[WARN {name}] only {len(features)} features available", file=sys.stderr)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(features, indent=2) + "\n")
            print(f"[OK {name}] wrote top-{len(features)} -> {out}")
            successes += 1
        except Exception as exc:
            print(f"[FAIL {name}] {exc}", file=sys.stderr)
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
