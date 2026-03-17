"""Issue 01 — GLM coefficient table.

Loads the trained GLM, extracts Ridge coefficients, and saves a ranked
CSV to outputs/glm_coefficients.csv.

Usage:
    uv run python -m src.interpretability.glm_coefficients \
        --checkpoint checkpoints/glm-baseline/model-<timestamp>.joblib \
        --features outputs/feature_names.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def extract_coefficients(model, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame with feature, coefficient, and abs_magnitude columns."""
    glm = model.model  # unwrap GLMBaseline → Ridge
    coefs = glm.coef_

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "abs_magnitude": np.abs(coefs),
    })
    df = df.sort_values("abs_magnitude", ascending=False).reset_index(drop=True)
    return df


def run(checkpoint_path: Path, features_path: Path, output_dir: Path = Path("outputs")) -> Path:
    import joblib

    model = joblib.load(checkpoint_path)
    feature_names = json.loads(features_path.read_text())

    df = extract_coefficients(model, feature_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "glm_coefficients.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} coefficients to {out_path}")

    # Sanity check: look for BMI/weight-related features
    health_kws = ["bmi", "weight", "ht_", "wt_", "ins_age"]
    for kw in health_kws:
        matches = df[df["feature"].str.lower().str.contains(kw)]
        if not matches.empty:
            top = matches.iloc[0]
            sign = "+" if top["coefficient"] > 0 else "-"
            print(f"  Sanity check '{kw}': {top['feature']} coef={sign}{abs(top['coefficient']):.4f}")

    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract GLM coefficients")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--features", type=Path, default=Path("outputs/feature_names.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.checkpoint, args.features, args.output_dir)
