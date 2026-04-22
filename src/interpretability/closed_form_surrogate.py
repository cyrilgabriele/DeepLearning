"""Closed-form polynomial surrogate for simplified KAN predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import cohen_kappa_score, r2_score
from sklearn.preprocessing import PolynomialFeatures

from src.interpretability.utils.paths import reports as reports_dir


def _predict_scores(module, X_df: pd.DataFrame) -> np.ndarray:
    X_tensor = torch.tensor(X_df.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)
    with torch.no_grad():
        return module(X_tensor).cpu().numpy().flatten()


def _rounded_classes(scores: np.ndarray) -> np.ndarray:
    return np.clip(np.round(scores), 1, 8).astype(int)


def _format_polynomial_formula(
    feature_names: list[str],
    poly: PolynomialFeatures,
    model: RidgeCV,
) -> str:
    terms = []
    if abs(float(model.intercept_)) > 1e-12:
        terms.append(f"{float(model.intercept_):.12g}")

    for coefficient, term in zip(model.coef_, poly.get_feature_names_out(feature_names)):
        coeff_value = float(coefficient)
        if abs(coeff_value) <= 1e-12:
            continue
        sympy_term = term.replace("^", "**").replace(" ", " * ")
        terms.append(f"({coeff_value:.12g}) * ({sympy_term})")

    return " + ".join(terms) if terms else "0.0"


def run(
    module,
    X_eval: pd.DataFrame,
    *,
    output_dir: Path,
    feature_names: list[str] | None = None,
    y_eval: pd.Series | None = None,
    flavor: str = "chebykan",
) -> dict[str, object]:
    feature_names = list(feature_names or X_eval.columns.tolist())
    X_surrogate = X_eval.loc[:, feature_names].copy()
    target_scores = _predict_scores(module, X_eval)

    best_payload: dict[str, object] | None = None
    for degree in (1, 2, 3):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_surrogate.to_numpy(dtype=np.float64, copy=False))
        model = RidgeCV(alphas=np.logspace(-6, 3, 10))
        model.fit(X_poly, target_scores)
        predictions = model.predict(X_poly)
        fidelity_r2 = float(r2_score(target_scores, predictions))

        payload = {
            "degree": degree,
            "alpha": float(model.alpha_),
            "poly": poly,
            "model": model,
            "predictions": predictions,
            "fidelity_r2": fidelity_r2,
        }
        if best_payload is None or fidelity_r2 > float(best_payload["fidelity_r2"]):
            best_payload = payload

    assert best_payload is not None
    surrogate_predictions = np.asarray(best_payload["predictions"], dtype=float)
    surrogate_classes = _rounded_classes(surrogate_predictions)
    qwk_surrogate = None
    if y_eval is not None:
        qwk_surrogate = float(cohen_kappa_score(y_eval, surrogate_classes, weights="quadratic"))

    formula = _format_polynomial_formula(
        feature_names,
        best_payload["poly"],
        best_payload["model"],
    )
    report = {
        "flavor": flavor,
        "label": "surrogate of the simplified KAN",
        "feature_names": feature_names,
        "degree": int(best_payload["degree"]),
        "alpha": float(best_payload["alpha"]),
        "fidelity_r2": round(float(best_payload["fidelity_r2"]), 6),
        "qwk_surrogate": round(qwk_surrogate, 6) if qwk_surrogate is not None else None,
        "formula": formula,
    }

    report_dir = reports_dir(output_dir)
    json_path = report_dir / f"{flavor}_closed_form_surrogate.json"
    md_path = report_dir / f"{flavor}_closed_form_surrogate.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_lines = [
        f"# {flavor} — Closed-Form Surrogate",
        "",
        f"- Label: {report['label']}",
        f"- Degree: {report['degree']}",
        f"- Ridge alpha: {report['alpha']}",
        f"- Fidelity R^2: {report['fidelity_r2']}",
        f"- Surrogate rounded QWK: {report['qwk_surrogate']}",
        "",
        "## Formula",
        "",
        "```text",
        formula,
        "```",
    ]
    md_path.write_text("\n".join(md_lines) + "\n")

    return {
        "report": report,
        "json_path": json_path,
        "md_path": md_path,
    }
