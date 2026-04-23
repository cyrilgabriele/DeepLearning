"""Case-level finite-difference explanations for trained TabKAN models."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.interpretability.utils.paths import data as data_dir, reports as reports_dir


def _predict_scores(module, X_df: pd.DataFrame) -> np.ndarray:
    X_tensor = torch.tensor(X_df.to_numpy(dtype=np.float32, copy=False), dtype=torch.float32)
    with torch.no_grad():
        return module(X_tensor).cpu().numpy().flatten()


def _predict_changed_score(
    module,
    case_frame: pd.DataFrame,
    *,
    feature: str,
    new_value: float,
) -> float:
    candidate = case_frame.copy()
    candidate.at[candidate.index[0], feature] = new_value
    return float(_predict_scores(module, candidate)[0])


def _rounded_class(score: float) -> int:
    return int(np.clip(np.round(score), 1, 8))


def _base_feature_name(feature: str) -> str:
    base = feature
    for prefix in ("cb_", "qt_", "mm_"):
        if base.startswith(prefix):
            return base[len(prefix):]
    if base.startswith("missing_"):
        return base[len("missing_"):]
    return base


def _sanitize_row_id(row_id: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row_id))


def _infer_processed_to_raw_mapping(
    *,
    feature: str,
    X_eval: pd.DataFrame,
    X_eval_raw: pd.DataFrame | None,
) -> dict[float, Any]:
    if X_eval_raw is None or feature not in X_eval.columns:
        return {}
    base_feature = _base_feature_name(feature)
    if base_feature not in X_eval_raw.columns:
        return {}

    joined = pd.DataFrame(
        {
            "processed": X_eval[feature].astype(float),
            "raw": X_eval_raw[base_feature],
        }
    ).dropna(subset=["processed", "raw"])
    mapping: dict[float, Any] = {}
    for processed_value, group in joined.groupby("processed", sort=False):
        mode = group["raw"].mode(dropna=True)
        mapping[float(processed_value)] = mode.iloc[0] if not mode.empty else group["raw"].iloc[0]
    return mapping


def _raw_display_for_target(
    *,
    feature: str,
    target_processed: float,
    current_raw_value: Any,
    processed_to_raw: dict[float, Any],
) -> Any:
    if feature.startswith("missing_"):
        return "missing" if target_processed >= 0.5 else "observed"
    if processed_to_raw:
        nearest_key = min(processed_to_raw, key=lambda key: abs(float(key) - float(target_processed)))
        if abs(float(nearest_key) - float(target_processed)) < 1e-6:
            return processed_to_raw[nearest_key]
    if isinstance(current_raw_value, (int, float, np.integer, np.floating)) and math.isfinite(float(current_raw_value)):
        return float(target_processed)
    return target_processed


def _nearest_observed_value(values: np.ndarray, target: float) -> float:
    observed = np.unique(values.astype(float))
    return float(observed[np.argmin(np.abs(observed - target))])


def _analyze_feature(
    *,
    module,
    feature: str,
    feature_type: str,
    X_eval: pd.DataFrame,
    case_frame: pd.DataFrame,
    current_score: float,
    current_class: int,
    current_raw_value: Any,
    processed_to_raw: dict[float, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = X_eval[feature].astype(float).to_numpy()
    current_processed = float(case_frame.iloc[0][feature])
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    median_value = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)

    record: dict[str, Any] = {
        "feature": feature,
        "feature_type": feature_type,
        "current_processed_value": current_processed,
        "current_raw_value": current_raw_value,
        "local_slope": None,
        "finite_difference_step": None,
        "reference_label": None,
        "reference_processed_value": None,
        "reference_raw_value": None,
        "delta_to_reference_output": None,
        "delta_to_reference_class": None,
        "contribution_vs_reference": None,
        "delta_plus_one_iqr_output": None,
        "delta_minus_one_iqr_output": None,
        "delta_one_iqr_output": None,
        "delta_one_iqr_class": None,
        "one_iqr_direction": None,
        "one_iqr_target_processed_value": None,
        "one_iqr_target_raw_value": None,
    }
    scenarios: list[dict[str, Any]] = []

    def add_scenario(label: str, target_processed: float) -> tuple[float, int]:
        new_score = _predict_changed_score(
            module,
            case_frame,
            feature=feature,
            new_value=float(target_processed),
        )
        output_delta = float(new_score - current_score)
        class_delta = int(_rounded_class(new_score) - current_class)
        target_raw_value = _raw_display_for_target(
            feature=feature,
            target_processed=float(target_processed),
            current_raw_value=current_raw_value,
            processed_to_raw=processed_to_raw,
        )
        scenarios.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "scenario": label,
                "current_score": current_score,
                "new_score": new_score,
                "output_delta": output_delta,
                "class_delta": class_delta,
                "current_processed_value": current_processed,
                "target_processed_value": float(target_processed),
                "current_raw_value": current_raw_value,
                "target_raw_value": target_raw_value,
            }
        )
        return output_delta, class_delta

    if feature_type == "continuous":
        span = max_value - min_value
        step = max(1e-3, 0.05 * iqr, 0.01 * span if span > 0 else 0.0)
        lo = max(min_value, current_processed - step)
        hi = min(max_value, current_processed + step)
        if hi > lo:
            score_lo = _predict_changed_score(module, case_frame, feature=feature, new_value=lo)
            score_hi = _predict_changed_score(module, case_frame, feature=feature, new_value=hi)
            record["local_slope"] = float((score_hi - score_lo) / (hi - lo))
            record["finite_difference_step"] = float((hi - lo) / 2)

        median_target = float(np.clip(median_value, min_value, max_value))
        delta_reference_output, delta_reference_class = add_scenario("move_to_median", median_target)
        record["reference_label"] = "median"
        record["reference_processed_value"] = median_target
        record["reference_raw_value"] = _raw_display_for_target(
            feature=feature,
            target_processed=median_target,
            current_raw_value=current_raw_value,
            processed_to_raw=processed_to_raw,
        )
        record["delta_to_reference_output"] = delta_reference_output
        record["delta_to_reference_class"] = delta_reference_class
        record["contribution_vs_reference"] = float(-delta_reference_output)

        if iqr > 0:
            plus_target = float(np.clip(current_processed + iqr, min_value, max_value))
            minus_target = float(np.clip(current_processed - iqr, min_value, max_value))
            plus_delta, plus_class = add_scenario("plus_one_iqr", plus_target)
            minus_delta, minus_class = add_scenario("minus_one_iqr", minus_target)
            record["delta_plus_one_iqr_output"] = plus_delta
            record["delta_minus_one_iqr_output"] = minus_delta
            if abs(plus_delta) >= abs(minus_delta):
                record["delta_one_iqr_output"] = plus_delta
                record["delta_one_iqr_class"] = plus_class
                record["one_iqr_direction"] = "plus"
                record["one_iqr_target_processed_value"] = plus_target
                record["one_iqr_target_raw_value"] = _raw_display_for_target(
                    feature=feature,
                    target_processed=plus_target,
                    current_raw_value=current_raw_value,
                    processed_to_raw=processed_to_raw,
                )
            else:
                record["delta_one_iqr_output"] = minus_delta
                record["delta_one_iqr_class"] = minus_class
                record["one_iqr_direction"] = "minus"
                record["one_iqr_target_processed_value"] = minus_target
                record["one_iqr_target_raw_value"] = _raw_display_for_target(
                    feature=feature,
                    target_processed=minus_target,
                    current_raw_value=current_raw_value,
                    processed_to_raw=processed_to_raw,
                )
        return record, scenarios

    if feature_type == "ordinal":
        median_target = _nearest_observed_value(values, median_value)
        delta_reference_output, delta_reference_class = add_scenario("move_to_median_step", median_target)
        record["reference_label"] = "median_step"
        record["reference_processed_value"] = median_target
        record["reference_raw_value"] = _raw_display_for_target(
            feature=feature,
            target_processed=median_target,
            current_raw_value=current_raw_value,
            processed_to_raw=processed_to_raw,
        )
        record["delta_to_reference_output"] = delta_reference_output
        record["delta_to_reference_class"] = delta_reference_class
        record["contribution_vs_reference"] = float(-delta_reference_output)

        lower_target = _nearest_observed_value(values, float(np.percentile(values, 25)))
        upper_target = _nearest_observed_value(values, float(np.percentile(values, 75)))
        lower_delta, lower_class = add_scenario("move_to_lower_quartile_step", lower_target)
        upper_delta, upper_class = add_scenario("move_to_upper_quartile_step", upper_target)
        record["delta_plus_one_iqr_output"] = upper_delta
        record["delta_minus_one_iqr_output"] = lower_delta
        if abs(upper_delta) >= abs(lower_delta):
            record["delta_one_iqr_output"] = upper_delta
            record["delta_one_iqr_class"] = upper_class
            record["one_iqr_direction"] = "upper_quartile"
            record["one_iqr_target_processed_value"] = upper_target
            record["one_iqr_target_raw_value"] = _raw_display_for_target(
                feature=feature,
                target_processed=upper_target,
                current_raw_value=current_raw_value,
                processed_to_raw=processed_to_raw,
            )
        else:
            record["delta_one_iqr_output"] = lower_delta
            record["delta_one_iqr_class"] = lower_class
            record["one_iqr_direction"] = "lower_quartile"
            record["one_iqr_target_processed_value"] = lower_target
            record["one_iqr_target_raw_value"] = _raw_display_for_target(
                feature=feature,
                target_processed=lower_target,
                current_raw_value=current_raw_value,
                processed_to_raw=processed_to_raw,
            )
        return record, scenarios

    if feature_type in {"binary", "missing_indicator"}:
        toggle_target = float(0.0 if current_processed >= 0.5 else 1.0)
        delta_reference_output, delta_reference_class = add_scenario("toggle", toggle_target)
        record["reference_label"] = "toggle"
        record["reference_processed_value"] = toggle_target
        record["reference_raw_value"] = _raw_display_for_target(
            feature=feature,
            target_processed=toggle_target,
            current_raw_value=current_raw_value,
            processed_to_raw=processed_to_raw,
        )
        record["delta_to_reference_output"] = delta_reference_output
        record["delta_to_reference_class"] = delta_reference_class
        record["contribution_vs_reference"] = float(-delta_reference_output)
        return record, scenarios

    observed_values = pd.Series(values).value_counts()
    alt_values = [float(value) for value in observed_values.index.tolist() if float(value) != current_processed]
    target_value = alt_values[0] if alt_values else current_processed
    delta_reference_output, delta_reference_class = add_scenario("move_to_mode_alternative", target_value)
    record["reference_label"] = "mode_alternative"
    record["reference_processed_value"] = target_value
    record["reference_raw_value"] = _raw_display_for_target(
        feature=feature,
        target_processed=target_value,
        current_raw_value=current_raw_value,
        processed_to_raw=processed_to_raw,
    )
    record["delta_to_reference_output"] = delta_reference_output
    record["delta_to_reference_class"] = delta_reference_class
    record["contribution_vs_reference"] = float(-delta_reference_output)
    return record, scenarios


def run(
    module,
    X_eval: pd.DataFrame,
    *,
    output_dir: Path,
    flavor: str = "chebykan",
    feature_types: dict[str, str] | None = None,
    X_eval_raw: pd.DataFrame | None = None,
    candidate_features: list[str] | None = None,
    row_position: int = 0,
) -> dict[str, Path]:
    feature_types = feature_types or {}
    candidate_features = list(candidate_features or X_eval.columns.tolist())
    case_frame = X_eval.iloc[[row_position]].copy()
    base_score = float(_predict_scores(module, case_frame)[0])
    base_class = _rounded_class(base_score)

    raw_row = None
    if X_eval_raw is not None and row_position < len(X_eval_raw):
        raw_row = X_eval_raw.iloc[row_position]
    row_id = raw_row["Id"] if raw_row is not None and "Id" in raw_row.index else row_position
    row_slug = _sanitize_row_id(row_id)

    sensitivity_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    for feature in candidate_features:
        feature_type = feature_types.get(feature, "unknown")
        processed_to_raw = _infer_processed_to_raw_mapping(
            feature=feature,
            X_eval=X_eval,
            X_eval_raw=X_eval_raw,
        )
        current_raw_value = raw_row[_base_feature_name(feature)] if (
            raw_row is not None and _base_feature_name(feature) in raw_row.index
        ) else float(case_frame.iloc[0][feature])
        sensitivity_row, scenarios = _analyze_feature(
            module=module,
            feature=feature,
            feature_type=feature_type,
            X_eval=X_eval,
            case_frame=case_frame,
            current_score=base_score,
            current_class=base_class,
            current_raw_value=current_raw_value,
            processed_to_raw=processed_to_raw,
        )
        sensitivity_rows.append(sensitivity_row)
        scenario_rows.extend(scenarios)

    sensitivities_df = pd.DataFrame(sensitivity_rows).sort_values(
        "contribution_vs_reference",
        ascending=False,
        na_position="last",
        ignore_index=True,
    )
    what_if_df = pd.DataFrame(scenario_rows)
    if not what_if_df.empty:
        what_if_df["abs_output_delta"] = what_if_df["output_delta"].abs()
        what_if_df = what_if_df.sort_values(
            "abs_output_delta",
            ascending=False,
            ignore_index=True,
        )

    positive_df = sensitivities_df[sensitivities_df["contribution_vs_reference"].fillna(0) > 0].head(5)
    negative_df = sensitivities_df[sensitivities_df["contribution_vs_reference"].fillna(0) < 0].nsmallest(
        5,
        "contribution_vs_reference",
    )
    top_what_if_df = what_if_df.head(5)

    summary_lines = [
        f"# Case Summary — {row_id}",
        "",
        f"- Base model score: {base_score:.6f}",
        f"- Rounded risk class: {base_class}",
        "",
        "## Top Positive Contributors",
    ]
    if positive_df.empty:
        summary_lines.append("- None identified.")
    else:
        for _, row in positive_df.iterrows():
            summary_lines.append(
                f"- {row['feature']}: contribution_vs_reference={row['contribution_vs_reference']:.6f}"
            )

    summary_lines.extend(["", "## Top Negative Contributors"])
    if negative_df.empty:
        summary_lines.append("- None identified.")
    else:
        for _, row in negative_df.iterrows():
            summary_lines.append(
                f"- {row['feature']}: contribution_vs_reference={row['contribution_vs_reference']:.6f}"
            )

    summary_lines.extend(["", "## Top What-If Changes"])
    if top_what_if_df.empty:
        summary_lines.append("- None generated.")
    else:
        for _, row in top_what_if_df.iterrows():
            summary_lines.append(
                f"- {row['feature']} ({row['scenario']}): output_delta={row['output_delta']:.6f}, "
                f"class_delta={int(row['class_delta'])}"
            )

    report_path = reports_dir(output_dir) / f"{flavor}_case_summary_{row_slug}.md"
    sensitivities_path = data_dir(output_dir) / f"{flavor}_local_sensitivities_{row_slug}.csv"
    what_if_path = data_dir(output_dir) / f"{flavor}_case_what_if_{row_slug}.csv"

    report_path.write_text("\n".join(summary_lines) + "\n")
    sensitivities_df.to_csv(sensitivities_path, index=False)
    what_if_df.to_csv(what_if_path, index=False)

    return {
        "case_summary": report_path,
        "local_sensitivities": sensitivities_path,
        "what_if": what_if_path,
    }
