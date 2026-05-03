"""Attention attribution for applicant 55728 against the TabPFN training set.

Forward-hooks the last attention layer of the highest-validation-QWK
Run-A seed's underlying TabPFNRegressor, captures attention weights on a
single forward pass over the training set + applicant 55728's row,
averages across heads, and saves the top-N most attended training
applicants with a small summary.

Timeboxed: if the hook fails (API drift, ensemble structure not exposing
member transformers, etc.), fall back to a 'mention only' artifact and
log the failure to runs/2026-05-02-tabpfn/04_attention.log.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import preprocess_tabpfn_paper as tabpfn_prep


REPO = _REPO_ROOT
VAL_QWK_SUMMARY = REPO / "outputs/interpretability/tabpfn_paper/val_qwk_summary.json"
APPLICANT_ID = 55728
TOP_N = 20
OUTPUT_DIR = REPO / "outputs/interpretability/tabpfn_paper"
LOG_PATH = REPO / "runs/2026-05-02-tabpfn/04_attention.log"


def _log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)


def _pick_best_seed_record() -> dict:
    summary = json.loads(VAL_QWK_SUMMARY.read_text())
    return max(summary["per_seed"], key=lambda r: r["val_qwk"])


def _resolve_checkpoint(experiment_name: str) -> Path:
    candidates = sorted((REPO / "checkpoints" / experiment_name).glob("model-*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint under checkpoints/{experiment_name}/")
    return candidates[-1]


def _resolve_inner_transformer(model):
    """Drill into the TabPFN wrapper to find the inner torch transformer.

    Tries a few attribute paths in case the package version exposes the
    transformer differently. Returns (inner_estimator, transformer_module)
    or raises if none of the paths exist.
    """

    inner = model._estimator  # base TabPFNRegressor
    for path in ("model_", "model", "_model", "predictor_._model"):
        try:
            obj = inner
            for part in path.split("."):
                obj = getattr(obj, part)
            if obj is not None:
                return inner, obj
        except AttributeError:
            continue
    raise RuntimeError("Could not resolve the inner TabPFN transformer.")


def _attention_hook(captured: dict):
    def _hook(module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2:
            captured["attn"] = output[1]
        elif hasattr(output, "attentions"):
            captured["attn"] = output.attentions
        else:
            captured["attn"] = None
    return _hook


def _run_attention_extraction() -> None:
    record = _pick_best_seed_record()
    experiment_name = record["experiment_name"]
    seed = record["seed"]
    _log(f"Using {experiment_name} (val_qwk={record['val_qwk']:.4f})")

    raw_df = pd.read_csv("data/prudential-life-insurance-assessment/train.csv")
    if APPLICANT_ID not in raw_df["Id"].values:
        raise RuntimeError(f"Applicant {APPLICANT_ID} not found in raw training CSV.")

    out = tabpfn_prep.run_pipeline(
        "data/prudential-life-insurance-assessment/train.csv", random_seed=seed,
    )
    X_train_outer = out["X_train_outer"]
    X_test_outer = out["X_test_outer"]
    full_X = pd.concat([X_train_outer, X_test_outer], axis=0)

    if APPLICANT_ID not in full_X.index:
        raise RuntimeError(f"Applicant {APPLICANT_ID} not in preprocessed feature index.")

    X_query = full_X.loc[[APPLICANT_ID]]
    X_pool = X_train_outer  # in-context training rows the model attends over

    ckpt = _resolve_checkpoint(experiment_name)
    model = joblib.load(ckpt)

    inner, transformer = _resolve_inner_transformer(model)
    _log(f"Resolved inner transformer: {type(transformer).__name__}")

    import torch  # noqa: F401

    attn_layers = [m for m in transformer.modules() if "Attention" in type(m).__name__]
    if not attn_layers:
        raise RuntimeError("No attention layers found in the resolved transformer.")
    last_attn = attn_layers[-1]
    captured: dict = {"attn": None}
    handle = last_attn.register_forward_hook(_attention_hook(captured))
    try:
        inner.predict(X_query)
    finally:
        handle.remove()

    if captured["attn"] is None:
        raise RuntimeError("Forward hook captured no attention tensor.")

    attn = captured["attn"]
    if hasattr(attn, "detach"):
        attn_np = attn.detach().cpu().numpy()
    else:
        attn_np = np.asarray(attn)

    while attn_np.ndim > 3:
        attn_np = attn_np.mean(axis=0)
    if attn_np.ndim == 3:
        attn_mean = attn_np.mean(axis=0)
    else:
        attn_mean = attn_np
    query_attn = attn_mean[-1, : len(X_pool)]
    if query_attn.size != len(X_pool):
        _log(f"WARNING: attention vector size {query_attn.size} != training pool size {len(X_pool)}")
        query_attn = query_attn[: len(X_pool)]

    order = np.argsort(query_attn)[::-1]
    top_idx = order[:TOP_N]
    top_train_ids = [int(X_pool.index[i]) for i in top_idx]
    top_weights = [float(query_attn[i]) for i in top_idx]

    raw_responses = raw_df.set_index("Id").loc[top_train_ids, "Response"].astype(int).tolist()

    csv_rows = []
    for tid, weight, resp in zip(top_train_ids, top_weights, raw_responses):
        row = {"train_applicant_id": tid, "attention_weight": weight, "true_response": resp}
        for feat in ("BMI", "Ins_Age", "Medical_History_15", "Product_Info_4"):
            if feat in raw_df.columns:
                row[feat] = float(raw_df.set_index("Id").loc[tid, feat])
        csv_rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(OUTPUT_DIR / "applicant_55728_attention.csv", index=False)
    _log(f"Wrote {OUTPUT_DIR / 'applicant_55728_attention.csv'}")

    predicted_class = int(model.predict(X_query)[0])
    responses = np.asarray(raw_responses, dtype=float)
    summary = {
        "applicant_id": APPLICANT_ID,
        "predicted_class": predicted_class,
        "top_n_attended_count": TOP_N,
        "mean_true_response_among_top_n": float(responses.mean()),
        "median_true_response_among_top_n": float(np.median(responses)),
        "std_true_response_among_top_n": float(responses.std(ddof=1)),
        "fraction_top_n_matching_predicted_class": float((responses == predicted_class).mean()),
    }
    (OUTPUT_DIR / "applicant_55728_attention_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    _log(f"Wrote {OUTPUT_DIR / 'applicant_55728_attention_summary.json'}")


def main() -> None:
    start = time.time()
    try:
        _run_attention_extraction()
        _log("attention extraction succeeded")
    except Exception as exc:
        elapsed = time.time() - start
        _log(f"attention extraction FAILED after {elapsed:.0f}s: {exc}")
        _log(traceback.format_exc())
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "applicant_55728_attention_fallback.json").write_text(
            json.dumps({
                "fallback_reason": str(exc),
                "elapsed_seconds": elapsed,
            }, indent=2)
        )
        _log("wrote fallback marker; paper §3.2.1 attention sentence will use mention-only wording")


if __name__ == "__main__":
    main()
