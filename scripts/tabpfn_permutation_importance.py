"""Feature importance for TabPFN via batched mean-replacement ablation.

Uses the seed-42 Stage C TabPFN checkpoint. For each of the 126
features, replaces that feature's column with its mean across the chosen
outer-test subsample, then measures the QWK drop relative to the unperturbed
baseline. Higher QWK drop = higher feature importance.

Why mean-replacement ablation instead of sklearn's `permutation_importance`:
TabPFN's `predict()` is dominated by per-call context-embedding overhead
(~75 s per call on the 10k-row in-context buffer), so a 631-call permutation
loop runs for >12 hours. Batched ablation stacks 126 perturbed datasets
into one large predict call, reducing total predict calls from 631 to 2 and
total wall-clock to ~30 minutes.

Outputs:
  - outputs/interpretability/tabpfn_paper/stage-c-tabpfn-full-seed42/data/tabpfn_feature_ranking.csv
  - configs/experiment_stages/stage_c_explanation_package/feature_lists/tabpfn_top20_features.json

The top-20 list feeds the Run B 20-feature config.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.preprocessing import preprocess_tabpfn_paper as tabpfn_prep


REPO = _REPO_ROOT
EXPERIMENT_NAME = "stage-c-tabpfn-full-seed42"
SEED = 42
RANKING_OUT = REPO / f"outputs/interpretability/tabpfn_paper/{EXPERIMENT_NAME}/data/tabpfn_feature_ranking.csv"
TOP20_OUT = REPO / "configs/experiment_stages/stage_c_explanation_package/feature_lists/tabpfn_top20_features.json"
SUBSAMPLE_SIZE = 100
RANDOM_STATE = 42


def _resolve_checkpoint(experiment_name: str) -> Path:
    candidates = sorted((REPO / "checkpoints" / experiment_name).glob("model-*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under checkpoints/{experiment_name}/")
    return candidates[-1]


def main() -> None:
    experiment_name = EXPERIMENT_NAME
    seed = SEED
    print(f"Using fixed figure seed: {experiment_name}", flush=True)

    ckpt = _resolve_checkpoint(experiment_name)
    print(f"Loading {ckpt}", flush=True)
    model = joblib.load(ckpt)

    print("Reconstructing outer-test split via tabpfn_paper preprocessing", flush=True)
    out = tabpfn_prep.run_pipeline(
        "data/prudential-life-insurance-assessment/train.csv", random_seed=seed,
    )
    X_eval = out["X_test_outer"]
    y_eval = out["y_test_outer"].to_numpy().astype(int)
    print(f"X_eval shape: {X_eval.shape}", flush=True)

    rng = np.random.default_rng(RANDOM_STATE)
    if len(X_eval) > SUBSAMPLE_SIZE:
        idx = np.sort(rng.choice(len(X_eval), size=SUBSAMPLE_SIZE, replace=False))
        X_sub = X_eval.iloc[idx].reset_index(drop=True)
        y_sub = y_eval[idx]
        print(f"Subsampled to {SUBSAMPLE_SIZE} rows for ablation analysis", flush=True)
    else:
        X_sub = X_eval.reset_index(drop=True)
        y_sub = y_eval

    feature_names = list(X_sub.columns)
    n_features = len(feature_names)

    # Step 1: baseline predict on the subsample
    print(f"Step 1/2: baseline predict on {len(X_sub)} rows", flush=True)
    t0 = time.time()
    baseline_preds = model.predict(X_sub)
    baseline_qwk = cohen_kappa_score(y_sub, baseline_preds, weights="quadratic")
    print(f"  baseline QWK = {baseline_qwk:.4f} ({time.time() - t0:.1f}s)", flush=True)

    # Step 2: build a stacked batch of 126 perturbed copies, predict once
    print(f"Step 2/2: stacked predict on {n_features} * {len(X_sub)} = {n_features * len(X_sub)} rows", flush=True)
    feature_means = X_sub.mean(axis=0)
    stacked = []
    for feat in feature_names:
        perturbed = X_sub.copy()
        perturbed[feat] = feature_means[feat]
        stacked.append(perturbed)
    stacked_df = pd.concat(stacked, axis=0, ignore_index=True)
    print(f"  stacked shape: {stacked_df.shape}", flush=True)

    t0 = time.time()
    stacked_preds = model.predict(stacked_df)
    print(f"  stacked predict done ({time.time() - t0:.1f}s)", flush=True)

    # Step 3: split predictions per feature, compute QWK drop
    importance = []
    for i, feat in enumerate(feature_names):
        chunk = stacked_preds[i * len(X_sub):(i + 1) * len(X_sub)]
        perturbed_qwk = cohen_kappa_score(y_sub, chunk, weights="quadratic")
        importance.append({
            "feature": feat,
            "importance": baseline_qwk - perturbed_qwk,
            "perturbed_qwk": perturbed_qwk,
        })

    df = pd.DataFrame(importance).sort_values("importance", ascending=False).reset_index(drop=True)
    df["importance_std"] = 0.0  # ablation is deterministic; std column kept for schema parity

    RANKING_OUT.parent.mkdir(parents=True, exist_ok=True)
    df[["feature", "importance", "importance_std", "perturbed_qwk"]].to_csv(RANKING_OUT, index=False)
    print(f"Wrote {RANKING_OUT}", flush=True)

    top20 = df["feature"].head(20).tolist()
    TOP20_OUT.parent.mkdir(parents=True, exist_ok=True)
    TOP20_OUT.write_text(json.dumps(top20, indent=2))
    print(f"Wrote {TOP20_OUT}", flush=True)
    print("Top-20 features:", top20, flush=True)


if __name__ == "__main__":
    main()
