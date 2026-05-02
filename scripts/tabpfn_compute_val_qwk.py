"""Compute validation QWK for each Run-A TabPFN seed.

For each of the three seed checkpoints, reload the wrapped estimator,
re-derive the validation split via the trainer's preprocessing pipeline,
predict on it (using the threshold that was fit during training), and
record QWK. Writes a JSON summary used to fill the Table 1 row in the
paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.config import load_experiment_config
from src.preprocessing import preprocess_tabpfn_paper as tabpfn_prep


REPO = Path(__file__).resolve().parent.parent
RUNS = [
    ("stage-c-tabpfn-full-seed42", 42, "tabpfn_full_seed42.yaml"),
    ("stage-c-tabpfn-full-seed1337", 1337, "tabpfn_full_seed1337.yaml"),
    ("stage-c-tabpfn-full-seed2024", 2024, "tabpfn_full_seed2024.yaml"),
]
CONFIG_DIR = REPO / "configs/experiment_stages/stage_c_explanation_package/tabpfn/all_features/train"
OUTPUT = REPO / "outputs/interpretability/tabpfn_paper/val_qwk_summary.json"


def _resolve_checkpoint(experiment_name: str) -> Path:
    candidates = sorted((REPO / "checkpoints" / experiment_name).glob("model-*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under checkpoints/{experiment_name}/")
    return candidates[-1]


def _val_qwk_for_seed(experiment_name: str, seed: int, config_filename: str) -> float:
    cfg = load_experiment_config(CONFIG_DIR / config_filename)
    train_csv = cfg.trainer.train_csv
    outputs = tabpfn_prep.run_pipeline(train_csv, random_seed=seed)
    inner_splits = outputs["inner_splits"]
    if not inner_splits:
        raise RuntimeError(f"No inner validation split produced for seed {seed}.")
    _, X_val, _, y_val = inner_splits[0]

    ckpt_path = _resolve_checkpoint(experiment_name)
    model = joblib.load(ckpt_path)

    preds = model.predict(X_val)
    return float(cohen_kappa_score(y_val, preds, weights="quadratic"))


def main() -> None:
    rows = []
    for experiment_name, seed, config_filename in RUNS:
        kappa = _val_qwk_for_seed(experiment_name, seed, config_filename)
        rows.append({"experiment_name": experiment_name, "seed": seed, "val_qwk": kappa})
        print(f"{experiment_name} (seed={seed}): val_qwk = {kappa:.4f}", flush=True)

    qwks = np.asarray([row["val_qwk"] for row in rows], dtype=float)
    summary = {
        "per_seed": rows,
        "mean_val_qwk": float(qwks.mean()),
        "std_val_qwk": float(qwks.std(ddof=1)),
        "n_seeds": len(rows),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
