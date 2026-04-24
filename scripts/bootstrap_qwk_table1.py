"""Bootstrap 95% CI on outer-test QWK for all six Table 1 models.

For each model: load the saved checkpoint, run inference on the held-out
outer-test split (X_eval), apply the fitted ordinal thresholds (if any)
to get integer class predictions, compare to y_eval, and bootstrap the
QWK statistic with 1000 resamples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score

# Patch torch.load to default weights_only=False (we trust our own checkpoints).
# Needed because PyTorch 2.6 flipped the default, and XGBBaseline's joblib
# payload includes torch tensors that trip the strict unpickler.
_orig_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

from src.config import load_experiment_config
from src.metrics.qwk import _apply_thresholds
from src.models.tabkan import TabKAN


REPO = Path("/Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning")
EVAL_ROOT = REPO / "outputs" / "eval"
CHECKPOINT_DIR = REPO / "checkpoints"

MODELS = [
    # (exp_name, ckpt_stem, eval_recipe, config_yaml)
    ("stage-c-glm-baseline", "model-20260423-122752", "kan_paper",
     "configs/experiment_stages/stage_c_explanation_package/glm_baseline.yaml"),
    ("stage-c-xgboost-best", "model-20260412-165531", "xgboost_paper",
     None),
    ("stage-c-xgboost-top20-retuned", "model-20260424-135751", "kan_paper",
     None),
    ("stage-c-xgb-best", "model-20260424-144108", "xgboost_paper",
     None),
    ("stage-c-xgb-top20", "model-20260424-144138", "kan_paper",
     None),
    ("stage-c-xgb-top20-tune-tuned", "model-20260424-172456", "kan_paper",
     None),
    ("stage-c-chebykan-top20-tune-tuned", "model-20260424-172539", "kan_paper",
     "sweeps/stage-c-chebykan-top20-tune_best.yaml"),
    ("stage-c-fourierkan-top20-tune-tuned", "model-20260424-172802", "kan_paper",
     "sweeps/stage-c-fourierkan-top20-tune_best.yaml"),
    ("stage-c-chebykan-best", "model-20260423-123113", "kan_paper",
     "configs/experiment_stages/stage_c_explanation_package/chebykan_best.yaml"),
    ("stage-c-fourierkan-best", "model-20260423-123404", "kan_paper",
     "configs/experiment_stages/stage_c_explanation_package/fourierkan_best.yaml"),
    ("stage-c-chebykan-pareto-q0583-top20-noln", "model-20260422-103844", "kan_paper",
     "configs/experiment_stages/stage_c_explanation_package/chebykan_pareto_q0583_top20_noln.yaml"),
    ("stage-c-fourierkan-pareto-top20-noln", "model-20260423-092231", "kan_paper",
     "configs/experiment_stages/stage_c_explanation_package/fourierkan_pareto_top20_noln.yaml"),
]


def _load_eval(exp_name: str, recipe: str) -> tuple[pd.DataFrame, np.ndarray]:
    eval_dir = EVAL_ROOT / recipe / exp_name
    X_eval = pd.read_parquet(eval_dir / "X_eval.parquet")
    y_eval = pd.read_parquet(eval_dir / "y_eval.parquet").squeeze("columns").to_numpy()
    return X_eval, y_eval.astype(int)


def _apply_or_round(raw: np.ndarray, thresholds: np.ndarray | None) -> np.ndarray:
    if thresholds is not None and len(thresholds) > 0:
        return np.clip(_apply_thresholds(raw, thresholds), 1, 8).astype(int)
    return np.clip(np.round(raw), 1, 8).astype(int)


def _predict_tabkan(ckpt_path: Path, config_path: Path,
                    X_eval: pd.DataFrame, thresholds: np.ndarray | None) -> np.ndarray:
    cfg = load_experiment_config(config_path)
    state = torch.load(ckpt_path, map_location="cpu")

    # Remove "module." prefix if Lightning wrapped it
    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            clean_state[k[len("module."):]] = v
        else:
            clean_state[k] = v
    state = clean_state

    # Infer in_features from first KAN layer weight shape
    first_key = next(k for k in state.keys() if "cheby_coeffs" in k or "fourier_a" in k)
    in_features = state[first_key].shape[1]

    widths = cfg.model.resolved_hidden_widths()
    kwargs = dict(
        in_features=in_features,
        widths=widths,
        kan_type=cfg.model.flavor,
        use_layernorm=cfg.model.use_layernorm,
    )
    if cfg.model.flavor == "chebykan":
        kwargs["degree"] = cfg.model.degree or 3
    else:
        kwargs["grid_size"] = cfg.model.params.get("grid_size", 4)

    module = TabKAN(**kwargs)
    module.load_state_dict(state)
    module.eval()

    X_tensor = torch.tensor(X_eval.to_numpy(dtype=np.float32, copy=False),
                            dtype=torch.float32)
    with torch.no_grad():
        raw = module(X_tensor).cpu().numpy().flatten()
    return _apply_or_round(raw, thresholds)


def _predict_glm(ckpt_path: Path, X_eval: pd.DataFrame,
                 thresholds: np.ndarray | None) -> np.ndarray:
    import joblib
    obj = joblib.load(ckpt_path)
    # obj is a GLMBaseline instance; use its predict() which applies thresholds
    return obj.predict(X_eval)


def _predict_xgb(ckpt_path: Path, X_eval: pd.DataFrame,
                 thresholds: np.ndarray | None) -> np.ndarray:
    import joblib
    obj = joblib.load(ckpt_path)  # relies on module-level torch.load patch
    return obj.predict(X_eval)


def _load_thresholds_from_manifest(manifest_path: Path) -> np.ndarray | None:
    m = json.loads(manifest_path.read_text())
    oc = m.get("ordinal_calibration")
    if oc and oc.get("thresholds"):
        return np.array(oc["thresholds"], dtype=np.float32)
    return None


def predict_on_outer_test(exp_name: str, ckpt_stem: str, recipe: str,
                          config_path: str) -> tuple[np.ndarray, np.ndarray]:
    X_eval, y_eval = _load_eval(exp_name, recipe)

    ckpt_dir = CHECKPOINT_DIR / exp_name
    for suffix in (".pt", ".joblib", ".json"):
        p = ckpt_dir / f"{ckpt_stem}{suffix}"
        if p.exists():
            ckpt_path = p
            break
    else:
        raise FileNotFoundError(f"No checkpoint in {ckpt_dir}/{ckpt_stem}")

    manifest_path = ckpt_dir / f"{ckpt_stem}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    thresholds = _load_thresholds_from_manifest(manifest_path)

    model_name = manifest["config"]["model"]["name"]
    cfg_path = (REPO / config_path) if config_path is not None else None

    if model_name == "glm":
        y_pred = _predict_glm(ckpt_path, X_eval, thresholds)
    elif model_name in {"xgb", "xgboost", "xgb_paper", "xgboost-paper", "xgboost-paper"}:
        y_pred = _predict_xgb(ckpt_path, X_eval, thresholds)
    else:
        y_pred = _predict_tabkan(ckpt_path, cfg_path, X_eval, thresholds)

    return y_eval, y_pred


def bootstrap_qwk(y_true: np.ndarray, y_pred: np.ndarray,
                  n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = cohen_kappa_score(y_true[idx], y_pred[idx], weights="quadratic")
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main() -> None:
    results = []
    for exp_name, ckpt_stem, recipe, config_path in MODELS:
        print(f"Evaluating {exp_name} ...")
        try:
            y_true, y_pred = predict_on_outer_test(exp_name, ckpt_stem, recipe, config_path)
            point, lo, hi = bootstrap_qwk(y_true, y_pred)
            print(f"  QWK = {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  n={len(y_true)}")
            results.append({"experiment": exp_name, "qwk": point,
                            "ci_low": lo, "ci_high": hi, "n_test": len(y_true)})
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({"experiment": exp_name, "error": str(exc)})

    print("\n=== Summary ===")
    print(f"{'Model':50s} {'QWK':>8s} {'[lower95':>10s} {'upper95]':>10s}")
    for r in results:
        if "qwk" in r:
            print(f"{r['experiment']:50s} {r['qwk']:8.4f} {r['ci_low']:10.4f} {r['ci_high']:10.4f}")
        else:
            print(f"{r['experiment']:50s}  ERROR: {r['error']}")

    out_path = REPO / "outputs" / "reports" / "table1_bootstrap_qwk.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
