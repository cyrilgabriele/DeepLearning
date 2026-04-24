"""Quick gamma-grid sweep for XGBoost to find the actual outer-test ceiling.

The Optuna-reported best_qwk=0.6546 used an old `model=xgb` implementation
that is no longer in the registry, with gamma=3.8957. Retraining with the
current `xgboost-paper` model and that gamma gives worse outer-test than
gamma=0. This script sweeps gamma on a small log-grid to establish the
true best-case outer-test QWK reachable with current code.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import yaml
import subprocess


REPO = Path("/Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning")
BASE_CONFIG = REPO / "configs/experiment_stages/stage_c_explanation_package/xgboost_best.yaml"

GAMMAS = [0.0, 0.5, 1.0, 2.0, 3.8957, 6.0, 10.0]


def main() -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    results = []
    for g in GAMMAS:
        cfg = {**base}
        cfg["trainer"] = {**base["trainer"], "experiment_name": f"xgb-gamma-{g:.3f}"}
        cfg["model"] = {
            **base["model"],
            "params": {**base["model"]["params"], "gamma": g},
        }
        # Reduce n_estimators for speed
        cfg["model"]["params"]["n_estimators"] = 300
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            tmp_path = f.name

        print(f"\n=== gamma = {g:.3f} ===")
        result = subprocess.run(
            ["uv", "run", "python", "main.py", "--stage", "train", "--config", tmp_path],
            cwd=str(REPO), capture_output=True, text=True, timeout=1200,
        )
        # Parse QWK from stdout
        qwk = None
        for line in result.stdout.splitlines():
            if line.strip().startswith("qwk:"):
                qwk = float(line.split()[1])
                break
        print(f"  QWK = {qwk}")
        results.append((g, qwk))

    print("\n=== Summary ===")
    for g, qwk in results:
        marker = "  <-- Optuna" if abs(g - 3.8957) < 0.01 else ""
        print(f"  gamma = {g:6.3f}  QWK = {qwk}{marker}")


if __name__ == "__main__":
    main()
