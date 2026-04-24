"""Scan sparsity_lambda for the tuned-config sparse ChebyKAN hero.

The Optuna-tuned ChebyKAN-20 config ([64,64], degree 7, lr 0.00095)
carries sparsity_lambda=0.0108 inherited from the original Pareto
sweep; at that lambda the model produces 2913 active edges with
QWK 0.593 but is not prune-robust (QWK collapses above threshold
0.01). We scan lambda ∈ {0.03, 0.05, 0.1, 0.2, 0.3} to find a
point with meaningful sparsity while preserving QWK.

For each lambda: train → apply L1 pruning at threshold 0.01 →
report (QWK_before, QWK_after, edges_after, sparsity_ratio).
"""

from __future__ import annotations

from pathlib import Path
import tempfile, subprocess, yaml, json


REPO = Path("/Users/gian1/CODE/HSG/FS26/DeepLearning/DeepLearning")
BASE = REPO / "configs/experiment_stages/stage_c_explanation_package/chebykan_tuned_sparse_hero.yaml"

LAMBDAS = [0.03, 0.05, 0.1, 0.2, 0.3]


def run(cmd, cwd=None, timeout=900):
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)


def main() -> None:
    base = yaml.safe_load(BASE.read_text())
    results = []

    for lam in LAMBDAS:
        cfg = {**base}
        cfg["trainer"] = {**base["trainer"], "experiment_name": f"chebykan-lambda-{lam:.3f}"}
        cfg["model"] = {**base["model"],
                        "params": {**base["model"]["params"], "sparsity_lambda": lam}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            tmp = Path(f.name)

        print(f"\n=== lambda = {lam} ===")
        print(f"[train]")
        r = run(["uv", "run", "python", "main.py", "--stage", "train",
                 "--config", str(tmp)], cwd=REPO, timeout=900)
        train_qwk = None
        for line in r.stdout.splitlines():
            s = line.strip()
            if s.startswith("qwk:"):
                train_qwk = float(s.split()[1])
                break
        print(f"  train QWK = {train_qwk}")

        # Now run interpret to prune
        print(f"[interpret/prune at threshold 0.01]")
        r2 = run(["uv", "run", "python", "main.py", "--stage", "interpret",
                  "--config", str(tmp),
                  "--pruning-threshold", "0.01",
                  "--candidate-library", "scipy",
                  "--max-features", "20"], cwd=REPO, timeout=900)
        # Parse pruning summary from stdout
        pruning_json_path = REPO / "outputs" / "interpretability" / "kan_paper" / f"chebykan-lambda-{lam:.3f}" / "reports" / "chebykan_pruning_summary.json"
        if pruning_json_path.exists():
            psum = json.loads(pruning_json_path.read_text())
            print(f"  edges_before={psum['edges_before']} edges_after={psum['edges_after']} "
                  f"sparsity={psum['sparsity_ratio']:.4f} "
                  f"qwk_before={psum['qwk_before']:.4f} qwk_after={psum['qwk_after']:.4f}")
            results.append({
                "lambda": lam,
                "train_qwk": train_qwk,
                **psum,
            })
        else:
            print(f"  pruning summary not found at {pruning_json_path}")

    print("\n\n=== SUMMARY ===")
    print(f"{'lambda':>8} {'train_qwk':>10} {'edges_after':>12} {'sparsity%':>10} {'qwk_before':>11} {'qwk_after':>10}")
    for r in results:
        print(f"{r['lambda']:>8.3f} {r['train_qwk']:>10.4f} {r['edges_after']:>12} "
              f"{r['sparsity_ratio']*100:>9.1f}% {r['qwk_before']:>11.4f} {r['qwk_after']:>10.4f}")

    (REPO / "outputs" / "reports" / "chebykan_lambda_scan.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
