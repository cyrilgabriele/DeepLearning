"""Pick the strong sparse-reference sparsity_lambda from a 20-feature Pareto JSON.

Rule: among all Pareto points with sparsity_ratio >= MIN_SPARSITY, take the one
with the highest QWK. This matches Cyril's full-feature 'strong sparse reference'
selection (high sparsity, negligible QWK loss vs. the dense top point).

Usage:
    python scripts/pick_strong_sparse_lambda.py <pareto_json> [--min-sparsity 0.5]

Prints the chosen lambda to stdout. Exit 0 on success, 1 on no eligible point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pareto_json", type=Path)
    parser.add_argument("--min-sparsity", type=float, default=0.5)
    args = parser.parse_args()

    if not args.pareto_json.exists():
        print(f"Pareto JSON not found: {args.pareto_json}", file=sys.stderr)
        return 1

    payload = json.loads(args.pareto_json.read_text())
    front = payload.get("pareto_front", [])
    eligible = [p for p in front if p.get("sparsity_ratio", 0.0) >= args.min_sparsity]
    if not eligible:
        print(
            f"No Pareto point with sparsity_ratio >= {args.min_sparsity} "
            f"in {args.pareto_json} (front size {len(front)})",
            file=sys.stderr,
        )
        return 1

    chosen = max(eligible, key=lambda p: p.get("qwk", float("-inf")))
    lam = chosen["params"]["sparsity_lambda"]
    print(
        f"Chosen trial {chosen.get('trial_number')}: "
        f"qwk={chosen.get('qwk'):.6f} sparsity_ratio={chosen.get('sparsity_ratio'):.4f} "
        f"sparsity_lambda={lam}",
        file=sys.stderr,
    )
    print(lam)
    return 0


if __name__ == "__main__":
    sys.exit(main())
