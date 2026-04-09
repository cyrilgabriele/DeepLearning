"""Legacy script retained only to fail loudly.

The supported workflow is `main.py` with the `train`, `tune`, `interpret`,
`retrain`, and `select` stages.
"""

from __future__ import annotations

import sys


def main() -> None:
    raise SystemExit(
        "src/evaluate.py is legacy and no longer supported. "
        "Use `uv run python main.py --stage interpret ...` or the stage-level "
        "pipeline in `main.py` instead."
    )


if __name__ == "__main__":
    main()
