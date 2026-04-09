"""Legacy script retained only to fail loudly.

The supported workflow is `main.py` with the typed config pipeline.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "src/submit.py is legacy and no longer supported. "
        "Use `uv run python main.py --stage train --config ...` so artifacts, "
        "checkpoints, and test predictions follow the current pipeline."
    )


if __name__ == "__main__":
    main()
