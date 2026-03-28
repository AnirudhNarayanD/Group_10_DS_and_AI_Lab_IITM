"""Unified training entry point.

Dispatches to the appropriate training script based on --model-type:
  baseline   → models/baseline/train.py   (Encoder + Linear, CE only)
  middleware → models/middleware/train.py  (Encoder + MiddleBlock + Decoder + SafetyGate, CE + BCE)

Usage:
  python -m models.train --model-type baseline   --train-data ... --val-data ... --output-dir ...
  python -m models.train --model-type middleware  --train-data ... --val-data ... --output-dir ...

Or run each directly:
  python -m models.baseline.train   --train-data ... --val-data ... --output-dir ...
  python -m models.middleware.train --train-data ... --val-data ... --output-dir ...
"""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--model-type":
        print(__doc__)
        sys.exit(1)

    model_type = sys.argv[2]
    # Remove --model-type <value> so the sub-script's argparse sees only its args
    sys.argv = [sys.argv[0]] + sys.argv[3:]

    if model_type == "baseline":
        from models.baseline.train import main as baseline_main
        baseline_main()
    elif model_type == "middleware":
        from models.middleware.train import main as middleware_main
        middleware_main()
    else:
        print(f"Unknown model type: {model_type!r}. Choose 'baseline' or 'middleware'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
