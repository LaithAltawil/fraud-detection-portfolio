"""Train and evaluate the supervised fraud models.

Usage:
    uv run python scripts/03_train.py
"""

from __future__ import annotations

import json
import time

from fraud_detection.config import load_config
from fraud_detection.train import train


def main() -> None:
    cfg = load_config()
    if not cfg.artifacts.features.exists():
        raise SystemExit(
            f"Feature table not found at {cfg.artifacts.features}. "
            "Run scripts/02_build_features.py first."
        )
    t = time.time()
    metrics = train(cfg)
    print(json.dumps(metrics, indent=2))
    print(f"total training time: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
