"""Run the unsupervised anomaly-detection extension.

Usage:
    uv run python scripts/04_unsupervised.py
"""

from __future__ import annotations

import json

from fraud_detection.config import load_config
from fraud_detection.unsupervised import score_unsupervised, write_unsupervised_predictions


def main() -> None:
    cfg = load_config()
    if not cfg.artifacts.features.exists():
        raise SystemExit("Build features first: scripts/02_build_features.py")

    results = score_unsupervised(cfg)
    print(json.dumps(results, indent=2))
    write_unsupervised_predictions(cfg)
    print("anomaly scores written to artifacts/unsupervised_scores.parquet")


if __name__ == "__main__":
    main()
