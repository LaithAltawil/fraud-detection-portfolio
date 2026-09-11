"""Build the causal feature table from the raw files.

Usage:
    uv run python scripts/02_build_features.py            # full 13.3M rows
    uv run python scripts/02_build_features.py --sample 2000000 --force
"""

from __future__ import annotations

import argparse
import time

from fraud_detection import data, features
from fraud_detection.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fraud feature table.")
    parser.add_argument("--sample", type=int, default=None, help="Use only the first N raw rows.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if Parquet exists.")
    args = parser.parse_args()

    cfg = load_config()
    if args.sample:
        cfg.features.sample_rows = args.sample

    t = time.time()
    con = data.prepare(cfg)
    print(f"registered raw views in {time.time() - t:.1f}s")

    t = time.time()
    path = features.build_features(con, cfg, force=args.force)
    print(f"features written to {path} in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
