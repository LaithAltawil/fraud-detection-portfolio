"""Leakage-safe, time-based train/validation/test split.

Fraud is a temporal process and the 2,000 cardholders transact repeatedly, so a
random split would leak near-duplicate behaviour across folds and inflate scores.
Instead we sort the *labeled* timeline and cut it chronologically:

    |------------ train ------------|---- validation ----|---- test ----|
    70%                              15%                  15%

Reported metrics are therefore "future" performance, which is what a deployed
model actually faces.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .config import Config


@dataclass
class Splits:
    train: pl.DataFrame
    valid: pl.DataFrame
    test: pl.DataFrame
    train_end: object
    valid_end: object

    def sizes(self) -> dict[str, int]:
        return {"train": self.train.height, "valid": self.valid.height, "test": self.test.height}


def time_split(features_path: str, cfg: Config, label_col: str = "is_fraud") -> Splits:
    """Split the labeled feature table chronologically."""
    lf = pl.scan_parquet(features_path).filter(pl.col(label_col).is_not_null())

    # Determine cut points on the labeled timeline (position-based so it works
    # with datetime columns regardless of quantile interpolation support).
    ts = lf.select("ts").collect().to_series().sort()
    n = len(ts)
    if n == 0:
        raise ValueError("No labeled rows found; check the labels file.")
    i1 = int(n * cfg.split.train_frac)
    i2 = int(n * (cfg.split.train_frac + cfg.split.valid_frac))
    train_end, valid_end = ts[i1], ts[i2]

    train = lf.filter(pl.col("ts") <= train_end).collect()
    valid = lf.filter((pl.col("ts") > train_end) & (pl.col("ts") <= valid_end)).collect()
    test = lf.filter(pl.col("ts") > valid_end).collect()

    return Splits(train, valid, test, train_end, valid_end)
