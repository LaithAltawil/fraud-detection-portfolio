"""Configuration loading and path resolution.

The project keeps every tunable in ``config/config.yaml`` so experiments are
reproducible from a single source of truth. Paths inside the config are resolved
relative to the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# repo_root = .../fraud-detection-portfolio  (file is at src/fraud_detection/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass
class DataConfig:
    raw_dir: Path
    transactions: Path
    users: Path
    cards: Path
    mcc: Path
    labels: Path


@dataclass
class ArtifactsConfig:
    dir: Path
    features: Path
    predictions: Path
    model: Path
    metrics: Path


@dataclass
class FeaturesConfig:
    sample_rows: int | None = None
    rolling_window: int = 5
    use_categoricals: bool = False


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    valid_frac: float = 0.15

    @property
    def test_frac(self) -> float:
        return round(1.0 - self.train_frac - self.valid_frac, 10)


@dataclass
class ModelConfig:
    random_state: int = 42
    params: dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 100
    metric: str = "average_precision"


@dataclass
class EvaluationConfig:
    cost_false_negative: float = 200.0
    cost_false_positive: float = 5.0
    fpr_targets: list[float] = field(default_factory=lambda: [0.001, 0.005, 0.01, 0.05])


@dataclass
class Config:
    data: DataConfig
    artifacts: ArtifactsConfig
    features: FeaturesConfig
    split: SplitConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    seed: int = 42
    project_root: Path = PROJECT_ROOT


def _resolve(base: Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the YAML config, returning typed dataclasses."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    base = PROJECT_ROOT
    files = raw["data"]["files"]
    raw_dir = _resolve(base, raw["data"]["raw_dir"])

    data = DataConfig(
        raw_dir=raw_dir,
        transactions=raw_dir / files["transactions"],
        users=raw_dir / files["users"],
        cards=raw_dir / files["cards"],
        mcc=raw_dir / files["mcc"],
        labels=raw_dir / files["labels"],
    )

    art_dir = _resolve(base, raw["artifacts"]["dir"])
    artifacts = ArtifactsConfig(
        dir=art_dir,
        features=art_dir / raw["artifacts"]["features"],
        predictions=art_dir / raw["artifacts"]["predictions"],
        model=art_dir / raw["artifacts"]["model"],
        metrics=art_dir / raw["artifacts"]["metrics"],
    )

    features = FeaturesConfig(**raw["features"])
    split = SplitConfig(**raw["split"])
    model = ModelConfig(**raw["model"])
    evaluation = EvaluationConfig(**raw["evaluation"])

    return Config(
        data=data,
        artifacts=artifacts,
        features=features,
        split=split,
        model=model,
        evaluation=evaluation,
        seed=raw.get("seed", 42),
    )
