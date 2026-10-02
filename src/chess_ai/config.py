"""Validated experiment configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    required = {
        "model",
        "training",
        "search",
        "hardware",
        "dataset",
        "teacher",
        "self_play",
        "logging",
        "seed",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(sorted(missing))}")
    if config["model"]["channels"] <= 0 or config["model"]["blocks"] < 0:
        raise ValueError("Invalid model dimensions")
    if config["training"]["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    return config
