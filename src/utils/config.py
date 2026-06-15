"""Load and validate model_parameters.yml."""

import yaml
import os

from src.utils.schema import AppConfig, EarlyStoppingConfig


def _dict_to_config(raw: dict) -> AppConfig:
    """Convert raw YAML dict to AppConfig, handling nested early_stopping."""
    # Handle nested early_stopping dict
    es_raw = raw.pop("early_stopping", {})
    if isinstance(es_raw, dict):
        raw["early_stopping"] = EarlyStoppingConfig(
            patience=es_raw.get("patience", 20),
            threshold=es_raw.get("threshold", 0),
        )
    elif isinstance(es_raw, EarlyStoppingConfig):
        raw["early_stopping"] = es_raw
    else:
        raw["early_stopping"] = EarlyStoppingConfig()

    # Drop YAML nulls that map to Python None to let dataclass defaults apply
    # but only for fields where None is not a valid value
    known_fields = {f.name for f in AppConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw.items() if k in known_fields}

    return AppConfig(**filtered)


def load_config(path: str = "configs/model_parameters.yml") -> dict:
    """Load YAML config, validate, return dict for backward compat.

    Raises FileNotFoundError if the config file is missing.
    Raises ValueError if required fields are empty or mode is invalid.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"  Copy configs/model_parameters.example.yml → configs/model_parameters.yml and edit."
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.full_load(f)

    if raw is None:
        raise ValueError(f"{path} is empty. Please fill in the configuration.")

    cfg = _dict_to_config(raw)
    cfg.validate()
    return cfg.to_dict()
