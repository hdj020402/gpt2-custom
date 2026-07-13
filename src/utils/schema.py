"""Type schema for model_parameters.yml — validation + IDE autocompletion.

Usage::

    from src.utils.config import load_config
    cfg = load_config("configs/model_parameters.yml")  # validates required fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EarlyStoppingConfig:
    patience: int = 20
    threshold: float = 0.0


@dataclass
class AppConfig:
    """Top-level config schema.  ``load_config()`` fills this from YAML."""

    # ── General ──
    mode: str = ""           # training | hpo | generation
    jobtype: str = ""
    seed: int = 42

    # ── Dataset ──
    train_file: str | None = None
    val_file: str | None = None
    test_file: str | None = None
    target: str = "moltxt"
    corpus_file: str | None = None
    custom_tokens_file: str | None = None
    custom_tokenize: str | None = None
    custom_metrics: str | None = None
    vocab_size: int = 1000
    tk_num_proc: int = 4
    per_device_train_batch_size: int = 32
    dataloader_num_workers: int = 4

    # ── Model ──
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    dropout: float = 0.1

    # ── Training ──
    resume_from_checkpoint: str | None = None
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 30
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    warmup_steps: int | None = None
    eval_steps: int = 1000
    model_save_steps: int = 1000
    save_total_limit: int = 3
    logging_steps: int = 1000
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    # ── Mixed precision ──
    bf16: bool = True       # bfloat16 (recommended for Ampere+ GPUs: A100, RTX 3090/4090)
    fp16: bool = False      # float16 (use only if GPU doesn't support bf16, e.g. V100/T4)

    # ── Generation ──
    device: str = "cuda"
    pretrained_model: str | None = None
    start_token: str = "<Energy:>"
    stop_token: str = "</ei1a>"
    temperature: float = 0.0
    max_tokens: int = 100

    # ── Set at runtime ──
    time: str = ""
    output_dir: str = ""

    _REQUIRED_KEYS: tuple[str, ...] = (
        "mode",
        "jobtype",
    )

    def validate(self) -> None:
        """Crash early with a clear message when required fields are missing."""
        for key in self._REQUIRED_KEYS:
            value = getattr(self, key, None)
            if value is None or value == "":
                raise ValueError(
                    f"model_parameters.yml: '{key}' is required but is empty or missing.\n"
                    f"  Copy configs/model_parameters.example.yml → configs/model_parameters.yml and fill in all fields."
                )

        if ".." in self.jobtype or self.jobtype.startswith("/"):
            raise ValueError(
                f"model_parameters.yml: 'jobtype' must not contain '..' or start with '/' "
                f"(got '{self.jobtype}')."
            )

        valid_modes = ("training", "fine-tuning", "hpo", "generation")
        if self.mode not in valid_modes:
            raise ValueError(
                f"model_parameters.yml: 'mode' must be one of {valid_modes}, got '{self.mode}'."
            )

        if self.mode in ("training", "fine-tuning"):
            if not self.train_file or not self.val_file:
                raise ValueError(
                    f"model_parameters.yml: mode='{self.mode}' requires 'train_file' and 'val_file' to be set."
                )

        if self.mode == "generation":
            if not self.test_file:
                raise ValueError(
                    "model_parameters.yml: mode='generation' requires 'test_file' to be set."
                )
            if not self.pretrained_model:
                raise ValueError(
                    "model_parameters.yml: mode='generation' requires 'pretrained_model' to be set."
                )

    def to_dict(self) -> dict[str, Any]:
        """Flatten to dict for backward compatibility with existing code."""
        result = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, EarlyStoppingConfig):
                result[field_name] = {"patience": value.patience, "threshold": value.threshold}
            else:
                result[field_name] = value
        return result
