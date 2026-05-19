"""
Custom module template for gpt2-custom framework.

Copy this file to custom/custom_module.py and modify as needed.
Set in configs/model_parameters.yml:
    custom_module: custom/custom_module.py

All definitions below are optional. Remove or comment out anything you don't need.
"""

import numpy as np
from typing import Any
from transformers import PreTrainedTokenizer, EvalPrediction

# ═══════════════════════════════════════════════════════════════════════════════
# Module-level config variables (read by the framework)
# ═══════════════════════════════════════════════════════════════════════════════

# Whether dataset.map() should pass batches (True) or single samples (False)
# Set to False if your tokenize function needs per-sample offset_mapping
tk_batched: bool = True

# Metric used for best model selection and early stopping
metric_for_best_model: str = "eval_loss"
greater_is_better: bool = False

# Injected at runtime by the framework — available for use in compute_metrics
# tokenizer: PreTrainedTokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# tokenize — Override default tokenization
# ═══════════════════════════════════════════════════════════════════════════════

# def tokenize(
#     tokenizer: PreTrainedTokenizer,
#     batch: dict[str, Any],
#     target: str,
#     context_length: int,
# ) -> dict[str, list[int]]:
#     """
#     Custom tokenization function.
#
#     Args:
#         tokenizer: tokenizer with registered special tokens
#         batch: sample dict (single sample if tk_batched=False, batch if True)
#         target: column name containing text
#         context_length: max token length
#
#     Returns:
#         Must include "input_ids". Optionally include "attention_mask" and "labels".
#         If "labels" is returned, the framework will auto-switch to a data collator
#         that pads labels with -100 (ignore_index).
#     """
#     text = batch[target]
#     encoding = tokenizer(text, truncation=True, max_length=context_length)
#     return {"input_ids": encoding["input_ids"]}

# ═══════════════════════════════════════════════════════════════════════════════
# preprocess_logits_for_metrics
# ═══════════════════════════════════════════════════════════════════════════════

# def preprocess_logits_for_metrics(logits, labels):
#     """Convert logits to token ids to reduce memory during evaluation."""
#     return logits.argmax(dim=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# compute_metrics — Custom evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════════

# def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
#     """
#     Custom metric computation.
#
#     Access the tokenizer via:
#         import sys
#         _tokenizer = sys.modules[__name__].tokenizer
#
#     Args:
#         eval_pred: contains .predictions and .label_ids (numpy arrays)
#
#     Returns:
#         dict of metric_name -> value
#     """
#     pass
