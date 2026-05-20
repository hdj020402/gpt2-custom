"""
Custom metrics module template for gpt2-custom framework.

Copy this file to custom/custom_metrics.py and modify as needed.
Set in configs/model_parameters.yml:
    custom_metrics: custom/custom_metrics.py

Changes to this file do NOT invalidate the tokenized dataset cache.
"""

import numpy as np
from transformers import PreTrainedTokenizer, EvalPrediction

# ═══════════════════════════════════════════════════════════════════════════════
# Module-level config variables (read by the framework)
# ═══════════════════════════════════════════════════════════════════════════════

# Metric used for best model selection and early stopping
metric_for_best_model: str = "eval_loss"
greater_is_better: bool = False

# Injected at runtime by the framework — available for use in compute_metrics
# tokenizer: PreTrainedTokenizer

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
