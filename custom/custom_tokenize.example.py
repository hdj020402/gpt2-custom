"""
Custom tokenize module template for gpt2-custom framework.

Copy this file to custom/custom_tokenize.py and modify as needed.
Set in configs/model_parameters.yml:
    custom_tokenize: custom/custom_tokenize.py

Changes to this file invalidate the tokenized dataset cache.
"""

from typing import Any
from transformers import PreTrainedTokenizer

# ═══════════════════════════════════════════════════════════════════════════════
# Module-level config variables (read by the framework)
# ═══════════════════════════════════════════════════════════════════════════════

# Whether dataset.map() should pass batches (True) or single samples (False)
# Set to False if your tokenize function needs per-sample offset_mapping
tk_batched: bool = True

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
