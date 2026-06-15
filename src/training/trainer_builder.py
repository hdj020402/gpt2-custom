import math
import os
import logging
import torch
from typing import Callable
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EvalPrediction,
    )
from datasets import DatasetDict, load_from_disk

from src.dataset.data_processing import gen_dataset, hash_dataset
from src.dataset.tokenizer import gen_tokenizer, tokenize
from src.dataset.data_collator import DataCollatorForCLMWithLabels
from src.model.model_utils import gen_model
from src.utils.custom_module_loader import load_custom_module
from src.utils.utils import hash_files

class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

def gen_trainer(
    param: dict,
    model: GPT2LMHeadModel | None,
    model_init: Callable[[], GPT2LMHeadModel] | None,
    tokenizer: PreTrainedTokenizer,
    train_val_dataset: DatasetDict,
    data_collator=None,
    compute_metrics: Callable[[EvalPrediction], dict] | None = None,
    preprocess_logits_for_metrics: Callable | None = None,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    ) -> Trainer:
    args = TrainingArguments(
        output_dir=param['output_dir'],
        seed=param['seed'],
        remove_unused_columns=False,

        dataloader_num_workers=param['dataloader_num_workers'],
        per_device_train_batch_size=param['per_device_train_batch_size'],
        per_device_eval_batch_size=param['per_device_train_batch_size']*param['gradient_accumulation_steps'],
        gradient_accumulation_steps=param['gradient_accumulation_steps'],

        num_train_epochs=param['num_train_epochs'],
        fp16=True,

        learning_rate=param['learning_rate'],
        weight_decay=param['weight_decay'],
        max_grad_norm=param['max_grad_norm'],
        warmup_steps=param['warmup_steps'],
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,

        eval_strategy='steps',
        eval_steps=param['eval_steps'],
        save_steps=param['model_save_steps'],
        save_total_limit=param['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,

        logging_steps=param['logging_steps'],
        report_to="tensorboard",
        disable_tqdm=True,
        )

    if data_collator is None:
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        model_init=model_init,
        processing_class=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_val_dataset["train"],
        eval_dataset=train_val_dataset["val"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=param['early_stopping']['patience'],
                early_stopping_threshold=param['early_stopping']['threshold']
                ),
            ClearCacheCallback()
            ]
        )

    return trainer


logger = logging.getLogger(__name__)

def build_trainer(param: dict) -> Trainer:
    custom_tk = load_custom_module(param.get('custom_tokenize'), name="custom_tokenize")
    custom_mt = load_custom_module(param.get('custom_metrics'), name="custom_metrics")

    datasets = gen_dataset(param)
    tokenizer = gen_tokenizer(
        corpus=datasets['corpus']['corpus']['text'],
        vocab_size=param['vocab_size'],
        added_tokens=datasets['custom_tokens']['custom_tokens']['token'])

    # Determine tokenize function and batched mode
    tokenize_fn = getattr(custom_tk, 'tokenize', None) or tokenize
    tk_batched = getattr(custom_tk, 'tk_batched', True)

    # Build cache hash (include custom tokenize file if present)
    data_hash = hash_dataset(param)
    cache_key = data_hash['train_val']
    custom_tk_path = param.get('custom_tokenize')
    if custom_tk_path:
        custom_hash = hash_files(os.path.abspath(custom_tk_path))
        cache_key = f"{cache_key}_{custom_hash}"

    os.makedirs('./cache/tokenized', exist_ok=True)
    cache_path = f"./cache/tokenized/{cache_key}"
    if os.path.exists(cache_path):
        logger.info(f"Loading cached tokenized dataset from {cache_path} ...")
        train_val_dataset = load_from_disk(cache_path)
    else:
        logger.info("Tokenizing dataset ...")
        train_val_dataset = datasets['train_val'].map(
            lambda x: tokenize_fn(
                tokenizer=tokenizer,
                batch=x,
                target=param['target'],
                context_length=param['n_ctx']
                ),
            batched=tk_batched,
            remove_columns=datasets['train_val']['train'].column_names,
            num_proc=param['tk_num_proc']
            )
        train_val_dataset.save_to_disk(cache_path, num_proc=param['tk_num_proc'])

    # Auto-detect data collator based on whether labels exist
    has_labels = "labels" in train_val_dataset["train"].column_names
    data_collator = DataCollatorForCLMWithLabels(tokenizer) if has_labels else None

    # Inject tokenizer into custom metrics module so compute_metrics can access it
    if custom_mt is not None:
        custom_mt.tokenizer = tokenizer

    # Extract custom metrics settings
    compute_metrics_fn = getattr(custom_mt, 'compute_metrics', None)
    preprocess_logits_fn = getattr(custom_mt, 'preprocess_logits_for_metrics', None)
    metric_name = getattr(custom_mt, 'metric_for_best_model', 'eval_loss')
    greater = getattr(custom_mt, 'greater_is_better', False)

    # Resolve warmup_steps — auto-calculate from warmup_ratio if not explicitly set
    if param.get('warmup_steps') is None:
        warmup_ratio = param.get('warmup_ratio', 0.0)
        if warmup_ratio > 0:
            effective_batch = param['per_device_train_batch_size'] * param.get('gradient_accumulation_steps', 1)
            steps_per_epoch = len(train_val_dataset['train']) // effective_batch
            total_steps = steps_per_epoch * param['num_train_epochs']
            param['warmup_steps'] = math.ceil(warmup_ratio * total_steps)
            logger.info(f"Auto-calculated warmup_steps={param['warmup_steps']} "
                        f"(ratio={warmup_ratio}, total_steps={total_steps})")
        else:
            param['warmup_steps'] = 0

    model = gen_model(param, tokenizer)
    def model_init():
        return gen_model(param, tokenizer)

    if param['mode'] == 'training':
        trainer = gen_trainer(
            param=param,
            model=model,
            model_init=None,
            tokenizer=tokenizer,
            train_val_dataset=train_val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            preprocess_logits_for_metrics=preprocess_logits_fn,
            metric_for_best_model=metric_name,
            greater_is_better=greater,
            )
    elif param['mode'] == 'hpo':
        trainer = gen_trainer(
            param=param,
            model=None,
            model_init=model_init,
            tokenizer=tokenizer,
            train_val_dataset=train_val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            preprocess_logits_for_metrics=preprocess_logits_fn,
            metric_for_best_model=metric_name,
            greater_is_better=greater,
            )

    return trainer
