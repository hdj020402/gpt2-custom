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
    )
from datasets import DatasetDict

class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

def gen_trainer(
    param: dict,
    model: GPT2LMHeadModel | None,
    model_init: Callable[[], GPT2LMHeadModel] | None,
    tokenizer: PreTrainedTokenizer,
    train_val_dataset: DatasetDict,
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=param['logging_steps'],
        report_to="tensorboard",
        disable_tqdm=True,
        )

    tokenizer.pad_token = tokenizer.pad_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        model_init=model_init,
        processing_class=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_val_dataset["train"],
        eval_dataset=train_val_dataset["val"],
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=param['early_stopping']['patience'],
                early_stopping_threshold=param['early_stopping']['threshold']
                ),
            ClearCacheCallback()
            ]
        )

    return trainer


import os
from datasets import load_from_disk
from dataset.data_processing import gen_dataset, hash_dataset
from dataset.tokenizer import gen_tokenizer, tokenize
from model.model_utils import gen_model

def build_trainer(param: dict) -> Trainer:
    datasets = gen_dataset(param)
    tokenizer = gen_tokenizer(
        corpus=datasets['corpus']['corpus']['text'],
        vocab_size=param['vocab_size'],
        added_tokens=datasets['custom_tokens']['custom_tokens']['token'])

    data_hash = hash_dataset(param)
    os.path.makedirs('./cache/tokenized', exist_ok=True)
    cache_path = f"./cache/tokenized/{data_hash['train_val']}"
    if os.path.exists(cache_path):
        print(f"Loading cached tokenized dataset from {cache_path} ...")
        train_val_dataset = load_from_disk(cache_path)
    else:
        print("Tokenizing dataset ...")
        train_val_dataset = datasets['train_val'].map(
            lambda x: tokenize(
                tokenizer=tokenizer,
                batch=x,
                target=param['target'],
                context_length=param['n_ctx']
                ),
            batched=True,
            remove_columns=datasets['train_val']['train'].column_names,
            num_proc=param['tk_num_proc']
            )

    model = gen_model(param, tokenizer)
    def model_init():
        return gen_model(param, tokenizer)

    if param['mode'] == 'training':
        trainer = gen_trainer(
            param=param,
            model=model,
            model_init=None,
            tokenizer=tokenizer,
            train_val_dataset=train_val_dataset
            )
    elif param['mode'] == 'hpo':
        trainer = gen_trainer(
            param=param,
            model=None,
            model_init=model_init,
            tokenizer=tokenizer,
            train_val_dataset=train_val_dataset
            )

    return trainer
