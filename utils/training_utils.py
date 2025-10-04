import torch
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

def gen_trainer(
    param: dict,
    model: GPT2LMHeadModel,
    tokenizer: PreTrainedTokenizer,
    train_val_dataset: DatasetDict,
    ):
    args = TrainingArguments(
        output_dir=param['output_dir'],
        seed=param['seed'],
        remove_unused_columns=False,

        dataloader_num_workers=param['num_workers'],
        per_device_train_batch_size=param['batch_size'],
        per_device_eval_batch_size=param['batch_size']*param['accumulation_steps'],
        gradient_accumulation_steps=param['accumulation_steps'],

        num_train_epochs=param['epoch'],
        fp16=True,

        learning_rate=param['lr'],
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

class ClearCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
