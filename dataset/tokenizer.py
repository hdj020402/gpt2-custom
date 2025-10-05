from typing import Iterable
from transformers import AutoTokenizer, PreTrainedTokenizer

def gen_tokenizer(
    corpus: Iterable[str],
    vocab_size: int,
    added_tokens: Iterable[str]
) -> PreTrainedTokenizer:
    old_tokenizer = AutoTokenizer.from_pretrained('gpt2-original')
    tokenizer = old_tokenizer.train_new_from_iterator(corpus, vocab_size=vocab_size)
    tokenizer.add_tokens(list(added_tokens))
    tokenizer.add_special_tokens(
        special_tokens_dict={
            'bos_token': '<|endoftext|>',
            'eos_token': '<|endoftext|>',
            'pad_token': ' ',
            'unk_token': '<|endoftext|>'})

    return tokenizer

def tokenize(
    tokenizer: PreTrainedTokenizer,
    batch: dict,
    target: str,
    context_length: int
    ) -> dict[str, list[list[int]]]:
    outputs = tokenizer(
        batch[target],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    valid_input_ids = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length <= context_length:
            valid_input_ids.append(input_ids)

    return {"input_ids": valid_input_ids}

