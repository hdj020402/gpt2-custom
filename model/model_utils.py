import torch
from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedTokenizer

def gen_model(param: dict, tokenizer: PreTrainedTokenizer):
    config = AutoConfig.from_pretrained(
        "gpt2-original",
        vocab_size=len(tokenizer),
        n_ctx=param['n_ctx'],
        n_embd=param['n_embd'],
        n_head=param['n_head'],
        n_layer=param['n_layer'],
        n_positions=param['n_positions'],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GPT2LMHeadModel(config)
    model = model.to(device)
    return model
