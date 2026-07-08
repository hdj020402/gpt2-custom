from transformers import GPT2LMHeadModel, AutoConfig, PreTrainedTokenizer

def gen_model(param: dict, tokenizer: PreTrainedTokenizer):
    config = AutoConfig.from_pretrained(
        "configs/gpt2-original",
        vocab_size=len(tokenizer),
        n_ctx=param['n_ctx'],
        n_embd=param['n_embd'],
        n_head=param['n_head'],
        n_layer=param['n_layer'],
        n_positions=param['n_positions'],
        attn_pdrop=param['dropout'],
        resid_pdrop=param['dropout'],
        embd_pdrop=param['dropout'],
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(config)
    # Device placement is handled by HuggingFace Trainer (single-GPU and DDP).
    return model
