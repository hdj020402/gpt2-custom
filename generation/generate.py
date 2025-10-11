import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from dataset.data_processing import gen_dataset
from utils.utils import LogManager

def gen_prompts(param: dict) -> list[str]:
    datasets = gen_dataset(param)
    prompts_list = []
    for sample in datasets['test']['test']:
        full_text: str = sample[param['target']]
        energy_tag_start_index = full_text.find(param['start_token'], 1)
        if energy_tag_start_index != -1:
            prompt = full_text[:energy_tag_start_index + len(param['start_token'])]
            prompts_list.append(prompt)
    return prompts_list

def generation(param: dict):
    log_manager = LogManager(param)
    log_manager.start_logging()
    os.environ['VLLM_LOGGING_CONFIG_PATH'] = 'generation/logging_config.json'
    from vllm import LLM, SamplingParams

    print("Generating using GPU ...")

    prompts_list = gen_prompts(param)
    print(f'Extracted {len(prompts_list)} prompts for generation ...')

    llm_engine = LLM(
        model=param['pretrained_model'],
        tokenizer_mode='auto',
        dtype='float16',
        max_model_len=param['n_ctx'],
        )
    generation_config = SamplingParams(
        temperature=param['temperature'],
        max_tokens=param['max_tokens'],
        stop=param['stop_token'],
        )
    generation_results = llm_engine.generate(
        prompts_list,
        generation_config,
        )

    generated_texts = [result.outputs[0].text for result in generation_results]
    full_generated_sequences = [
        prompt + generated_text
        for prompt, generated_text in zip(prompts_list, generated_texts)
        ]
    pd.DataFrame(
        {param['target']: full_generated_sequences}
        ).to_csv(f"{param['output_dir']}/generated_texts.csv", index=False)

    log_manager.end_logging()

def generation_cpu(param: dict):
    log_manager = LogManager(param)
    log_manager.start_logging()

    print("Generating using CPU ...")

    prompts_list = gen_prompts(param)
    print(f'Extracted {len(prompts_list)} prompts for generation ...')

    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(param['pretrained_model'])
    model = AutoModelForCausalLM.from_pretrained(param['pretrained_model'])
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=device,
        )

    generated_texts = []
    for i, prompt in enumerate(prompts_list):
        print(f"Generating {i+1}/{len(prompts_list)} ...")
        outputs = generator(
            prompt,
            max_new_tokens=param['max_tokens'],
            temperature=param['temperature'],
            do_sample=True if param['temperature'] > 0 else False,
            pad_token_id=tokenizer.pad_token_id,
            )
        generated_text = outputs[0]['generated_text'][len(prompt):]  # 去掉 prompt
        generated_texts.append(generated_text)

    full_generated_sequences = [
        prompt + generated_text
        for prompt, generated_text in zip(prompts_list, generated_texts)
        ]

    pd.DataFrame(
        {param['target']: full_generated_sequences}
        ).to_csv(f"{param['output_dir']}/generated_texts.csv", index=False)

    log_manager.end_logging()
