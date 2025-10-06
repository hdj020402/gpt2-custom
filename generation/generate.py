from vllm import LLM, SamplingParams
from dataset.data_processing import gen_dataset
from utils.utils import LogManager

def gen_prompts(param: dict):
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

    log_manager.end_logging()
