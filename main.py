import time, yaml, os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["PYTHONHASHSEED"] = "0"
os.environ["HF_DATASETS_CACHE"] = os.path.abspath("./cache")
os.environ["HF_HOME"] = os.path.abspath("./cache")
import torch
from dataset.data_processing import gen_dataset
from model.model_utils import gen_model
from utils.tokenizer import gen_tokenizer, tokenize
from utils.training_utils import gen_trainer
from utils.setup_seed import setup_seed


def training(param: dict):
    param['output_dir'] = f"Training_Recording/{param['jobtype']}/{param['time']}"
    datasets = gen_dataset(param)
    tokenizer = gen_tokenizer(
        corpus=datasets['corpus']['corpus']['text'],
        vocab_size=param['vocab_size'],
        added_tokens=datasets['custom_tokens']['custom_tokens']['token'])
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
    trainer = gen_trainer(param, model, tokenizer, train_val_dataset)
    trainer.train()

def main():
    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())
    with open('model_parameters.yml', 'r', encoding='utf-8') as mp:
        param: dict = yaml.full_load(mp)
    param['time'] = TIME

    seed = param['seed']
    setup_seed(seed)
    torch.use_deterministic_algorithms(True)

    if param['mode'] in ['training', 'fine-tuning']:
        training(param)
    # elif param['mode'] == 'generation':
    #     generation(param)
    # elif param['mode'] == 'evaluation':
    #     evaluation(param)

if __name__ == '__main__':
    main()
