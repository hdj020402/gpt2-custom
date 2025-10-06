import time, yaml, os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["PYTHONHASHSEED"] = "0"
os.environ["HF_DATASETS_CACHE"] = os.path.abspath("./cache")
os.environ["HF_HOME"] = os.path.abspath("./cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from training.train import training
from training.hpo import hpo
from utils.setup_seed import setup_seed
from utils.utils import LogManager


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
    elif param['mode'] == 'hpo':
        with open('hparam_tuning.yml', 'r', encoding='utf-8') as ht:
            ht_param: dict[str, dict] = yaml.full_load(ht)
        hpo(param, ht_param)
    # elif param['mode'] == 'generation':
    #     generation(param)

if __name__ == '__main__':
    main()
