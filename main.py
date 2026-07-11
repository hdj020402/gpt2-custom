import time, os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["PYTHONHASHSEED"] = "0"
os.environ["HF_DATASETS_CACHE"] = os.path.abspath("./cache")
os.environ["HF_HOME"] = os.path.abspath("./cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch

from src.utils.config import load_config
from src.training.train import training
from src.training.hpo import hpo
from src.generation.generate import generation, generation_cpu
from src.utils.setup_seed import setup_seed


def _get_rank() -> int:
    """Return the rank of the current process in distributed mode, or 0."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def _get_world_size() -> int:
    """Return the number of processes in distributed mode, or 1."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def main():
    param = load_config()

    # Detect distributed environment (set by torchrun / torch.distributed.launch).
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_distributed = local_rank != -1

    seed = param['seed']
    setup_seed(seed, rank=max(0, local_rank))

    # torch.use_deterministic_algorithms must be called BEFORE
    # init_process_group (NCCL init triggers CUDA context creation).
    torch.use_deterministic_algorithms(True)

    # Initialize process group early so barriers in build_trainer() work.
    # Trainer detects this and skips its own init.
    if is_distributed and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    rank = _get_rank()
    world_size = _get_world_size()

    TIME = time.strftime('%b_%d_%Y_%H%M%S', time.localtime())
    if is_distributed:
        _buf = [TIME]
        torch.distributed.broadcast_object_list(_buf, src=0)
        TIME = _buf[0]
    param['time'] = TIME

    if rank == 0:
        if is_distributed:
            print(f"[DDP] World size: {world_size}, Local rank: {local_rank}")
        print(f"Mode: {param['mode']}, Job: {param['jobtype']}, Seed: {seed}")

    if param['mode'] in ['training', 'fine-tuning']:
        training(param)
    elif param['mode'] == 'hpo':
        with open('configs/hpo.yml', 'r', encoding='utf-8') as ht:
            import yaml
            ht_param: dict[str, dict] = yaml.full_load(ht)
        hpo(param, ht_param)
    elif param['mode'] == 'generation':
        if param['device'] == 'cuda':
            generation(param)
        elif param['device'] == 'cpu':
            generation_cpu(param)

if __name__ == '__main__':
    main()
