import torch
import random
import numpy as np


def setup_seed(seed: int, rank: int = 0):
    """Set random seeds for reproducibility.

    In DDP mode, each rank gets ``seed + rank`` so that data shuffling
    differs across workers (prevents all ranks from processing identical
    batches), while training remains deterministic for a given seed+rank
    combination.
    """
    effective_seed = seed + rank
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    np.random.seed(effective_seed)
    random.seed(effective_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

