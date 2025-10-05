from datasets import load_dataset, DatasetDict
from utils.utils import hash_files

def gen_dataset(param: dict) -> dict[str, DatasetDict]:
    train_file = param['train_file']
    val_file = param['val_file']
    try:
        train_val_dataset = load_dataset(
            'csv',
            data_files={'train':train_file,'val':val_file},
            )
    except TypeError:
        train_val_dataset = None

    test_file = param['test_file']
    try:
        test_dataset = load_dataset(
            'csv',
            data_files={'test':test_file},
            )
    except TypeError:
        test_dataset = None

    corpus_file = param['corpus_file']
    try:
        corpus_dataset = load_dataset(
            'csv',
            data_files={'corpus':corpus_file},
            )
    except TypeError:
        corpus_dataset = None

    custom_tokens_file = param['custom_tokens_file']
    try:
        custom_tokens_dataset = load_dataset(
            'csv',
            data_files={'custom_tokens':custom_tokens_file},
            )
    except TypeError:
        custom_tokens_dataset = None

    return {
        'train_val':train_val_dataset,
        'test':test_dataset,
        'corpus':corpus_dataset,
        'custom_tokens':custom_tokens_dataset
        }

def hash_dataset(param: dict):
    train_file = param['train_file']
    val_file = param['val_file']
    try:
        train_val_hash = hash_files(train_file, val_file)
    except TypeError:
        train_val_hash = None

    test_file = param['test_file']
    try:
        test_hash = hash_files(test_file)
    except TypeError:
        test_hash = None

    corpus_file = param['corpus_file']
    try:
        corpus_hash = hash_files(corpus_file)
    except TypeError:
        corpus_hash = None

    custom_tokens_file = param['custom_tokens_file']
    try:
        custom_tokens_hash = hash_files(custom_tokens_file)
    except TypeError:
        custom_tokens_hash = None

    return {
        'train_val':train_val_hash,
        'test':test_hash,
        'corpus':corpus_hash,
        'custom_tokens':custom_tokens_hash
        }
