from datasets import load_dataset, DatasetDict
from src.utils.utils import hash_files


# Map dataset key → (param_key_for_file, data_files_keys)
_DATASET_SPECS: dict[str, tuple[str | None, list[str]]] = {
    "train_val":     ("train_file",      ["train", "val"]),
    "test":          ("test_file",       ["test"]),
    "corpus":        ("corpus_file",     ["corpus"]),
    "custom_tokens": ("custom_tokens_file", ["custom_tokens"]),
}


def _load_one(param: dict, key: str) -> DatasetDict | None:
    """Load a single CSV dataset.  Returns None if the param key is not set."""
    param_key, data_keys = _DATASET_SPECS[key]
    file_path = param.get(param_key)
    if file_path is None:
        return None

    # train_val takes two files; others take one
    if key == "train_val":
        data_files = {"train": param["train_file"], "val": param["val_file"]}
        if any(v is None for v in data_files.values()):
            return None
    else:
        data_files = {data_keys[0]: file_path}

    return load_dataset("csv", data_files=data_files)


def gen_dataset(param: dict) -> dict[str, DatasetDict | None]:
    """Load all configured datasets.  Returns a dict keyed by dataset name,
    with None for any dataset whose file path was not provided."""
    return {key: _load_one(param, key) for key in _DATASET_SPECS}


def hash_dataset(param: dict) -> dict[str, str | None]:
    """Hash the data files for each dataset, for cache key generation."""
    result = {}
    for key, (param_key, data_keys) in _DATASET_SPECS.items():
        if key == "train_val":
            f1 = param.get("train_file")
            f2 = param.get("val_file")
            result[key] = hash_files(f1, f2) if f1 and f2 else None
        else:
            f = param.get(param_key)
            result[key] = hash_files(f) if f else None
    return result
