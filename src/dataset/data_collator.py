import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

@dataclass
class DataCollatorForCLMWithLabels:
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: list[dict]) -> dict:
        batch = self.tokenizer.pad(
            [{"input_ids": f["input_ids"]} for f in features],
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            lab = f["labels"]
            padded = lab + [-100] * (max_len - len(lab))
            labels.append(padded)
        batch["labels"] = torch.tensor(labels)
        return batch
