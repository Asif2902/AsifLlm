"""
Efficient streaming dataloader for AchLLM pre-training.
Supports streaming from HuggingFace datasets with dynamic mixing by weight.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from transformers import AutoTokenizer
import numpy as np


class StreamingTextDataset(Dataset):
    """Streaming dataset that samples from multiple sources by weight"""
    def __init__(self, tokenizer_path, seq_len=2048, max_samples=1_000_000):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.seq_len = seq_len
        self.max_samples = max_samples

        # Load datasets with weights from config
        self.datasets = {
            "fineweb-edu": load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True),
            "the-stack-smol": load_dataset("bigcode/the-stack-smol", split="train", streaming=True),
            "open-web-math": load_dataset("open-web-math/open-web-math", split="train", streaming=True),
        }
        self.weights = {"fineweb-edu": 0.6, "the-stack-smol": 0.2, "open-web-math": 0.2}

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        # Sample dataset by weight
        dataset_name = np.random.choice(
            list(self.datasets.keys()),
            p=list(self.weights.values())
        )
        dataset = self.datasets[dataset_name]

        # Get a sample
        sample = next(iter(dataset))
        text = sample.get("text", "")

        # Tokenize
        tokens = self.tokenizer.encode(text, max_length=self.seq_len, truncation=True)

        # Pad or truncate to seq_len
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[:self.seq_len]

        return torch.tensor(tokens, dtype=torch.long)


def create_dataloader(tokenizer_path, batch_size=512, seq_len=2048, num_workers=4):
    """Create dataloader for pre-training"""
    dataset = StreamingTextDataset(tokenizer_path, seq_len=seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader
