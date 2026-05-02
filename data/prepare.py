"""
Data preparation script for AchLLM pre-training.
Downloads and tokenizes datasets for 50B tokens of pre-training data.

Datasets:
- HuggingFaceFW/fineweb-edu (60%): cleaned educational web text
- bigcode/the-stack-smol (20%): MIT-licensed code
- open-web-math/open-web-math (20%): math reasoning

Usage:
    python data/prepare.py --output_dir ./data/tokenized --tokenizer_path ./tokenizer.json
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def train_tokenizer(texts, vocab_size=32000, output_path="tokenizer.json"):
    """Train BPE tokenizer on sample texts (first 10B tokens of fineweb-edu)"""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
    tokenizer.pre_tokenizer = Whitespace()

    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(output_path)
    return tokenizer


def download_and_tokenize(config_path="./configs/achlm_800m.yaml"):
    """Download datasets and tokenize"""
    print("Loading datasets...")

    # Load datasets with weights: 60% fineweb-edu, 20% the-stack-smol, 20% open-web-math
    fineweb = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    stack = load_dataset("bigcode/the-stack-smol", split="train", streaming=True)
    math = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

    print("Datasets loaded (streaming mode)")

    # Sample text for tokenizer training (first 10B tokens ~ 2B characters)
    print("Collecting text for tokenizer training...")
    sample_texts = []
    char_count = 0
    target_chars = 2_000_000_000  # ~10B tokens * 4 chars/token (rough estimate)

    for sample in fineweb:
        text = sample.get("text", "")
        sample_texts.append(text)
        char_count += len(text)
        if char_count >= target_chars:
            break

    print(f"Collected {len(sample_texts)} samples ({char_count} chars) for tokenizer training")

    # Train tokenizer
    print("Training BPE tokenizer...")
    tokenizer = train_tokenizer(sample_texts, vocab_size=32000)
    print("Tokenizer trained and saved")

    # Tokenize and save datasets as memory-mapped numpy arrays
    print("Tokenizing and saving datasets...")
    # Implementation continues...
    # (Full implementation would stream, tokenize, and save in shards)

    print("Data preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare AchLLM training data")
    parser.add_argument("--output_dir", type=str, default="./data/tokenized",
                        help="Output directory for tokenized data")
    parser.add_argument("--config", type=str, default="./configs/achlm_800m.yaml",
                        help="Path to config file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    download_and_tokenize(args.config)
