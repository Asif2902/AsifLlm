"""
Supervised Fine-Tuning (SFT) script for AchLLM.
Uses instruction datasets: OpenHermes-2.5, CodeAlpaca-20k, Alpaca.

Usage:
    python train/sft.py --config configs/achlm_800m.yaml --model_checkpoint path/to/checkpoint.pt
"""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import yaml

from model.config import AchLLMConfig
from model.model import AchLLM


def load_instruction_data():
    """Load instruction datasets"""
    datasets = []

    # OpenHermes-2.5
    datasets.append(load_dataset("teknium/OpenHermes-2.5", split="train"))

    # CodeAlpaca-20k
    datasets.append(load_dataset("sahil2801/CodeAlpaca-20k", split="train"))

    # Alpaca
    datasets.append(load_dataset("tatsu-lab/alpaca", split="train"))

    return concatenate_datasets(datasets)


def sft_train(config_path, model_checkpoint):
    """Main SFT training loop"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load model
    model_cfg = AchLLMConfig()
    model = AchLLM(model_cfg)

    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint["model"])
    model.train()

    # Load data
    print("Loading instruction datasets...")
    dataset = load_instruction_data()

    tokenizer = AutoTokenizer.from_pretrained("tokenizer.json")
    tokenizer.pad_token = tokenizer.eos_token

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    print(f"Starting SFT on {len(dataset)} samples")

    # Training loop (simplified)
    for epoch in range(3):
        for idx, sample in enumerate(dataset):
            # Format instruction
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            output = sample.get("output", "")

            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"

            # Tokenize
            tokens = tokenizer(prompt, max_length=2048, truncation=True, padding="max_length")
            input_ids = torch.tensor(tokens["input_ids"]).unsqueeze(0)

            # Forward
            logits, aux_loss = model(input_ids, training=True)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, cfg["model"]["vocab_size"]),
                input_ids[:, 1:].contiguous().view(-1),
            ) + aux_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if idx % 100 == 0:
                print(f"Epoch {epoch} | Sample {idx} | Loss: {loss.item():.4f}")

    # Save SFT model
    torch.save({"model": model.state_dict()}, "checkpoints/sft_final.pt")
    print("SFT complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT for AchLLM")
    parser.add_argument("--config", type=str, default="configs/achlm_800m.yaml")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    args = parser.parse_args()
    sft_train(args.config, args.model_checkpoint)
