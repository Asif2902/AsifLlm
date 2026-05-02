"""
Pre-training script for AchLLM.
Trains on 50B tokens from fineweb-edu, the-stack-smol, and open-web-math.

Usage:
    python train/pretrain.py --config configs/achlm_800m.yaml
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import wandb

from model.config import AchLLMConfig
from model.model import AchLLM
from data.dataloader import create_dataloader


def train(config_path="configs/achlm_800m.yaml"):
    """Main pre-training loop"""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Model config
    model_cfg = AchLLMConfig(
        hidden_dim=cfg["model"]["hidden_dim"],
        num_blocks=cfg["model"]["num_blocks"],
        vocab_size=cfg["model"]["vocab_size"],
        max_seq_len=cfg["model"]["max_seq_len"],
    )

    # Initialize model
    model = AchLLM(model_cfg)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Learning rate scheduler
    total_steps = cfg["training"]["total_tokens"] // cfg["training"]["batch_size"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Dataloader
    tokenizer = AutoTokenizer.from_pretrained("tokenizer.json")
    dataloader = create_dataloader(
        tokenizer_path="tokenizer.json",
        batch_size=cfg["training"]["batch_size"] // cfg["training"]["gradient_accumulation_steps"],
        seq_len=cfg["training"]["seq_len"],
    )

    # Initialize wandb
    wandb.init(project=cfg["logging"]["wandb_project"], config=cfg)

    # Training loop
    global_step = 0
    total_tokens = 0

    print(f"Starting pre-training: target {cfg['training']['total_tokens']:,} tokens")

    for batch_idx, input_ids in enumerate(dataloader):
        # Forward pass
        logits, aux_loss = model(input_ids, training=True)

        # Compute loss (cross-entropy)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, cfg["model"]["vocab_size"]),
            input_ids[:, 1:].contiguous().view(-1),
        )
        loss = loss + aux_loss

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % cfg["training"]["gradient_accumulation_steps"] == 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % cfg["logging"]["log_every"] == 0:
                perplexity = torch.exp(loss).item()
                wandb.log({
                    "loss": loss.item(),
                    "perplexity": perplexity,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "tokens": total_tokens,
                })
                print(f"Step {global_step} | Loss: {loss.item():.4f} | PPL: {perplexity:.2f}")

            # Checkpoint
            if total_tokens > 0 and total_tokens % cfg["training"]["checkpoint_every_tokens"] == 0:
                checkpoint_path = f"checkpoints/step_{global_step}.pt"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "tokens": total_tokens,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        total_tokens += input_ids.numel()

        if total_tokens >= cfg["training"]["total_tokens"]:
            print("Pre-training complete!")
            break

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train AchLLM")
    parser.add_argument("--config", type=str, default="configs/achlm_800m.yaml")
    args = parser.parse_args()
    train(args.config)
