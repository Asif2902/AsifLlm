"""
DPO (Direct Preference Optimization) training for AchLLM.
Uses UltraFeedback binarized dataset.

Usage:
    python train/dpo.py --model_checkpoint checkpoints/sft_final.pt
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer
import yaml

from model.config import AchLLMConfig
from model.model import AchLLM


def dpo_train(config_path, model_checkpoint):
    """Main DPO training loop"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load model (need to convert to HF format for TRL)
    print("Loading model for DPO training...")

    # Load preference dataset
    from datasets import load_dataset
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train")

    tokenizer = AutoTokenizer.from_pretrained("tokenizer.json")
    tokenizer.pad_token = tokenizer.eos_token

    # Note: Full DPO implementation requires converting AchLLM to HuggingFace format
    # This is a placeholder for the DPO training script
    print("DPO training requires HuggingFace compatibility layer")
    print("See: https://github.com/huggingface/trl for full DPO implementation")

    # Placeholder for actual DPO training
    # trainer = DPOTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset,
    #     tokenizer=tokenizer,
    # )
    # trainer.train()

    print("DPO training script created (requires HF compatibility layer)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO training for AchLLM")
    parser.add_argument("--config", type=str, default="configs/achlm_800m.yaml")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    args = parser.parse_args()
    dpo_train(args.config, args.model_checkpoint)
