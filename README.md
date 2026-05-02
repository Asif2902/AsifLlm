# AchLLM — Phone-Native Language Model

A phone-native language model that runs on CPU-only Android and iOS devices with less than 500MB RAM. Features ~800M total parameters with ~120M active per token via sparse MoE.

## Architecture

AchLLM combines 5 research breakthroughs:
1. **BitLinear** — Ternary {-1, 0, 1} weights (1.58-bit) for addition-only operations
2. **Mamba-2 SSM** — Linear O(N) complexity, constant memory at inference
3. **Hybrid GQA Attention** — Every 6th block for long-range recall
4. **Sparse MoE FFN** — 8 experts, top-1 active (DS-MoE training)
5. **Task-Aware Prefix Router** — 2M params, adjusts behavior per task
6. **Speculative Draft Head** — 40M params, 2x speed via speculative decoding

## Specs

| Property | Value |
|---|---|
| Total parameters | ~800M |
| Active parameters/token | ~120M |
| RAM at inference | < 500MB |
| Disk size (quantized) | ~180MB |
| Context length | 4096 tokens |
| Vocabulary | 32,000 BPE tokens |

## File Structure

```
achlm/
├── model/           # Model architecture
│   ├── config.py          # AchLLMConfig dataclass
│   ├── bitlinear.py       # BitNet ternary weight layer
│   ├── mamba_block.py    # Mamba-2 SSM with BitLinear
│   ├── gqa_attention.py  # GQA + ALiBi + QK-Norm
│   ├── moe_ffn.py        # Sparse MoE with DS-MoE
│   ├── router.py          # Prefix router (2M params)
│   ├── draft_head.py     # Speculative draft head (40M)
│   └── model.py          # Full AchLLM assembly
├── train/           # Training scripts
│   ├── pretrain.py       # 50B token pre-training
│   ├── sft.py            # Instruction fine-tuning
│   ├── dpo.py            # DPO alignment
│   └── train_router.py  # Router training
├── data/            # Data preparation
│   ├── prepare.py         # Download + tokenize
│   └── dataloader.py     # Streaming dataloader
├── inference/       # Inference
│   ├── generate.py       # CPU inference with speculative decoding
│   └── export_gguf.py   # Export to GGUF/bitnet.cpp
├── configs/         # Configurations
│   └── achlm_800m.yaml
├── tests/           # Tests
│   ├── test_bitlinear.py
│   ├── test_mamba.py
│   ├── test_gqa.py
│   ├── test_moe.py
│   ├── test_model.py
│   └── test_router_draft.py
└── README.md
```

## Quick Start

### Install Dependencies
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets tokenizers wandb pyyaml
```

### Run Tests
```bash
PYTHONPATH=/root/achllm python3 tests/test_bitlinear.py
PYTHONPATH=/root/achllm python3 tests/test_mamba.py
PYTHONPATH=/root/achllm python3 tests/test_model.py
```

### Train
```bash
# Pre-train (50B tokens)
python train/pretrain.py --config configs/achlm_800m.yaml

# SFT
python train/sft.py --model_checkpoint checkpoints/step_XXX.pt

# DPO
python train/dpo.py --model_checkpoint checkpoints/sft_final.pt
```

### Generate
```bash
python inference/generate.py \
    --model_checkpoint checkpoints/sft_final.pt \
    --prompt "Write a hello world program in Python" \
    --use_speculative
```

## License
- Model: Apache 2.0
- Training Data: MIT licensed

*AchLLM — Built by @0xasif_eth — Arc testnet native — May 2026*
