import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class AchLLMConfig:
    # Core model dimensions
    hidden_dim: int = 896
    num_blocks: int = 24
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # MoE configuration
    num_experts: int = 20  # 20 experts per layer (only 1 active = sparse)
    top_k_experts_train: int = 2  # DS-MoE: top-2 during training
    top_k_experts_infer: int = 1  # Top-1 during inference
    moe_positions: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 10, 13, 16, 19])  # 8 MoE layers (no overlap with GQA)
    moe_aux_loss_weight: float = 0.01

    # Mamba-2 configuration
    mamba_state_dim: int = 64
    mamba_conv_width: int = 4
    mamba_expand_factor: int = 2
    mamba_dt_rank: int = 32

    # GQA Attention configuration
    gqa_positions: List[int] = field(default_factory=lambda: [5, 11, 18, 23])
    gqa_num_q_heads: int = 8
    gqa_num_kv_heads: int = 4
    gqa_head_dim: int = 112  # 896 / 8 = 112

    # Prefix Router configuration
    router_num_classes: int = 4  # {code, chat, math, reason}

    # Speculative Draft Head configuration
    draft_hidden_dim: int = 256

    # General architecture settings
    tie_embeddings: bool = True
    use_alibi: bool = True  # No RoPE
    use_qk_norm: bool = True  # Mandatory for GQA stability
    activation_dtype: torch.dtype = torch.bfloat16

    # Tokenizer IDs
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
