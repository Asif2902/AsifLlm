import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import AchLLMConfig
from .mamba_block import MambaBlock
from .rms_norm import RMSNorm

class DraftHead(nn.Module):
    """
    Speculative draft head: small model that runs ahead to guess next 4 tokens.
    Main model verifies all 4 in a single forward pass.
    Uses 2 Mamba-2 blocks with hidden=256.
    Shares embedding with main model (no extra memory).
    Total params: ~40M
    """
    def __init__(self, config: AchLLMConfig):
        super().__init__()
        # Create a smaller config for draft model
        self.draft_config = AchLLMConfig(
            hidden_dim=config.draft_hidden_dim,  # 256
            num_blocks=2,  # Only 2 Mamba blocks
            vocab_size=config.vocab_size,  # 32000 (shared)
            max_seq_len=config.max_seq_len,
            mamba_state_dim=32,  # Smaller state
            mamba_conv_width=2,
            mamba_expand_factor=2,
            mamba_dt_rank=16,
            gqa_positions=[],  # No GQA in draft
            moe_positions=[],  # No MoE in draft
        )

        # 2 Mamba-2 blocks
        self.blocks = nn.ModuleList([
            MambaBlock(self.draft_config) for _ in range(2)
        ])

        # Final norm
        self.norm = RMSNorm(config.draft_hidden_dim)

        # LM head: projects from draft_hidden (256) to vocab (32000)
        self.lm_head = nn.Linear(config.draft_hidden_dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, draft_hidden_dim] - embeddings projected to draft space
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)
