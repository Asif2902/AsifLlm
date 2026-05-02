import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import AchLLMConfig
from .bitlinear import BitLinear
from .mamba_block import MambaBlock
from .gqa_attention import GQAAttention
from .moe_ffn import SparseMoE
from .rms_norm import RMSNorm
from .router import PrefixRouter
from .draft_head import DraftHead

class TiedLinear(nn.Module):
    """LM head tied to embedding weights"""
    def __init__(self, embed: nn.Embedding):
        super().__init__()
        self.embed = embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.embed.weight)

class AchLLM(nn.Module):
    def __init__(self, config: AchLLMConfig):
        super().__init__()
        self.config = config

        # 1. Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)

        # 2. Build block sequence (24 blocks total)
        self.blocks = nn.ModuleList()
        for i in range(config.num_blocks):
            if i in config.gqa_positions:
                self.blocks.append(GQAAttention(config))
            elif i in config.moe_positions:
                self.blocks.append(SparseMoE(config))
            else:
                self.blocks.append(MambaBlock(config))

        # 3. Final RMSNorm
        self.norm = RMSNorm(config.hidden_dim)

        # 4. Tied LM head (shares weights with embedding)
        self.lm_head = TiedLinear(self.embed)

        # 5. Prefix Router (runs once per prompt, not per token)
        # Uses fp16 Linear (not BitLinear) per rules
        self.router = PrefixRouter(config.hidden_dim, config.router_num_classes)

        # 6. Draft Head for speculative decoding
        # Shares embedding with main model (no extra memory)
        self.draft = DraftHead(config)
        # Projection from main hidden (1024) to draft hidden (256)
        self.draft_proj = nn.Linear(config.hidden_dim, config.draft_hidden_dim, bias=False)

    def forward(self, input_ids: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        # input_ids: [batch, seq_len]
        x = self.embed(input_ids)  # [batch, seq_len, 512]

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            if isinstance(block, SparseMoE):
                x, aux_loss = block(x, training=training)
                total_aux_loss += aux_loss
            else:
                x = block(x)  # Mamba/GQA don't return aux loss

        x = self.norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, 32000]

        return logits, total_aux_loss
