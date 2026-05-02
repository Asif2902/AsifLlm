import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear
from .rms_norm import RMSNorm
from .config import AchLLMConfig

class GQAAttention(nn.Module):
    def __init__(self, config: AchLLMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim  # 512
        self.num_q_heads = config.gqa_num_q_heads  # 8
        self.num_kv_heads = config.gqa_num_kv_heads  # 4
        self.head_dim = config.gqa_head_dim  # 64
        self.max_seq_len = config.max_seq_len  # 4096

        # Validate dimensions
        assert self.hidden_dim == self.num_q_heads * self.head_dim, \
            "hidden_dim must equal num_q_heads * head_dim"
        assert self.num_q_heads % self.num_kv_heads == 0, \
            "num_q_heads must be divisible by num_kv_heads"
        self.q_per_kv = self.num_q_heads // self.num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim  # 4*64=256

        # Projections (all BitLinear)
        self.q_proj = BitLinear(self.hidden_dim, self.hidden_dim)  # 512 → 512 (8*64)
        self.k_proj = BitLinear(self.hidden_dim, self.kv_dim)      # 512 → 256 (4*64)
        self.v_proj = BitLinear(self.hidden_dim, self.kv_dim)      # 512 → 256 (4*64)
        self.o_proj = BitLinear(self.hidden_dim, self.hidden_dim)  # 512 → 512

        # QK-Norm: normalize Q and K before attention (per head L2 norm)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        # Pre-compute ALiBi slopes and bias
        self._register_alibi_parameters()

    def _register_alibi_parameters(self):
        # ALiBi: linear bias with slopes per head, increasing with distance
        # Slopes for 8 attention heads: 2^(-8/8), 2^(-7/8), ..., 2^(-1/8)
        slopes = torch.tensor([
            2 ** (-i / self.num_q_heads) for i in range(1, self.num_q_heads + 1)
        ])  # [num_q_heads]
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def _get_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Generate ALiBi bias: [num_q_heads, seq_len, seq_len]
        # Bias for position i attending to j: -slope * |i - j|
        positions = torch.arange(seq_len, device=device)
        relative_pos = positions.unsqueeze(1) - positions.unsqueeze(0)  # [seq_len, seq_len]
        relative_pos = torch.abs(relative_pos)  # [seq_len, seq_len]

        # Apply slopes per head: [num_q_heads, 1, 1] * [1, seq_len, seq_len]
        alibi_bias = -self.alibi_slopes.view(-1, 1, 1) * relative_pos.unsqueeze(0)
        return alibi_bias  # [num_q_heads, seq_len, seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_dim] (512)
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, 512]
        k = self.k_proj(x)  # [batch, seq_len, 512]
        v = self.v_proj(x)  # [batch, seq_len, 512]

        # Reshape to heads: [batch, seq_len, num_heads, head_dim]
        q = q.view(batch, seq_len, self.num_q_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # QK-Norm: normalize per head
        q = self.q_norm(q)  # [batch, seq_len, 8, 64]
        k = self.k_norm(k)  # [batch, seq_len, 4, 64]

        # Repeat K, V for grouped queries: [batch, seq_len, 8, 64]
        k = k.repeat_interleave(self.q_per_kv, dim=2)
        v = v.repeat_interleave(self.q_per_kv, dim=2)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores: [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Add ALiBi bias: [num_heads, seq_len, seq_len] → broadcast to batch
        alibi_bias = self._get_alibi_bias(seq_len, x.device)
        attn_scores = attn_scores + alibi_bias.unsqueeze(0)

        # Attention softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to V
        attn_output = torch.matmul(attn_probs, v)  # [batch, num_heads, seq_len, head_dim]

        # Reshape back: [batch, seq_len, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.hidden_dim)

        # Output projection (BitLinear)
        return self.o_proj(attn_output)
