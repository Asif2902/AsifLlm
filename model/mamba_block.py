import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear
from .rms_norm import RMSNorm
from .config import AchLLMConfig

class MambaBlock(nn.Module):
    def __init__(self, config: AchLLMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim  # 512
        self.inner_dim = self.hidden_dim * config.mamba_expand_factor  # 1024
        self.state_dim = config.mamba_state_dim  # 64
        self.conv_width = config.mamba_conv_width  # 4
        self.dt_rank = config.mamba_dt_rank  # 32

        # Pre-norm (required per Step 2)
        self.norm = RMSNorm(self.hidden_dim)

        # In projection: hidden_dim → inner_dim (1024) using BitLinear
        self.in_proj = BitLinear(self.hidden_dim, self.inner_dim)

        # 1D depthwise convolution over sequence (conv_width=4)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=self.conv_width,
            padding=self.conv_width - 1,
            groups=self.inner_dim,  # Depthwise convolution
            bias=False
        )

        # SSM projections (all BitLinear)
        self.ssm_B = BitLinear(self.inner_dim, self.state_dim)  # B matrix
        self.ssm_C = BitLinear(self.state_dim, self.inner_dim)  # C matrix
        self.dt_proj = BitLinear(self.inner_dim, self.dt_rank)  # Delta time projection

        # Out projection: inner_dim → hidden_dim (BitLinear)
        self.out_proj = BitLinear(self.inner_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_dim]
        residual = x
        x = self.norm(x)  # Pre-norm

        # In projection
        x = self.in_proj(x)  # [batch, seq_len, inner_dim]

        # 1D convolution (Conv1d expects [batch, channels, seq_len])
        x_conv = x.transpose(1, 2)  # [batch, inner_dim, seq_len]
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv[:, :, :x.shape[1]]  # Trim padding to original seq_len
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, inner_dim]
        x_conv = F.silu(x_conv)  # Activation after conv

        # SSM: B → state → C
        ssm_b = self.ssm_B(x_conv)  # [batch, seq_len, state_dim]
        ssm_c = self.ssm_C(F.silu(ssm_b))  # [batch, seq_len, inner_dim]

        # Delta time gating
        dt = self.dt_proj(x_conv)  # [batch, seq_len, dt_rank]
        dt = dt.repeat(1, 1, self.inner_dim // self.dt_rank)  # Expand to match inner_dim
        dt_gate = torch.sigmoid(dt)

        # Combine SSM output with delta gating
        ssm_out = ssm_c * dt_gate  # [batch, seq_len, inner_dim]

        # Out projection
        out = self.out_proj(ssm_out)  # [batch, seq_len, hidden_dim]

        # Residual connection
        return residual + out
