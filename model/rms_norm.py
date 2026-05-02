import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, dim] or [batch, dim]
        power_mean = torch.mean(x ** 2, dim=-1, keepdim=True)
        norm = torch.sqrt(power_mean + self.eps)
        return x / norm * self.weight
