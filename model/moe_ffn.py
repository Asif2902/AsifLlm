import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear
from .rms_norm import RMSNorm
from .config import AchLLMConfig

class MoEExpert(nn.Module):
    """Single MoE expert with SwiGLU activation"""
    def __init__(self, config: AchLLMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim  # 512
        self.inner_dim = self.hidden_dim * config.mamba_expand_factor  # 1024

        # SwiGLU: gate_proj * up_proj (both 512→1024), then down_proj (1024→512)
        self.gate_proj = BitLinear(self.hidden_dim, self.inner_dim)
        self.up_proj = BitLinear(self.hidden_dim, self.inner_dim)
        self.down_proj = BitLinear(self.inner_dim, self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch*seq_len, 512]
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        swiglu = F.silu(gate) * up
        return self.down_proj(swiglu)

class SparseMoE(nn.Module):
    def __init__(self, config: AchLLMConfig):
        super().__init__()
        self.num_experts = config.num_experts  # 8
        self.top_k_train = config.top_k_experts_train  # 2 (DS-MoE)
        self.top_k_infer = config.top_k_experts_infer  # 1
        self.aux_loss_weight = config.moe_aux_loss_weight  # 0.01

        # Experts
        self.experts = nn.ModuleList([
            MoEExpert(config) for _ in range(self.num_experts)
        ])

        # Router: 512 → 8 experts, with QK-Norm on logits
        self.router = BitLinear(config.hidden_dim, self.num_experts)
        self.router_norm = RMSNorm(self.num_experts)  # QK-Norm on router logits

    def forward(self, x: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, hidden_dim] (512)
        batch, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)  # [batch*seq_len, 512]
        num_tokens = x_flat.shape[0]

        # Router logits: [num_tokens, 8]
        router_logits = self.router(x_flat)
        router_logits = self.router_norm(router_logits)  # QK-Norm

        if training:
            top_k = self.top_k_train
            # DS-MoE: top-2, weight by router score
            router_probs = F.softmax(router_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(router_probs, top_k, dim=-1)
            # Normalize probabilities
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        else:
            top_k = self.top_k_infer
            # Top-1 only
            top_k_probs, top_k_indices = torch.topk(router_logits, top_k, dim=-1)
            top_k_probs = F.softmax(top_k_probs, dim=-1)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for i in range(top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_probs[:, i].unsqueeze(-1)

            # Gather tokens per expert
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.sum() == 0:
                    continue
                selected_x = x_flat[mask]
                expert_output = self.experts[expert_id](selected_x)
                output[mask] += expert_weight[mask] * expert_output

        # Reshape back to [batch, seq_len, hidden_dim]
        output = output.reshape(batch, seq_len, hidden_dim)

        # Auxiliary load balance loss
        if training:
            # Compute expert counts
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for expert_id in range(self.num_experts):
                expert_counts[expert_id] = (top_k_indices == expert_id).sum().float()
            expert_frac = expert_counts / num_tokens
            router_prob_mean = router_probs.mean(dim=0)
            aux_loss = self.num_experts * (expert_frac * router_prob_mean).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return output, self.aux_loss_weight * aux_loss
