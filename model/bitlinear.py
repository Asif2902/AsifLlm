import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Original float weight tensor (quantized during forward)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization for linear layers
        nn.init.kaiming_uniform_(self.weight, a=0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _quantize_weight(self, weight: torch.Tensor):
        """Quantize weights to ternary {-1, 0, 1} using absmean scaling (BitNet 1.58-bit)"""
        # Compute absmean scale
        scale = torch.mean(torch.abs(weight))
        if scale == 0:
            return weight, scale

        # Quantize: round(weight / scale) then clamp to [-1, 1]
        quantized = torch.round(weight / scale).clamp(-1, 1)

        # Straight-Through Estimator (STE): forward uses quantized values,
        # backward passes gradient to original weight
        quantized = quantized + (weight - weight.detach())
        return quantized, scale

    def _quantize_activation(self, x: torch.Tensor):
        """Quantize activations to int8 using per-token absmax scaling"""
        # Compute per-token absmax (over feature dimension)
        if x.dim() == 3:
            # x: [batch, seq_len, features]
            absmax = torch.amax(torch.abs(x), dim=-1, keepdim=True)  # [batch, seq_len, 1]
        else:
            # x: [batch, features]
            absmax = torch.amax(torch.abs(x), dim=-1, keepdim=True)  # [batch, 1]

        # Scale to int8 range (symmetric, using 127 for positive max)
        scale = absmax / 127.0
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        # Quantize to int8
        x_quant = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
        return x_quant, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize input activation to int8
        x_quant, x_scale = self._quantize_activation(x)

        # Quantize weight to ternary
        w_quant, w_scale = self._quantize_weight(self.weight)

        # Reshape for linear operation
        original_shape = x.shape
        if x.dim() == 3:
            batch, seq_len, _ = x.shape
            x_flat = x_quant.reshape(batch * seq_len, -1)
            # Linear: (batch*seq_len, in_features) @ (out_features, in_features).T
            output_flat = F.linear(x_flat.float(), w_quant.float())
            # Dequantize: multiply by weight scale and activation scale
            output_flat = output_flat * (w_scale * x_scale.reshape(-1, 1))
            output = output_flat.reshape(batch, seq_len, -1)
        else:
            # x: [batch, features]
            output = F.linear(x_quant.float(), w_quant.float()) * (w_scale * x_scale)

        if self.bias is not None:
            output = output + self.bias
        return output
