import torch
from model.config import AchLLMConfig
from model.moe_ffn import SparseMoE

def test_moe_output_shape():
    """Verify MoE preserves input shape"""
    config = AchLLMConfig()
    moe = SparseMoE(config)

    # Input: batch=2, seq_len=128, hidden_dim=512
    x = torch.randn(2, 128, config.hidden_dim)
    output, aux_loss = moe(x, training=True)

    assert output.shape == (2, 128, config.hidden_dim), \
        f"Shape mismatch: expected (2,128,512), got {output.shape}"
    print("✓ MoE output shape correct")
    print(f"  Aux loss: {aux_loss.item():.4f}")

def test_moe_experts_bitlinear():
    """Verify all expert projections are BitLinear"""
    config = AchLLMConfig()
    moe = SparseMoE(config)

    from model.bitlinear import BitLinear
    for expert_id, expert in enumerate(moe.experts):
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            proj = getattr(expert, proj_name)
            assert isinstance(proj, BitLinear), \
                f"Expert {expert_id} {proj_name} is not BitLinear"
    print("✓ All expert projections are BitLinear")

def test_moe_router_norm():
    """Verify router uses QK-Norm (RMSNorm)"""
    config = AchLLMConfig()
    moe = SparseMoE(config)

    from model.rms_norm import RMSNorm
    assert isinstance(moe.router_norm, RMSNorm), "Router norm is not RMSNorm"
    print("✓ Router QK-Norm present")

if __name__ == "__main__":
    test_moe_output_shape()
    test_moe_experts_bitlinear()
    test_moe_router_norm()
    print("\nAll MoE tests passed!")
