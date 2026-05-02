import torch
from model.config import AchLLMConfig
from model.gqa_attention import GQAAttention

def test_gqa_output_shape():
    """Verify GQA attention preserves input shape"""
    config = AchLLMConfig()
    attn = GQAAttention(config)

    # Input: batch=2, seq_len=128, hidden_dim=512
    x = torch.randn(2, 128, config.hidden_dim)
    output = attn(x)

    assert output.shape == (2, 128, config.hidden_dim), \
        f"Shape mismatch: expected (2,128,512), got {output.shape}"
    print("✓ GQA attention output shape correct")

def test_gqa_bitlinear_projections():
    """Verify all projections are BitLinear"""
    config = AchLLMConfig()
    attn = GQAAttention(config)

    from model.bitlinear import BitLinear
    projections = [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]
    for proj in projections:
        assert isinstance(proj, BitLinear), f"Projection {proj} is not BitLinear"
    print("✓ All GQA projections are BitLinear")

def test_gqa_qk_norm():
    """Verify QK-Norm layers exist and are RMSNorm"""
    config = AchLLMConfig()
    attn = GQAAttention(config)

    from model.rms_norm import RMSNorm
    assert isinstance(attn.q_norm, RMSNorm), "Q-Norm is not RMSNorm"
    assert isinstance(attn.k_norm, RMSNorm), "K-Norm is not RMSNorm"
    print("✓ QK-Norm layers present and correct")

if __name__ == "__main__":
    test_gqa_output_shape()
    test_gqa_bitlinear_projections()
    test_gqa_qk_norm()
    print("\nAll GQA attention tests passed!")
