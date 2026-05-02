import torch
from model.config import AchLLMConfig
from model.mamba_block import MambaBlock

def test_mamba_output_shape():
    """Verify Mamba-2 block preserves input shape"""
    config = AchLLMConfig()
    block = MambaBlock(config)

    # Input: batch=2, seq_len=128, hidden_dim=512
    x = torch.randn(2, 128, config.hidden_dim)
    output = block(x)

    assert output.shape == (2, 128, config.hidden_dim), \
        f"Shape mismatch: expected (2,128,512), got {output.shape}"
    print("✓ Mamba-2 block output shape correct")

def test_mamba_bitlinear_usage():
    """Verify all linear layers in Mamba block are BitLinear"""
    config = AchLLMConfig()
    block = MambaBlock(config)

    from model.bitlinear import BitLinear
    linear_layers = [
        block.in_proj, block.ssm_B, block.ssm_C,
        block.dt_proj, block.out_proj
    ]
    for layer in linear_layers:
        assert isinstance(layer, BitLinear), \
            f"Layer {layer} is not BitLinear"
    print("✓ All linear layers are BitLinear")

if __name__ == "__main__":
    test_mamba_output_shape()
    test_mamba_bitlinear_usage()
    print("\nAll Mamba block tests passed!")
