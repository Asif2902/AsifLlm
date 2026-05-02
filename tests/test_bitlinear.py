import torch
from model.bitlinear import BitLinear

def test_bitlinear_ternary_constraint():
    """Verify BitLinear weights are ternary {-1, 0, 1} after quantization"""
    layer = BitLinear(in_features=512, out_features=512)

    # Forward pass to trigger quantization
    x = torch.randn(2, 128, 512)
    output = layer(x)

    # Get quantized weight values
    w_quant, _ = layer._quantize_weight(layer.weight)
    unique_vals = w_quant.unique().tolist()

    # Check all values are in {-1, 0, 1}
    for val in unique_vals:
        assert val in [-1.0, 0.0, 1.0], f"Non-ternary weight value: {val}"
    print("✓ Ternary weight constraint satisfied")

def test_bitlinear_output_shape():
    """Verify output shape matches input shape for 3D input"""
    layer = BitLinear(in_features=512, out_features=512)
    x = torch.randn(2, 128, 512)
    output = layer(x)
    assert output.shape == (2, 128, 512), f"Output shape mismatch: {output.shape}"
    print("✓ Output shape correct")

def test_bitlinear_no_bias():
    """Verify BitLinear has no bias by default"""
    layer = BitLinear(512, 512)
    assert layer.bias is None, "BitLinear should have no bias by default"
    print("✓ No bias by default")

if __name__ == "__main__":
    test_bitlinear_ternary_constraint()
    test_bitlinear_output_shape()
    test_bitlinear_no_bias()
    print("\nAll BitLinear tests passed!")
