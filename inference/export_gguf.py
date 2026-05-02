"""
Export AchLLM to GGUF format for llama.cpp / bitnet.cpp inference.

The GGUF format is used by:
- llama.cpp: https://github.com/ggerganov/llama.cpp
- bitnet.cpp: https://github.com/microsoft/bitnet.cpp

Usage:
    python inference/export_gguf.py --model_checkpoint checkpoints/sft_final.pt --output achllm_800m.gguf
"""
import argparse
import struct
import torch
import numpy as np

from model.config import AchLLMConfig
from model.model import AchLLM


def export_to_gguf(model, output_path, vocab_size=32000):
    """
    Export AchLLM to GGUF format.
    Note: This is a simplified version. Full GGUF export requires
    implementing the full GGUF spec.
    """
    print(f"Exporting model to GGUF: {output_path}")

    # GGUF header
    with open(output_path, "wb") as f:
        # Magic number: "GGUF"
        f.write(b"GGUF")

        # Version
        f.write(struct.pack("<I", 3))  # Version 3

        # Tensor count (simplified - just write embedding and LM head)
        tensor_count = sum(1 for _ in model.parameters())
        f.write(struct.pack("<Q", tensor_count))

        # Write each tensor
        for name, param in model.named_parameters():
            # Tensor name
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # Shape
            shape = list(param.shape)
            f.write(struct.pack("<I", len(shape)))
            for dim in shape:
                f.write(struct.pack("<Q", dim))

            # Data type (0 = F32, 1 = F16, 2 = Q4_0, etc.)
            # For BitLinear, weights are ternary {-1, 0, 1}
            f.write(struct.pack("<I", 0))  # F32 for now

            # Tensor data
            data = param.detach().cpu().numpy().astype(np.float32)
            f.write(data.tobytes())

    print(f"Model exported to {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def export_to_bitnet(model, output_path):
    """
    Export to bitnet.cpp format.
    BitNet uses ternary weights {-1, 0, 1} which our model already has.
    """
    print("Exporting to BitNet format...")
    print("See: https://github.com/microsoft/bitnet.cpp")
    # Placeholder for bitnet.cpp export
    # The bitnet.cpp format expects ternary weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export AchLLM to GGUF")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="achllm_800m.gguf")
    parser.add_argument("--format", type=str, choices=["gguf", "bitnet"], default="gguf")
    args = parser.parse_args()

    import os

    # Load model
    config = AchLLMConfig()
    model = AchLLM(config)
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Export
    if args.format == "gguf":
        export_to_gguf(model, args.output)
    else:
        export_to_bitnet(model, args.output)

    print("Export complete!")
