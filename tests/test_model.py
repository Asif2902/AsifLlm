import torch
from model.config import AchLLMConfig
from model.model import AchLLM

def count_params(model: torch.nn.Module, requires_grad_only: bool = False) -> int:
    total = 0
    for p in model.parameters():
        if requires_grad_only and not p.requires_grad:
            continue
        total += p.numel()
    return total

def count_active_params(model: AchLLM) -> int:
    """Approximate active params: exclude dormant MoE experts (top-1 inference)"""
    total = 0
    # Embedding + tied LM head (count once)
    total += model.embed.weight.numel()
    # Final norm
    total += sum(p.numel() for p in model.norm.parameters())
    # Blocks: count all non-MoE params + 1 expert per MoE layer
    for block in model.blocks:
        if hasattr(block, 'experts'):  # SparseMoE
            # Count only 1 active expert (top-1)
            for p in block.experts[0].parameters():
                total += p.numel()
            # Router params
            total += sum(p.numel() for p in block.router.parameters())
        else:
            # MambaBlock or GQAAttention
            total += sum(p.numel() for p in block.parameters())
    return total

def test_model_logits_shape():
    """Verify model outputs correct logits shape"""
    config = AchLLMConfig()
    model = AchLLM(config)

    # Input: batch=2, seq_len=128
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    logits, aux_loss = model(input_ids, training=False)

    assert logits.shape == (2, 128, config.vocab_size), \
        f"Logits shape mismatch: expected (2,128,32000), got {logits.shape}"
    print(f"✓ Model logits shape correct: {logits.shape}")
    print(f"  Aux loss (inference): {aux_loss.item():.4f}")

def test_model_param_counts():
    """Check total and active parameter counts"""
    config = AchLLMConfig()
    model = AchLLM(config)

    total_params = count_params(model)
    active_params = count_active_params(model)

    print(f"Total parameters: {total_params:,} (~{total_params/1e9:.2f}B)")
    print(f"Active parameters: {active_params:,} (~{active_params/1e9:.2f}B)")

    # Check active params per prompt specs (~120M)
    assert 100_000_000 < active_params < 150_000_000, \
        f"Active params out of range: {active_params}"
    print("✓ Active parameter count within expected range (~120M)")

if __name__ == "__main__":
    test_model_logits_shape()
    test_model_param_counts()
    print("\nFull model smoke test passed!")
