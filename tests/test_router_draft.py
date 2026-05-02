import torch
from model.config import AchLLMConfig
from model.router import PrefixRouter
from model.draft_head import DraftHead

def test_router():
    """Test PrefixRouter"""
    router = PrefixRouter(hidden_dim=512, num_classes=4)

    # Create fake embeddings: batch=2, seq_len=64, hidden=512
    embeddings = torch.randn(2, 64, 512)

    # Run router
    result = router.forward(input_ids=None, embeddings=embeddings)

    assert "task_class" in result
    assert "configs" in result
    assert result["task_class"].shape == (2,)
    print("✓ PrefixRouter output correct")
    print(f"  Task classes: {result['task_class'].tolist()}")
    print(f"  Configs: {[c['name'] for c in result['configs']]}")

def test_draft_head():
    """Test DraftHead"""
    config = AchLLMConfig()
    draft = DraftHead(config)

    # Input: batch=2, seq_len=32, draft_hidden=256
    x = torch.randn(2, 32, config.draft_hidden_dim)
    logits = draft(x)

    assert logits.shape == (2, 32, config.vocab_size), \
        f"Shape mismatch: {logits.shape}"
    print("✓ DraftHead output shape correct")
    print(f"  Logits shape: {logits.shape}")

if __name__ == "__main__":
    test_router()
    test_draft_head()
    print("\nRouter and Draft Head tests passed!")
