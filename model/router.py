import torch
import torch.nn as nn
from .config import AchLLMConfig

class PrefixRouter(nn.Module):
    """
    Task-aware prefix router: reads first 32 tokens and classifies task type.
    Runs once per prompt, adjusts generation parameters based on task class.
    Output classes: {code, chat, math, reason}
    Total params: ~2M (512 → 256 → 4)
    """
    def __init__(self, hidden_dim: int = 512, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.max_prefix_len = 32

        # Small MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),  # 512 → 256
            nn.ReLU(),
            nn.Linear(256, num_classes)   # 256 → 4
        )

        # Task-specific generation settings
        self.task_configs = {
            0: {"name": "code",   "temperature": 0.2, "top_k": 40,
                 "expert_bias": [0, 0, 1, 0, 0, 0, 0, 1]},
            1: {"name": "chat",   "temperature": 0.8, "top_k": 50,
                 "expert_bias": [1, 1, 0, 0, 0, 0, 0, 0]},
            2: {"name": "math",   "temperature": 0.1, "top_k": 20,
                 "expert_bias": [0, 0, 0, 1, 1, 0, 0, 0]},
            3: {"name": "reason", "temperature": 0.4, "top_k": 40,
                 "expert_bias": [0, 0, 0, 0, 1, 1, 0, 0]},
        }

    def forward(self, input_ids: torch.Tensor, embeddings: torch.Tensor = None) -> dict:
        """
        Args:
            input_ids: [batch, seq_len] - not used directly, using embeddings
            embeddings: [batch, seq_len, hidden_dim] - token embeddings
        Returns:
            dict with task_class, task_name, and generation settings
        """
        if embeddings is None:
            raise ValueError("Embeddings must be provided for router")

        # Take first min(32, seq_len) tokens
        seq_len = embeddings.shape[1]
        prefix_len = min(self.max_prefix_len, seq_len)
        prefix_embeddings = embeddings[:, :prefix_len, :]  # [batch, prefix_len, hidden]

        # Mean-pool the prefix tokens
        pooled = prefix_embeddings.mean(dim=1)  # [batch, hidden]

        # Classify
        logits = self.classifier(pooled)  # [batch, 4]
        probs = torch.softmax(logits, dim=-1)
        task_class = torch.argmax(probs, dim=-1)  # [batch]

        # Return task info for each batch item
        batch_size = embeddings.shape[0]
        results = []
        for b in range(batch_size):
            tc = task_class[b].item()
            results.append(self.task_configs[tc])

        return {
            "task_class": task_class,
            "probs": probs,
            "configs": results
        }

    def get_generation_params(self, task_class: torch.Tensor) -> dict:
        """Get generation parameters for a given task class"""
        tc = task_class.item() if task_class.numel() == 1 else task_class[0].item()
        return self.task_configs[tc]
