"""
Separate training script for PrefixRouter.
Trains the router to classify task types: {code, chat, math, reason}.

Usage:
    python train/train_router.py --model_checkpoint path/to/checkpoint.pt
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model.config import AchLLMConfig
from model.model import AchLLM
from model.router import PrefixRouter


class RouterDataset(Dataset):
    """Dataset for router training - 10K labeled examples"""
    def __init__(self, tokenizer_path, num_samples=10000):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # In practice, use Claude API to auto-label 10K diverse prompts
        # Here we create a placeholder dataset
        self.samples = []
        self.labels = {"code": 0, "chat": 1, "math": 2, "reason": 3}

        # Placeholder: create dummy samples
        for _ in range(num_samples):
            task = np.random.choice(["code", "chat", "math", "reason"])
            self.samples.append(f"Example {task} prompt")
            self.labels.append(self.labels[task])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        label = self.labels[idx]

        tokens = self.tokenizer(text, max_length=32, padding="max_length", truncation=True)
        return torch.tensor(tokens["input_ids"]), torch.tensor(label)


def train_router(model_checkpoint, num_epochs=5):
    """Train the PrefixRouter"""
    # Load model
    config = AchLLMConfig()
    model = AchLLM(config)
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint["model"])

    # Freeze main model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Only train router
    router = model.router
    for param in router.parameters():
        param.requires_grad = True

    # Dataset and dataloader
    dataset = RouterDataset("tokenizer.json")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(router.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"Training router for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for input_ids, labels in dataloader:
            # Get embeddings from main model (frozen)
            with torch.no_grad():
                embeddings = model.embed(input_ids)

            # Router forward
            logits = router.classifier(embeddings.mean(dim=1))
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloader):.4f} | Acc: {acc:.2%}")

    # Save router only
    torch.save({"router": router.state_dict()}, "checkpoints/router_final.pt")
    print("Router training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PrefixRouter")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    train_router(args.model_checkpoint, args.epochs)
