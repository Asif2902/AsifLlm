"""
CPU inference script for AchLLM with speculative decoding.
Uses draft head to accelerate generation 2-3x on mobile CPU.

Usage:
    python inference/generate.py --model_checkpoint checkpoints/sft_final.pt --prompt "Hello, how are you?"
"""
import argparse
import torch
import torch.nn.functional as F

from model.config import AchLLMConfig
from model.model import AchLLM


def speculative_generate(model, input_ids, max_new_tokens=100, draft_verify_count=4):
    """
    Speculative decoding: draft head guesses next N tokens, main model verifies all at once.
    """
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get current sequence
            seq = generated[:, -model.config.max_seq_len:]

            # Draft head: generate N draft tokens autoregressively
            draft_tokens = []
            draft_input = seq.clone()
            for i in range(draft_verify_count):
                # Project to draft space
                embeddings = model.embed(draft_input)
                draft_embeds = model.draft_proj(embeddings[:, -1:, :])  # Last token only
                draft_logits = model.draft(draft_embeds)
                next_token = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                draft_input = torch.cat([draft_input, next_token], dim=1)

            # Main model: verify all N draft tokens in one forward pass
            main_embeds = model.embed(draft_input)
            main_logits, _ = model(draft_input, training=False)

            # Acceptance criterion: compare draft predictions with main model
            accepted = 0
            for i, draft_tok in enumerate(draft_tokens):
                main_tok = torch.argmax(main_logits[:, len(seq) + i, :], dim=-1, keepdim=True)
                if draft_tok.item() == main_tok.item():
                    accepted += 1
                else:
                    break

            # Append accepted tokens (or fall back to main model)
            if accepted > 0:
                generated = torch.cat([generated, torch.cat(draft_tokens[:accepted], dim=1)], dim=1)
            else:
                # Fall back to main model prediction
                next_tok = torch.argmax(main_logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)

            # Check for EOS
            if generated[0, -1].item() == model.config.eos_token_id or model.config.pad_token_id:
                break

    return generated


def generate(model, input_ids, max_new_tokens=100, temperature=0.8, top_k=50):
    """Standard greedy/ sampled generation (no speculative decoding)"""
    model.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            seq = generated[:, -model.config.max_seq_len:]

            # Run model
            logits, _ = model(seq, training=False)

            # Apply temperature and top-k
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_logits[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token[0].item() in [model.config.eos_token_id, model.config.pad_token_id]:
                break

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate with AchLLM")
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--use_speculative", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    # Load model
    config = AchLLMConfig()
    model = AchLLM(config)
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Tokenize prompt
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("tokenizer.json")
    input_ids = torch.tensor([tokenizer.encode(args.prompt)])

    print(f"Prompt: {args.prompt}")
    print("Generating...")

    # Generate
    if args.use_speculative:
        output_ids = speculative_generate(model, input_ids, args.max_new_tokens)
    else:
        output_ids = generate(model, input_ids, args.max_new_tokens, args.temperature, args.top_k)

    # Decode
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated:\n{output_text}")


if __name__ == "__main__":
    main()
