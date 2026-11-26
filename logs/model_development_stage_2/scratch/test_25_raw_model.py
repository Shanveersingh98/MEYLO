"""
test_25_raw_model.py
====================
Runs the 25 specific questions through FromScratchLM (checkpoints/ultimate_improvement/best.pt)
without web search, tools, or templates.
"""
import sys
import os
import json
import time

sys.path.insert(0, r"D:\DEEP-LEARNING-PROJECT\src")

import torch
from tokenizers import Tokenizer
from model import FromScratchLM

CHECKPOINT_PATH = r"D:\DEEP-LEARNING-PROJECT\checkpoints\ultimate_improvement\best.pt"
TOKENIZER_PATH = r"D:\DEEP-LEARNING-PROJECT\tokenizer\tokenizer.json"

QUESTIONS = [
    # General Conversation
    ("General Conversation", 1, "Hello, how are you?"),
    ("General Conversation", 2, "What are you doing?"),
    ("General Conversation", 3, "Can we talk?"),
    ("General Conversation", 4, "Tell me something interesting."),
    ("General Conversation", 5, "How was your day?"),
    ("General Conversation", 6, "What can you help me with?"),
    # Educational Questions
    ("Educational", 7, "What is Artificial Intelligence?"),
    ("Educational", 8, "What is Machine Learning?"),
    ("Educational", 9, "What is Deep Learning?"),
    ("Educational", 10, "What is a Neural Network?"),
    ("Educational", 11, "What is Supervised Learning?"),
    ("Educational", 12, "What is a CNN?"),
    ("Educational", 13, "What is a Transformer?"),
    ("Educational", 14, "What is the difference between CNN and Transformer?"),
    # Programming
    ("Programming", 15, "What is Python?"),
    ("Programming", 16, "What is a variable?"),
    ("Programming", 17, "What is a loop?"),
    ("Programming", 18, "What is a function?"),
    # Everyday Conversation
    ("Everyday Conversation", 19, "How can I improve my communication skills?"),
    ("Everyday Conversation", 20, "How can I manage my time?"),
    ("Everyday Conversation", 21, "How can I prepare for examinations?"),
    # Emotional Conversation
    ("Emotional Conversation", 22, "I am feeling sad. Can we talk?"),
    ("Emotional Conversation", 23, "I feel lonely."),
    ("Emotional Conversation", 24, "I am stressed about my examination."),
    ("Emotional Conversation", 25, "I had a bad day.")
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from: {CHECKPOINT_PATH} on {device}")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    
    model = FromScratchLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    stop_tokens = {0, 2, 3, 4, 5}
    assistant_id = tokenizer.token_to_id("<|assistant|>")

    for cat, num, q in QUESTIONS:
        prompt_text = f"[BOS]\n<|user|>\n{q}\n<|assistant|>\n"
        encoded = tokenizer.encode(prompt_text)
        prompt_len = len(encoded.ids)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

        t0 = time.time()
        gen_tokens = []
        for _ in range(150):
            ctx = input_ids[:, -1024:]
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(ctx)
            logits = logits[:, -1, :].float()

            # Token repetition penalty
            if gen_tokens:
                for tid in set(gen_tokens):
                    if logits[0, tid] > 0:
                        logits[0, tid] /= 1.15
                    else:
                        logits[0, tid] *= 1.15

            # 3-gram repetition blocking
            if len(gen_tokens) >= 2:
                curr_prefix = tuple(gen_tokens[-2:])
                banned = set()
                for i in range(len(gen_tokens) - 2):
                    if tuple(gen_tokens[i:i+2]) == curr_prefix:
                        banned.add(gen_tokens[i+2])
                for b_tok in banned:
                    logits[0, b_tok] = -float("inf")

            # Temperature & top_k & top_p
            logits = logits / 0.35
            vals, idxs = torch.topk(logits, 20)
            filt = torch.full_like(logits, -float("inf"))
            filt.scatter_(1, idxs, vals)
            logits = filt

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.85
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float("inf"))

            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            token_id = next_tok.item()

            if token_id in stop_tokens:
                break
            gen_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_tok], dim=1)

        dur_ms = (time.time() - t0) * 1000
        resp_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print(f"\n[{num:2d}] ({cat}) Q: {q}")
        print(f"     A: {resp_text}")
        print(f"     ({len(gen_tokens)} tokens, {dur_ms:.1f}ms)")

        results.append({
            "category": cat,
            "number": num,
            "question": q,
            "response": resp_text,
            "tokens": len(gen_tokens),
            "latency_ms": round(dur_ms, 1)
        })

    out_path = r"D:\DEEP-LEARNING-PROJECT\logs\model_development_stage_2\scratch\raw_25_test_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw responses to {out_path}")

if __name__ == "__main__":
    main()
