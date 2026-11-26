import sys
import json
import argparse
import ast
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "eos_v2_pilot" / "best.pt"
HELDOUT_JSONL = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "curriculum_heldout_eval.jsonl"
HELDOUT_TOKENS = PROJECT_DIR / "data" / "experiment_2" / "training_sequences" / "heldout_tokens.bin"
HELDOUT_MASK = PROJECT_DIR / "data" / "experiment_2" / "training_sequences" / "heldout_mask.bin"
ORIGINAL_VAL_TOKENS = PROJECT_DIR / "data" / "processed" / "training_sequences_eos_v2" / "validation_tokens.bin"
ORIGINAL_VAL_MASK = PROJECT_DIR / "data" / "processed" / "training_sequences_eos_v2" / "validation_mask.bin"

BENCHMARK_PROMPTS = [
    "Hello",
    "What is artificial intelligence?",
    "What is machine learning?",
    "Explain Python in simple words.",
    "What is 2 + 2?",
    "Explain monopsony in simple words.",
    "Explain what a function is in Python.",
    "What is the difference between AI and machine learning?"
]

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FromScratchLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.1)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def evaluate_binary_split(model, tokens_path: Path, mask_path: Path, device: torch.device):
    if not tokens_path.exists() or not mask_path.exists():
        return None
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
    num_seqs = len(tokens) // CONTEXT_LENGTH

    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_active = 0

    batch_size = 16
    with torch.no_grad():
        for start_idx in range(0, num_seqs, batch_size):
            end_idx = min(start_idx + batch_size, num_seqs)
            b_tokens = torch.tensor(tokens[start_idx * CONTEXT_LENGTH : end_idx * CONTEXT_LENGTH].reshape(-1, CONTEXT_LENGTH), dtype=torch.long, device=device)
            b_mask = torch.tensor(mask[start_idx * CONTEXT_LENGTH : end_idx * CONTEXT_LENGTH].reshape(-1, CONTEXT_LENGTH), dtype=torch.bool, device=device)

            input_ids = b_tokens[:, :-1]
            target_ids = b_tokens[:, 1:]
            active_mask = b_mask[:, 1:]

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(input_ids)
                tok_loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_ids.reshape(-1)).reshape(target_ids.shape)
                loss_elements = tok_loss[active_mask]
                total_loss += loss_elements.sum().item()
                total_active += loss_elements.numel()

    mean_loss = total_loss / total_active if total_active > 0 else float("inf")
    ppl = float(np.exp(min(mean_loss, 20.0)))
    return {
        "sequences": num_seqs,
        "active_tokens": total_active,
        "mean_loss": round(mean_loss, 4),
        "perplexity": round(ppl, 2)
    }


def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    do_sample: bool = False
):
    stop_token_ids = {0, 2, 3, 4, 5}
    formatted = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(formatted)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    generated_tokens = []
    stop_reason = "max_new_tokens_reached"

    for _ in range(max_new_tokens):
        context = input_ids[:, -CONTEXT_LENGTH:]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(context)
        logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0 and generated_tokens:
            for tid in set(generated_tokens):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if not do_sample:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if 0 < top_k < logits.size(-1):
                topk_vals, _ = torch.topk(logits, top_k)
                kth_val = topk_vals[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth_val, torch.full_like(logits, -float("inf")), logits)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                filter_indices = sorted_indices[sorted_mask]
                logits[0, filter_indices] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if token_id in stop_token_ids:
            stop_reason = "stop_token_encountered"
            break

        generated_tokens.append(token_id)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    decoded = tokenizer.decode(generated_tokens)
    return {
        "text": decoded.strip(),
        "token_count": len(generated_tokens),
        "stop_reason": stop_reason,
        "token_ids": generated_tokens
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 2 Held-Out Evaluation")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_heldout_samples", type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)

    print("=" * 70)
    print("EXPERIMENT 2 HELD-OUT EVALUATION SUITE")
    print("=" * 70)
    print(f"Evaluating checkpoint: {ckpt_path}")
    print(f"Device:                {device}")

    model, ckpt_data = load_model(ckpt_path, device)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # 1. Perplexity evaluations
    print("\nEvaluating Cross-Entropy Loss & Perplexity:")
    heldout_loss_stats = evaluate_binary_split(model, HELDOUT_TOKENS, HELDOUT_MASK, device)
    if heldout_loss_stats:
        print(f"  [Held-Out Curriculum Suite ({heldout_loss_stats['sequences']} seqs)]: Loss = {heldout_loss_stats['mean_loss']:.4f}, PPL = {heldout_loss_stats['perplexity']:.2f}")

    orig_val_stats = evaluate_binary_split(model, ORIGINAL_VAL_TOKENS, ORIGINAL_VAL_MASK, device)
    if orig_val_stats:
        print(f"  [Original OASST1 Val Split ({orig_val_stats['sequences']} seqs)]: Loss = {orig_val_stats['mean_loss']:.4f}, PPL = {orig_val_stats['perplexity']:.2f}")

    # 2. Benchmark Prompt Evaluation (Original 8 Prompts)
    print("\n" + "=" * 70)
    print("EVALUATING ORIGINAL 8 BENCHMARK PROMPTS")
    print("=" * 70)
    for idx, prompt in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"\nPrompt {idx}: \"{prompt}\"")
        res_sample = generate(model, tokenizer, prompt, device, do_sample=True, temperature=0.8, top_p=0.9)
        print(f"  [Sampled]: {res_sample['text']} (tokens={res_sample['token_count']}, reason={res_sample['stop_reason']})")

    # 3. Held-out suite sampling
    print("\n" + "=" * 70)
    print("EVALUATING SAMPLES FROM HELD-OUT GENERALIZATION SUITE")
    print("=" * 70)
    with open(HELDOUT_JSONL, "r", encoding="utf-8") as f:
        heldout_items = [json.loads(l) for l in f if l.strip()]

    # Stratified sample: 5 arithmetic, 5 python, 5 concepts
    samples_by_cat = {"arithmetic_heldout": [], "python_heldout_task": [], "concept_heldout": []}
    for item in heldout_items:
        cat = item.get("category", "")
        for k in samples_by_cat:
            if k in cat and len(samples_by_cat[k]) < 5:
                samples_by_cat[k].append(item)

    for cat_name, items in samples_by_cat.items():
        print(f"\n--- Category: {cat_name} ---")
        for item in items:
            q = item["messages"][0]["text"]
            ground_truth = item["messages"][1]["text"]
            res = generate(model, tokenizer, q, device, do_sample=False)
            print(f"Q: {q}")
            print(f"  Generated: {res['text'][:180]}...")
            print(f"  Expected:  {ground_truth[:120]}...")


if __name__ == "__main__":
    main()
