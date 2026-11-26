import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

PILOT_BEST_CHECKPOINT = PROJECT_DIR / "checkpoints" / "eos_v2_pilot" / "best.pt"
BASELINE_BEST_CHECKPOINT = PROJECT_DIR / "checkpoints" / "clean_50k_best.pt"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
DEFAULT_OUTPUT_JSONL = PROJECT_DIR / "logs" / "eos_v2_pilot" / "pilot_generation_eval.jsonl"

TEST_PROMPTS = [
    "Hello",
    "What is artificial intelligence?",
    "What is machine learning?",
    "Explain Python in simple words.",
    "What is 2 + 2?",
    "Explain monopsony in simple words.",
    "Explain what a function is in Python.",
    "What is the difference between AI and machine learning?"
]

def validate_special_tokens(tokenizer):
    required_tokens = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
    }
    resolved = {}
    for token_name, expected_id in required_tokens.items():
        actual_id = tokenizer.token_to_id(token_name)
        if actual_id is None:
            raise ValueError(f"Required special token '{token_name}' not found in tokenizer!")
        if actual_id != expected_id:
            raise ValueError(f"Token '{token_name}' mismatch: expected {expected_id}, got {actual_id}")
        resolved[token_name] = actual_id
    return resolved

def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FromScratchLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint

def compute_repetition_metrics(token_ids):
    if not token_ids:
        return {
            "distinct_1": 1.0,
            "distinct_2": 1.0,
            "max_consecutive_repeat": 0,
            "has_repetition_loop": False
        }

    total_1 = len(token_ids)
    uniq_1 = len(set(token_ids))
    d1 = float(uniq_1 / total_1)

    if total_1 > 1:
        bigrams = [tuple(token_ids[i:i+2]) for i in range(total_1 - 1)]
        d2 = float(len(set(bigrams)) / len(bigrams))
    else:
        d2 = 1.0

    max_repeat = 1
    curr_repeat = 1
    for i in range(1, total_1):
        if token_ids[i] == token_ids[i - 1]:
            curr_repeat += 1
            max_repeat = max(max_repeat, curr_repeat)
        else:
            curr_repeat = 1

    has_loop = max_repeat >= 4 or (total_1 >= 20 and d2 < 0.4)

    return {
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "max_consecutive_repeat": int(max_repeat),
        "has_repetition_loop": bool(has_loop)
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
    special_ids = validate_special_tokens(tokenizer)
    stop_token_ids = {
        special_ids["[EOS]"],
        special_ids["<|user|>"],
        special_ids["<|assistant|>"],
        special_ids["[BOS]"]
    }

    formatted = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(formatted)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    generated_tokens = []
    stop_reason = "max_new_tokens_reached"

    for _ in range(max_new_tokens):
        input_context = input_ids[:, -1024:]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(input_context)

        logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for tid in set(generated_tokens):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if not do_sample:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0 and top_k < logits.size(-1):
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
        input_ids = torch.cat([input_ids, next_token], dim=1)

    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    metrics = compute_repetition_metrics(generated_tokens)
    metrics.update({
        "generated_token_count": len(generated_tokens),
        "stop_reason": stop_reason
    })
    return decoded_text, generated_tokens, metrics

def run_evaluation_suite(model, tokenizer, device: torch.device):
    results = []
    print("\n" + "=" * 80)
    print("EVALUATING MODEL ON 8 FIXED BENCHMARK PROMPTS")
    print("=" * 80)

    for idx, prompt in enumerate(TEST_PROMPTS, start=1):
        # 1. Greedy
        g_text, g_ids, g_metrics = generate(
            model, tokenizer, prompt, device,
            max_new_tokens=100,
            repetition_penalty=1.15,
            do_sample=False
        )

        # 2. Sampling
        s_text, s_ids, s_metrics = generate(
            model, tokenizer, prompt, device,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True
        )

        print(f"\n--- Prompt {idx}/8: '{prompt}' ---")
        print(f"[GREEDY] (Tokens: {g_metrics['generated_token_count']}, Stop: {g_metrics['stop_reason']}, Loop: {g_metrics['has_repetition_loop']})")
        print(f"  {g_text if g_text else '[EMPTY]'}")
        print(f"[SAMPLED] (Tokens: {s_metrics['generated_token_count']}, Stop: {s_metrics['stop_reason']}, Loop: {s_metrics['has_repetition_loop']})")
        print(f"  {s_text if s_text else '[EMPTY]'}")

        results.append({
            "prompt_index": idx,
            "prompt": prompt,
            "greedy": {"text": g_text, "token_ids": g_ids, "metrics": g_metrics},
            "sampled": {"text": s_text, "token_ids": s_ids, "metrics": s_metrics}
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pilot Checkpoint on Fixed Prompts")
    parser.add_argument("--checkpoint", type=str, default=str(PILOT_BEST_CHECKPOINT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_JSONL))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    validate_special_tokens(tokenizer)

    model, ckpt = load_checkpoint_model(ckpt_path, device)
    print(f"Loaded checkpoint from: {ckpt_path}")
    print(f"  Optimizer update:     {ckpt.get('optimizer_update', ckpt.get('step'))}")
    print(f"  Validation loss:      {ckpt.get('validation_loss'):.4f}")

    results = run_evaluation_suite(model, tokenizer, device)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nEvaluation complete. Results saved to: {out_path}")

if __name__ == "__main__":
    main()
