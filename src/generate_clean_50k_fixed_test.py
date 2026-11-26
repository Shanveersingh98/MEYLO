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

DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "clean_50k_best.pt"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
OUTPUT_JSONL = PROJECT_DIR / "logs" / "clean_50k_fixed_generation_test.jsonl"

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

def validate_parameters(temperature, top_k, top_p, repetition_penalty, max_new_tokens):
    """Validates decoding hyperparameters with clear assertions and error messages."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0.0, 1.0], got {top_p}")
    if repetition_penalty < 1.0:
        raise ValueError(f"repetition_penalty must be >= 1.0, got {repetition_penalty}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

def validate_special_tokens(tokenizer):
    """Validates special token IDs present in the tokenizer."""
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
            raise ValueError(
                f"Special token '{token_name}' id mismatch: expected {expected_id}, got {actual_id}"
            )
        resolved[token_name] = actual_id
    return resolved

def load_clean_model(checkpoint_path, device):
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
    """Calculates n-gram diversity metrics on generated tokens."""
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

    # Check for consecutive identical repeats
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

def generate_original_baseline(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100
):
    """
    Original uncorrected greedy generation matching generate_clean_50k.py exactly:
    - Repetition penalty bypassed in greedy mode
    - Stop condition checks ONLY [EOS]
    - Computations in float16 autocast
    """
    formatted = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(formatted)
    prompt_length = len(encoded.ids)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("[EOS]")

    stop_reason = "max_new_tokens_reached"
    generated_tokens = []

    for _ in range(max_new_tokens):
        input_context = input_ids[:, -1024:]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(input_context)
        
        logits = logits[:, -1, :]
        # Original: greedy argmax with ZERO repetition penalty
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        token_id = next_token.item()

        if token_id == eos_id:
            stop_reason = "eos_token"
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

def generate_corrected(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.15,
    do_sample=False
):
    """
    Corrected generation:
    1. Repetition penalty applied before selection for BOTH greedy and sampling.
    2. Penalizes generated tokens ONLY, not tokens from the prompt.
    3. Proper temperature scaling, top-k, and top-p (nucleus) filtering in float32.
    4. Multi-token stop condition: stops on [EOS] OR role boundary tokens (<|user|>, <|assistant|>).
    5. Excludes stop tokens from emitted text, preserving preceding valid tokens.
    """
    validate_parameters(temperature, top_k, top_p, repetition_penalty, max_new_tokens)
    special_ids = validate_special_tokens(tokenizer)

    stop_token_ids = {
        special_ids["[EOS]"],
        special_ids["<|user|>"],
        special_ids["<|assistant|>"],
        special_ids["[BOS]"]
    }

    formatted = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(formatted)
    prompt_length = len(encoded.ids)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    generated_tokens = []
    stop_reason = "max_new_tokens_reached"

    for _ in range(max_new_tokens):
        input_context = input_ids[:, -1024:]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(input_context)

        # Cast logits to float32 for high-precision arithmetic
        logits = logits[:, -1, :].float()

        # Step 1: Apply Repetition Penalty to GENERATED tokens only (for both greedy and sample)
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            gen_token_set = set(generated_tokens)
            for tid in gen_token_set:
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        if not do_sample:
            # Corrected Deterministic / Greedy Argmax (on penalized logits)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Corrected Stochastic Sampling with Temperature + Top-K + Top-P
            logits = logits / temperature

            # Top-K Filtering
            if top_k > 0 and top_k < logits.size(-1):
                topk_values, topk_indices = torch.topk(logits, top_k)
                kth_val = topk_values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < kth_val, torch.full_like(logits, float("-inf")), logits)

            # Top-P (Nucleus) Filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift mask right to ensure at least the top token is retained
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()

        # Check stopping conditions
        if token_id in stop_token_ids:
            if token_id == special_ids["[EOS]"]:
                stop_reason = "eos_token"
            elif token_id == special_ids["<|user|>"]:
                stop_reason = "user_role_boundary"
            elif token_id == special_ids["<|assistant|>"]:
                stop_reason = "assistant_role_boundary"
            else:
                stop_reason = "bos_token_boundary"
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

def run_comparison_suite(model, tokenizer, device, prompts=TEST_PROMPTS):
    print("=" * 80)
    print("RUNNING 3-WAY COMPARISON SUITE:")
    print("  Mode 1: Original Greedy (Uncorrected Baseline)")
    print("  Mode 2: Corrected Greedy (RepPenalty 1.15 + Multi-Stop in Float32)")
    print("  Mode 3: Corrected Sampling (Temp 0.8 + Top-K 50 + Top-P 0.9 + RepPenalty 1.15 + Multi-Stop)")
    print("=" * 80)

    results = []

    for idx, prompt in enumerate(prompts, 1):
        print(f"\n{'='*80}\nPROMPT [{idx}/{len(prompts)}]: '{prompt}'\n{'='*80}")

        # 1. Original Greedy
        orig_text, orig_ids, orig_metrics = generate_original_baseline(
            model, tokenizer, prompt, device, max_new_tokens=100
        )
        print(f"[ORIGINAL GREEDY] (Tokens: {orig_metrics['generated_token_count']}, Stop: {orig_metrics['stop_reason']}, Loop: {orig_metrics['has_repetition_loop']})")
        print(f"{orig_text if orig_text else '[EMPTY RESPONSE]'}\n")

        # 2. Corrected Greedy
        fix_greedy_text, fix_greedy_ids, fix_greedy_metrics = generate_corrected(
            model, tokenizer, prompt, device,
            max_new_tokens=100,
            repetition_penalty=1.15,
            do_sample=False
        )
        print(f"[CORRECTED GREEDY] (Tokens: {fix_greedy_metrics['generated_token_count']}, Stop: {fix_greedy_metrics['stop_reason']}, Loop: {fix_greedy_metrics['has_repetition_loop']})")
        print(f"{fix_greedy_text if fix_greedy_text else '[EMPTY RESPONSE]'}\n")

        # 3. Corrected Sampling
        fix_sample_text, fix_sample_ids, fix_sample_metrics = generate_corrected(
            model, tokenizer, prompt, device,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True
        )
        print(f"[CORRECTED SAMPLING] (Tokens: {fix_sample_metrics['generated_token_count']}, Stop: {fix_sample_metrics['stop_reason']}, Loop: {fix_sample_metrics['has_repetition_loop']})")
        print(f"{fix_sample_text if fix_sample_text else '[EMPTY RESPONSE]'}\n")

        record = {
            "prompt_index": idx,
            "prompt": prompt,
            "original_greedy": {
                "text": orig_text,
                "token_ids": orig_ids,
                "metrics": orig_metrics
            },
            "corrected_greedy": {
                "text": fix_greedy_text,
                "token_ids": fix_greedy_ids,
                "metrics": fix_greedy_metrics
            },
            "corrected_sampling": {
                "text": fix_sample_text,
                "token_ids": fix_sample_ids,
                "metrics": fix_sample_metrics
            }
        }
        results.append(record)

    return results

def main():
    parser = argparse.ArgumentParser(description="Test corrected generation logic against baseline")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output", type=str, default=str(OUTPUT_JSONL))
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    special_ids = validate_special_tokens(tokenizer)
    print(f"Special Tokens Validated: {special_ids}")

    print(f"Loading model from: {ckpt_path}")
    model, ckpt = load_clean_model(ckpt_path, device)
    print(f"Checkpoint Step:     {ckpt.get('step')}")
    print(f"Validation Loss:     {ckpt.get('validation_loss')}")

    results = run_comparison_suite(model, tokenizer, device, prompts=TEST_PROMPTS)

    print(f"\nWriting test records to: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done! {len(results)} prompt evaluations saved successfully.")

if __name__ == "__main__":
    main()
