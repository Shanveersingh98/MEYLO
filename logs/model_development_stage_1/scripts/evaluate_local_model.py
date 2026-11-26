"""
Independent Model Evaluator — evaluate_local_model.py
=====================================================
Standalone evaluation harness for FromScratchLM.
Mirrors src/generate.py decoding EXACTLY. Imports ONLY model.py + tokenizer.
No chat.py, web_retrieval.py, answer_synthesis.py, intent_router.py.
"""

import sys
import os
import json
import time

# ---- ensure we can import from src/ ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
from tokenizers import Tokenizer
from model import FromScratchLM

# ======================================================
# Paths
# ======================================================
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "ultimate_improvement", "best.pt")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "tokenizer.json")
BENCHMARK_PATH = os.path.join(PROJECT_ROOT, "logs", "model_development_stage_1", "independent_model_benchmark.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "logs", "model_development_stage_1", "baseline_model_responses.json")

# ======================================================
# Generation config (matches src/generate.py)
# ======================================================
GENERATION_CONFIG = {
    "temperature": 0.35,
    "top_k": 20,
    "top_p": 0.85,
    "repetition_penalty_factor": 1.15,
    "ngram_block": 3,
    "max_new_tokens": 200,  # extended for evaluation
    "stop_token_ids": [0, 2, 3, 4, 5],
}


def load_model_and_tokenizer():
    """Load model from checkpoint and tokenizer from file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FromScratchLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[eval] model loaded — step {checkpoint.get('optimizer_update', checkpoint.get('step', 'N/A'))}, "
          f"val_loss {checkpoint.get('validation_loss', 'N/A')}")

    return model, tokenizer, config, device


# ======================================================
# Generation — mirrors src/generate.py exactly
# ======================================================

def generate(model, tokenizer, config, device, prompt_text, max_new_tokens=200):
    """
    Generate a response given an already-formatted prompt string.
    Returns (response_text, tokens_generated, latency_ms).
    """
    encoded = tokenizer.encode(prompt_text)
    prompt_length = len(encoded.ids)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    stop_token_ids = set(GENERATION_CONFIG["stop_token_ids"])
    assistant_id = tokenizer.token_to_id("<|assistant|>")

    t0 = time.perf_counter()

    for _ in range(max_new_tokens):
        # keep only latest context window
        input_context = input_ids[:, -config["context_length"]:]

        with torch.no_grad():
            logits = model(input_context)

        logits = logits[:, -1, :].clone()

        # --- Repetition penalty (token-level) ---
        generated_tokens = input_ids[0, prompt_length:].tolist()
        if generated_tokens:
            for token_id in set(generated_tokens):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= 1.15
                else:
                    logits[0, token_id] *= 1.15

        # --- 3-gram repetition blocking ---
        if len(generated_tokens) >= 2:
            curr_prefix = tuple(generated_tokens[-2:])
            banned = set()
            for i in range(len(generated_tokens) - 2):
                if tuple(generated_tokens[i:i+2]) == curr_prefix:
                    banned.add(generated_tokens[i+2])
            for b_tok in banned:
                logits[0, b_tok] = -float("inf")

        # --- Temperature ---
        temperature = GENERATION_CONFIG["temperature"]
        logits = logits / max(temperature, 1e-4)

        # --- Top-k ---
        top_k = GENERATION_CONFIG["top_k"]
        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits.scatter_(1, indices, values)
            logits = filtered_logits

        # --- Top-p (nucleus) ---
        top_p = GENERATION_CONFIG["top_p"]
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float("inf"))

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() in stop_token_ids:
            break

    latency_ms = (time.perf_counter() - t0) * 1000.0

    generated_ids = input_ids[0].tolist()

    # Find assistant marker
    assistant_start = 0
    for i, tid in enumerate(generated_ids):
        if tid == assistant_id:
            assistant_start = i + 1
            break

    assistant_ids = generated_ids[assistant_start:]
    tokens_generated = len(assistant_ids)

    response = tokenizer.decode(assistant_ids, skip_special_tokens=True).strip()

    return response, tokens_generated, latency_ms


def format_single_turn(prompt):
    """Format a single-turn prompt exactly like training format."""
    return "[BOS]\n<|user|>\n" + prompt + "\n<|assistant|>\n"


def format_multi_turn(history):
    """
    Format multi-turn as described in the brief.
    history: list of (role, text) tuples.
    Returns the formatted string for generation at the last assistant turn.
    """
    parts = []
    for i, (role, text) in enumerate(history):
        if role == "user":
            if i == 0:
                parts.append("[BOS]\n<|user|>\n" + text + "\n<|assistant|>\n")
            else:
                parts.append("<|user|>\n" + text + "\n<|assistant|>\n")
        elif role == "assistant":
            parts.append(text + "\n[EOS]\n")
    return "".join(parts)


# ======================================================
# Quality analysis helpers
# ======================================================

def analyse_response(response):
    """Return a list of quality issue notes for a single response."""
    notes = []
    if not response or len(response.strip()) == 0:
        notes.append("EMPTY_RESPONSE")
        return notes

    words = response.split()

    # Premature EOS (very short)
    if len(words) < 3:
        notes.append(f"VERY_SHORT ({len(words)} words)")

    # Simple word-level repetition detection
    if len(words) > 6:
        for window in [3, 4, 5]:
            for i in range(len(words) - 2 * window):
                chunk = " ".join(words[i:i+window])
                rest = " ".join(words[i+window:])
                count = rest.count(chunk)
                if count >= 2:
                    notes.append(f"REPETITION_LOOP ('{chunk[:40]}...' repeated {count+1}x)")
                    break
            else:
                continue
            break

    # Incomplete sentence (ends mid-word or no terminal punctuation)
    stripped = response.rstrip()
    if stripped and stripped[-1] not in ".!?\"')":
        notes.append("INCOMPLETE_SENTENCE")

    # Very long output with no periods — possible degeneration
    if len(words) > 40 and "." not in response:
        notes.append("NO_SENTENCE_BOUNDARIES")

    # Topic drift proxy: check for random unicode / garbage
    non_ascii = sum(1 for c in response if ord(c) > 127 and ord(c) < 8192)
    if non_ascii > len(response) * 0.1 and len(response) > 20:
        notes.append(f"POSSIBLE_GARBAGE ({non_ascii} non-ascii chars)")

    if not notes:
        notes.append("OK")
    return notes


# ======================================================
# Main evaluation loop
# ======================================================

def main():
    print("=" * 70)
    print("FromScratchLM Independent Baseline Evaluation")
    print("=" * 70)

    model, tokenizer, config, device = load_model_and_tokenizer()

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    results = []
    categories = benchmark["categories"]

    # ---------- Single-turn categories ----------
    for cat_key in ["A_general_conversation", "B_educational_explanations",
                     "C_instruction_following", "D_basic_reasoning",
                     "F_text_quality_probes"]:
        cat = categories[cat_key]
        print(f"\n--- {cat['label']} ---")
        for prompt in cat["prompts"]:
            formatted = format_single_turn(prompt)
            response, tok_gen, latency = generate(
                model, tokenizer, config, device,
                formatted, max_new_tokens=GENERATION_CONFIG["max_new_tokens"]
            )
            quality_notes = analyse_response(response)
            print(f"  Q: {prompt[:60]}")
            print(f"  A: {response[:120]}{'...' if len(response)>120 else ''}")
            print(f"     [{tok_gen} tokens, {latency:.0f}ms] notes={quality_notes}")

            results.append({
                "category": cat["label"],
                "prompt": prompt,
                "response": response,
                "tokens_generated": tok_gen,
                "latency_ms": round(latency, 1),
                "generation_config": GENERATION_CONFIG,
                "notes": quality_notes,
            })

    # ---------- Multi-turn ----------
    mt_cat = categories["E_multi_turn"]
    print(f"\n--- {mt_cat['label']} ---")
    for conv in mt_cat["conversations"]:
        print(f"\n  [Conversation {conv['id']}]")
        history = []
        conversation_log = []

        for turn_idx, user_msg in enumerate(conv["turns"]):
            history.append(("user", user_msg))
            formatted = format_multi_turn(history)
            response, tok_gen, latency = generate(
                model, tokenizer, config, device,
                formatted, max_new_tokens=GENERATION_CONFIG["max_new_tokens"]
            )
            quality_notes = analyse_response(response)
            print(f"    User: {user_msg}")
            print(f"    Bot:  {response[:120]}{'...' if len(response)>120 else ''}")
            print(f"          [{tok_gen} tokens, {latency:.0f}ms] notes={quality_notes}")

            conversation_log.append({"role": "user", "text": user_msg})
            conversation_log.append({
                "role": "assistant",
                "text": response,
                "tokens_generated": tok_gen,
                "latency_ms": round(latency, 1),
                "notes": quality_notes,
            })

            history.append(("assistant", response))

        results.append({
            "category": mt_cat["label"],
            "conversation_id": conv["id"],
            "conversation": conversation_log,
            "generation_config": GENERATION_CONFIG,
            "notes": "multi-turn",
        })

    # ---------- Save ----------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"Saved {len(results)} results to {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
