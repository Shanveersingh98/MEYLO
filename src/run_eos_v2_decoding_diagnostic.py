import sys
import json
import argparse
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
from model import FromScratchLM

CKPT_PATH = PROJECT_DIR / "checkpoints" / "eos_v2_pilot" / "best.pt"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
OUTPUT_DIR = PROJECT_DIR / "logs" / "eos_v2_pilot" / "decoding_diagnostic"
OUTPUT_JSONL = OUTPUT_DIR / "decoding_results.jsonl"

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
    required = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[BOS]": 2,
        "[EOS]": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
    }
    resolved = {}
    for tok, eid in required.items():
        aid = tokenizer.token_to_id(tok)
        assert aid == eid, f"Token {tok} mismatch: expected {eid}, got {aid}"
        resolved[tok] = aid
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
            "distinct_3": 1.0,
            "max_consecutive_repeat": 0,
            "has_repetition_loop": False,
            "repeated_4grams": 0
        }

    total_1 = len(token_ids)
    uniq_1 = len(set(token_ids))
    d1 = float(uniq_1 / total_1)

    if total_1 > 1:
        bigrams = [tuple(token_ids[i:i+2]) for i in range(total_1 - 1)]
        d2 = float(len(set(bigrams)) / len(bigrams))
    else:
        d2 = 1.0

    if total_1 > 2:
        trigrams = [tuple(token_ids[i:i+3]) for i in range(total_1 - 2)]
        d3 = float(len(set(trigrams)) / len(trigrams))
    else:
        d3 = 1.0

    rep_4grams = 0
    if total_1 >= 4:
        fourgrams = [tuple(token_ids[i:i+4]) for i in range(total_1 - 3)]
        c4 = Counter(fourgrams)
        rep_4grams = sum(cnt for cnt in c4.values() if cnt > 1)

    max_repeat = 1
    curr_repeat = 1
    for i in range(1, total_1):
        if token_ids[i] == token_ids[i - 1]:
            curr_repeat += 1
            max_repeat = max(max_repeat, curr_repeat)
        else:
            curr_repeat = 1

    has_loop = max_repeat >= 4 or (total_1 >= 20 and d2 < 0.40) or (rep_4grams >= 15)

    return {
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "distinct_3": round(d3, 4),
        "max_consecutive_repeat": int(max_repeat),
        "has_repetition_loop": bool(has_loop),
        "repeated_4grams": int(rep_4grams)
    }

def generate_sample(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    config_type: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    contrastive_alpha: float = 0.6,
    contrastive_k: int = 5,
    contrastive_context_mode: str = "generated_only",
    seed: int = 42
):
    torch.manual_seed(seed)
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

    if config_type in ("greedy_existing", "greedy_pure"):
        rp = repetition_penalty if config_type == "greedy_existing" else 1.0

        for _ in range(max_new_tokens):
            input_context = input_ids[:, -1024:]
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(input_context)
            logits = logits[:, -1, :].float()

            if rp != 1.0 and len(generated_tokens) > 0:
                for tid in set(generated_tokens):
                    if logits[0, tid] > 0:
                        logits[0, tid] /= rp
                    else:
                        logits[0, tid] *= rp

            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = next_token.item()
            if token_id in stop_token_ids:
                stop_reason = "stop_token_encountered"
                break
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    elif config_type == "nucleus_sampling":
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

    elif config_type == "contrastive_search":
        last_hidden = None
        def hook_fn(module, inp, out):
            nonlocal last_hidden
            last_hidden = out

        handle = model.final_norm.register_forward_hook(hook_fn)
        prev_hidden_states = []

        if contrastive_context_mode == "full_context":
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    _ = model(input_ids)
                prompt_h = last_hidden[0].float()
                prev_hidden_states = [prompt_h[i:i+1] for i in range(prompt_h.shape[0])]

        k = contrastive_k
        alpha = contrastive_alpha

        for _ in range(max_new_tokens):
            input_context = input_ids[:, -1023:]
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(input_context)
            logits = logits[:, -1, :].float()

            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k, dim=-1) # (1, k)

            cand_inputs = input_context.repeat(k, 1)
            cand_tokens = topk_indices.view(k, 1)
            batch_cand = torch.cat([cand_inputs, cand_tokens], dim=1)

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    _ = model(batch_cand)
                cand_hiddens = last_hidden[:, -1, :].float() # (k, d_model)

            if len(prev_hidden_states) > 0:
                all_prev_h = torch.cat(prev_hidden_states, dim=0) # (N_prev, d_model)
                norm_cand = F.normalize(cand_hiddens, p=2, dim=-1)
                norm_prev = F.normalize(all_prev_h, p=2, dim=-1)
                sim_matrix = torch.mm(norm_cand, norm_prev.t()) # (k, N_prev)
                max_sims = sim_matrix.max(dim=1).values         # (k,)
            else:
                max_sims = torch.zeros(k, device=device)

            scores = (1.0 - alpha) * topk_probs[0] - alpha * max_sims
            best_cand_idx = torch.argmax(scores).item()
            best_token = topk_indices[0, best_cand_idx].view(1, 1)

            chosen_hidden = cand_hiddens[best_cand_idx:best_cand_idx+1].clone()
            prev_hidden_states.append(chosen_hidden)

            token_id = best_token.item()
            if token_id in stop_token_ids:
                stop_reason = "stop_token_encountered"
                break

            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, best_token], dim=1)

        handle.remove()

    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    metrics = compute_repetition_metrics(generated_tokens)
    metrics.update({
        "generated_token_count": len(generated_tokens),
        "stop_reason": stop_reason
    })
    return decoded_text, generated_tokens, metrics

def run_all_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("EXPERIMENT 1: EOS V2 DECODING DIAGNOSTIC")
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT_PATH}")
    print("=" * 80)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    validate_special_tokens(tokenizer)
    model, checkpoint = load_checkpoint_model(CKPT_PATH, device)

    print(f"Optimizer update: {checkpoint.get('optimizer_update')}")
    print(f"Validation loss:  {checkpoint.get('validation_loss'):.4f}")

    configs = [
        {
            "id": "greedy_pure",
            "name": "Pure Greedy (Rep Penalty 1.0)",
            "type": "greedy_pure",
            "kwargs": {"config_type": "greedy_pure", "repetition_penalty": 1.0}
        },
        {
            "id": "greedy_existing",
            "name": "Existing Greedy (Rep Penalty 1.15)",
            "type": "greedy_existing",
            "kwargs": {"config_type": "greedy_existing", "repetition_penalty": 1.15}
        },
        {
            "id": "contrastive_search",
            "name": "Contrastive Search (alpha=0.6, k=5, generated prefix)",
            "type": "contrastive_search",
            "kwargs": {"config_type": "contrastive_search", "contrastive_alpha": 0.6, "contrastive_k": 5, "contrastive_context_mode": "generated_only"}
        },
        {
            "id": "contrastive_full_context",
            "name": "Contrastive Search (alpha=0.6, k=5, full context with prompt)",
            "type": "contrastive_search",
            "kwargs": {"config_type": "contrastive_search", "contrastive_alpha": 0.6, "contrastive_k": 5, "contrastive_context_mode": "full_context"}
        },
        {
            "id": "nucleus_temp_0.6",
            "name": "Nucleus Sampling (T=0.6, top-p=0.9, top-k=50, rep-pen=1.15)",
            "type": "nucleus_sampling",
            "kwargs": {"config_type": "nucleus_sampling", "temperature": 0.6, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.15}
        },
        {
            "id": "nucleus_temp_0.8",
            "name": "Nucleus Sampling (T=0.8, top-p=0.9, top-k=50, rep-pen=1.15)",
            "type": "nucleus_sampling",
            "kwargs": {"config_type": "nucleus_sampling", "temperature": 0.8, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.15}
        },
        {
            "id": "nucleus_temp_1.0",
            "name": "Nucleus Sampling (T=1.0, top-p=0.9, top-k=50, rep-pen=1.15)",
            "type": "nucleus_sampling",
            "kwargs": {"config_type": "nucleus_sampling", "temperature": 1.0, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.15}
        }
    ]

    all_results = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        print(f"\n" + "-" * 70)
        print(f"RUNNING CONFIGURATION: {cfg_name}")
        print("-" * 70)

        prompt_results = []
        for p_idx, prompt in enumerate(TEST_PROMPTS, 1):
            text, tokens, metrics = generate_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=100,
                seed=42,
                **cfg["kwargs"]
            )

            prompt_results.append({
                "prompt_index": p_idx,
                "prompt": prompt,
                "text": text,
                "token_ids": tokens,
                "metrics": metrics
            })

            print(f"[{p_idx}/8] '{prompt}'")
            print(f"     Tokens: {metrics['generated_token_count']:3d} | Stop: {metrics['stop_reason']} | Loop: {metrics['has_repetition_loop']} | Rep4g: {metrics['repeated_4grams']} | d2: {metrics['distinct_2']:.4f}")
            print(f"     Text: {text[:100]!r}...")

        # Summary statistics across 8 prompts
        avg_tokens = round(sum(p["metrics"]["generated_token_count"] for p in prompt_results) / len(prompt_results), 1)
        avg_d1 = round(sum(p["metrics"]["distinct_1"] for p in prompt_results) / len(prompt_results), 4)
        avg_d2 = round(sum(p["metrics"]["distinct_2"] for p in prompt_results) / len(prompt_results), 4)
        avg_rep4g = round(sum(p["metrics"]["repeated_4grams"] for p in prompt_results) / len(prompt_results), 2)
        loop_count = sum(1 for p in prompt_results if p["metrics"]["has_repetition_loop"])
        eos_count = sum(1 for p in prompt_results if p["metrics"]["stop_reason"] == "stop_token_encountered")
        max_limit_count = sum(1 for p in prompt_results if p["metrics"]["stop_reason"] == "max_new_tokens_reached")

        summary = {
            "avg_tokens": avg_tokens,
            "avg_distinct_1": avg_d1,
            "avg_distinct_2": avg_d2,
            "avg_repeated_4grams": avg_rep4g,
            "loops_detected": loop_count,
            "stopped_on_eos_or_special": eos_count,
            "hit_max_limit": max_limit_count,
            "eos_rate": round(eos_count / len(prompt_results), 4)
        }

        print(f"\n--> SUMMARY FOR {cfg_id}:")
        print(f"    Avg Tokens: {avg_tokens} | Avg d2: {avg_d2} | Avg Rep 4-grams: {avg_rep4g}")
        print(f"    Loops: {loop_count}/8 | Stopped by EOS: {eos_count}/8 ({summary['eos_rate']*100:.1f}%) | Max tokens: {max_limit_count}/8")

        all_results.append({
            "config_id": cfg_id,
            "config_name": cfg_name,
            "config_type": cfg["type"],
            "parameters": cfg["kwargs"],
            "summary_metrics": summary,
            "prompt_results": prompt_results
        })

    # Write out JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print(f"All diagnostic runs finished successfully! Results saved to: {OUTPUT_JSONL}")
    print("=" * 80)

if __name__ == "__main__":
    run_all_experiments()
