import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import json
from collections import Counter
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "clean_50k_best.pt"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
OUTPUT_FILE = PROJECT_DIR / "logs" / "clean_50k_contrastive_search.jsonl"

TEST_PROMPTS = [
    "Hello",
    "What is artificial intelligence?",
    "What is machine learning?",
    "Explain Python in simple words.",
    "What is 2 + 2?",
    "Explain what a function is in Python.",
    "What is the difference between AI and machine learning?",
    "Explain monopsony in simple words."
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_model_and_tokenizer():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
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
    return model, tokenizer, config

def compute_metrics(token_ids, tokenizer, max_new_tokens=100, ended_with_eos=False):
    token_count = len(token_ids)
    unique_tokens = len(set(token_ids))
    unique_ratio = round(unique_tokens / token_count, 4) if token_count > 0 else 0.0

    # Repeated n-grams: sum of counts for n-grams appearing > 1
    def count_repeated_ngrams(tokens, n):
        if len(tokens) < n:
            return 0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        c = Counter(ngrams)
        return sum(cnt for cnt in c.values() if cnt > 1)

    repeated_2grams = count_repeated_ngrams(token_ids, 2)
    repeated_3grams = count_repeated_ngrams(token_ids, 3)
    repeated_4grams = count_repeated_ngrams(token_ids, 4)

    # Maximum consecutive repetition
    max_consecutive = 0
    if token_count > 0:
        max_consecutive = 1
        cur_consecutive = 1
        for i in range(1, token_count):
            if token_ids[i] == token_ids[i-1]:
                cur_consecutive += 1
                if cur_consecutive > max_consecutive:
                    max_consecutive = cur_consecutive
            else:
                cur_consecutive = 1

    # Most repeated token
    most_repeated_token = "None"
    most_repeated_count = 0
    if token_count > 0:
        c = Counter(token_ids)
        most_common_id, most_repeated_count = c.most_common(1)[0]
        tok_str = tokenizer.id_to_token(most_common_id)
        if tok_str:
            tok_str = tok_str.replace("Ġ", " ")
        most_repeated_token = f"ID {most_common_id} ('{tok_str}')"

    hit_max_limit = (token_count >= max_new_tokens) and (not ended_with_eos)

    return {
        "token_count": token_count,
        "unique_tokens": unique_tokens,
        "unique_ratio": unique_ratio,
        "most_repeated_token": most_repeated_token,
        "most_repeated_count": most_repeated_count,
        "max_consecutive_repetition": max_consecutive,
        "repeated_2grams": repeated_2grams,
        "repeated_3grams": repeated_3grams,
        "repeated_4grams": repeated_4grams,
        "ended_with_eos": ended_with_eos,
        "hit_max_limit": hit_max_limit
    }

def generate_prompt(model, tokenizer, prompt, mode, max_new_tokens=100, seed=42):
    torch.manual_seed(seed)
    formatted = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(formatted)
    prompt_length = len(encoded.ids)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("[EOS]")

    ended_with_eos = False

    if mode == "A_Greedy":
        # temperature = 1.0, top_k = none, top_p = none, rep_penalty = 1.0
        for _ in range(max_new_tokens):
            input_context = input_ids[:, -1024:]
            with torch.no_grad():
                logits = model(input_context)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_id:
                ended_with_eos = True
                break

    elif mode == "B_Sampling":
        # temperature = 0.8, top_k = 50, top_p = none, rep_penalty = 1.15
        for _ in range(max_new_tokens):
            input_context = input_ids[:, -1024:]
            with torch.no_grad():
                logits = model(input_context)[:, -1, :]

            # Repetition penalty applied strictly to generated tokens
            generated_tokens = set(input_ids[0, prompt_length:].tolist())
            for token_id in generated_tokens:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= 1.15
                else:
                    logits[0, token_id] *= 1.15

            logits = logits / 0.8
            values, indices = torch.topk(logits, min(50, logits.size(-1)))
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits.scatter_(1, indices, values)

            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_id:
                ended_with_eos = True
                break

    elif mode == "C_TopP":
        # temperature = 0.8, top_k = none, top_p = 0.9, rep_penalty = 1.0
        for _ in range(max_new_tokens):
            input_context = input_ids[:, -1024:]
            with torch.no_grad():
                logits = model(input_context)[:, -1, :]

            logits = logits / 0.8
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > 0.9
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            filtered_logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == eos_id:
                ended_with_eos = True
                break

    elif mode == "D_Contrastive":
        # top_k = 5, alpha = 0.6, no temperature sampling
        last_hidden = None
        def hook_fn(module, inp, out):
            nonlocal last_hidden
            last_hidden = out

        handle = model.final_norm.register_forward_hook(hook_fn)
        prev_hidden_states = []

        for _ in range(max_new_tokens):
            input_context = input_ids[:, -1023:]
            with torch.no_grad():
                logits = model(input_context)[:, -1, :]

            # Model probability distribution (no temperature)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 5, dim=-1) # (1, 5)

            # Evaluate each candidate's actual hidden representation
            cand_inputs = input_context.repeat(5, 1)
            cand_tokens = topk_indices.view(5, 1)
            batch_cand = torch.cat([cand_inputs, cand_tokens], dim=1)

            with torch.no_grad():
                _ = model(batch_cand)
                cand_hiddens = last_hidden[:, -1, :] # (5, d_model)

            if len(prev_hidden_states) > 0:
                all_prev_h = torch.cat(prev_hidden_states, dim=0) # (N_prev, d_model)
                norm_cand = F.normalize(cand_hiddens, p=2, dim=-1)
                norm_prev = F.normalize(all_prev_h, p=2, dim=-1)
                sim_matrix = torch.mm(norm_cand, norm_prev.t()) # (5, N_prev)
                max_sims = sim_matrix.max(dim=1).values         # (5,)
            else:
                max_sims = torch.zeros(5, device=device)

            scores = (1.0 - 0.6) * topk_probs[0] - 0.6 * max_sims
            best_cand_idx = torch.argmax(scores).item()
            best_token = topk_indices[0, best_cand_idx].view(1, 1)

            # Append chosen candidate's hidden state
            chosen_hidden = cand_hiddens[best_cand_idx:best_cand_idx+1].clone()
            prev_hidden_states.append(chosen_hidden)

            input_ids = torch.cat([input_ids, best_token], dim=1)
            if best_token.item() == eos_id:
                ended_with_eos = True
                break

        handle.remove()

    generated_ids = input_ids[0, prompt_length:].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    stats = compute_metrics(generated_ids, tokenizer, max_new_tokens, ended_with_eos)
    return text, generated_ids, stats

def run_experiment():
    model, tokenizer, config = load_model_and_tokenizer()

    configs = [
        {
            "config_name": "A_Greedy",
            "params": {"temperature": 1.0, "top_k": None, "top_p": None, "repetition_penalty": 1.0, "method": "greedy"}
        },
        {
            "config_name": "B_Sampling",
            "params": {"temperature": 0.8, "top_k": 50, "top_p": None, "repetition_penalty": 1.15, "method": "sampling"}
        },
        {
            "config_name": "C_TopP",
            "params": {"temperature": 0.8, "top_k": None, "top_p": 0.9, "repetition_penalty": 1.0, "method": "top_p"}
        },
        {
            "config_name": "D_Contrastive",
            "params": {"top_k": 5, "alpha": 0.6, "temperature": None, "method": "contrastive_search"}
        }
    ]

    all_results = []

    print("=" * 80)
    print("STARTING DECODING EXPERIMENT: CONTRASTIVE SEARCH VS BASELINES")
    print("=" * 80)

    for cfg in configs:
        mode = cfg["config_name"]
        print(f"\nRunning Configuration: {mode} ({cfg['params']})")
        prompts_output = []

        for idx, prompt in enumerate(TEST_PROMPTS, 1):
            text, token_ids, stats = generate_prompt(model, tokenizer, prompt, mode, max_new_tokens=100, seed=42)
            print(f"  [{idx}/8] Prompt: '{prompt}'")
            print(f"        Tokens: {stats['token_count']} | Unique Ratio: {stats['unique_ratio']} | Rep 4-grams: {stats['repeated_4grams']} | EOS: {stats['ended_with_eos']}")
            print(f"        Output snippet: {text[:80]!r}...")

            prompts_output.append({
                "prompt": prompt,
                "text": text,
                "token_ids": token_ids,
                "stats": stats
            })

        # Calculate summary metrics across 8 prompts
        avg_unique_ratio = round(sum(p["stats"]["unique_ratio"] for p in prompts_output) / len(prompts_output), 4)
        avg_repeated_4grams = round(sum(p["stats"]["repeated_4grams"] for p in prompts_output) / len(prompts_output), 4)
        eos_rate = round(sum(1 for p in prompts_output if p["stats"]["ended_with_eos"]) / len(prompts_output), 4)
        max_token_rate = round(sum(1 for p in prompts_output if p["stats"]["hit_max_limit"]) / len(prompts_output), 4)

        summary_stats = {
            "avg_unique_ratio": avg_unique_ratio,
            "avg_repeated_4grams": avg_repeated_4grams,
            "eos_rate": eos_rate,
            "max_token_rate": max_token_rate
        }

        print(f"\n---> SUMMARY FOR {mode}:")
        print(f"     Avg Unique Ratio:    {avg_unique_ratio:.4f}")
        print(f"     Avg Repeated 4-grams: {avg_repeated_4grams:.4f}")
        print(f"     EOS Rate:            {eos_rate * 100:.1f}%")
        print(f"     Max-token Rate:      {max_token_rate * 100:.1f}%")

        cfg_result = {
            "config_name": mode,
            "params": cfg["params"],
            "summary_stats": summary_stats,
            "prompts": prompts_output
        }
        all_results.append(cfg_result)

    # Save to clean_50k_contrastive_search.jsonl
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    print(f"\nAll results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiment()
