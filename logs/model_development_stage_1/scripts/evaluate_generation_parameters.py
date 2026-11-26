"""
evaluate_generation_parameters.py
=================================
Controlled comparison of generation decoding parameters on FromScratchLM.
Tests 4 configurations across 5 representative prompts to measure:
- text coherence
- repetition
- response completeness
- topic relevance
- generation length (tokens)
- latency (ms)

Produces: logs/model_development_stage_1/generation_comparison.json
"""

import sys
import os
import json
import time
from typing import List, Dict, Any

PROJECT_ROOT = r"D:\DEEP-LEARNING-PROJECT"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import torch
from tokenizers import Tokenizer
from model import FromScratchLM

CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "ultimate_improvement", "best.pt")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "tokenizer", "tokenizer.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "logs", "model_development_stage_1", "generation_comparison.json")

TEST_PROMPTS = [
    {
        "id": "p1_educational",
        "category": "Educational",
        "prompt": "What is machine learning?"
    },
    {
        "id": "p2_instruction",
        "category": "Instruction Following",
        "prompt": "Explain what a computer program is in exactly three points."
    },
    {
        "id": "p3_reasoning",
        "category": "Basic Reasoning",
        "prompt": "If I have 5 apples and give away 2, how many do I have left?"
    },
    {
        "id": "p4_open_conv",
        "category": "General Conversation",
        "prompt": "Tell me something interesting."
    },
    {
        "id": "p5_constraint",
        "category": "Instruction Following",
        "prompt": "Explain photosynthesis in two sentences."
    }
]

CONFIGURATIONS = {
    "config_1_production": {
        "description": "Production Baseline (calibrated nucleus sampling with repetition penalty & 3-gram block)",
        "temperature": 0.35,
        "top_k": 20,
        "top_p": 0.85,
        "repetition_penalty": 1.15,
        "ngram_block": 3
    },
    "config_2_creative": {
        "description": "Higher Diversity / Exploration (higher temp & top_k)",
        "temperature": 0.70,
        "top_k": 40,
        "top_p": 0.90,
        "repetition_penalty": 1.15,
        "ngram_block": 3
    },
    "config_3_greedy_low_temp": {
        "description": "Low Temperature / Near-Greedy (minimal sampling noise)",
        "temperature": 0.10,
        "top_k": 5,
        "top_p": 0.70,
        "repetition_penalty": 1.15,
        "ngram_block": 3
    },
    "config_4_no_ngram_block": {
        "description": "No Repetition Blocking (raw nucleus sampling without 3-gram block or penalty)",
        "temperature": 0.35,
        "top_k": 20,
        "top_p": 0.85,
        "repetition_penalty": 1.00,
        "ngram_block": 0
    }
}

def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen_eval] Loading model on {device}...")
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
    return model, tokenizer, cfg, device

def generate_with_config(model, tokenizer, config_model, device, prompt: str, gen_cfg: dict, max_new_tokens=150):
    text = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(text)
    prompt_length = len(encoded.ids)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    stop_token_ids = {0, 2, 3, 4, 5}
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    
    temp = gen_cfg["temperature"]
    top_k = gen_cfg["top_k"]
    top_p = gen_cfg["top_p"]
    rep_pen = gen_cfg["repetition_penalty"]
    ngram_block = gen_cfg["ngram_block"]
    
    t0 = time.perf_counter()
    
    for _ in range(max_new_tokens):
        context = input_ids[:, -config_model["context_length"]:]
        with torch.no_grad():
            logits = model(context)
            
        logits = logits[:, -1, :].clone()
        
        # Token repetition penalty
        gen_tokens = input_ids[0, prompt_length:].tolist()
        if gen_tokens and rep_pen != 1.0:
            for tid in set(gen_tokens):
                if logits[0, tid] > 0:
                    logits[0, tid] /= rep_pen
                else:
                    logits[0, tid] *= rep_pen
                    
        # N-gram repetition blocking
        if ngram_block > 0 and len(gen_tokens) >= ngram_block - 1:
            prefix_len = ngram_block - 1
            curr_prefix = tuple(gen_tokens[-prefix_len:])
            banned = set()
            for i in range(len(gen_tokens) - prefix_len):
                if tuple(gen_tokens[i:i+prefix_len]) == curr_prefix:
                    banned.add(gen_tokens[i+prefix_len])
            for b_tok in banned:
                logits[0, b_tok] = -float("inf")
                
        # Temperature
        logits = logits / max(temp, 1e-4)
        
        # Top-k
        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered = torch.full_like(logits, -float("inf"))
            filtered.scatter_(1, indices, values)
            logits = filtered
            
        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float("inf"))
            
        probs = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        
        if next_tok.item() in stop_token_ids:
            break
            
    dur_ms = (time.perf_counter() - t0) * 1000.0
    generated_ids = input_ids[0].tolist()
    
    asst_start = 0
    for i, tid in enumerate(generated_ids):
        if tid == assistant_id:
            asst_start = i + 1
            break
    resp_ids = generated_ids[asst_start:]
    resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
    
    # Check for repetition loops
    words = resp_text.split()
    repetition_detected = False
    if len(words) > 6:
        for w_size in [3, 4]:
            for i in range(len(words) - 2 * w_size):
                chunk = " ".join(words[i:i+w_size])
                rest = " ".join(words[i+w_size:])
                if rest.count(chunk) >= 2:
                    repetition_detected = True
                    break
    
    return {
        "response": resp_text,
        "token_count": len(resp_ids),
        "latency_ms": round(dur_ms, 1),
        "repetition_detected": repetition_detected,
        "terminal_punctuation": bool(resp_text and resp_text[-1] in ".!?\"')")
    }

def main():
    model, tokenizer, config_model, device = load_system()
    
    comparison_results = {
        "metadata": {
            "model": "FromScratchLM (14.1M)",
            "checkpoint": "checkpoints/ultimate_improvement/best.pt",
            "prompt_count": len(TEST_PROMPTS),
            "config_count": len(CONFIGURATIONS)
        },
        "configurations": CONFIGURATIONS,
        "results": []
    }
    
    for prompt_info in TEST_PROMPTS:
        pid = prompt_info["id"]
        prompt = prompt_info["prompt"]
        cat = prompt_info["category"]
        print(f"\n[Prompt: {prompt}]")
        
        prompt_record = {
            "prompt_id": pid,
            "category": cat,
            "prompt": prompt,
            "outputs": {}
        }
        
        for cfg_id, cfg_params in CONFIGURATIONS.items():
            out = generate_with_config(model, tokenizer, config_model, device, prompt, cfg_params)
            prompt_record["outputs"][cfg_id] = out
            print(f"  {cfg_id}: ({out['token_count']} tok, {out['latency_ms']}ms) -> {out['response'][:80]}...")
            
        comparison_results["results"].append(prompt_record)
        
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved comparison results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
