import sys
import json
import time
import re
import ast
import math
import hashlib
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
EVAL_SUITE_DIR = PROJECT_DIR / "data" / "ultimate_improvement" / "eval_suite"

CONTEXT_LENGTH = 1024
VOCAB_SIZE = 8000
STOP_TOKEN_IDS = {0, 2, 3, 4, 5}  # PAD, BOS, EOS, USER, ASSISTANT


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FromScratchLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg.get("dropout", 0.0)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 0,
    do_sample: bool = False,
    custom_formatted_prefix: str = None
):
    if custom_formatted_prefix is not None:
        formatted = custom_formatted_prefix
    else:
        # Sanitize prompt of special tokens to prevent injection
        clean_p = prompt.replace("<|user|>", "").replace("<|assistant|>", "").replace("[EOS]", "").replace("[BOS]", "")
        formatted = f"[BOS]\n<|user|>\n{clean_p.strip()}\n<|assistant|>\n"

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

        # Token-level repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for tid in set(generated_tokens):
                if logits[0, tid] > 0:
                    logits[0, tid] /= repetition_penalty
                else:
                    logits[0, tid] *= repetition_penalty

        # N-gram repetition blocking if enabled
        if no_repeat_ngram_size > 0 and len(generated_tokens) >= no_repeat_ngram_size - 1:
            gen_list = generated_tokens
            ngram_len = no_repeat_ngram_size
            curr_prefix = tuple(gen_list[-(ngram_len - 1):])
            banned_next_tokens = set()
            for i in range(len(gen_list) - ngram_len + 1):
                prev_ngram = tuple(gen_list[i : i + ngram_len - 1])
                if prev_ngram == curr_prefix:
                    banned_next_tokens.add(gen_list[i + ngram_len - 1])
            for b_tok in banned_next_tokens:
                logits[0, b_tok] = -float("inf")

        if not do_sample:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-4)
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
        if token_id in STOP_TOKEN_IDS:
            stop_reason = "stop_token_encountered"
            break

        generated_tokens.append(token_id)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    decoded = tokenizer.decode(generated_tokens)

    # Calculate diversity statistics
    uniq1 = len(set(generated_tokens))
    d1 = (uniq1 / len(generated_tokens)) if generated_tokens else 1.0
    bigrams = [tuple(generated_tokens[i:i+2]) for i in range(len(generated_tokens)-1)]
    d2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 1.0
    trigrams = [tuple(generated_tokens[i:i+3]) for i in range(len(generated_tokens)-2)]
    d3 = (len(set(trigrams)) / len(trigrams)) if trigrams else 1.0

    # Degeneracy loop detector: 4+ identical consecutive tokens or repeated 4-grams >= 12
    consec_rep = False
    for i in range(len(generated_tokens) - 3):
        if generated_tokens[i] == generated_tokens[i+1] == generated_tokens[i+2] == generated_tokens[i+3]:
            consec_rep = True
            break
    fourgrams = [tuple(generated_tokens[i:i+4]) for i in range(len(generated_tokens)-3)]
    rep_4g = len(fourgrams) - len(set(fourgrams))
    is_loop = consec_rep or (len(generated_tokens) >= 20 and d2 < 0.45) or rep_4g >= 12

    return {
        "text": decoded.strip(),
        "token_count": len(generated_tokens),
        "stop_reason": stop_reason,
        "is_loop": is_loop,
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "distinct_3": round(d3, 4),
        "token_ids": generated_tokens
    }


def parse_arithmetic_answer(text: str):
    clean_text = text.replace(",", "")
    patterns = [
        r"=\s*(-?\d+)",
        r"(?:is|equals|equal to|leaves|leaves us with|remain|remaining|total of|gives|yields|result is)\s*(-?\d+)",
        r"(-?\d+)\s*(?:apples|books|cookies|marbles|gallons|cartons|items|students|shares|boxes|miles|meters|loaves|passengers|dollars|crates|stickers|cars|pages)",
        r"^\s*(-?\d+)\b",
        r"\b(-?\d+)\b"
    ]
    for pat in patterns:
        m = re.search(pat, clean_text, re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
    return None


def execute_sandboxed_python(code_str: str, func_name: str, test_cases: list):
    try:
        ast.parse(code_str)
    except Exception as e:
        return False, False, f"SyntaxError: {e}"

    sandbox = {
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
            "range": range, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "str": str, "int": int, "float": float, "bool": bool,
            "round": round, "sorted": sorted, "enumerate": enumerate, "zip": zip,
            "True": True, "False": False, "None": None, "Exception": Exception, "ValueError": ValueError
        }
    }

    try:
        exec(code_str, sandbox)
    except Exception as e:
        return True, False, f"RuntimeError during definition: {e}"

    if func_name not in sandbox or not callable(sandbox[func_name]):
        return True, False, f"Function '{func_name}' not defined or not callable"

    fn = sandbox[func_name]
    passed_count = 0
    for idx, tc in enumerate(test_cases):
        args = tc["args"]
        expected = tc["expected"]
        try:
            actual = fn(*args)
            if actual == expected:
                passed_count += 1
            elif isinstance(expected, float) and isinstance(actual, (int, float)) and abs(actual - expected) < 1e-4:
                passed_count += 1
            else:
                return True, False, f"Test {idx+1} failed: {func_name}{tuple(args)} -> {actual} != {expected}"
        except Exception as e:
            return True, False, f"Test {idx+1} raised exception: {e}"

    return True, (passed_count == len(test_cases)), f"Passed {passed_count}/{len(test_cases)} tests"


def evaluate_suite_01_benchmarks(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_01_benchmarks.json", "r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    results = []
    for b in benchmarks:
        gen = generate(model, tokenizer, b["prompt"], device, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        results.append({
            "id": b["id"],
            "prompt": b["prompt"],
            "category": b["category"],
            "text": gen["text"],
            "stop_reason": gen["stop_reason"],
            "is_loop": gen["is_loop"],
            "distinct_2": gen["distinct_2"]
        })
    eos_count = sum(1 for r in results if r["stop_reason"] == "stop_token_encountered")
    loops_count = sum(1 for r in results if r["is_loop"])
    return {
        "total": len(results),
        "eos_count": eos_count,
        "eos_rate_pct": round((eos_count / len(results)) * 100, 2),
        "loop_count": loops_count,
        "loop_rate_pct": round((loops_count / len(results)) * 100, 2),
        "items": results
    }


def evaluate_suite_02_arithmetic(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_02_arithmetic_single_step.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    correct_by_tier = {}
    total_by_tier = {}
    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=60, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        ans = parse_arithmetic_answer(gen["text"])
        is_correct = (ans == it["expected_answer"])
        tier = it["tier"]
        total_by_tier[tier] = total_by_tier.get(tier, 0) + 1
        if is_correct:
            correct_by_tier[tier] = correct_by_tier.get(tier, 0) + 1
        results.append({
            "id": it["id"],
            "prompt": it["prompt"],
            "expected": it["expected_answer"],
            "predicted": ans,
            "is_correct": is_correct,
            "tier": tier,
            "text": gen["text"],
            "is_loop": gen["is_loop"],
            "stop_reason": gen["stop_reason"]
        })
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": round((correct / total) * 100, 2),
        "by_tier": {t: {"correct": correct_by_tier.get(t, 0), "total": total_by_tier[t], "acc_pct": round((correct_by_tier.get(t, 0)/total_by_tier[t])*100, 2)} for t in total_by_tier},
        "items": results
    }


def evaluate_suite_03_multistep_arithmetic(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_03_arithmetic_multi_step.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=80, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        ans = parse_arithmetic_answer(gen["text"])
        is_correct = (ans == it["expected_answer"])
        results.append({
            "id": it["id"],
            "prompt": it["prompt"],
            "expected": it["expected_answer"],
            "predicted": ans,
            "is_correct": is_correct,
            "steps": it["steps"],
            "text": gen["text"],
            "is_loop": gen["is_loop"]
        })
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    return {
        "total": total,
        "correct": correct,
        "accuracy_pct": round((correct / total) * 100, 2),
        "items": results
    }


def evaluate_suite_04_python(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_04_python_tasks.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    code_block_re = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
    results = []
    syntax_valid_count = 0
    func_correct_count = 0
    exec_total = 0

    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=150, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        text = gen["text"]
        rec = {
            "id": it["id"],
            "type": it["type"],
            "prompt": it["prompt"],
            "text": text,
            "is_loop": gen["is_loop"],
            "stop_reason": gen["stop_reason"]
        }
        if it["type"] == "executable_code":
            exec_total += 1
            blocks = code_block_re.findall(text)
            code_to_test = blocks[0] if blocks else text
            s_val, f_corr, note = execute_sandboxed_python(code_to_test, it["func_name"], it["tests"])
            if s_val:
                syntax_valid_count += 1
            if f_corr:
                func_correct_count += 1
            rec["syntax_valid"] = s_val
            rec["func_correct"] = f_corr
            rec["note"] = note
        elif "expected_key_points" in it:
            kps = it["expected_key_points"]
            matched = [kp for kp in kps if kp.lower() in text.lower()]
            rec["matched_kps"] = matched
            rec["total_kps"] = len(kps)
            rec["score"] = round(len(matched) / len(kps), 2)
        results.append(rec)

    return {
        "exec_total": exec_total,
        "exec_syntax_valid": syntax_valid_count,
        "exec_syntax_valid_pct": round((syntax_valid_count / exec_total) * 100, 2) if exec_total else 0,
        "exec_func_correct": func_correct_count,
        "exec_func_correct_pct": round((func_correct_count / exec_total) * 100, 2) if exec_total else 0,
        "items": results
    }


def evaluate_suite_05_concepts(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_05_concepts.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    total_kps = 0
    matched_kps = 0
    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=120, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        text = gen["text"]
        kps = it["ground_truth_key_points"]
        matched = [kp for kp in kps if any(w in text.lower() for w in kp.lower().split())]
        matched_kps += len(matched)
        total_kps += len(kps)
        results.append({
            "id": it["id"],
            "concept": it["concept"],
            "prompt": it["prompt"],
            "text": text,
            "matched_kps": matched,
            "total_kps": len(kps),
            "match_ratio": round(len(matched) / len(kps), 2),
            "is_loop": gen["is_loop"]
        })
    recall = round((matched_kps / total_kps) * 100, 2) if total_kps else 0
    return {
        "total_concepts": len(results),
        "total_key_points": total_kps,
        "matched_key_points": matched_kps,
        "key_point_recall_pct": recall,
        "items": results
    }


def evaluate_suite_06_factual(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_06_factual_qa.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    correct_count = 0
    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=50, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        text = gen["text"]
        kps = it["expected_key_points"]
        is_hit = any(kp.lower() in text.lower() for kp in kps)
        if is_hit:
            correct_count += 1
        results.append({
            "id": it["id"],
            "prompt": it["prompt"],
            "expected_key_points": kps,
            "text": text,
            "is_correct": is_hit,
            "is_loop": gen["is_loop"]
        })
    acc = round((correct_count / len(results)) * 100, 2)
    return {
        "total": len(results),
        "correct": correct_count,
        "accuracy_pct": acc,
        "items": results
    }


def evaluate_suite_07_instructions(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_07_instruction_following.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    passed_count = 0
    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=60, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        text = gen["text"].strip()
        etype = it["eval_type"]
        passed = False
        reason = ""

        if etype == "word_count":
            words = text.split()
            passed = len(words) == it["expected_count"]
            reason = f"Words: {len(words)} (Expected {it['expected_count']})"
        elif etype == "exact_choice":
            clean = text.lower().strip()
            passed = any(clean == vc or clean.startswith(vc) for vc in it["valid_choices"])
            reason = f"Choice: '{clean}'"
        elif etype == "numbered_list":
            nums = re.findall(r"^\s*([1-9])[\.\)]\s+", text, re.MULTILINE)
            passed = len(nums) == it["expected_items"]
            reason = f"Found {len(nums)} numbered items"
        elif etype == "negative_constraint_char":
            f_char = it["forbidden_char"].lower()
            passed = f_char not in text.lower() and len(text) > 2
            reason = f"Forbidden char '{f_char}' in text: {f_char in text.lower()}"
        elif etype == "sentence_count":
            sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
            passed = len(sents) <= it["max_sentences"]
            reason = f"Found {len(sents)} sentences"
        elif etype == "all_caps":
            passed = text.isupper() and (it["expected_keyword"] in text)
            reason = f"IsUpper: {text.isupper()}"
        elif etype == "only_number":
            nums = re.findall(r"\b(-?\d+)\b", text)
            passed = (len(nums) == 1) and (int(nums[0]) == it["expected_number"])
            reason = f"Numbers found: {nums}"
        elif etype == "bullet_points":
            bullets = [l for l in text.splitlines() if l.strip().startswith("-") or l.strip().startswith("*")]
            passed = len(bullets) >= it["expected_bullets"]
            reason = f"Bullets found: {len(bullets)}"
        elif etype == "exact_phrase":
            passed = it["expected_phrase"].lower() in text.lower()
            reason = f"Phrase match: {passed}"
        elif etype == "forbidden_word":
            passed = it["forbidden_word"].lower() not in text.lower() and len(text) > 10
            reason = f"Forbidden word present: {it['forbidden_word'].lower() in text.lower()}"
        elif etype == "comma_separated":
            parts = [p.strip() for p in text.split(",") if p.strip()]
            passed = len(parts) >= it["min_items"]
            reason = f"Comma items: {len(parts)}"

        if passed:
            passed_count += 1
        results.append({
            "id": it["id"],
            "prompt": it["prompt"],
            "text": text,
            "passed": passed,
            "reason": reason,
            "is_loop": gen["is_loop"]
        })

    acc = round((passed_count / len(results)) * 100, 2)
    return {
        "total": len(results),
        "passed": passed_count,
        "accuracy_pct": acc,
        "items": results
    }


def evaluate_suite_08_structured(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_08_structured_answers.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    valid_count = 0

    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=100, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        text = gen["text"]
        fmt = it["format"]
        is_valid = False
        note = ""

        if fmt == "json":
            # Extract json snippet
            json_m = re.search(r"\{.*?\}", text, re.DOTALL)
            if json_m:
                try:
                    obj = json.loads(json_m.group(0))
                    if isinstance(obj, dict):
                        req = it.get("required_keys", [])
                        is_valid = all(k in obj for k in req)
                        note = f"JSON parsed, keys present: {all(k in obj for k in req)}"
                except Exception as e:
                    note = f"JSON parse error: {e}"
        elif fmt == "json_array":
            json_m = re.search(r"\[.*?\]", text, re.DOTALL)
            if json_m:
                try:
                    arr = json.loads(json_m.group(0))
                    is_valid = isinstance(arr, list) and len(arr) > 0
                    note = f"JSON array parsed: {arr}"
                except Exception as e:
                    note = f"JSON array error: {e}"
        elif fmt == "markdown_table":
            lines = [l.strip() for l in text.splitlines() if l.strip().startswith("|") and l.strip().endswith("|")]
            has_pipe_rows = len(lines) >= 2
            req_h = it.get("required_headers", [])
            headers_ok = any(all(h.lower() in l.lower() for h in req_h) for l in lines) if req_h else True
            is_valid = has_pipe_rows and headers_ok
            note = f"Table rows: {len(lines)}, headers present: {headers_ok}"
        elif fmt in ["numbered_list", "bullet_list"]:
            items_found = [l for l in text.splitlines() if re.match(r"^(\d+[\.\)]|[-*])\s+", l.strip())]
            is_valid = len(items_found) >= it.get("expected_count", 3)
            note = f"List items found: {len(items_found)}"

        if is_valid:
            valid_count += 1
        results.append({
            "id": it["id"],
            "prompt": it["prompt"],
            "text": text,
            "is_valid": is_valid,
            "note": note,
            "is_loop": gen["is_loop"]
        })

    acc = round((valid_count / len(results)) * 100, 2)
    return {
        "total": len(results),
        "valid": valid_count,
        "valid_pct": acc,
        "items": results
    }


def evaluate_suite_09_multiturn(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_09_multiturn.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    consistent_count = 0

    for it in items:
        turn1_user = it["turns"][0]["user"]
        gen1 = generate(model, tokenizer, turn1_user, device, max_new_tokens=60, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        turn1_asst = gen1["text"]

        # Construct multi-turn context
        turn2_user = it["turns"][1]["user"]
        prefix = f"[BOS]\n<|user|>\n{turn1_user}\n<|assistant|>\n{turn1_asst}\n[EOS]\n<|user|>\n{turn2_user}\n<|assistant|>\n"
        gen2 = generate(model, tokenizer, turn2_user, device, max_new_tokens=60, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram, custom_formatted_prefix=prefix)
        turn2_asst = gen2["text"]

        expected_kws = it["turns"][1]["expected_keywords"]
        hit = any(kw.lower() in turn2_asst.lower() for kw in expected_kws)
        if hit:
            consistent_count += 1

        results.append({
            "id": it["id"],
            "turn1_user": turn1_user,
            "turn1_asst": turn1_asst,
            "turn2_user": turn2_user,
            "turn2_asst": turn2_asst,
            "expected_keywords": expected_kws,
            "consistent": hit,
            "is_loop": gen2["is_loop"]
        })

    acc = round((consistent_count / len(results)) * 100, 2)
    return {
        "total": len(results),
        "consistent": consistent_count,
        "consistency_pct": acc,
        "items": results
    }


def evaluate_suite_10_refusals(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_10_refusal_uncertainty.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    appropriate_count = 0

    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=60, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        text = gen["text"]
        kws = it["refusal_keywords"]
        hit = any(kw.lower() in text.lower() for kw in kws)
        if hit:
            appropriate_count += 1
        results.append({
            "id": it["id"],
            "prompt": it["prompt"],
            "type": it["type"],
            "text": text,
            "refused_or_uncertain": hit,
            "refusal_keywords": kws,
            "is_loop": gen["is_loop"]
        })

    acc = round((appropriate_count / len(results)) * 100, 2)
    return {
        "total": len(results),
        "appropriate_refusals": appropriate_count,
        "refusal_accuracy_pct": acc,
        "items": results
    }


def evaluate_suite_11_robustness(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_11_robustness_malformed.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    handled_count = 0

    for it in items:
        prompt = it["prompt"]
        try:
            gen = generate(model, tokenizer, prompt, device, max_new_tokens=40, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
            handled = True
            out_text = gen["text"]
            stop = gen["stop_reason"]
        except Exception as e:
            handled = False
            out_text = f"CRASH: {e}"
            stop = "exception"

        if handled:
            handled_count += 1
        results.append({
            "id": it["id"],
            "description": it["description"],
            "prompt_sample": prompt[:50],
            "handled_without_crash": handled,
            "text": out_text[:100],
            "stop_reason": stop
        })

    acc = round((handled_count / len(results)) * 100, 2)
    return {
        "total": len(results),
        "handled_count": handled_count,
        "handled_pct": acc,
        "items": results
    }


def evaluate_suite_12_context(model, tokenizer, device, do_sample=False, rep_penalty=1.15, no_repeat_ngram=0):
    with open(EVAL_SUITE_DIR / "suite_12_context_boundary.jsonl", "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    results = []
    correct_count = 0

    for it in items:
        gen = generate(model, tokenizer, it["prompt"], device, max_new_tokens=40, do_sample=do_sample, repetition_penalty=rep_penalty, no_repeat_ngram_size=no_repeat_ngram)
        text = gen["text"]
        kw = it["expected_keyword"].lower()
        hit = kw in text.lower()
        if hit:
            correct_count += 1
        results.append({
            "id": it["id"],
            "target_tokens": it["target_tokens"],
            "expected_keyword": kw,
            "text": text,
            "hit": hit,
            "stop_reason": gen["stop_reason"]
        })

    acc = round((correct_count / len(results)) * 100, 2)
    return {
        "total": len(results),
        "correct": correct_count,
        "accuracy_pct": acc,
        "items": results
    }


def run_complete_evaluation(
    checkpoint_path: Path,
    out_json_path: Path,
    device: torch.device,
    name: str = "Reference Checkpoint",
    do_sample: bool = False,
    seed: int = 42,
    rep_penalty: float = 1.15,
    no_repeat_ngram: int = 0
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"RUNNING COMPREHENSIVE EVALUATION ON: {name}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    tokenizer_hash = get_file_sha256(TOKENIZER_PATH)
    ckpt_hash = get_file_sha256(checkpoint_path)

    print(f"Verified Checkpoint SHA-256: {ckpt_hash}")
    print(f"Verified Tokenizer SHA-256:  {tokenizer_hash}")

    model, ckpt = load_model(checkpoint_path, device)

    t0 = time.time()
    print("\n[1/12] Evaluating Historical Benchmarks...")
    s1 = evaluate_suite_01_benchmarks(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[2/12] Evaluating Single-Step Arithmetic (100 items)...")
    s2 = evaluate_suite_02_arithmetic(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[3/12] Evaluating Multi-Step Arithmetic (20 items)...")
    s3 = evaluate_suite_03_multistep_arithmetic(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[4/12] Evaluating Python Programming (30 items)...")
    s4 = evaluate_suite_04_python(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[5/12] Evaluating Concepts & Key Point Recall (30 items)...")
    s5 = evaluate_suite_05_concepts(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[6/12] Evaluating Basic Factual QA (20 items)...")
    s6 = evaluate_suite_06_factual(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[7/12] Evaluating Instruction Following (15 items)...")
    s7 = evaluate_suite_07_instructions(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[8/12] Evaluating Structured Formatting (10 items)...")
    s8 = evaluate_suite_08_structured(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[9/12] Evaluating Multi-Turn Consistency (10 dialogues)...")
    s9 = evaluate_suite_09_multiturn(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[10/12] Evaluating Refusal / Uncertainty (10 items)...")
    s10 = evaluate_suite_10_refusals(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[11/12] Evaluating Robustness & Malformed Inputs (10 items)...")
    s11 = evaluate_suite_11_robustness(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    print("[12/12] Evaluating Context Boundary Adherence (5 items)...")
    s12 = evaluate_suite_12_context(model, tokenizer, device, do_sample, rep_penalty, no_repeat_ngram)

    elapsed = time.time() - t0

    # Aggregate overall statistics across all generations
    all_gen_items = (
        s1["items"] + s2["items"] + s3["items"] + s4["items"] + s5["items"] +
        s6["items"] + s7["items"] + s8["items"] + s10["items"] + s12["items"]
    )
    total_gens = len(all_gen_items)
    total_eos = sum(1 for g in all_gen_items if g.get("stop_reason") == "stop_token_encountered")
    total_loops = sum(1 for g in all_gen_items if g.get("is_loop", False))

    summary = {
        "model_name": name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_hash,
        "tokenizer_sha256": tokenizer_hash,
        "evaluation_timestamp": time.time(),
        "elapsed_seconds": round(elapsed, 2),
        "settings": {
            "do_sample": do_sample,
            "seed": seed,
            "repetition_penalty": rep_penalty,
            "no_repeat_ngram_size": no_repeat_ngram,
            "max_new_tokens_default": 120
        },
        "aggregate_behavior": {
            "total_prompts_evaluated": total_gens,
            "total_eos_terminations": total_eos,
            "overall_eos_termination_pct": round((total_eos / total_gens) * 100, 2) if total_gens else 0,
            "total_loops_detected": total_loops,
            "overall_loop_rate_pct": round((total_loops / total_gens) * 100, 2) if total_gens else 0
        },
        "metrics_by_suite": {
            "suite_01_benchmarks": {
                "total": s1["total"],
                "eos_rate_pct": s1["eos_rate_pct"],
                "loop_rate_pct": s1["loop_rate_pct"]
            },
            "suite_02_arithmetic_single_step": {
                "total": s2["total"],
                "correct": s2["correct"],
                "accuracy_pct": s2["accuracy_pct"],
                "by_tier": s2["by_tier"]
            },
            "suite_03_arithmetic_multi_step": {
                "total": s3["total"],
                "correct": s3["correct"],
                "accuracy_pct": s3["accuracy_pct"]
            },
            "suite_04_python": {
                "exec_total": s4["exec_total"],
                "syntax_valid": s4["exec_syntax_valid"],
                "syntax_valid_pct": s4["exec_syntax_valid_pct"],
                "func_correct": s4["exec_func_correct"],
                "func_correct_pct": s4["exec_func_correct_pct"]
            },
            "suite_05_concepts": {
                "total_key_points": s5["total_key_points"],
                "matched_key_points": s5["matched_key_points"],
                "key_point_recall_pct": s5["key_point_recall_pct"]
            },
            "suite_06_factual_qa": {
                "total": s6["total"],
                "correct": s6["correct"],
                "accuracy_pct": s6["accuracy_pct"]
            },
            "suite_07_instruction_following": {
                "total": s7["total"],
                "passed": s7["passed"],
                "accuracy_pct": s7["accuracy_pct"]
            },
            "suite_08_structured_formatting": {
                "total": s8["total"],
                "valid": s8["valid"],
                "valid_pct": s8["valid_pct"]
            },
            "suite_09_multiturn_consistency": {
                "total": s9["total"],
                "consistent": s9["consistent"],
                "consistency_pct": s9["consistency_pct"]
            },
            "suite_10_refusal_uncertainty": {
                "total": s10["total"],
                "appropriate_refusals": s10["appropriate_refusals"],
                "refusal_accuracy_pct": s10["refusal_accuracy_pct"]
            },
            "suite_11_robustness_malformed": {
                "total": s11["total"],
                "handled_pct": s11["handled_pct"]
            },
            "suite_12_context_boundary": {
                "total": s12["total"],
                "correct": s12["correct"],
                "accuracy_pct": s12["accuracy_pct"]
            }
        },
        "raw_details": {
            "suite_01_benchmarks": s1["items"],
            "suite_02_arithmetic_single_step": s2["items"],
            "suite_03_arithmetic_multi_step": s3["items"],
            "suite_04_python": s4["items"],
            "suite_05_concepts": s5["items"],
            "suite_06_factual_qa": s6["items"],
            "suite_07_instruction_following": s7["items"],
            "suite_08_structured_formatting": s8["items"],
            "suite_09_multiturn_consistency": s9["items"],
            "suite_10_refusal_uncertainty": s10["items"],
            "suite_11_robustness_malformed": s11["items"],
            "suite_12_context_boundary": s12["items"]
        }
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nCompleted evaluation in {elapsed:.1f}s. Results saved to: {out_json_path}")
    print("\n--- SUMMARY METRICS ---")
    print(f"Overall EOS Terminations:       {total_eos}/{total_gens} ({summary['aggregate_behavior']['overall_eos_termination_pct']}%)")
    print(f"Overall Repetition Loops:       {total_loops}/{total_gens} ({summary['aggregate_behavior']['overall_loop_rate_pct']}%)")
    print(f"Arithmetic Single-Step:         {s2['correct']}/{s2['total']} ({s2['accuracy_pct']}%)")
    print(f"Arithmetic Multi-Step:          {s3['correct']}/{s3['total']} ({s3['accuracy_pct']}%)")
    print(f"Python Syntax Valid:            {s4['exec_syntax_valid']}/{s4['exec_total']} ({s4['exec_syntax_valid_pct']}%)")
    print(f"Python Functional Correctness:  {s4['exec_func_correct']}/{s4['exec_total']} ({s4['exec_func_correct_pct']}%)")
    print(f"Concept Key-Point Recall:       {s5['matched_key_points']}/{s5['total_key_points']} ({s5['key_point_recall_pct']}%)")
    print(f"Factual QA Accuracy:            {s6['correct']}/{s6['total']} ({s6['accuracy_pct']}%)")
    print(f"Instruction Following:          {s7['passed']}/{s7['total']} ({s7['accuracy_pct']}%)")
    print(f"Structured Formatting:          {s8['valid']}/{s8['total']} ({s8['valid_pct']}%)")
    print(f"Multi-Turn Consistency:         {s9['consistent']}/{s9['total']} ({s9['consistency_pct']}%)")
    print(f"Refusal / Uncertainty:          {s10['appropriate_refusals']}/{s10['total']} ({s10['refusal_accuracy_pct']}%)")
    print(f"Robustness (Malformed inputs):  {s11['handled_count']}/{s11['total']} ({s11['handled_pct']}%)")
    print(f"Context Boundary Adherence:     {s12['correct']}/{s12['total']} ({s12['accuracy_pct']}%)")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/experiment_2/best.pt")
    parser.add_argument("--out_file", type=str, default="logs/ultimate_improvement/baseline_metrics.json")
    parser.add_argument("--name", type=str, default="Experiment 2 Reference Baseline")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rep_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_complete_evaluation(
        checkpoint_path=PROJECT_DIR / args.checkpoint,
        out_json_path=PROJECT_DIR / args.out_file,
        device=device,
        name=args.name,
        do_sample=args.sample,
        seed=args.seed,
        rep_penalty=args.rep_penalty,
        no_repeat_ngram=args.no_repeat_ngram
    )
