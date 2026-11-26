import sys
import json
import ast
import re
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

REF_CHECKPOINT = PROJECT_DIR / "checkpoints" / "eos_v2_pilot" / "best.pt"
EXP2_CHECKPOINT = PROJECT_DIR / "checkpoints" / "experiment_2" / "best.pt"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"

TEST_DIR = PROJECT_DIR / "data" / "experiment_2" / "test_untouched"
BENCHMARK_PROMPTS_FILE = TEST_DIR / "benchmark_8_prompts.json"
UNTOUCHED_ARITH_FILE = TEST_DIR / "untouched_arithmetic_test.jsonl"
UNTOUCHED_PYTHON_FILE = TEST_DIR / "untouched_python_test.jsonl"
UNTOUCHED_CONCEPTS_FILE = TEST_DIR / "untouched_concepts_test.jsonl"

OASST1_VAL_TOKENS = PROJECT_DIR / "data" / "processed" / "training_sequences_eos_v2" / "validation_tokens.bin"
OASST1_VAL_MASK = PROJECT_DIR / "data" / "processed" / "training_sequences_eos_v2" / "validation_mask.bin"

OUTPUT_RESULTS_JSON = PROJECT_DIR / "logs" / "eos_v2_pilot" / "experiment_2" / "evaluation_results.json"

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024


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
        dropout=cfg.get("dropout", 0.1)
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
    
    # Repetition metrics
    uniq1 = len(set(generated_tokens))
    d1 = (uniq1 / len(generated_tokens)) if generated_tokens else 1.0
    bigrams = [tuple(generated_tokens[i:i+2]) for i in range(len(generated_tokens)-1)]
    d2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 1.0
    
    return {
        "text": decoded.strip(),
        "token_count": len(generated_tokens),
        "stop_reason": stop_reason,
        "distinct_1": round(d1, 4),
        "distinct_2": round(d2, 4),
        "token_ids": generated_tokens
    }


def parse_arithmetic_answer(text: str):
    """
    Strictly extracts numerical answer from model output using equation and phrasing regexes.
    Prevents false positives from numbers in the prompt.
    """
    clean_text = text.replace(",", "")
    patterns = [
        r"=\s*(-?\d+)",
        r"(?:is|equals|equal to|leaves|leaves us with|remain|remaining|total of|gives|yields)\s*(-?\d+)",
        r"(-?\d+)\s*(?:apples|books|cookies|marbles|gallons|cartons|items|students|shares|boxes|miles|meters)",
        r"^\s*(-?\d+)\b"
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
    """
    Executes Python code in a restricted dictionary sandbox with timeout and test cases.
    Returns (syntax_valid, all_passed, test_results).
    """
    # 1. Syntax check
    try:
        ast.parse(code_str)
    except Exception as e:
        return False, False, f"SyntaxError: {e}"

    # 2. Execution check
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


def evaluate_checkpoint_suite(model, tokenizer, device: torch.device, name: str):
    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {name.upper()}")
    print(f"=======================================================")

    results = {"model_name": name}

    # 1. Benchmark 8 Prompts (Greedy and Sampled)
    print("\n1. Evaluating Benchmark 8 Prompts...")
    with open(BENCHMARK_PROMPTS_FILE, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    bench_results = []
    for b in benchmarks:
        prompt = b["prompt"]
        gen_greedy = generate(model, tokenizer, prompt, device, do_sample=False)
        gen_sample = generate(model, tokenizer, prompt, device, do_sample=True, temperature=0.8, top_p=0.9)
        bench_results.append({
            "id": b["id"],
            "prompt": prompt,
            "category": b["category"],
            "contaminated": b["training_contaminated"],
            "greedy": gen_greedy,
            "sampled": gen_sample
        })
        print(f"  Prompt: \"{prompt}\"")
        print(f"    [Greedy]: {gen_greedy['text'][:120]}... (EOS: {gen_greedy['stop_reason'] == 'stop_token_encountered'})")
    results["benchmarks"] = bench_results

    # 2. Untouched Arithmetic Suite (100 items)
    print("\n2. Evaluating Untouched Arithmetic Suite (100 questions)...")
    with open(UNTOUCHED_ARITH_FILE, "r", encoding="utf-8") as f:
        arith_items = [json.loads(l) for l in f if l.strip()]

    arith_results = []
    correct_by_tier = {}
    total_by_tier = {}
    correct_by_op = {"+": 0, "-": 0}
    total_by_op = {"+": 0, "-": 0}

    for item in arith_items:
        prompt = item["prompt"]
        expected = item["expected_answer"]
        tier = item["tier"]
        op = item["operation"]

        gen = generate(model, tokenizer, prompt, device, do_sample=False, max_new_tokens=60)
        parsed_ans = parse_arithmetic_answer(gen["text"])
        is_correct = (parsed_ans == expected)

        total_by_tier[tier] = total_by_tier.get(tier, 0) + 1
        total_by_op[op] += 1
        if is_correct:
            correct_by_tier[tier] = correct_by_tier.get(tier, 0) + 1
            correct_by_op[op] += 1

        arith_results.append({
            "id": item["id"],
            "prompt": prompt,
            "expected": expected,
            "predicted": parsed_ans,
            "is_correct": is_correct,
            "tier": tier,
            "op": op,
            "text": gen["text"]
        })

    total_arith = len(arith_items)
    total_correct = sum(1 for r in arith_results if r["is_correct"])
    overall_arith_acc = (total_correct / total_arith) * 100

    print(f"  Overall Untouched Arithmetic Accuracy: {total_correct}/{total_arith} ({overall_arith_acc:.2f}%)")
    for t in sorted(total_by_tier.keys()):
        corr = correct_by_tier.get(t, 0)
        tot = total_by_tier[t]
        print(f"    - {t:22}: {corr:2d}/{tot:2d} ({(corr/tot)*100:5.1f}%)")
    print(f"    - Addition (+):        {correct_by_op['+']}/{total_by_op['+']} ({(correct_by_op['+']/total_by_op['+'])*100:.1f}%)")
    print(f"    - Subtraction (-):     {correct_by_op['-']}/{total_by_op['-']} ({(correct_by_op['-']/total_by_op['-'])*100:.1f}%)")

    results["arithmetic"] = {
        "accuracy": round(overall_arith_acc, 2),
        "total": total_arith,
        "correct": total_correct,
        "by_tier": {t: {"correct": correct_by_tier.get(t, 0), "total": total_by_tier[t], "acc": round((correct_by_tier.get(t, 0)/total_by_tier[t])*100, 2)} for t in total_by_tier},
        "by_op": {op: {"correct": correct_by_op[op], "total": total_by_op[op], "acc": round((correct_by_op[op]/total_by_op[op])*100, 2)} for op in total_by_op},
        "details": arith_results
    }

    # 3. Untouched Python Suite (30 items)
    print("\n3. Evaluating Untouched Python Suite (30 tasks)...")
    with open(UNTOUCHED_PYTHON_FILE, "r", encoding="utf-8") as f:
        py_items = [json.loads(l) for l in f if l.strip()]

    py_results = []
    exec_syntax_valid = 0
    exec_func_correct = 0
    code_block_re = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)

    for item in py_items:
        prompt = item["prompt"]
        ttype = item["type"]
        gen = generate(model, tokenizer, prompt, device, do_sample=False, max_new_tokens=150)
        text = gen["text"]

        record = {
            "id": item["id"],
            "type": ttype,
            "prompt": prompt,
            "generated": text,
            "eos_terminated": gen["stop_reason"] == "stop_token_encountered"
        }

        if ttype == "executable_code":
            blocks = code_block_re.findall(text)
            if not blocks:
                code_to_test = text
            else:
                code_to_test = blocks[0]

            syntax_valid, func_correct, note = execute_sandboxed_python(code_to_test, item["func_name"], item["tests"])
            if syntax_valid:
                exec_syntax_valid += 1
            if func_correct:
                exec_func_correct += 1
            record["syntax_valid"] = syntax_valid
            record["func_correct"] = func_correct
            record["note"] = note
        elif "expected_key_points" in item:
            kps = item["expected_key_points"]
            found_kps = [kp for kp in kps if kp.lower() in text.lower()]
            record["key_points_found"] = found_kps
            record["key_points_total"] = len(kps)
            record["score"] = round(len(found_kps) / len(kps), 2)

        py_results.append(record)

    total_exec = sum(1 for it in py_items if it["type"] == "executable_code")
    print(f"  Executable Python Tasks: {total_exec}")
    print(f"    - Syntax Valid:         {exec_syntax_valid}/{total_exec} ({(exec_syntax_valid/total_exec)*100:.1f}%)")
    print(f"    - Functionally Correct: {exec_func_correct}/{total_exec} ({(exec_func_correct/total_exec)*100:.1f}%)")

    results["python"] = {
        "exec_total": total_exec,
        "exec_syntax_valid": exec_syntax_valid,
        "exec_syntax_valid_pct": round((exec_syntax_valid/total_exec)*100, 2),
        "exec_func_correct": exec_func_correct,
        "exec_func_correct_pct": round((exec_func_correct/total_exec)*100, 2),
        "details": py_results
    }

    # 4. Untouched Concepts Suite (30 items)
    print("\n4. Evaluating Untouched Concepts Suite (30 questions)...")
    with open(UNTOUCHED_CONCEPTS_FILE, "r", encoding="utf-8") as f:
        concept_items = [json.loads(l) for l in f if l.strip()]

    concept_results = []
    total_kp_matches = 0
    total_kps = 0

    for item in concept_items:
        prompt = item["prompt"]
        kps = item["ground_truth_key_points"]
        gen = generate(model, tokenizer, prompt, device, do_sample=False, max_new_tokens=120)
        text = gen["text"]

        matched = [kp for kp in kps if any(w in text.lower() for w in kp.lower().split())]
        total_kp_matches += len(matched)
        total_kps += len(kps)

        concept_results.append({
            "id": item["id"],
            "concept": item["concept"],
            "domain": item["domain"],
            "prompt": prompt,
            "generated": text,
            "key_points_matched": matched,
            "key_points_total": len(kps),
            "match_ratio": round(len(matched) / len(kps), 2),
            "distinct_2": gen["distinct_2"],
            "eos_terminated": gen["stop_reason"] == "stop_token_encountered"
        })

    concept_recall = (total_kp_matches / total_kps) * 100
    print(f"  Concept Key-Point Recall: {total_kp_matches}/{total_kps} ({concept_recall:.1f}%)")
    results["concepts"] = {
        "key_point_recall": round(concept_recall, 2),
        "total_key_points": total_kps,
        "matched_key_points": total_kp_matches,
        "details": concept_results
    }

    # 5. Generation Behavior Summary
    all_gens = [b["greedy"] for b in bench_results] + [b["sampled"] for b in bench_results]
    eos_count = sum(1 for g in all_gens if g["stop_reason"] == "stop_token_encountered")
    mean_d1 = float(np.mean([g["distinct_1"] for g in all_gens]))
    mean_d2 = float(np.mean([g["distinct_2"] for g in all_gens]))
    mean_len = float(np.mean([g["token_count"] for g in all_gens]))

    results["generation_behavior"] = {
        "eos_termination_pct": round((eos_count / len(all_gens)) * 100, 2),
        "mean_distinct_1": round(mean_d1, 4),
        "mean_distinct_2": round(mean_d2, 4),
        "mean_token_length": round(mean_len, 1)
    }

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # Evaluate Reference Model
    ref_model, _ = load_model(REF_CHECKPOINT, device)
    ref_results = evaluate_checkpoint_suite(ref_model, tokenizer, device, "Reference Checkpoint (Pilot best.pt)")
    del ref_model
    torch.cuda.empty_cache()

    # Evaluate Experiment 2 Model
    exp2_model, _ = load_model(EXP2_CHECKPOINT, device)
    exp2_results = evaluate_checkpoint_suite(exp2_model, tokenizer, device, "Experiment 2 Fine-Tuned (best.pt)")
    del exp2_model
    torch.cuda.empty_cache()

    combined_eval = {
        "timestamp": time.time(),
        "reference": ref_results,
        "experiment_2": exp2_results
    }

    OUTPUT_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(combined_eval, f, indent=2, ensure_ascii=False)
    print(f"\nAll detailed evaluation results saved to: {OUTPUT_RESULTS_JSON}")


if __name__ == "__main__":
    main()
