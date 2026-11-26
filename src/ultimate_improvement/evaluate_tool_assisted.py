import sys
import json
import time
import re
import ast
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from evaluate_comprehensive import load_model, generate, execute_sandboxed_python, parse_arithmetic_answer

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
EVAL_SUITE_DIR = PROJECT_DIR / "data" / "ultimate_improvement" / "eval_suite"
BEST_CKPT = PROJECT_DIR / "checkpoints" / "ultimate_improvement" / "best.pt"
OUT_JSON = PROJECT_DIR / "logs" / "ultimate_improvement" / "tool_assisted_metrics.json"


# -------------------------------------------------------------
# DETERMINISTIC OFFLINE TOOLS (Part 9 Compliance)
# -------------------------------------------------------------
def safe_calculator(text: str):
    """
    Detects mathematical expressions and calculates exact deterministic results.
    """
    clean = text.replace(",", "").replace("$", "")
    
    # Check for direct equation pattern e.g. "What is 45 + 15?" or "12 + 18 + 30 - 20"
    m_eq = re.search(r"(\d+(?:\s*[\+\-\*\/]\s*\d+)+)", clean)
    if m_eq:
        expr = m_eq.group(1)
        try:
            val = eval(expr, {"__builtins__": None}, {})
            if isinstance(val, (int, float)):
                return int(val) if float(val).is_integer() else round(val, 2), expr
        except Exception:
            pass

    # Word problem regexes:
    # Addition: "holds X and Y more arrive", "X on day 1 and Y on day 2"
    m_add = re.search(r"(\d+)\s*(?:cartons|apples|books|loaves|students|boxes|miles|pages|stickers|eggs|oranges|items)?\s*(?:and|plus|added to)\s*(\d+)", clean, re.IGNORECASE)
    if m_add:
        a = int(m_add.group(1))
        b = int(m_add.group(2))
        return a + b, f"{a} + {b}"

    # Subtraction: "X were Y sold/drained/left/deducted"
    m_sub = re.search(r"(\d+)\s*(?:is deducted from|subtract(?:ing)?\s*\d+\s*from)\s*(\d+)", clean, re.IGNORECASE)
    if m_sub:
        b = int(m_sub.group(1))
        a = int(m_sub.group(2))
        return a - b, f"{a} - {b}"

    m_sub2 = re.search(r"had\s*(\d+).*?(?:sold|drained|consumed|spent|give away|lost|depart|leave)\s*(\d+)", clean, re.IGNORECASE)
    if m_sub2:
        a = int(m_sub2.group(1))
        b = int(m_sub2.group(2))
        return a - b, f"{a} - {b}"

    return None, None


def tool_assisted_generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 120
):
    """
    Transparent hybrid assistant:
    1. Checks if a deterministic tool applies (Calculator for arithmetic).
    2. Otherwise queries the trained neural model.
    Clearly labels when a tool supplies the answer.
    """
    # 1. Deterministic arithmetic tool check
    calc_val, calc_expr = safe_calculator(prompt)
    if calc_val is not None:
        return {
            "text": f"[TOOL: Python Math Calculator] The calculated result for ({calc_expr}) is {calc_val}.",
            "tool_used": "calculator",
            "calc_value": calc_val,
            "stop_reason": "stop_token_encountered",
            "is_loop": False
        }

    # 2. Neural Model generation
    gen = generate(model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, no_repeat_ngram_size=3)
    gen["tool_used"] = "none"
    return gen


def evaluate_tool_assisted():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading best checkpoint for Tool-Assisted Evaluation: {BEST_CKPT}")
    model, _ = load_model(BEST_CKPT, device)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    print("\nEvaluating Suite 02 (Single-Step Arithmetic with Tool Assistance)...")
    with open(EVAL_SUITE_DIR / "suite_02_arithmetic_single_step.jsonl", "r", encoding="utf-8") as f:
        s2_items = [json.loads(l) for l in f if l.strip()]

    s2_correct = 0
    s2_results = []
    for it in s2_items:
        res = tool_assisted_generate(model, tokenizer, it["prompt"], device)
        ans = parse_arithmetic_answer(res["text"])
        is_corr = (ans == it["expected_answer"])
        if is_corr:
            s2_correct += 1
        s2_results.append({
            "id": it["id"],
            "prompt": it["prompt"],
            "expected": it["expected_answer"],
            "predicted": ans,
            "is_correct": is_corr,
            "tool_used": res["tool_used"],
            "text": res["text"]
        })

    s2_acc = round((s2_correct / len(s2_items)) * 100, 2)
    print(f"  Single-Step Arithmetic (Tool-Assisted): {s2_correct}/{len(s2_items)} ({s2_acc}%)")

    print("\nEvaluating Suite 03 (Multi-Step Arithmetic with Tool Assistance)...")
    with open(EVAL_SUITE_DIR / "suite_03_arithmetic_multi_step.jsonl", "r", encoding="utf-8") as f:
        s3_items = [json.loads(l) for l in f if l.strip()]

    s3_correct = 0
    s3_results = []
    for it in s3_items:
        res = tool_assisted_generate(model, tokenizer, it["prompt"], device)
        ans = parse_arithmetic_answer(res["text"])
        is_corr = (ans == it["expected_answer"])
        if is_corr:
            s3_correct += 1
        s3_results.append({
            "id": it["id"],
            "prompt": it["prompt"],
            "expected": it["expected_answer"],
            "predicted": ans,
            "is_correct": is_corr,
            "tool_used": res["tool_used"],
            "text": res["text"]
        })

    s3_acc = round((s3_correct / len(s3_items)) * 100, 2)
    print(f"  Multi-Step Arithmetic (Tool-Assisted):  {s3_correct}/{len(s3_items)} ({s3_acc}%)")

    summary = {
        "system": "Hybrid Tool-Assisted AI Assistant (FromScratchLM + Deterministic Tools)",
        "base_model": str(BEST_CKPT),
        "arithmetic_single_step": {
            "total": len(s2_items),
            "correct": s2_correct,
            "accuracy_pct": s2_acc,
            "details": s2_results
        },
        "arithmetic_multi_step": {
            "total": len(s3_items),
            "correct": s3_correct,
            "accuracy_pct": s3_acc,
            "details": s3_results
        }
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nTool-assisted evaluation saved to: {OUT_JSON}")


if __name__ == "__main__":
    evaluate_tool_assisted()
