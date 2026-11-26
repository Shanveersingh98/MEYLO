"""
scratch/dry_run_purification.py
===============================
Dry run of dataset purification logic on ultimate_train.jsonl and category1_oasst1_cleaned.jsonl.
Classifies records into companion-retained vs companion-excluded.
"""

import json
import re
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
ULT_TRAIN = PROJECT_DIR / "data" / "ultimate_improvement" / "curriculum_raw" / "ultimate_train.jsonl"
OASST1_CLEAN = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category1_oasst1_cleaned.jsonl"

CODE_PATTERNS = [
    re.compile(r"```(?:python|c|cpp|java|javascript|bash|sql|html|css)", re.IGNORECASE),
    re.compile(r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\(", re.IGNORECASE),
    re.compile(r"write a python\s+(?:script|function|code|program)", re.IGNORECASE),
    re.compile(r"implement a python\s+function", re.IGNORECASE),
    re.compile(r"in python,\s+what is the role of parameter", re.IGNORECASE),
    re.compile(r"how do you write a python function", re.IGNORECASE),
    re.compile(r"define a python function", re.IGNORECASE),
    re.compile(r"create a python function", re.IGNORECASE),
    re.compile(r"write a code snippet", re.IGNORECASE),
    re.compile(r"debug (?:this|the) code", re.IGNORECASE)
]

ARITHMETIC_PATTERNS = [
    re.compile(r"calculate\s+\d+\s*[\+\-\*\/]\s*\d+", re.IGNORECASE),
    re.compile(r"what is\s+\d+\s*(?:plus|minus|multiplied by|divided by)\s*\d+", re.IGNORECASE),
    re.compile(r"compute the sum of\s+\d+\s+and\s+\d+", re.IGNORECASE),
    re.compile(r"subtract\s+\d+\s+from\s+\d+", re.IGNORECASE),
    re.compile(r"what do you get when you (?:subtract|add|multiply)", re.IGNORECASE),
    re.compile(r"to calculate\s+\d+\s*[\+\-]\s*\d+\s+step by step", re.IGNORECASE),
    re.compile(r"1\.\s+calculate items remaining after", re.IGNORECASE),
    re.compile(r"1\.\s+first,? subtract the money spent", re.IGNORECASE),
    re.compile(r"therefore, there are \d+ (?:items|students|oranges) left", re.IGNORECASE)
]

def classify_record(record):
    """
    Returns (status, category, reason)
    status: 'RETAIN' or 'EXCLUDE'
    """
    # Extract all text in record
    text_content = ""
    if "messages" in record:
        for m in record["messages"]:
            text_content += " " + m.get("text", "")
    elif "text" in record:
        text_content = record["text"]
    elif "instruction" in record:
        text_content = record.get("instruction", "") + " " + record.get("response", "")

    # Check for arithmetic
    for pat in ARITHMETIC_PATTERNS:
        if pat.search(text_content):
            return "EXCLUDE", "arithmetic_calculation_cot", "Contains chain-of-thought arithmetic problem or synthetic math calculation"

    # Check for code/programming
    for pat in CODE_PATTERNS:
        if pat.search(text_content):
            return "EXCLUDE", "python_algorithmic_code", "Contains python/programming function implementation or coding exercise"

    # Check for JSON/markdown tables from synthetic instruction set
    if "messages" in record and len(record["messages"]) >= 2:
        prompt = record["messages"][0].get("text", "").lower()
        if "output a json" in prompt or "create a markdown table" in prompt or "do not use the letter 'e'" in prompt:
            return "EXCLUDE", "synthetic_formatting_constraint", "Contains synthetic formatting constraint not part of natural conversation"

    return "RETAIN", "conversational_companion", "Natural human dialogue, emotional disclosure, social interaction, or conversational Q&A"

def dry_run():
    print("Testing purification on ultimate_train.jsonl (6,000 records)...")
    retained = 0
    excluded = 0
    excl_reasons = Counter()
    
    with open(ULT_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            status, cat, reason = classify_record(rec)
            if status == "RETAIN":
                retained += 1
            else:
                excluded += 1
                excl_reasons[cat] += 1

    print(f"  Retained for active companion: {retained:,} records ({retained/6000*100:.1f}%)")
    print(f"  Excluded from active companion: {excluded:,} records ({excluded/6000*100:.1f}%)")
    print("  Exclusion categories:")
    for c, cnt in excl_reasons.items():
        print(f"    - {c:32}: {cnt:,}")

    # Also test on category1_oasst1_cleaned.jsonl
    print("\nTesting purification on category1_oasst1_cleaned.jsonl (2,972 records)...")
    oasst_retained = 0
    oasst_excluded = 0
    oasst_excl = Counter()
    with open(OASST1_CLEAN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            status, cat, reason = classify_record(rec)
            if status == "RETAIN":
                oasst_retained += 1
            else:
                oasst_excluded += 1
                oasst_excl[cat] += 1
    print(f"  Retained from OASST1: {oasst_retained:,} records")
    print(f"  Excluded from OASST1: {oasst_excluded:,} records")
    for c, cnt in oasst_excl.items():
        print(f"    - {c:32}: {cnt:,}")

if __name__ == "__main__":
    dry_run()
