import sys
import json
import ast
import re
import numpy as np
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
RAW_DIR = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw"
SEQ_DIR = PROJECT_DIR / "data" / "experiment_2" / "training_sequences"

print("=" * 70)
print("PHASE A: INDEPENDENT AUDIT OF PREPARED DATA")
print("=" * 70)

# 1. Audit raw JSONL files
raw_files = sorted(RAW_DIR.glob("*.jsonl"))
print(f"\n1. Raw JSONL File Verification ({len(raw_files)} files found):")
file_counts = {}
for rf in raw_files:
    count = 0
    with open(rf, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    file_counts[rf.name] = count
    print(f"  - {rf.name:36} : {count:6,} records | {rf.stat().st_size:10,} bytes")

# 2. Arithmetic verification & contamination analysis
print("\n2. Arithmetic Category Audit:")
with open(RAW_DIR / "category2_arithmetic_train.jsonl", "r", encoding="utf-8") as f:
    train_arith = [json.loads(l) for l in f if l.strip()]
with open(RAW_DIR / "category2_arithmetic_heldout.jsonl", "r", encoding="utf-8") as f:
    heldout_arith = [json.loads(l) for l in f if l.strip()]

train_pairs = set((x["a"], x["b"], x["operation"]) for x in train_arith)
heldout_pairs = set((x["a"], x["b"], x["operation"]) for x in heldout_arith)
pair_overlap = train_pairs.intersection(heldout_pairs)
print(f"  - Train arithmetic examples:    {len(train_arith):,}")
print(f"  - Held-out arithmetic examples: {len(heldout_arith):,}")
print(f"  - Exact (a, b, op) overlap:     {len(pair_overlap)}")
is_2_plus_2_in_train = (2, 2, "+") in train_pairs
is_2_plus_2_in_heldout = (2, 2, "+") in heldout_pairs
print(f"  - Is (2, 2, '+') in train?      {is_2_plus_2_in_train}")
print(f"  - Is (2, 2, '+') in held-out?   {is_2_plus_2_in_heldout}")

# Template overlap in arithmetic
train_arith_prompts = set(x["messages"][0]["text"] for x in train_arith)
heldout_arith_prompts = set(x["messages"][0]["text"] for x in heldout_arith)
prompt_overlap = train_arith_prompts.intersection(heldout_arith_prompts)
print(f"  - Prompt text overlap:          {len(prompt_overlap)}")

# Verify all arithmetic calculations
arith_errors = 0
for ex in train_arith + heldout_arith:
    a, b, op, ans = ex["a"], ex["b"], ex["operation"], ex["answer"]
    expected = a + b if op == "+" else a - b
    if ans != expected:
        arith_errors += 1
print(f"  - Calculation errors detected:  {arith_errors}")

# 3. Python category audit
print("\n3. Python Category Audit:")
with open(RAW_DIR / "category3_python_train.jsonl", "r", encoding="utf-8") as f:
    train_py = [json.loads(l) for l in f if l.strip()]
with open(RAW_DIR / "category3_python_heldout.jsonl", "r", encoding="utf-8") as f:
    heldout_py = [json.loads(l) for l in f if l.strip()]

train_py_prompts = set(x["messages"][0]["text"] for x in train_py)
heldout_py_prompts = set(x["messages"][0]["text"] for x in heldout_py)
py_prompt_overlap = train_py_prompts.intersection(heldout_py_prompts)
print(f"  - Train Python examples:        {len(train_py):,}")
print(f"  - Held-out Python examples:     {len(heldout_py):,}")
print(f"  - Prompt text overlap:          {len(py_prompt_overlap)}")

# AST parse all code blocks
code_re = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)
py_syntax_errors = 0
total_code_blocks = 0
for ex in train_py + heldout_py:
    asst_text = ex["messages"][1]["text"]
    blocks = code_re.findall(asst_text)
    for b in blocks:
        total_code_blocks += 1
        try:
            ast.parse(b)
        except SyntaxError:
            py_syntax_errors += 1
print(f"  - Total code blocks parsed:     {total_code_blocks:,}")
print(f"  - AST Syntax errors:            {py_syntax_errors}")

# Task isolation check
t_tasks = set(x.get("task") for x in train_py if x.get("task"))
h_tasks = set(x.get("task") for x in heldout_py if x.get("task"))
task_overlap = t_tasks.intersection(h_tasks)
print(f"  - Task name overlap:            {len(task_overlap)}")

# 4. Concepts category audit
print("\n4. Concepts Category Audit:")
with open(RAW_DIR / "category4_concepts_train.jsonl", "r", encoding="utf-8") as f:
    train_concepts = [json.loads(l) for l in f if l.strip()]
with open(RAW_DIR / "category4_concepts_heldout.jsonl", "r", encoding="utf-8") as f:
    heldout_concepts = [json.loads(l) for l in f if l.strip()]

t_cids = set(x["concept_id"] for x in train_concepts)
h_cids = set(x["concept_id"] for x in heldout_concepts)
cid_overlap = t_cids.intersection(h_cids)
print(f"  - Train concept examples:       {len(train_concepts):,}")
print(f"  - Held-out concept examples:    {len(heldout_concepts):,}")
print(f"  - Concept ID overlap:           {len(cid_overlap)}")

# 5. OASST1 cleaned audit
print("\n5. OASST1 Cleaned Audit:")
with open(RAW_DIR / "category1_oasst1_cleaned.jsonl", "r", encoding="utf-8") as f:
    oasst_items = [json.loads(l) for l in f if l.strip()]
print(f"  - OASST1 clean sequences:       {len(oasst_items):,}")

# Check turn alternation and ending
oasst_structural_errors = 0
canned_re = re.compile(r"^(as an ai|i am an ai|as a large language model)", re.IGNORECASE)
canned_found = 0
for ex in oasst_items:
    msgs = ex["messages"]
    if len(msgs) == 0 or msgs[-1]["role"] != "assistant":
        oasst_structural_errors += 1
    for i, m in enumerate(msgs):
        exp = "prompter" if i % 2 == 0 else "assistant"
        if m["role"] != exp:
            oasst_structural_errors += 1
        if m["role"] == "assistant" and canned_re.search(m["text"][:100]):
            canned_found += 1
print(f"  - Structural errors:            {oasst_structural_errors}")
print(f"  - Canned AI disclaimers found:  {canned_found}")

# 6. Binary file audit
print("\n6. Binary Sequences Audit:")
bin_files = sorted(SEQ_DIR.glob("*.bin"))
for bf in bin_files:
    size = bf.stat().st_size
    is_tokens = "tokens" in bf.name
    dtype = np.uint16 if is_tokens else np.uint8
    elem_size = 2 if is_tokens else 1
    num_elems = size // elem_size
    num_seqs = num_elems // 1024
    rem = num_elems % 1024
    
    arr = np.memmap(bf, dtype=dtype, mode="r")
    min_val = int(arr.min())
    max_val = int(arr.max())
    
    print(f"  - {bf.name:22} : {size:12,} bytes | {num_seqs:5,} seqs | Remainder={rem} | Range=[{min_val}, {max_val}]")
    if is_tokens and max_val >= 8000:
        print(f"    WARNING: Token ID >= 8000 found ({max_val})!")
    if not is_tokens and max_val > 1:
        print(f"    WARNING: Mask values outside {{0, 1}} found ({max_val})!")

print("\n" + "=" * 70)
print("PHASE A AUDIT FINISHED")
print("=" * 70)
