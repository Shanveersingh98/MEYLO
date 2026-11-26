import sys
import json
import re
from pathlib import Path
from collections import Counter
import numpy as np
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
CURR_DIR = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw"
BIN_DIR = PROJECT_DIR / "data" / "experiment_2" / "training_sequences"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
EVAL_DIR = PROJECT_DIR / "data" / "ultimate_improvement" / "eval_suite"

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

files_to_audit = [
    ("OASST1 Cleaned", CURR_DIR / "category1_oasst1_cleaned.jsonl"),
    ("Arithmetic Train", CURR_DIR / "category2_arithmetic_train.jsonl"),
    ("Python Train", CURR_DIR / "category3_python_train.jsonl"),
    ("Concepts Train", CURR_DIR / "category4_concepts_train.jsonl"),
    ("Arithmetic Heldout", CURR_DIR / "category2_arithmetic_heldout.jsonl"),
    ("Python Heldout", CURR_DIR / "category3_python_heldout.jsonl"),
    ("Concepts Heldout", CURR_DIR / "category4_concepts_heldout.jsonl")
]

audit_results = {}
all_train_prompts = set()

print("=" * 70)
print("AUDITING CURRICULUM RAW DATASETS")
print("=" * 70)

for name, path in files_to_audit:
    if not path.exists():
        continue
    with open(path, "r", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]

    prompt_lengths = []
    asst_lengths = []
    total_token_lengths = []
    tree_ids = set()
    prompts = []
    empty_count = 0

    for it in items:
        msgs = it["messages"]
        if not msgs:
            empty_count += 1
            continue
        p_text = msgs[0]["text"]
        a_text = msgs[1]["text"] if len(msgs) > 1 else ""
        if "tree_id" in it:
            tree_ids.add(it["tree_id"])

        p_tokens = len(tokenizer.encode(p_text).ids)
        a_tokens = len(tokenizer.encode(a_text).ids)
        prompt_lengths.append(p_tokens)
        asst_lengths.append(a_tokens)

        # formatted sequence length
        formatted = f"[BOS]\n<|user|>\n{p_text.strip()}\n<|assistant|>\n{a_text.strip()}\n[EOS]"
        seq_len = len(tokenizer.encode(formatted).ids)
        total_token_lengths.append(seq_len)

        prompts.append(p_text.strip())
        if "Train" in name or "OASST1" in name:
            all_train_prompts.add(p_text.strip().lower())

    uniq_prompts = len(set(prompts))
    dup_count = len(prompts) - uniq_prompts

    audit_results[name] = {
        "count": len(items),
        "unique_prompts": uniq_prompts,
        "duplicate_prompts": dup_count,
        "empty_samples": empty_count,
        "distinct_trees": len(tree_ids),
        "mean_prompt_tokens": round(float(np.mean(prompt_lengths)), 1) if prompt_lengths else 0,
        "mean_asst_tokens": round(float(np.mean(asst_lengths)), 1) if asst_lengths else 0,
        "mean_seq_tokens": round(float(np.mean(total_token_lengths)), 1) if total_token_lengths else 0,
        "max_seq_tokens": int(np.max(total_token_lengths)) if total_token_lengths else 0,
        "min_seq_tokens": int(np.min(total_token_lengths)) if total_token_lengths else 0,
        "oversized_gt_1024": sum(1 for l in total_token_lengths if l > 1024)
    }
    print(f"[{name:20}] Count: {len(items):5d} | Duplicates: {dup_count:3d} | Mean Seq Len: {audit_results[name]['mean_seq_tokens']:5.1f} | Max: {audit_results[name]['max_seq_tokens']:4d}")

print("\n" + "=" * 70)
print("AUDITING EVALUATION SUITE LEAKAGE AGAINST TRAINING PROMPTS")
print("=" * 70)

leakage_report = {}
for suite_path in EVAL_DIR.glob("*.json*"):
    with open(suite_path, "r", encoding="utf-8") as f:
        if suite_path.suffix == ".json":
            eval_items = json.load(f)
        else:
            eval_items = [json.loads(l) for l in f if l.strip()]

    exact_matches = []
    for it in eval_items:
        p = it.get("prompt")
        if not p and "turns" in it:
            p = it["turns"][0]["user"]
        if p and p.strip().lower() in all_train_prompts:
            exact_matches.append(p)

    leakage_report[suite_path.name] = {
        "suite_items": len(eval_items),
        "exact_matches_in_train": len(exact_matches),
        "matches": exact_matches
    }
    print(f"[{suite_path.name:35}] Items: {len(eval_items):3d} | Exact Train Matches: {len(exact_matches):2d}")
    if exact_matches:
        for m in exact_matches:
            print(f"    -> MATCH: {repr(m)}")

summary = {
    "datasets": audit_results,
    "leakage": leakage_report
}

out_json = PROJECT_DIR / "logs" / "ultimate_improvement" / "dataset_audit_stats.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("\nAudit stats written to:", out_json)
