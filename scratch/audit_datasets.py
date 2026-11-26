import json
import os
import glob
from collections import Counter

def inspect_dataset(path, max_records=5):
    print(f"\n==========================================")
    print(f"Dataset: {path}")
    if not os.path.exists(path):
        print("Does not exist.")
        return
    size = os.path.getsize(path)
    print(f"File size: {size:,} bytes ({size / (1024*1024):.2f} MB)")
    
    count = 0
    categories = Counter()
    sources = Counter()
    sample_records = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                rec = json.loads(line)
                cat = rec.get("category", rec.get("source", "unknown"))
                categories[cat] += 1
                if "source" in rec:
                    sources[rec["source"]] += 1
                if len(sample_records) < max_records:
                    sample_records.append(rec)
            except Exception as e:
                pass
                
    print(f"Total records: {count}")
    print(f"Categories distribution: {dict(categories)}")
    if sources:
        print(f"Sources distribution: {dict(sources)}")
    print("--- First 2 Samples Preview ---")
    for i, s in enumerate(sample_records[:2]):
        print(f"Sample {i+1}: keys={list(s.keys())}")
        if "text" in s:
            print(f"  text snippet: {repr(s['text'][:120])}")
        elif "messages" in s:
            print(f"  messages count: {len(s['messages'])}")
            print(f"  msg 0: {repr(s['messages'][0])[:120]}")
        else:
            print(f"  raw: {repr(s)[:120]}")

datasets_to_check = [
    "data/ultimate_improvement/curriculum_raw/ultimate_train.jsonl",
    "data/ultimate_improvement/curriculum_raw/ultimate_heldout.jsonl",
    "data/experiment_2/curriculum_raw/curriculum_train_all.jsonl",
    "data/experiment_2/curriculum_raw/category1_oasst1_cleaned.jsonl",
    "data/experiment_2/curriculum_raw/category2_arithmetic_train.jsonl",
    "data/experiment_2/curriculum_raw/category3_python_train.jsonl",
    "data/experiment_2/curriculum_raw/category4_concepts_train.jsonl",
    "data/stage_2_candidates/raw/dolly_15k/databricks-dolly-15k.jsonl",
    "data/stage_2_candidates/raw/gsm8k/train.jsonl",
]

for d in datasets_to_check:
    inspect_dataset(d)
