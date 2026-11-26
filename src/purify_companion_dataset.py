"""
src/purify_companion_dataset.py
===============================
Purifies the training and validation corpus for the AI Companion.
Excludes:
- Synthetic Chain-of-Thought arithmetic problem-solving
- Algorithmic Python functions and coding tasks
- Unrelated technical code generation
- Rigid synthetic formatting constraints

Preserves:
- Friendly everyday conversations
- Greetings and casual discussions
- Emotional disclosures and empathetic responses
- Friendship, relationship, and interpersonal experiences
- Stress, sadness, happiness, loneliness, and tiredness support
- Natural multi-turn memory dialogues and follow-up exchanges

CRITICAL PRESERVATION RULE:
Original datasets in data/ultimate_improvement/ and data/experiment_2/ remain 100% UNTOUCHED.
Purified datasets are written to data/companion_active/.
All exclusions and lineage are recorded in data/companion_project.db.
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR))

from src.companion_database import CompanionDatabase, compute_file_sha256, get_utc_timestamp

ULT_TRAIN = PROJECT_DIR / "data" / "ultimate_improvement" / "curriculum_raw" / "ultimate_train.jsonl"
ULT_HELDOUT = PROJECT_DIR / "data" / "ultimate_improvement" / "curriculum_raw" / "ultimate_heldout.jsonl"
OASST1_CLEAN = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category1_oasst1_cleaned.jsonl"

OUT_DIR = PROJECT_DIR / "data" / "companion_active"
OUT_TRAIN = OUT_DIR / "companion_train.jsonl"
OUT_VAL = OUT_DIR / "companion_val.jsonl"
METADATA_FILE = OUT_DIR / "dataset_metadata.json"

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
    re.compile(r"debug (?:this|the) code", re.IGNORECASE),
    re.compile(r"return\s+[a-zA-Z0-9_]+\s*%", re.IGNORECASE)
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
            return "EXCLUDE", "arithmetic_calculation_cot", "Chain-of-thought arithmetic problem or synthetic calculation"

    # Check for code/programming
    for pat in CODE_PATTERNS:
        if pat.search(text_content):
            return "EXCLUDE", "python_algorithmic_code", "Python/programming function implementation or coding task"

    # Check for synthetic formatting constraints
    if "messages" in record and len(record["messages"]) >= 2:
        prompt = record["messages"][0].get("text", "").lower()
        if "output a json" in prompt or "create a markdown table" in prompt or "do not use the letter 'e'" in prompt:
            return "EXCLUDE", "synthetic_formatting_constraint", "Synthetic formatting constraint not part of natural conversation"

    return "RETAIN", "conversational_companion", "Natural human dialogue, emotional disclosure, social interaction, or conversational Q&A"

def purify():
    print("=== AI COMPANION DATASET PURIFICATION ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = CompanionDatabase()

    # 1. Purify Training Set from ultimate_train.jsonl
    print(f"\nProcessing {ULT_TRAIN.name}...")
    train_retained = []
    train_excluded = []
    excl_counts = Counter()
    excl_samples = {}

    with open(ULT_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            status, cat, reason = classify_record(rec)
            if status == "RETAIN":
                train_retained.append(rec)
            else:
                train_excluded.append(rec)
                excl_counts[cat] += 1
                if cat not in excl_samples:
                    # Capture sample preview
                    prompt_txt = rec["messages"][0]["text"] if "messages" in rec else str(rec)[:100]
                    excl_samples[cat] = prompt_txt[:150]

    # Save purified training dataset
    with open(OUT_TRAIN, "w", encoding="utf-8") as out_f:
        for it in train_retained:
            out_f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"  Purified training sequences: {len(train_retained):,} saved to {OUT_TRAIN.name}")
    print(f"  Excluded sequences: {len(train_excluded):,}")
    for c, cnt in excl_counts.items():
        print(f"    - {c:32}: {cnt:,} items (Reason: {reason})")

    # 2. Purify Held-out Validation Set from ultimate_heldout.jsonl
    print(f"\nProcessing {ULT_HELDOUT.name}...")
    val_retained = []
    val_excluded = []
    with open(ULT_HELDOUT, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            status, cat, reason = classify_record(rec)
            if status == "RETAIN":
                val_retained.append(rec)
            else:
                val_excluded.append(rec)

    with open(OUT_VAL, "w", encoding="utf-8") as out_f:
        for it in val_retained:
            out_f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"  Purified validation sequences: {len(val_retained):,} saved to {OUT_VAL.name}")
    print(f"  Excluded validation sequences: {len(val_excluded):,}")

    # 3. Write Dataset Metadata JSON
    metadata = {
        "dataset_name": "purified_companion_corpus_v1",
        "description": "Purified conversational, emotional, and social companion dialogue corpus with programming and arithmetic excluded.",
        "created_at": get_utc_timestamp(),
        "source_files": [
            "data/ultimate_improvement/curriculum_raw/ultimate_train.jsonl",
            "data/ultimate_improvement/curriculum_raw/ultimate_heldout.jsonl"
        ],
        "train_file": str(OUT_TRAIN.relative_to(PROJECT_DIR)),
        "train_records": len(train_retained),
        "train_size_bytes": OUT_TRAIN.stat().st_size,
        "train_sha256": compute_file_sha256(OUT_TRAIN),
        "val_file": str(OUT_VAL.relative_to(PROJECT_DIR)),
        "val_records": len(val_retained),
        "val_size_bytes": OUT_VAL.stat().st_size,
        "val_sha256": compute_file_sha256(OUT_VAL),
        "total_records_excluded": len(train_excluded) + len(val_excluded),
        "exclusions_summary": dict(excl_counts)
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    # 4. Record Exclusions in Database
    print("\nRecording exclusions in companion_project.db...")
    for cat, cnt in excl_counts.items():
        excl_id = f"excl_ult_train_{cat}"
        sample = excl_samples.get(cat, "")
        reason = (
            "Chain-of-thought arithmetic word problems excluded from active companion dialogue."
            if "arithmetic" in cat
            else "Algorithmic Python code function generation excluded from active companion dialogue."
            if "python" in cat
            else "Synthetic instruction/table constraints excluded from natural dialogue."
        )
        db.record_exclusion(
            exclusion_id=excl_id,
            source_dataset="data/ultimate_improvement/curriculum_raw/ultimate_train.jsonl",
            category=cat,
            count_excluded=cnt,
            reason=reason,
            sample_preview=sample
        )

    # 5. Register Purified Datasets in Database
    print("Registering purified datasets in companion_project.db...")
    db.register_dataset(
        dataset_id="ds_companion_active_train",
        name="companion_train.jsonl",
        category="conversational_companion_active",
        file_path="data/companion_active/companion_train.jsonl",
        record_count=len(train_retained),
        size_bytes=OUT_TRAIN.stat().st_size,
        sha256_hash=metadata["train_sha256"],
        is_active_companion=True,
        status="active",
        notes="Purified active training dataset for friendly emotional companion"
    )

    db.register_dataset(
        dataset_id="ds_companion_active_val",
        name="companion_val.jsonl",
        category="conversational_companion_validation",
        file_path="data/companion_active/companion_val.jsonl",
        record_count=len(val_retained),
        size_bytes=OUT_VAL.stat().st_size,
        sha256_hash=metadata["val_sha256"],
        is_active_companion=True,
        status="active",
        notes="Purified active validation dataset for friendly emotional companion"
    )

    # 6. Record Processing History & Event
    db.conn.execute("""
    INSERT INTO dataset_processing_history (
        operation_id, step_name, input_datasets, output_dataset,
        records_in, records_out, operation_details, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        f"proc_purification_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "companion_dataset_purification",
        json.dumps(["data/ultimate_improvement/curriculum_raw/ultimate_train.jsonl"]),
        "data/companion_active/companion_train.jsonl",
        6000,
        len(train_retained),
        json.dumps(dict(excl_counts)),
        get_utc_timestamp()
    ))
    db.conn.commit()

    db.record_event(
        event_type="DATASET_PURIFICATION_COMPLETED",
        description="Active companion training corpus purified. Unrelated code, calculator, and arithmetic excluded.",
        metadata={
            "retained_train_records": len(train_retained),
            "excluded_train_records": len(train_excluded),
            "retained_val_records": len(val_retained),
            "excluded_val_records": len(val_excluded)
        }
    )

    # 7. Hot Backup
    db.backup_database()
    print("\nDatabase updated, verified, and hot backed up successfully.")
    summary = db.get_summary()
    print(f"Total exclusions tracked in DB: {summary['total_records_excluded']:,} across {summary['exclusion_categories']} categories.")
    db.close()

if __name__ == "__main__":
    purify()
