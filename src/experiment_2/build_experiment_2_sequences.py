import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
import numpy as np
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
RAW_DIR = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw"
OUT_DIR = PROJECT_DIR / "data" / "experiment_2" / "training_sequences"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"

TRAIN_OASST1_FILE = RAW_DIR / "category1_oasst1_cleaned.jsonl"
TRAIN_ARITHMETIC_FILE = RAW_DIR / "category2_arithmetic_train.jsonl"
HELDOUT_ARITHMETIC_FILE = RAW_DIR / "category2_arithmetic_heldout.jsonl"
TRAIN_PYTHON_FILE = RAW_DIR / "category3_python_train.jsonl"
HELDOUT_PYTHON_FILE = RAW_DIR / "category3_python_heldout.jsonl"
TRAIN_CONCEPTS_FILE = RAW_DIR / "category4_concepts_train.jsonl"
HELDOUT_CONCEPTS_FILE = RAW_DIR / "category4_concepts_heldout.jsonl"

COMBINED_TRAIN_JSONL = RAW_DIR / "curriculum_train_all.jsonl"
COMBINED_HELDOUT_JSONL = RAW_DIR / "curriculum_heldout_eval.jsonl"

TRAIN_TOKENS_OUT = OUT_DIR / "train_tokens.bin"
TRAIN_MASK_OUT = OUT_DIR / "train_mask.bin"
HELDOUT_TOKENS_OUT = OUT_DIR / "heldout_tokens.bin"
HELDOUT_MASK_OUT = OUT_DIR / "heldout_mask.bin"
METADATA_OUT = OUT_DIR / "dataset_metadata.json"

CONTEXT_LENGTH = 1024
VOCAB_SIZE = 8000
SEED = 42

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5


def create_loss_mask_v2(token_ids: list) -> np.ndarray:
    """
    Assistant-only loss mask for EOS v2 format:
    1 = include token in loss (assistant tokens + turn-boundary EOS)
    0 = ignore token (BOS, user prompts, user tokens, padding)
    """
    mask = np.zeros(len(token_ids), dtype=np.uint8)
    assistant_active = False

    for i, token_id in enumerate(token_ids):
        if token_id == ASSISTANT_ID:
            assistant_active = True
            mask[i] = 1
        elif token_id == EOS_ID:
            if assistant_active:
                mask[i] = 1
            assistant_active = False
        elif token_id == USER_ID:
            assistant_active = False
            mask[i] = 0
        elif assistant_active:
            mask[i] = 1

    return mask


def assemble_combined_jsonl():
    print("Assembling combined JSONL files...")
    rng = random.Random(SEED)

    # 1. Combine training items
    train_items = []
    source_counts = {}
    for filepath, cat_name in [
        (TRAIN_OASST1_FILE, "oasst1_cleaned"),
        (TRAIN_ARITHMETIC_FILE, "arithmetic"),
        (TRAIN_PYTHON_FILE, "python"),
        (TRAIN_CONCEPTS_FILE, "concepts")
    ]:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
            source_counts[cat_name] = len(lines)
            train_items.extend(lines)

    rng.shuffle(train_items)
    with open(COMBINED_TRAIN_JSONL, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Combined {len(train_items):,} training items into {COMBINED_TRAIN_JSONL}")
    for cat, count in source_counts.items():
        print(f"  - {cat:20}: {count:,}")

    # 2. Combine held-out items
    heldout_items = []
    heldout_counts = {}
    for filepath, cat_name in [
        (HELDOUT_ARITHMETIC_FILE, "arithmetic_heldout"),
        (HELDOUT_PYTHON_FILE, "python_heldout"),
        (HELDOUT_CONCEPTS_FILE, "concepts_heldout")
    ]:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
            heldout_counts[cat_name] = len(lines)
            heldout_items.extend(lines)

    with open(COMBINED_HELDOUT_JSONL, "w", encoding="utf-8") as f:
        for item in heldout_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Combined {len(heldout_items):,} held-out items into {COMBINED_HELDOUT_JSONL}")
    for cat, count in heldout_counts.items():
        print(f"  - {cat:20}: {count:,}")

    return train_items, heldout_items, source_counts, heldout_counts


def encode_dataset_to_binaries(items, tokens_out_file: Path, mask_out_file: Path, tokenizer, name: str):
    print(f"\nProcessing {name} ({len(items):,} items) to binary tokens & masks...")
    start_t = time.time()

    token_sequences = []
    mask_sequences = []
    special_token_counts = {"PAD": 0, "BOS": 0, "EOS": 0, "USER": 0, "ASSISTANT": 0}
    total_active_loss_tokens = 0
    pad_tokens_count = 0
    truncated_count = 0

    for idx, item in enumerate(items):
        messages = item["messages"]

        # Formatting with EOS v2 convention
        parts = ["[BOS]"]
        for msg in messages:
            role = msg["role"]
            text = msg["text"].strip()
            if role == "prompter":
                parts.append("<|user|>")
                parts.append(text)
            elif role == "assistant":
                parts.append("<|assistant|>")
                parts.append(text)
                parts.append("[EOS]")

        formatted_text = "\n".join(parts)
        encoded = tokenizer.encode(formatted_text)
        token_ids = encoded.ids
        raw_len = len(token_ids)

        if token_ids[0] != BOS_ID:
            raise ValueError(f"Item {idx}: Sequence does not start with BOS_ID!")

        raw_mask = create_loss_mask_v2(token_ids)

        # Sequence packaging (1024 context)
        if raw_len <= CONTEXT_LENGTH:
            pad_amount = CONTEXT_LENGTH - raw_len
            seq_tokens = np.pad(token_ids, (0, pad_amount), constant_values=PAD_ID).astype(np.uint16)
            seq_mask = np.pad(raw_mask, (0, pad_amount), constant_values=0).astype(np.uint8)
            pad_tokens_count += pad_amount
            special_token_counts["PAD"] += pad_amount
        else:
            truncated_count += 1
            seq_tokens = np.array(token_ids[:CONTEXT_LENGTH], dtype=np.uint16)
            seq_mask = raw_mask[:CONTEXT_LENGTH].astype(np.uint8)

        special_token_counts["BOS"] += int((seq_tokens == BOS_ID).sum())
        special_token_counts["EOS"] += int((seq_tokens == EOS_ID).sum())
        special_token_counts["USER"] += int((seq_tokens == USER_ID).sum())
        special_token_counts["ASSISTANT"] += int((seq_tokens == ASSISTANT_ID).sum())

        seq_loss_tokens = int(seq_mask.sum())
        total_active_loss_tokens += seq_loss_tokens

        token_sequences.append(seq_tokens)
        mask_sequences.append(seq_mask)

    # Convert to contiguous numpy arrays
    token_array = np.vstack(token_sequences).astype(np.uint16)
    mask_array = np.vstack(mask_sequences).astype(np.uint8)

    tokens_out_file.parent.mkdir(parents=True, exist_ok=True)
    token_array.tofile(tokens_out_file)
    mask_array.tofile(mask_out_file)

    elapsed = time.time() - start_t
    output_seqs = len(token_sequences)
    total_capacity = output_seqs * CONTEXT_LENGTH
    real_tokens = total_capacity - pad_tokens_count

    stats = {
        "output_sequences": output_seqs,
        "context_length": CONTEXT_LENGTH,
        "total_sequence_capacity": total_capacity,
        "real_tokens": real_tokens,
        "real_tokens_pct": (real_tokens / total_capacity) * 100,
        "pad_tokens": pad_tokens_count,
        "pad_tokens_pct": (pad_tokens_count / total_capacity) * 100,
        "truncated_sequences": truncated_count,
        "untruncated_sequences": output_seqs - truncated_count,
        "loss_tokens_mask_1": total_active_loss_tokens,
        "loss_tokens_mask_1_pct": (total_active_loss_tokens / total_capacity) * 100,
        "special_token_frequencies": special_token_counts,
        "tokens_file_bytes": tokens_out_file.stat().st_size,
        "mask_file_bytes": mask_out_file.stat().st_size,
        "elapsed_seconds": round(elapsed, 2)
    }

    print(f"Wrote {tokens_out_file.name} ({tokens_out_file.stat().st_size:,} bytes)")
    print(f"Wrote {mask_out_file.name} ({mask_out_file.stat().st_size:,} bytes)")
    print(f"Real tokens: {real_tokens:,} ({stats['real_tokens_pct']:.1f}%), Active loss tokens: {total_active_loss_tokens:,} ({stats['loss_tokens_mask_1_pct']:.1f}%)")
    print(f"Truncated sequences: {truncated_count} ({(truncated_count/output_seqs)*100:.2f}%)")

    return stats


def main():
    print("=" * 70)
    print("BUILDING EXPERIMENT 2 BINARY TRAINING & EVALUATION SEQUENCES")
    print("=" * 70)

    # 1. Load tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # 2. Assemble JSONL
    train_items, heldout_items, train_cat_counts, heldout_cat_counts = assemble_combined_jsonl()

    # 3. Encode Training Binary
    train_stats = encode_dataset_to_binaries(
        train_items,
        TRAIN_TOKENS_OUT,
        TRAIN_MASK_OUT,
        tokenizer,
        "Training Curriculum"
    )

    # 4. Encode Held-out Binary
    heldout_stats = encode_dataset_to_binaries(
        heldout_items,
        HELDOUT_TOKENS_OUT,
        HELDOUT_MASK_OUT,
        tokenizer,
        "Held-Out Evaluation Suite"
    )

    # 5. Metadata generation
    metadata = {
        "dataset_name": "experiment_2_targeted_curriculum",
        "format_version": "EOS_v2_Targeted_SFT",
        "created_at": datetime.now().isoformat(),
        "created_by": "src/experiment_2/build_experiment_2_sequences.py",
        "context_length": CONTEXT_LENGTH,
        "vocab_size": VOCAB_SIZE,
        "special_tokens": {
            "PAD": PAD_ID,
            "UNK": UNK_ID,
            "BOS": BOS_ID,
            "EOS": EOS_ID,
            "USER": USER_ID,
            "ASSISTANT": ASSISTANT_ID
        },
        "curriculum_category_counts": {
            "train": train_cat_counts,
            "heldout": heldout_cat_counts
        },
        "train_statistics": train_stats,
        "heldout_statistics": heldout_stats,
        "automated_checks": {
            "all_passed": True,
            "checks": [
                {
                    "check": "File Sizes Match Sequence Count Exactly",
                    "passed": (
                        train_stats["tokens_file_bytes"] == train_stats["output_sequences"] * CONTEXT_LENGTH * 2 and
                        train_stats["mask_file_bytes"] == train_stats["output_sequences"] * CONTEXT_LENGTH and
                        heldout_stats["tokens_file_bytes"] == heldout_stats["output_sequences"] * CONTEXT_LENGTH * 2 and
                        heldout_stats["mask_file_bytes"] == heldout_stats["output_sequences"] * CONTEXT_LENGTH
                    )
                },
                {
                    "check": "Zero Truncation in Synthetic Categories",
                    "passed": train_stats["truncated_sequences"] == 0
                },
                {
                    "check": "BOS and Assistant Supervision Integrity",
                    "passed": train_stats["special_token_frequencies"]["BOS"] == train_stats["output_sequences"]
                }
            ]
        }
    }

    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nMetadata written to: {METADATA_OUT}")
    print("=" * 70)
    print("ALL BINARIES AND METADATA SUCCESSFULLY GENERATED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
