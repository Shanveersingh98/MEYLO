from pathlib import Path
import json
import time
from datetime import datetime
import numpy as np

# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
INPUT_DIR = PROJECT_DIR / "data" / "processed" / "tokenized_conversations"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "training_sequences_clean"

TRAIN_SOURCE = INPUT_DIR / "train_formatted.jsonl"
VAL_SOURCE = INPUT_DIR / "validation_formatted.jsonl"

TRAIN_TOKENS_OUT = OUTPUT_DIR / "train_tokens.bin"
TRAIN_MASK_OUT = OUTPUT_DIR / "train_mask.bin"
VAL_TOKENS_OUT = OUTPUT_DIR / "validation_tokens.bin"
VAL_MASK_OUT = OUTPUT_DIR / "validation_mask.bin"
METADATA_OUT = OUTPUT_DIR / "dataset_metadata.json"

# ============================================================
# CONSTANTS (Design A)
# ============================================================

CONTEXT_LENGTH = 1024
VOCAB_SIZE = 8000

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5

def create_loss_mask(token_ids):
    """
    Assistant-only loss mask:
    1 = include token in assistant loss
    0 = ignore token
    """
    mask = np.zeros(len(token_ids), dtype=np.uint8)
    assistant_active = False

    for i, token_id in enumerate(token_ids):
        if token_id == ASSISTANT_ID:
            assistant_active = True
            mask[i] = 1
        elif token_id == USER_ID:
            assistant_active = False
        elif token_id == EOS_ID:
            if assistant_active:
                mask[i] = 1
            assistant_active = False
        elif assistant_active:
            mask[i] = 1

    return mask

def process_file(source_file, tokens_out_file, mask_out_file, name):
    print("=" * 70)
    print(f"PROCESSING {name.upper()} DATASET (DESIGN A: 1 CONV = 1 SEQUENCE)")
    print("=" * 70)

    start_time = time.time()
    source_conv_count = 0

    orig_total_tokens = 0
    orig_total_asst_tokens = 0

    retained_tokens = 0
    discarded_tokens = 0
    discarded_asst_tokens = 0

    truncated_conv_count = 0
    untruncated_conv_count = 0
    zero_asst_conv_count = 0

    pad_tokens_count = 0
    loss_tokens_count = 0
    unmasked_tokens_count = 0

    token_sequences = []
    mask_sequences = []

    with open(source_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            token_ids = item["token_ids"]
            raw_len = len(token_ids)

            source_conv_count += 1
            orig_total_tokens += raw_len

            # Full mask before truncation/padding
            raw_mask = create_loss_mask(token_ids)
            raw_asst_tokens = int(raw_mask.sum())
            orig_total_asst_tokens += raw_asst_tokens

            # Verify BOS at position 0
            if token_ids[0] != BOS_ID:
                raise ValueError(
                    f"Line {line_idx+1}: Conversation does not start with BOS! "
                    f"Found {token_ids[0]}"
                )

            # Apply Design A
            if raw_len <= CONTEXT_LENGTH:
                untruncated_conv_count += 1
                pad_amount = CONTEXT_LENGTH - raw_len
                seq_tokens = np.pad(token_ids, (0, pad_amount), constant_values=PAD_ID).astype(np.uint16)
                seq_mask = np.pad(raw_mask, (0, pad_amount), constant_values=0).astype(np.uint8)

                pad_tokens_count += pad_amount
                retained_tokens += raw_len
            else:
                truncated_conv_count += 1
                seq_tokens = np.array(token_ids[:CONTEXT_LENGTH], dtype=np.uint16)
                seq_mask = raw_mask[:CONTEXT_LENGTH].astype(np.uint8)

                retained_tokens += CONTEXT_LENGTH
                discarded_tokens += (raw_len - CONTEXT_LENGTH)
                discarded_asst_tokens += int(raw_mask[CONTEXT_LENGTH:].sum())

            # Check if sequence has zero assistant tokens
            seq_asst_tokens = int(seq_mask.sum())
            if seq_asst_tokens == 0:
                zero_asst_conv_count += 1

            loss_tokens_count += seq_asst_tokens
            unmasked_tokens_count += (CONTEXT_LENGTH - seq_asst_tokens)

            token_sequences.append(seq_tokens)
            mask_sequences.append(seq_mask)

            if source_conv_count % 5000 == 0:
                print(f"Processed {source_conv_count:,} conversations...")

    output_seq_count = len(token_sequences)
    print(f"\nConversations processed: {source_conv_count:,}")
    print(f"Output sequences:        {output_seq_count:,}")

    # Convert to contiguous numpy arrays
    token_array = np.vstack(token_sequences).astype(np.uint16)
    mask_array = np.vstack(mask_sequences).astype(np.uint8)

    print(f"Writing binary tokens to: {tokens_out_file}")
    token_array.tofile(tokens_out_file)

    print(f"Writing binary masks to:  {mask_out_file}")
    mask_array.tofile(mask_out_file)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds.")

    total_capacity = output_seq_count * CONTEXT_LENGTH
    real_tokens = total_capacity - pad_tokens_count

    stats = {
        "source_conversations": source_conv_count,
        "output_sequences": output_seq_count,
        "context_length": CONTEXT_LENGTH,
        "total_sequence_capacity": total_capacity,
        "real_tokens": real_tokens,
        "real_tokens_pct": (real_tokens / total_capacity) * 100,
        "pad_tokens": pad_tokens_count,
        "pad_tokens_pct": (pad_tokens_count / total_capacity) * 100,
        "avg_pad_tokens_per_seq": pad_tokens_count / output_seq_count,
        "truncated_conversations": truncated_conv_count,
        "truncated_conversations_pct": (truncated_conv_count / source_conv_count) * 100,
        "untruncated_conversations": untruncated_conv_count,
        "untruncated_conversations_pct": (untruncated_conv_count / source_conv_count) * 100,
        "original_total_tokens": orig_total_tokens,
        "retained_tokens": retained_tokens,
        "discarded_tokens": discarded_tokens,
        "original_asst_tokens": orig_total_asst_tokens,
        "retained_asst_tokens": loss_tokens_count,
        "discarded_asst_tokens": discarded_asst_tokens,
        "discarded_asst_tokens_pct": (discarded_asst_tokens / orig_total_asst_tokens) * 100 if orig_total_asst_tokens > 0 else 0,
        "loss_tokens_mask_1": loss_tokens_count,
        "loss_tokens_mask_1_pct": (loss_tokens_count / total_capacity) * 100,
        "unmasked_tokens_mask_0": unmasked_tokens_count,
        "unmasked_tokens_mask_0_pct": (unmasked_tokens_count / total_capacity) * 100,
        "zero_asst_sequences": zero_asst_conv_count,
        "tokens_file_bytes": tokens_out_file.stat().st_size,
        "mask_file_bytes": mask_out_file.stat().st_size,
    }

    return stats

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building clean isolated dataset (Design A)...")
    train_stats = process_file(TRAIN_SOURCE, TRAIN_TOKENS_OUT, TRAIN_MASK_OUT, "train")
    print()
    val_stats = process_file(VAL_SOURCE, VAL_TOKENS_OUT, VAL_MASK_OUT, "validation")

    metadata = {
        "dataset_name": "training_sequences_clean",
        "design": "A",
        "description": "One conversation per 1024-token sequence. Zero cross-conversation contamination. Truncated at 1024 if longer, padded with 0 if shorter.",
        "created_at": datetime.now().isoformat(),
        "created_by_script": "src/build_clean_sequences.py",
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
        "token_dtype": "uint16",
        "mask_dtype": "uint8",
        "source_files": {
            "train": str(TRAIN_SOURCE),
            "validation": str(VAL_SOURCE)
        },
        "output_files": {
            "train_tokens": str(TRAIN_TOKENS_OUT),
            "train_mask": str(TRAIN_MASK_OUT),
            "validation_tokens": str(VAL_TOKENS_OUT),
            "validation_mask": str(VAL_MASK_OUT)
        },
        "train_statistics": train_stats,
        "validation_statistics": val_stats
    }

    print()
    print(f"Writing dataset metadata to: {METADATA_OUT}")
    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Dataset build and metadata generation complete!")

if __name__ == "__main__":
    main()
