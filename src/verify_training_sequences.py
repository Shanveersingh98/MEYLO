from pathlib import Path
import numpy as np


TRAIN_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/training_sequences/train.bin"
)

VALIDATION_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/training_sequences/validation.bin"
)

CONTEXT_LENGTH = 1024
VOCAB_SIZE = 8000

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5


def verify_file(path, name):

    print()
    print("=" * 60)
    print(f"VERIFYING {name.upper()}")
    print("=" * 60)

    data = np.fromfile(path, dtype=np.uint16)

    print(f"File: {path}")
    print(f"Total token IDs: {len(data):,}")

    # Check sequence divisibility.
    if len(data) % CONTEXT_LENGTH != 0:
        print("FAIL: Token count is not divisible by context length.")
        return False

    sequence_count = len(data) // CONTEXT_LENGTH

    print(f"Sequences: {sequence_count:,}")
    print(f"Context length: {CONTEXT_LENGTH}")

    # Check vocabulary range.
    minimum = int(data.min())
    maximum = int(data.max())

    print()
    print("Token ID range")
    print(f"Minimum ID: {minimum}")
    print(f"Maximum ID: {maximum}")
    print(f"Allowed IDs: 0 - {VOCAB_SIZE - 1}")

    if minimum < 0:
        print("FAIL: Negative token ID detected.")
        return False

    if maximum >= VOCAB_SIZE:
        print("FAIL: Token ID outside vocabulary.")
        return False

    # Special token counts.
    print()
    print("Special token counts")

    for token_id, name in [
        (PAD_ID, "[PAD]"),
        (BOS_ID, "[BOS]"),
        (EOS_ID, "[EOS]"),
        (USER_ID, "<|user|>"),
        (ASSISTANT_ID, "<|assistant|>")
    ]:
        count = int(np.sum(data == token_id))
        print(f"{name}: {count:,}")

    # Show first sequence.
    print()
    print("First 30 token IDs:")
    print(data[:30].tolist())

    # Show final sequence.
    print()
    print("Last 30 token IDs:")
    print(data[-30:].tolist())

    # Padding check.
    padding_count = int(np.sum(data == PAD_ID))

    print()
    print(f"Total padding tokens: {padding_count:,}")

    print()
    print(f"PASS: {name} binary dataset is valid.")

    return True


train_ok = verify_file(
    TRAIN_FILE,
    "training"
)

validation_ok = verify_file(
    VALIDATION_FILE,
    "validation"
)

print()
print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

if train_ok and validation_ok:
    print("PASS: Both binary datasets are valid.")
else:
    print("FAIL: Binary dataset verification failed.")