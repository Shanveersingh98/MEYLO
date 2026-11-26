from pathlib import Path
import json
import numpy as np


INPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/tokenized_conversations"
)

TRAIN_FILE = INPUT_DIR / "train_formatted.jsonl"
VALIDATION_FILE = INPUT_DIR / "validation_formatted.jsonl"


def load_lengths(file_path):
    lengths = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            lengths.append(item["token_count"])

    return np.array(lengths)


def analyze(lengths, name):

    print()
    print("=" * 60)
    print(f"{name.upper()} TOKEN LENGTH ANALYSIS")
    print("=" * 60)

    print(f"Conversations: {len(lengths)}")
    print(f"Total tokens: {lengths.sum():,}")

    print()
    print("Basic statistics")
    print("-" * 30)
    print(f"Minimum: {lengths.min():,}")
    print(f"Maximum: {lengths.max():,}")
    print(f"Mean:    {lengths.mean():.2f}")
    print(f"Median:  {np.median(lengths):.2f}")

    print()
    print("Percentiles")
    print("-" * 30)

    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]

    for p in percentiles:
        value = np.percentile(lengths, p)
        print(f"{p:>5}%: {value:,.2f} tokens")

    print()
    print("Context-length coverage")
    print("-" * 30)

    context_lengths = [128, 256, 512, 1024, 2048]

    for context in context_lengths:

        conversations_fit = np.sum(lengths <= context)
        percentage = conversations_fit / len(lengths) * 100

        tokens_kept = np.minimum(lengths, context).sum()
        total_tokens = lengths.sum()

        token_percentage = tokens_kept / total_tokens * 100

        print(
            f"{context:>4} tokens -> "
            f"{conversations_fit:,}/{len(lengths):,} conversations "
            f"({percentage:.2f}%) | "
            f"token coverage: {token_percentage:.2f}%"
        )


print("Loading tokenized conversations...")

train_lengths = load_lengths(TRAIN_FILE)
validation_lengths = load_lengths(VALIDATION_FILE)

analyze(train_lengths, "Training")
analyze(validation_lengths, "Validation")


all_lengths = np.concatenate(
    [train_lengths, validation_lengths]
)

analyze(all_lengths, "Combined")

print()
print("=" * 60)
print("TOKEN LENGTH ANALYSIS COMPLETE")
print("=" * 60)