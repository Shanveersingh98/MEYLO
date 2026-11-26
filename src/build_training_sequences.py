from pathlib import Path
import json
import numpy as np


INPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/tokenized_conversations"
)

OUTPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/training_sequences"
)

TRAIN_FILE = INPUT_DIR / "train_formatted.jsonl"
VALIDATION_FILE = INPUT_DIR / "validation_formatted.jsonl"

TRAIN_OUTPUT = OUTPUT_DIR / "train.bin"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation.bin"

CONTEXT_LENGTH = 1024


def process_file(input_file, output_file, name):

    print()
    print("=" * 60)
    print(f"PROCESSING {name.upper()}")
    print("=" * 60)

    conversation_count = 0
    total_tokens = 0

    sequences = []
    current_sequence = []

    for line in open(input_file, "r", encoding="utf-8"):

        item = json.loads(line)

        token_ids = item["token_ids"]

        conversation_count += 1
        total_tokens += len(token_ids)

        position = 0

        while position < len(token_ids):

            remaining_space = (
                CONTEXT_LENGTH - len(current_sequence)
            )

            remaining_tokens = (
                len(token_ids) - position
            )

            take = min(
                remaining_space,
                remaining_tokens
            )

            current_sequence.extend(
                token_ids[position:position + take]
            )

            position += take

            if len(current_sequence) == CONTEXT_LENGTH:

                sequences.append(current_sequence)

                current_sequence = []

        if conversation_count % 5000 == 0:
            print(
                f"Processed conversations: "
                f"{conversation_count}"
            )

    # Keep the final partial sequence by padding it.
    if current_sequence:

        pad_id = 0

        current_sequence.extend(
            [pad_id] *
            (CONTEXT_LENGTH - len(current_sequence))
        )

        sequences.append(current_sequence)

    print()
    print("Writing binary dataset...")

    array = np.asarray(
        sequences,
        dtype=np.uint16
    )

    array.tofile(output_file)

    sequence_count = len(sequences)

    stored_tokens = sequence_count * CONTEXT_LENGTH

    padding_tokens = stored_tokens - total_tokens

    print()
    print(f"Conversations:        {conversation_count:,}")
    print(f"Original tokens:      {total_tokens:,}")
    print(f"Training sequences:   {sequence_count:,}")
    print(f"Stored tokens:        {stored_tokens:,}")
    print(f"Padding tokens:       {padding_tokens:,}")
    print(f"Sequence length:      {CONTEXT_LENGTH}")
    print(f"Binary file:          {output_file}")
    print(
        f"File size:            "
        f"{output_file.stat().st_size / (1024 ** 2):.2f} MB"
    )

    return sequence_count


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_count = process_file(
    TRAIN_FILE,
    TRAIN_OUTPUT,
    "training"
)

validation_count = process_file(
    VALIDATION_FILE,
    VALIDATION_OUTPUT,
    "validation"
)

print()
print("=" * 60)
print("TRAINING SEQUENCE BUILD COMPLETE")
print("=" * 60)

print(f"Training sequences:   {train_count:,}")
print(f"Validation sequences: {validation_count:,}")
print(
    f"Total sequences:      "
    f"{train_count + validation_count:,}"
)

print()
print("Context length:       1024")
print()
print("PASS: Packed binary datasets created.")