from pathlib import Path
import json
import numpy as np


PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")

INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "tokenized_conversations"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "training_sequences_masked"
)

TRAIN_FILE = INPUT_DIR / "train_formatted.jsonl"
VALIDATION_FILE = INPUT_DIR / "validation_formatted.jsonl"

TRAIN_TOKENS_OUTPUT = OUTPUT_DIR / "train_tokens.bin"
TRAIN_MASK_OUTPUT = OUTPUT_DIR / "train_mask.bin"

VALIDATION_TOKENS_OUTPUT = OUTPUT_DIR / "validation_tokens.bin"
VALIDATION_MASK_OUTPUT = OUTPUT_DIR / "validation_mask.bin"

CONTEXT_LENGTH = 1024

PAD_ID = 0
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5


def create_loss_mask(token_ids):
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


def process_file(input_file, token_output, mask_output, name):

    print()
    print("=" * 60)
    print(f"PROCESSING {name.upper()}")
    print("=" * 60)

    conversation_count = 0
    total_tokens = 0
    total_loss_tokens = 0

    token_sequences = []
    mask_sequences = []

    current_tokens = []
    current_mask = []

    with open(input_file, "r", encoding="utf-8") as file:

        for line in file:

            item = json.loads(line)

            token_ids = item["token_ids"]

            conversation_count += 1
            total_tokens += len(token_ids)

            loss_mask = create_loss_mask(token_ids)

            total_loss_tokens += int(loss_mask.sum())

            position = 0

            while position < len(token_ids):

                remaining_space = (
                    CONTEXT_LENGTH - len(current_tokens)
                )

                remaining_tokens = (
                    len(token_ids) - position
                )

                take = min(
                    remaining_space,
                    remaining_tokens
                )

                current_tokens.extend(
                    token_ids[
                        position:position + take
                    ]
                )

                current_mask.extend(
                    loss_mask[
                        position:position + take
                    ].tolist()
                )

                position += take

                if len(current_tokens) == CONTEXT_LENGTH:

                    token_sequences.append(
                        current_tokens
                    )

                    mask_sequences.append(
                        current_mask
                    )

                    current_tokens = []
                    current_mask = []

            if conversation_count % 5000 == 0:

                print(
                    f"Processed conversations: "
                    f"{conversation_count:,}"
                )

    if current_tokens:

        padding_amount = (
            CONTEXT_LENGTH - len(current_tokens)
        )

        current_tokens.extend(
            [PAD_ID] * padding_amount
        )

        current_mask.extend(
            [0] * padding_amount
        )

        token_sequences.append(
            current_tokens
        )

        mask_sequences.append(
            current_mask
        )

    print()
    print("Writing masked binary datasets...")

    token_array = np.asarray(
        token_sequences,
        dtype=np.uint16
    )

    mask_array = np.asarray(
        mask_sequences,
        dtype=np.uint8
    )

    token_array.tofile(token_output)
    mask_array.tofile(mask_output)

    sequence_count = len(token_sequences)

    stored_tokens = (
        sequence_count * CONTEXT_LENGTH
    )

    padding_tokens = (
        stored_tokens - total_tokens
    )

    print()
    print(f"Conversations:        {conversation_count:,}")
    print(f"Original tokens:      {total_tokens:,}")
    print(f"Assistant loss tokens:{total_loss_tokens:,}")
    print(f"Training sequences:   {sequence_count:,}")
    print(f"Stored tokens:        {stored_tokens:,}")
    print(f"Padding tokens:       {padding_tokens:,}")
    print(f"Sequence length:      {CONTEXT_LENGTH}")

    print()
    print(f"Token binary: {token_output}")
    print(
        f"Token size:   "
        f"{token_output.stat().st_size / (1024 ** 2):.2f} MB"
    )

    print()
    print(f"Mask binary:  {mask_output}")
    print(
        f"Mask size:    "
        f"{mask_output.stat().st_size / (1024 ** 2):.2f} MB"
    )

    return sequence_count


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


train_count = process_file(
    TRAIN_FILE,
    TRAIN_TOKENS_OUTPUT,
    TRAIN_MASK_OUTPUT,
    "training"
)

validation_count = process_file(
    VALIDATION_FILE,
    VALIDATION_TOKENS_OUTPUT,
    VALIDATION_MASK_OUTPUT,
    "validation"
)


print()
print("=" * 60)
print("MASKED DATASET BUILD COMPLETE")
print("=" * 60)

print(
    f"Training sequences:   {train_count:,}"
)

print(
    f"Validation sequences: {validation_count:,}"
)

print(
    f"Total sequences:      "
    f"{train_count + validation_count:,}"
)

print()
print("Context length:       1024")
print()
print(
    "PASS: Assistant-only loss datasets created."
)