from pathlib import Path
import json


# --------------------------------------------------
# Project paths
# --------------------------------------------------

INPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_paths"
)

OUTPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/formatted_conversations"
)


# --------------------------------------------------
# Format one JSONL file
# --------------------------------------------------

def format_conversations(
    input_file,
    output_file,
    name
):

    print()
    print(f"=== FORMATTING {name.upper()} ===")

    count = 0

    with open(
        input_file,
        "r",
        encoding="utf-8"
    ) as source, open(
        output_file,
        "w",
        encoding="utf-8"
    ) as target:

        for line in source:

            conversation = json.loads(line)

            messages = conversation["messages"]

            # Start of conversation
            target.write("[BOS]\n")

            for message in messages:

                role = message["role"]
                text = message["text"].strip()

                if role == "prompter":

                    target.write(
                        "<|user|>\n"
                    )

                elif role == "assistant":

                    target.write(
                        "<|assistant|>\n"
                    )

                target.write(text + "\n")

            # End of conversation
            target.write("[EOS]\n")

            # Conversation separator
            target.write(
                "====================\n"
            )

            count += 1

    print(
        f"Conversations formatted: {count}"
    )

    print(
        f"Saved to: {output_file}"
    )

    return count


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Input files
# --------------------------------------------------

train_input = (
    INPUT_DIR / "train_paths.jsonl"
)

validation_input = (
    INPUT_DIR / "validation_paths.jsonl"
)


# --------------------------------------------------
# Output files
# --------------------------------------------------

train_output = (
    OUTPUT_DIR / "train.txt"
)

validation_output = (
    OUTPUT_DIR / "validation.txt"
)


# --------------------------------------------------
# Format both datasets
# --------------------------------------------------

train_count = format_conversations(
    train_input,
    train_output,
    "training"
)

validation_count = format_conversations(
    validation_input,
    validation_output,
    "validation"
)


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print("=" * 60)
print("FORMATTING COMPLETE")
print("=" * 60)

print(
    f"Training conversations:   "
    f"{train_count}"
)

print(
    f"Validation conversations: "
    f"{validation_count}"
)

print()
print(
    "Expected training:   17683"
)

print(
    "Expected validation: 1989"
)

if (
    train_count == 17683
    and validation_count == 1989
):

    print(
        "PASS: Conversation formatting is correct."
    )

else:

    print(
        "FAIL: Conversation count mismatch."
    )