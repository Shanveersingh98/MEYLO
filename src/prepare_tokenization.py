from pathlib import Path
import json
from tokenizers import Tokenizer


# --------------------------------------------------
# Project paths
# --------------------------------------------------

INPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_paths"
)

OUTPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/tokenized_conversations"
)

TOKENIZER_PATH = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "tokenizer/tokenizer.json"
)


# --------------------------------------------------
# Input files
# --------------------------------------------------

TRAIN_FILE = INPUT_DIR / "train_paths.jsonl"
VALIDATION_FILE = INPUT_DIR / "validation_paths.jsonl"


# --------------------------------------------------
# Output files
# --------------------------------------------------

TRAIN_OUTPUT = OUTPUT_DIR / "train_formatted.jsonl"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation_formatted.jsonl"


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

print(
    f"Vocabulary size: "
    f"{tokenizer.get_vocab_size()}"
)


# --------------------------------------------------
# Check special tokens
# --------------------------------------------------

SPECIAL_TOKENS = [
    "[BOS]",
    "[EOS]",
    "<|user|>",
    "<|assistant|>"
]

print()
print("=== SPECIAL TOKENS ===")

for token in SPECIAL_TOKENS:

    token_id = tokenizer.token_to_id(token)

    print(
        f"{token}: {token_id}"
    )


# --------------------------------------------------
# Format one conversation
# --------------------------------------------------

def format_conversation(conversation):

    parts = ["[BOS]"]

    for message in conversation["messages"]:

        role = message["role"]
        text = message["text"].strip()

        if role == "prompter":

            parts.append("<|user|>")

        elif role == "assistant":

            parts.append("<|assistant|>")

        else:

            raise ValueError(
                f"Unknown role: {role}"
            )

        parts.append(text)

    parts.append("[EOS]")

    return "\n".join(parts)


# --------------------------------------------------
# Process JSONL
# --------------------------------------------------

def process_file(
    input_file,
    output_file,
    expected_count,
    name
):

    print()
    print(
        f"=== PROCESSING {name.upper()} ==="
    )

    count = 0
    total_tokens = 0
    min_tokens = None
    max_tokens = 0
    unknown_tokens = 0

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

            formatted = format_conversation(
                conversation
            )

            encoding = tokenizer.encode(
                formatted
            )

            token_ids = encoding.ids

            token_count = len(token_ids)

            unk_id = tokenizer.token_to_id(
                "[UNK]"
            )

            unknown_tokens += token_ids.count(
                unk_id
            )

            total_tokens += token_count

            if (
                min_tokens is None
                or token_count < min_tokens
            ):
                min_tokens = token_count

            if token_count > max_tokens:
                max_tokens = token_count

            output = {
                "tree_id":
                    conversation["tree_id"],

                "text":
                    formatted,

                "token_ids":
                    token_ids,

                "token_count":
                    token_count
            }

            target.write(
                json.dumps(
                    output,
                    ensure_ascii=False
                )
                + "\n"
            )

            count += 1

            if count % 5000 == 0:

                print(
                    f"Processed: "
                    f"{count}/{expected_count}"
                )

    average_tokens = (
        total_tokens / count
    )

    print()
    print(
        f"Conversations: {count}"
    )

    print(
        f"Total tokens: {total_tokens}"
    )

    print(
        f"Minimum tokens: {min_tokens}"
    )

    print(
        f"Maximum tokens: {max_tokens}"
    )

    print(
        f"Average tokens: "
        f"{average_tokens:.2f}"
    )

    print(
        f"[UNK] tokens: "
        f"{unknown_tokens}"
    )

    if count == expected_count:
        print(
            "PASS: Conversation count correct."
        )
    else:
        print(
            "FAIL: Conversation count mismatch."
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
# Process training data
# --------------------------------------------------

train_count = process_file(
    TRAIN_FILE,
    TRAIN_OUTPUT,
    17683,
    "training"
)


# --------------------------------------------------
# Process validation data
# --------------------------------------------------

validation_count = process_file(
    VALIDATION_FILE,
    VALIDATION_OUTPUT,
    1989,
    "validation"
)


# --------------------------------------------------
# Final verification
# --------------------------------------------------

print()
print("=" * 60)
print("TOKENIZATION PREPARATION COMPLETE")
print("=" * 60)

print(
    f"Training conversations: "
    f"{train_count}"
)

print(
    f"Validation conversations: "
    f"{validation_count}"
)

print(
    f"Total conversations: "
    f"{train_count + validation_count}"
)

if (
    train_count == 17683
    and validation_count == 1989
):

    print(
        "PASS: All conversations processed."
    )

else:

    print(
        "FAIL: Processing verification failed."
    )