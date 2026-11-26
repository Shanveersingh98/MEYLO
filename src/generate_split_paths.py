from datasets import load_from_disk
from pathlib import Path
import sys
import json


# --------------------------------------------------
# Project paths
# --------------------------------------------------

TRAIN_PATH = (
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_split/train"
)

VALIDATION_PATH = (
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_split/validation"
)

OUTPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_paths"
)


# --------------------------------------------------
# Load our tested path-building helper
# --------------------------------------------------

sys.path.insert(
    0,
    "D:/DEEP-LEARNING-PROJECT/src"
)

from build_paths import process_tree


# --------------------------------------------------
# Generate conversation paths
# --------------------------------------------------

def generate_paths(dataset, name):

    print()
    print(f"=== PROCESSING {name.upper()} ===")
    print(f"Messages: {len(dataset)}")

    tree_groups = {}

    # Group indexes by conversation tree
    for i, tree_id in enumerate(
        dataset["message_tree_id"]
    ):
        tree_groups.setdefault(
            tree_id,
            []
        ).append(i)

    print(
        f"Conversation trees: "
        f"{len(tree_groups)}"
    )

    all_paths = []

    for count, indexes in enumerate(
        tree_groups.values(),
        start=1
    ):

        paths = process_tree(
            indexes,
            dataset
        )

        all_paths.extend(paths)

        if count % 500 == 0:
            print(
                f"Processed trees: "
                f"{count}/{len(tree_groups)}"
            )

    print(
        f"Generated paths: "
        f"{len(all_paths)}"
    )

    return all_paths


# --------------------------------------------------
# Save paths as JSONL
# --------------------------------------------------

def save_paths(
    paths,
    dataset,
    output_file
):

    print()
    print(
        f"Saving {len(paths)} paths..."
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for path in paths:

            messages = []

            for index in path:

                messages.append({
                    "message_id":
                        dataset[index]["message_id"],

                    "role":
                        dataset[index]["role"],

                    "text":
                        dataset[index]["text"]
                })

            conversation = {
                "tree_id":
                    dataset[path[0]][
                        "message_tree_id"
                    ],

                "messages":
                    messages
            }

            f.write(
                json.dumps(
                    conversation,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"Saved: {output_file}"
    )


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading training dataset...")
train = load_from_disk(TRAIN_PATH)

print("Loading validation dataset...")
validation = load_from_disk(
    VALIDATION_PATH
)


# --------------------------------------------------
# Generate paths
# --------------------------------------------------

train_paths = generate_paths(
    train,
    "training"
)

validation_paths = generate_paths(
    validation,
    "validation"
)


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Output files
# --------------------------------------------------

train_output = (
    OUTPUT_DIR / "train_paths.jsonl"
)

validation_output = (
    OUTPUT_DIR / "validation_paths.jsonl"
)


# --------------------------------------------------
# Save
# --------------------------------------------------

save_paths(
    train_paths,
    train,
    train_output
)

save_paths(
    validation_paths,
    validation,
    validation_output
)


# --------------------------------------------------
# Final verification
# --------------------------------------------------

total_paths = (
    len(train_paths)
    + len(validation_paths)
)

print()
print("=" * 60)
print("PATH GENERATION COMPLETE")
print("=" * 60)

print(
    f"Training paths:   "
    f"{len(train_paths)}"
)

print(
    f"Validation paths: "
    f"{len(validation_paths)}"
)

print(
    f"Total paths:      "
    f"{total_paths}"
)

print()
print(
    "Expected total:   19672"
)

if total_paths == 19672:

    print(
        "PASS: Path count is correct."
    )

else:

    print(
        "FAIL: Path count mismatch!"
    )