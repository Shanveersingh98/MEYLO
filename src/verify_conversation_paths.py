from pathlib import Path
import json


TRAIN_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_paths/"
    "train_paths.jsonl"
)

VALIDATION_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_paths/"
    "validation_paths.jsonl"
)


def verify_file(path, expected_count, name):

    print()
    print("=" * 60)
    print(f"VERIFYING {name.upper()}")
    print("=" * 60)

    count = 0
    invalid_json = 0
    missing_fields = 0
    empty_messages = 0
    invalid_roles = 0

    role_counts = {
        "prompter": 0,
        "assistant": 0
    }

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1
        ):

            count += 1

            # Check JSON
            try:
                conversation = json.loads(line)

            except json.JSONDecodeError:
                invalid_json += 1
                continue

            # Check top-level fields
            if (
                "tree_id" not in conversation
                or "messages" not in conversation
            ):
                missing_fields += 1
                continue

            messages = conversation["messages"]

            if not messages:
                empty_messages += 1
                continue

            # Check every message
            for message in messages:

                required = [
                    "message_id",
                    "role",
                    "text"
                ]

                if not all(
                    field in message
                    for field in required
                ):
                    missing_fields += 1

                role = message.get("role")

                if role not in role_counts:
                    invalid_roles += 1
                else:
                    role_counts[role] += 1

                if not message.get("text", "").strip():
                    empty_messages += 1

    print()
    print(f"Paths found:       {count}")
    print(f"Expected paths:    {expected_count}")

    print()
    print("=== CHECKS ===")

    print(
        f"JSON errors:       {invalid_json}"
    )

    print(
        f"Missing fields:    {missing_fields}"
    )

    print(
        f"Empty messages:    {empty_messages}"
    )

    print(
        f"Invalid roles:     {invalid_roles}"
    )

    print()
    print("=== ROLE COUNTS ===")

    print(
        f"Prompter messages: "
        f"{role_counts['prompter']}"
    )

    print(
        f"Assistant messages: "
        f"{role_counts['assistant']}"
    )

    passed = (
        count == expected_count
        and invalid_json == 0
        and missing_fields == 0
        and empty_messages == 0
        and invalid_roles == 0
    )

    print()

    if passed:
        print(
            f"PASS: {name} file is valid."
        )
    else:
        print(
            f"FAIL: {name} file has problems."
        )

    return count, passed


# --------------------------------------------------
# Verify both files
# --------------------------------------------------

train_count, train_pass = verify_file(
    TRAIN_FILE,
    17683,
    "training"
)

validation_count, validation_pass = verify_file(
    VALIDATION_FILE,
    1989,
    "validation"
)


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print("=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)

total = train_count + validation_count

print(f"Training paths:   {train_count}")
print(f"Validation paths: {validation_count}")
print(f"Total paths:      {total}")

print()

if (
    train_pass
    and validation_pass
    and total == 19672
):
    print(
        "PASS: ALL CONVERSATION PATHS ARE VALID."
    )
else:
    print(
        "FAIL: DATA VERIFICATION FAILED."
    )