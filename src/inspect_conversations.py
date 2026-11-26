from datasets import load_from_disk
from pathlib import Path

TRAIN_PATH = (
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_split/train"
)

PATHS_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_paths/train_paths.txt"
)

print("Loading training dataset...")
ds = load_from_disk(TRAIN_PATH)

print(f"Messages loaded: {len(ds)}")

print()
print("Reading conversation paths...")

with open(PATHS_FILE, "r", encoding="utf-8") as f:
    content = f.read()

conversations = content.split("==========")

# Remove empty conversations
conversations = [
    conv.strip()
    for conv in conversations
    if conv.strip()
]

print(f"Conversations found: {len(conversations)}")

print()
print("=" * 70)
print("FIRST 5 CONVERSATIONS")
print("=" * 70)

for number, conversation in enumerate(
    conversations[:5],
    start=1
):

    messages = conversation.splitlines()

    print()
    print(f"CONVERSATION {number}")
    print("-" * 70)
    print(f"Messages: {len(messages)}")
    print()

    for i, message in enumerate(messages, start=1):

        print(f"Message {i}:")
        print(message)
        print()


print()
print("=" * 70)
print("PATH LENGTH STATISTICS")
print("=" * 70)

lengths = []

for conversation in conversations:
    lengths.append(
        len(conversation.splitlines())
    )

print(f"Total conversations: {len(lengths)}")
print(f"Shortest conversation: {min(lengths)} messages")
print(f"Longest conversation: {max(lengths)} messages")
print(
    f"Average conversation length: "
    f"{sum(lengths) / len(lengths):.2f} messages"
)

print()
print("=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)