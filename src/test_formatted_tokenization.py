from pathlib import Path
from tokenizers import Tokenizer


# --------------------------------------------------
# Project paths
# --------------------------------------------------

TOKENIZER_PATH = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "tokenizer/tokenizer.json"
)

TRAIN_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/formatted_conversations/"
    "train.txt"
)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(
    str(TOKENIZER_PATH)
)

print("Tokenizer loaded successfully!")

print()
print("=" * 60)
print("TOKENIZER INFORMATION")
print("=" * 60)

print(
    f"Vocabulary size: "
    f"{tokenizer.get_vocab_size()}"
)


# --------------------------------------------------
# Check special tokens
# --------------------------------------------------

special_tokens = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "<|user|>",
    "<|assistant|>"
]

print()
print("=" * 60)
print("SPECIAL TOKEN CHECK")
print("=" * 60)

special_token_ids = {}

for token in special_tokens:

    token_id = tokenizer.token_to_id(token)

    special_token_ids[token] = token_id

    print(
        f"{token:15} -> {token_id}"
    )


# --------------------------------------------------
# Read first conversations
# --------------------------------------------------

print()
print("=" * 60)
print("READING FORMATTED DATA")
print("=" * 60)

with open(
    TRAIN_FILE,
    "r",
    encoding="utf-8"
) as f:

    content = f.read()


conversations = content.split(
    "===================="
)

conversations = [
    c.strip()
    for c in conversations
    if c.strip()
]

print(
    f"Conversations found: "
    f"{len(conversations)}"
)


# --------------------------------------------------
# Test first conversation
# --------------------------------------------------

test_conversation = conversations[0]

print()
print("=" * 60)
print("FIRST FORMATTED CONVERSATION")
print("=" * 60)

print(test_conversation)


# --------------------------------------------------
# Tokenize first conversation
# --------------------------------------------------

print()
print("=" * 60)
print("TOKENIZATION TEST")
print("=" * 60)

encoding = tokenizer.encode(
    test_conversation
)

print(
    f"Number of tokens: "
    f"{len(encoding.ids)}"
)

print()
print("First 50 token IDs:")

print(
    encoding.ids[:50]
)

print()
print("First 50 tokens:")

print(
    encoding.tokens[:50]
)


# --------------------------------------------------
# Check special tokens inside encoding
# --------------------------------------------------

print()
print("=" * 60)
print("SPECIAL TOKEN DETECTION")
print("=" * 60)

for token in special_tokens:

    token_id = tokenizer.token_to_id(token)

    occurrences = encoding.ids.count(
        token_id
    )

    print(
        f"{token:15} "
        f"ID={token_id:<5} "
        f"Occurrences={occurrences}"
    )


# --------------------------------------------------
# Check unknown tokens
# --------------------------------------------------

unk_id = tokenizer.token_to_id("[UNK]")

unk_count = encoding.ids.count(
    unk_id
)

print()
print("=" * 60)
print("UNKNOWN TOKEN CHECK")
print("=" * 60)

print(
    f"[UNK] occurrences: "
    f"{unk_count}"
)


# --------------------------------------------------
# Decode test
# --------------------------------------------------

decoded = tokenizer.decode(
    encoding.ids
)

print()
print("=" * 60)
print("DECODE TEST")
print("=" * 60)

print("Decoded text:")
print(decoded)


# --------------------------------------------------
# Token statistics
# --------------------------------------------------

print()
print("=" * 60)
print("TOKEN LENGTH STATISTICS")
print("=" * 60)

sample_size = min(
    1000,
    len(conversations)
)

total_tokens = 0
min_tokens = None
max_tokens = 0

for conversation in conversations[
    :sample_size
]:

    token_count = len(
        tokenizer.encode(
            conversation
        ).ids
    )

    total_tokens += token_count

    if (
        min_tokens is None
        or token_count < min_tokens
    ):
        min_tokens = token_count

    if token_count > max_tokens:
        max_tokens = token_count


average_tokens = (
    total_tokens / sample_size
)


print(
    f"Conversations tested: "
    f"{sample_size}"
)

print(
    f"Minimum tokens: "
    f"{min_tokens}"
)

print(
    f"Maximum tokens: "
    f"{max_tokens}"
)

print(
    f"Average tokens: "
    f"{average_tokens:.2f}"
)


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

if (
    tokenizer.get_vocab_size() == 8000
    and unk_count == 0
    and all(
        special_token_ids[token] is not None
        for token in special_tokens
    )
):

    print(
        "PASS: Tokenizer is ready for "
        "conversation tokenization."
    )

else:

    print(
        "FAIL: Tokenizer requires "
        "further checking."
    )