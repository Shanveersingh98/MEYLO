from tokenizers import Tokenizer
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

TOKENIZER_PATH = Path(
    "D:/DEEP-LEARNING-PROJECT/tokenizer/tokenizer.json"
)

CORPUS_PATH = Path(
    "D:/DEEP-LEARNING-PROJECT/data/processed/tokenizer_corpus.txt"
)


# --------------------------------------------------
# Load tokenizer
# --------------------------------------------------

print("Loading tokenizer...")

tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

print("Tokenizer loaded successfully!")


# --------------------------------------------------
# Basic information
# --------------------------------------------------

vocab_size = tokenizer.get_vocab_size()

print()
print("=== TOKENIZER INFORMATION ===")
print(f"Vocabulary size: {vocab_size}")


# --------------------------------------------------
# Test normal text
# --------------------------------------------------

test_text = "Hello, how are you? I am learning artificial intelligence."

print()
print("=== ENCODING TEST ===")
print(f"Original text: {test_text}")

encoding = tokenizer.encode(test_text)

print(f"Token IDs: {encoding.ids}")
print(f"Tokens: {encoding.tokens}")
print(f"Number of tokens: {len(encoding.ids)}")


# --------------------------------------------------
# Decode tokens
# --------------------------------------------------

decoded_text = tokenizer.decode(encoding.ids)

print()
print("=== DECODING TEST ===")
print(f"Decoded text: {decoded_text}")


# --------------------------------------------------
# Special tokens
# --------------------------------------------------

print()
print("=== SPECIAL TOKENS ===")

special_tokens = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "<|user|>",
    "<|assistant|>",
]

for token in special_tokens:
    token_id = tokenizer.token_to_id(token)
    print(f"{token}: {token_id}")


# --------------------------------------------------
# Unknown token test
# --------------------------------------------------

unknown_test = "ZXQW987654321"

unknown_encoding = tokenizer.encode(unknown_test)

print()
print("=== UNKNOWN-TEXT TEST ===")
print(f"Text: {unknown_test}")
print(f"Tokens: {unknown_encoding.tokens}")
print(f"Token IDs: {unknown_encoding.ids}")


# --------------------------------------------------
# Corpus statistics
# --------------------------------------------------

print()
print("=== CORPUS CHECK ===")

lines = CORPUS_PATH.read_text(
    encoding="utf-8"
).splitlines()

print(f"Corpus messages: {len(lines)}")

sample_size = min(1000, len(lines))

total_tokens = 0
max_tokens = 0
min_tokens = None

for text in lines[:sample_size]:
    token_count = len(tokenizer.encode(text).ids)

    total_tokens += token_count

    if token_count > max_tokens:
        max_tokens = token_count

    if min_tokens is None or token_count < min_tokens:
        min_tokens = token_count


average_tokens = total_tokens / sample_size

print(f"Messages tested: {sample_size}")
print(f"Average tokens/message: {average_tokens:.2f}")
print(f"Minimum tokens/message: {min_tokens}")
print(f"Maximum tokens/message: {max_tokens}")


print()
print("=== TOKENIZER TESTING COMPLETE ===")