from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

CORPUS_PATH = "D:/DEEP-LEARNING-PROJECT/data/processed/tokenizer_corpus.txt"
OUTPUT_DIR = Path("D:/DEEP-LEARNING-PROJECT/tokenizer")
OUTPUT_FILE = OUTPUT_DIR / "tokenizer.json"


# --------------------------------------------------
# Tokenizer configuration
# --------------------------------------------------

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]",
    "<|user|>",
    "<|assistant|>",
]


# --------------------------------------------------
# Create BPE tokenizer
# --------------------------------------------------

print("Creating BPE tokenizer...")

tokenizer = Tokenizer(
    BPE(unk_token="[UNK]")
)

tokenizer.pre_tokenizer = ByteLevel()

tokenizer.decoder = ByteLevelDecoder()


# --------------------------------------------------
# Create BPE trainer
# --------------------------------------------------

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=SPECIAL_TOKENS,
)


# --------------------------------------------------
# Train tokenizer
# --------------------------------------------------

print("Training tokenizer...")
print(f"Corpus: {CORPUS_PATH}")
print(f"Target vocabulary size: {VOCAB_SIZE}")
print(f"Minimum frequency: {MIN_FREQUENCY}")

tokenizer.train(
    files=[CORPUS_PATH],
    trainer=trainer,
)


# --------------------------------------------------
# Save tokenizer
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer.save(str(OUTPUT_FILE))


print()
print("Tokenizer training completed!")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
print(f"Saved to: {OUTPUT_FILE}")