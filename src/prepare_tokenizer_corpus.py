from datasets import load_from_disk
from pathlib import Path

# Project locations
DATASET_PATH = "D:/DEEP-LEARNING-PROJECT/data/processed/oasst1_en"
OUTPUT_PATH = Path("D:/DEEP-LEARNING-PROJECT/data/processed/tokenizer_corpus.txt")

print("Loading saved cleaned dataset...")
ds = load_from_disk(DATASET_PATH)

print(f"Messages loaded: {len(ds)}")

print("Preparing tokenizer corpus...")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for text in ds["text"]:
        # Keep each message as one training example.
        # Replace line breaks only in this tokenizer-training copy.
        text = text.replace("\r", " ").replace("\n", " ")
        text = " ".join(text.split())

        if text:
            f.write(text + "\n")

print("Tokenizer corpus created successfully!")
print(f"Saved to: {OUTPUT_PATH}")