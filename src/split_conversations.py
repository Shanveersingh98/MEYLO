from datasets import load_from_disk
from pathlib import Path
import random


# --------------------------------------------------
# Project paths
# --------------------------------------------------

DATASET_PATH = "D:/DEEP-LEARNING-PROJECT/data/processed/oasst1_en"
OUTPUT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/data/processed/conversation_split"
)


# --------------------------------------------------
# Split configuration
# --------------------------------------------------

VALIDATION_RATIO = 0.10
SEED = 42


# --------------------------------------------------
# Load saved cleaned dataset
# --------------------------------------------------

print("Loading saved cleaned dataset...")

ds = load_from_disk(DATASET_PATH)

print(f"Messages loaded: {len(ds)}")


# --------------------------------------------------
# Find unique conversation trees
# --------------------------------------------------

print()
print("Finding conversation trees...")

tree_ids = sorted(set(ds["message_tree_id"]))

print(f"Unique conversation trees: {len(tree_ids)}")


# --------------------------------------------------
# Shuffle trees reproducibly
# --------------------------------------------------

random.seed(SEED)

random.shuffle(tree_ids)


# --------------------------------------------------
# Calculate split
# --------------------------------------------------

validation_count = round(len(tree_ids) * VALIDATION_RATIO)

validation_tree_ids = set(tree_ids[:validation_count])
train_tree_ids = set(tree_ids[validation_count:])


print()
print("=== TREE SPLIT ===")
print(f"Training trees: {len(train_tree_ids)}")
print(f"Validation trees: {len(validation_tree_ids)}")


# --------------------------------------------------
# Create message masks
# --------------------------------------------------

train_indices = []
validation_indices = []

for i, tree_id in enumerate(ds["message_tree_id"]):

    if tree_id in validation_tree_ids:
        validation_indices.append(i)

    elif tree_id in train_tree_ids:
        train_indices.append(i)


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

print()
print("Creating training dataset...")

train_ds = ds.select(train_indices)

print("Creating validation dataset...")

validation_ds = ds.select(validation_indices)


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_path = OUTPUT_DIR / "train"
validation_path = OUTPUT_DIR / "validation"

print()
print("Saving training dataset...")

train_ds.save_to_disk(str(train_path))

print("Saving validation dataset...")

validation_ds.save_to_disk(str(validation_path))


# --------------------------------------------------
# Final information
# --------------------------------------------------

print()
print("=== SPLIT COMPLETE ===")

print(f"Training messages: {len(train_ds)}")
print(f"Validation messages: {len(validation_ds)}")

print(f"Training path: {train_path}")
print(f"Validation path: {validation_path}")