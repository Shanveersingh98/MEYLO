from datasets import load_from_disk


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ORIGINAL_PATH = "D:/DEEP-LEARNING-PROJECT/data/processed/oasst1_en"

TRAIN_PATH = (
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_split/train"
)

VALIDATION_PATH = (
    "D:/DEEP-LEARNING-PROJECT/"
    "data/processed/conversation_split/validation"
)


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("Loading datasets...")

original = load_from_disk(ORIGINAL_PATH)
train = load_from_disk(TRAIN_PATH)
validation = load_from_disk(VALIDATION_PATH)


print()
print("=== MESSAGE COUNTS ===")

print(f"Original:    {len(original)}")
print(f"Training:    {len(train)}")
print(f"Validation:  {len(validation)}")

combined_count = len(train) + len(validation)

print(f"Combined:    {combined_count}")


# --------------------------------------------------
# Check message count
# --------------------------------------------------

print()
print("=== MESSAGE COUNT CHECK ===")

if combined_count == len(original):
    print("PASS: Message counts match.")
else:
    print("FAIL: Message counts do not match!")


# --------------------------------------------------
# Find conversation trees
# --------------------------------------------------

print()
print("Finding tree IDs...")

original_trees = set(original["message_tree_id"])
train_trees = set(train["message_tree_id"])
validation_trees = set(validation["message_tree_id"])


print()
print("=== TREE COUNTS ===")

print(f"Original trees:    {len(original_trees)}")
print(f"Training trees:    {len(train_trees)}")
print(f"Validation trees:  {len(validation_trees)}")


# --------------------------------------------------
# Check overlap
# --------------------------------------------------

overlap = train_trees.intersection(validation_trees)


print()
print("=== TREE OVERLAP CHECK ===")

print(f"Overlapping trees: {len(overlap)}")


if len(overlap) == 0:
    print("PASS: No conversation tree appears in both splits.")
else:
    print("FAIL: Data leakage detected!")


# --------------------------------------------------
# Check tree count
# --------------------------------------------------

combined_trees = train_trees.union(validation_trees)


print()
print("=== TREE COVERAGE CHECK ===")

if combined_trees == original_trees:
    print("PASS: All original trees are accounted for.")
else:
    print("FAIL: Some trees are missing.")


# --------------------------------------------------
# Final result
# --------------------------------------------------

print()
print("=== VERIFICATION COMPLETE ===")