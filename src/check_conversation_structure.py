from datasets import load_from_disk
from collections import defaultdict

DATASET_PATH = "D:/DEEP-LEARNING-PROJECT/data/processed/oasst1_en"

print("Loading saved dataset...")
ds = load_from_disk(DATASET_PATH)

print(f"Messages: {len(ds)}")

# Group messages by conversation tree
trees = defaultdict(list)

for i in range(len(ds)):
    trees[ds[i]["message_tree_id"]].append(i)

print(f"Conversation trees: {len(trees)}")

valid_paths = 0
invalid_paths = 0

for indexes in trees.values():

    # Map message ID -> dataset index
    message_to_index = {
        ds[i]["message_id"]: i
        for i in indexes
    }

    # Find children for each message
    children = defaultdict(list)

    for i in indexes:
        parent_id = ds[i]["parent_id"]
        children[parent_id].append(i)

    # Find roots
    roots = children.get(None, [])

    # Walk every root-to-leaf path
    stack = [(root, [root]) for root in roots]

    while stack:
        current, path = stack.pop()
        child_indexes = children.get(ds[current]["message_id"], [])

        if not child_indexes:
            roles = [ds[i]["role"] for i in path]

            expected = [
                "prompter" if j % 2 == 0 else "assistant"
                for j in range(len(roles))
            ]

            if roles == expected:
                valid_paths += 1
            else:
                invalid_paths += 1

        else:
            for child in child_indexes:
                stack.append((child, path + [child]))

print()
print("========== RESULT ==========")
print(f"Valid conversation paths:   {valid_paths}")
print(f"Invalid conversation paths: {invalid_paths}")
print(f"Total checked:              {valid_paths + invalid_paths}")