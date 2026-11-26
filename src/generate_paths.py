from datasets import load_dataset
from collections import defaultdict
from build_paths import process_tree

print("Loading OASST1...")
ds = load_dataset("OpenAssistant/oasst1")

print("Filtering English, non-deleted messages...")
clean = ds["train"].filter(
    lambda x: x["lang"] == "en" and x["deleted"] == False
)

print(f"Clean messages: {len(clean)}")

print("Grouping messages by conversation tree...")
trees = defaultdict(list)

for i in range(len(clean)):
    tree_id = clean[i]["message_tree_id"]
    trees[tree_id].append(i)

print(f"Conversation trees: {len(trees)}")

print("Building conversation paths...")
all_paths = []

for count, indexes in enumerate(trees.values(), start=1):
    all_paths.extend(process_tree(indexes, clean))

    if count % 500 == 0:
        print(f"Processed {count}/{len(trees)} trees...")

print(f"Conversation paths: {len(all_paths)}")

print("Checking first path:")

for index in all_paths[0]:
    message = clean[index]
    print(f"{message['role'].upper()}: {message['text'][:200]}")