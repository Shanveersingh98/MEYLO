import sys
import json
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
INPUT_TRAIN_FILE = PROJECT_DIR / "data" / "processed" / "conversation_paths" / "train_paths.jsonl"
INPUT_VAL_FILE = PROJECT_DIR / "data" / "processed" / "conversation_paths" / "validation_paths.jsonl"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
OUTPUT_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category1_oasst1_cleaned.jsonl"

TARGET_MAX_SEQUENCES = 3000
MIN_ASST_WORDS = 18
MIN_ASST_CHARS = 60
MIN_TOTAL_TOKENS = 50
MAX_TOTAL_TOKENS = 950
MAX_PATHS_PER_TREE = 2

CANNED_PATTERNS = [
    r"^as an ai\b",
    r"^as a large language model\b",
    r"^as an ai language model\b",
    r"^i am an ai\b",
    r"^i am an artificial intelligence\b",
    r"^i do not have personal\b",
    r"^i cannot assist with that\b",
    r"^i'm sorry, but as an ai\b",
    r"^i am sorry, but as an ai\b",
    r"^as a machine learning model\b",
    r"^i don't possess personal\b",
    r"^i am open assistant\b",
]
CANNED_RE = re.compile("|".join(CANNED_PATTERNS), re.IGNORECASE)


def normalize_prompt(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    return " ".join(cleaned.split())[:80]


def format_sequence_text(messages):
    parts = ["[BOS]"]
    for msg in messages:
        if msg["role"] == "prompter":
            parts.append("<|user|>")
            parts.append(msg["text"].strip())
        elif msg["role"] == "assistant":
            parts.append("<|assistant|>")
            parts.append(msg["text"].strip())
            parts.append("[EOS]")
    return "\n".join(parts)


def main():
    print("=" * 70)
    print("EXPERIMENT 2: FILTERING OASST1 CLEAN CONVERSATIONS (CATEGORY 1)")
    print("=" * 70)
    start_time = time.time()

    # 1. Load Tokenizer
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # 2. Load validation tree IDs for strict isolation assertion
    val_tree_ids = set()
    if INPUT_VAL_FILE.exists():
        with open(INPUT_VAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                val_tree_ids.add(item["tree_id"])
    print(f"Loaded {len(val_tree_ids)} validation tree IDs to guarantee strict isolation.")

    # 3. Stream and filter train paths
    total_scanned = 0
    trimmed_prompters = 0
    discard_reasons = Counter()
    tree_path_counts = defaultdict(int)
    seen_prompt_keys = set()
    valid_conversations = []

    with open(INPUT_TRAIN_FILE, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            total_scanned += 1
            conv = json.loads(line)
            tree_id = conv["tree_id"]

            # Strict isolation check
            if tree_id in val_tree_ids:
                discard_reasons["val_tree_overlap"] += 1
                continue

            messages = conv["messages"]

            # Trim trailing prompter messages
            while len(messages) > 0 and messages[-1]["role"] == "prompter":
                messages.pop()
                trimmed_prompters += 1

            if len(messages) == 0 or messages[-1]["role"] != "assistant":
                discard_reasons["malformed_turn_ending"] += 1
                continue

            # Check alternation: prompter, assistant, prompter, assistant...
            alternating = True
            for i, msg in enumerate(messages):
                expected = "prompter" if i % 2 == 0 else "assistant"
                if msg["role"] != expected:
                    alternating = False
                    break
            if not alternating:
                discard_reasons["non_alternating_roles"] += 1
                continue

            # Cap paths per tree for diversity
            if tree_path_counts[tree_id] >= MAX_PATHS_PER_TREE:
                discard_reasons["tree_path_cap_exceeded"] += 1
                continue

            # Assistant quality & canned check
            asst_failed = False
            for msg in messages:
                if msg["role"] == "assistant":
                    text = msg["text"].strip()
                    words = text.split()
                    if len(words) < MIN_ASST_WORDS or len(text) < MIN_ASST_CHARS:
                        asst_failed = True
                        discard_reasons["short_assistant_turn"] += 1
                        break
                    if CANNED_RE.search(text[:200]):
                        asst_failed = True
                        discard_reasons["canned_ai_disclaimer"] += 1
                        break
            if asst_failed:
                continue

            # Deduplication on initial user prompt
            prompt_key = normalize_prompt(messages[0]["text"])
            if prompt_key in seen_prompt_keys:
                discard_reasons["duplicate_prompt"] += 1
                continue

            # Format and check token count
            formatted = format_sequence_text(messages)
            enc = tokenizer.encode(formatted)
            token_count = len(enc.ids)

            if token_count < MIN_TOTAL_TOKENS:
                discard_reasons["below_min_tokens"] += 1
                continue
            if token_count > MAX_TOTAL_TOKENS:
                discard_reasons["above_max_tokens"] += 1
                continue

            # Valid candidate accepted
            seen_prompt_keys.add(prompt_key)
            tree_path_counts[tree_id] += 1
            valid_conversations.append({
                "tree_id": tree_id,
                "token_count": token_count,
                "turn_count": len(messages),
                "messages": messages
            })

            if len(valid_conversations) >= TARGET_MAX_SEQUENCES:
                break

    # 4. Summary & Verification
    print(f"\nTotal raw conversations scanned:     {total_scanned:,}")
    print(f"Discarded by filter:")
    for reason, count in discard_reasons.most_common():
        print(f"  - {reason:30}: {count:,}")
    print(f"\nRetained valid unique sequences:     {len(valid_conversations):,}")
    print(f"Distinct trees represented:          {len(tree_path_counts):,}")

    token_lengths = [c["token_count"] for c in valid_conversations]
    print(f"Token length min/mean/median/max:    {min(token_lengths)} / {sum(token_lengths)/len(token_lengths):.1f} / {sorted(token_lengths)[len(token_lengths)//2]} / {max(token_lengths)}")

    # 5. Write output JSONL
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for item in valid_conversations:
            record = {
                "category": "oasst1_cleaned",
                "tree_id": item["tree_id"],
                "token_count": item["token_count"],
                "messages": item["messages"]
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"Successfully wrote {len(valid_conversations):,} clean sequences to: {OUTPUT_FILE}")
    print(f"Completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
