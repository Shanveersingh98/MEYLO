import sys
import json
import random
import time
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
OUT_TRAIN_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category2_arithmetic_train.jsonl"
OUT_HELDOUT_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category2_arithmetic_heldout.jsonl"

SEED = 42
TARGET_TRAIN_COUNT = 1800
TARGET_HELDOUT_COUNT = 200

# Question templates
ADD_TEMPLATES_DIRECT = [
    "What is {a} + {b}?",
    "What is {a} plus {b}?",
    "Calculate {a} + {b}.",
    "Calculate the sum of {a} and {b}.",
    "What is the sum of {a} and {b}?",
    "Add {a} and {b}.",
    "Compute {a} + {b}.",
    "What do you get when you add {a} and {b}?",
    "Please evaluate: {a} + {b}",
    "{a} + {b} = ?",
    "Find the sum: {a} + {b}",
]

SUB_TEMPLATES_DIRECT = [
    "What is {a} - {b}?",
    "What is {a} minus {b}?",
    "Calculate {a} - {b}.",
    "Calculate the difference between {a} and {b}.",
    "What is the difference between {a} and {b}?",
    "Subtract {b} from {a}.",
    "Compute {a} - {b}.",
    "What do you get when you subtract {b} from {a}?",
    "Please evaluate: {a} - {b}",
    "{a} - {b} = ?",
    "Find the difference: {a} - {b}",
]

ADD_WORD_PROBLEMS = [
    ("If you have {a} apples and collect {b} more, how many apples do you have in total?",
     "You have {ans} apples in total ({a} + {b} = {ans})."),
    ("A bookstore sold {a} books on Friday and {b} books on Saturday. How many books were sold over the two days?",
     "Over the two days, a total of {ans} books were sold, since {a} + {b} = {ans}."),
    ("A cyclist rides {a} miles in the morning and {b} miles in the afternoon. What is the total distance traveled?",
     "The total distance traveled is {ans} miles ({a} + {b} = {ans})."),
    ("There are {a} students in Classroom A and {b} students in Classroom B. How many students are there altogether?",
     "Altogether there are {ans} students ({a} + {b} = {ans})."),
    ("A recipe requires {a} grams of flour and {b} grams of sugar. What is the combined weight of these ingredients?",
     "The combined weight is {ans} grams, because {a} + {b} = {ans}."),
    ("A warehouse received a shipment of {a} boxes in the morning and {b} boxes in the afternoon. How many boxes arrived?",
     "A total of {ans} boxes arrived at the warehouse ({a} + {b} = {ans})."),
    ("An investor bought {a} shares of company X and {b} shares of company Y. How many total shares were purchased?",
     "A total of {ans} shares were purchased ({a} + {b} = {ans})."),
]

SUB_WORD_PROBLEMS = [
    ("If you have {a} marbles and give away {b} of them, how many marbles do you have left?",
     "You have {ans} marbles left ({a} - {b} = {ans})."),
    ("A store had {a} shirts in stock and sold {b} during a sale. How many shirts remain in stock?",
     "There are {ans} shirts remaining in stock ({a} - {b} = {ans})."),
    ("A train started its journey with {a} passengers. At the first stop, {b} passengers got off. How many passengers remained on board?",
     "There were {ans} passengers remaining on board ({a} - {b} = {ans})."),
    ("You had ${a} in your wallet and spent ${b} on lunch. How much money do you have left?",
     "You have ${ans} left in your wallet ({a} - {b} = {ans})."),
    ("A full fuel tank has {a} liters. After driving, {b} liters were consumed. How many liters are left in the tank?",
     "There are {ans} liters remaining in the tank ({a} - {b} = {ans})."),
    ("A library had {a} copies of a novel, and {b} copies were checked out. How many copies are still on the shelf?",
     "There are {ans} copies still on the shelf ({a} - {b} = {ans})."),
    ("A runner planned a run of {a} meters and has already covered {b} meters. How many meters are left to run?",
     "There are {ans} meters left to run ({a} - {b} = {ans})."),
]

def make_add_answer(a: int, b: int, ans: int, style: str) -> str:
    assert a + b == ans
    if style == "direct":
        return f"{a} + {b} = {ans}."
    elif style == "sentence":
        return f"The sum of {a} and {b} is {ans}."
    elif style == "wordy":
        return f"{a} plus {b} equals {ans}."
    elif style == "steps":
        if b < 10:
            return f"To calculate {a} + {b}, we add {b} to {a}, giving {ans}. Therefore, {a} + {b} = {ans}."
        b_tens = (b // 10) * 10
        b_ones = b % 10
        if b_ones == 0:
            return f"To calculate {a} + {b}: adding {b_tens} to {a} gives {ans}. Therefore, {a} + {b} = {ans}."
        mid = a + b_tens
        return f"To calculate {a} + {b}: first add {b_tens} to {a} to get {mid}, then add {b_ones} to reach {ans}. Therefore, {a} + {b} = {ans}."
    return f"{a} + {b} = {ans}."

def make_sub_answer(a: int, b: int, ans: int, style: str) -> str:
    assert a - b == ans
    if style == "direct":
        return f"{a} - {b} = {ans}."
    elif style == "sentence":
        return f"The difference between {a} and {b} is {ans}."
    elif style == "wordy":
        return f"Subtracting {b} from {a} gives {ans}."
    elif style == "steps":
        if b < 10:
            return f"To calculate {a} - {b}, subtracting {b} from {a} leaves {ans}. Therefore, {a} - {b} = {ans}."
        b_tens = (b // 10) * 10
        b_ones = b % 10
        if b_ones == 0:
            return f"To calculate {a} - {b}: subtracting {b_tens} from {a} leaves {ans}. Therefore, {a} - {b} = {ans}."
        mid = a - b_tens
        return f"To calculate {a} - {b}: first subtract {b_tens} from {a} to get {mid}, then subtract {b_ones} to leave {ans}. Therefore, {a} - {b} = {ans}."
    return f"{a} - {b} = {ans}."


def generate_number_pairs(rng: random.Random):
    """
    Generates a pool of unique arithmetic triples: (a, b, op, tier)
    Tiers:
      1: 1-digit (0-9)
      2: 2-digit with 1-digit (10-99, 1-9)
      3: 2-digit with 2-digit (10-99, 10-99)
      4: 3-digit with 2-digit (100-999, 10-99)
      5: 3-digit with 3-digit (100-999, 100-999)
      6: 4-digit (1000-9999, 10-999)
    """
    all_triples = set()

    # Tier 1: 1-digit (cover ALL pairs 0..9)
    for a in range(10):
        for b in range(10):
            all_triples.add((a, b, "+", "tier1"))
            if a >= b:
                all_triples.add((a, b, "-", "tier1"))

    # Tier 2: 2-digit with 1-digit
    for a in range(10, 100):
        for b in range(1, 10):
            all_triples.add((a, b, "+", "tier2"))
            all_triples.add((a, b, "-", "tier2"))

    # Sample Tier 3: 2-digit with 2-digit
    t3_pairs = set()
    while len(t3_pairs) < 1200:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        op = rng.choice(["+", "-"])
        if op == "-" and a < b:
            a, b = b, a
        t3_pairs.add((a, b, op, "tier3"))
    all_triples.update(t3_pairs)

    # Sample Tier 4: 3-digit with 2-digit
    t4_pairs = set()
    while len(t4_pairs) < 800:
        a = rng.randint(100, 999)
        b = rng.randint(10, 99)
        op = rng.choice(["+", "-"])
        t4_pairs.add((a, b, op, "tier4"))
    all_triples.update(t4_pairs)

    # Sample Tier 5: 3-digit with 3-digit
    t5_pairs = set()
    while len(t5_pairs) < 800:
        a = rng.randint(100, 999)
        b = rng.randint(100, 999)
        op = rng.choice(["+", "-"])
        if op == "-" and a < b:
            a, b = b, a
        t5_pairs.add((a, b, op, "tier5"))
    all_triples.update(t5_pairs)

    # Sample Tier 6: 4-digit
    t6_pairs = set()
    while len(t6_pairs) < 400:
        a = rng.randint(1000, 9999)
        b = rng.randint(10, 999)
        op = rng.choice(["+", "-"])
        t6_pairs.add((a, b, op, "tier6"))
    all_triples.update(t6_pairs)

    return list(all_triples)


def build_examples():
    rng = random.Random(SEED)
    triples = generate_number_pairs(rng)
    rng.shuffle(triples)

    # Partition triples into train vs held-out to guarantee ZERO numerical pair leakage!
    # Explicitly guarantee that benchmark pair (2, 2, '+') is in TRAIN so model learns 2+2!
    heldout_triples = []
    train_triples = []

    # Benchmark pair check
    benchmark_triple = (2, 2, "+", "tier1")

    # Reserve held-out triples across tiers
    heldout_counts = Counter()
    for triple in triples:
        if triple == benchmark_triple:
            train_triples.append(triple)
            continue

        tier = triple[3]
        target_tier_heldout = {
            "tier1": 20,
            "tier2": 40,
            "tier3": 60,
            "tier4": 40,
            "tier5": 30,
            "tier6": 10
        }[tier]

        if heldout_counts[tier] < target_tier_heldout:
            heldout_triples.append(triple)
            heldout_counts[tier] += 1
        else:
            train_triples.append(triple)

    # Trim train triples to target size while preserving benchmark triple
    rng.shuffle(train_triples)
    if benchmark_triple in train_triples:
        train_triples.remove(benchmark_triple)
    train_triples = [benchmark_triple] + train_triples[:TARGET_TRAIN_COUNT - 1]
    heldout_triples = heldout_triples[:TARGET_HELDOUT_COUNT]

    # Verification: Disjoint sets of pairs
    train_pair_set = set((t[0], t[1], t[2]) for t in train_triples)
    heldout_pair_set = set((t[0], t[1], t[2]) for t in heldout_triples)
    overlap = train_pair_set.intersection(heldout_pair_set)
    assert len(overlap) == 0, f"Critical error: numerical pair leakage detected! Overlap: {overlap}"
    assert (2, 2, "+") in train_pair_set, "Critical error: (2, 2, '+') missing from train!"

    styles = ["direct", "sentence", "wordy", "steps"]

    # Build Training Examples
    train_examples = []
    for a, b, op, tier in train_triples:
        ans = a + b if op == "+" else a - b
        is_word_problem = rng.random() < 0.20  # 20% word problems
        if is_word_problem:
            wp_list = ADD_WORD_PROBLEMS if op == "+" else SUB_WORD_PROBLEMS
            q_template, a_template = rng.choice(wp_list)
            q = q_template.format(a=a, b=b)
            ans_text = a_template.format(a=a, b=b, ans=ans)
        else:
            style = rng.choice(styles)
            if op == "+":
                q = rng.choice(ADD_TEMPLATES_DIRECT).format(a=a, b=b)
                ans_text = make_add_answer(a, b, ans, style)
            else:
                q = rng.choice(SUB_TEMPLATES_DIRECT).format(a=a, b=b)
                ans_text = make_sub_answer(a, b, ans, style)

        train_examples.append({
            "category": "arithmetic",
            "tier": tier,
            "operation": op,
            "a": a,
            "b": b,
            "answer": ans,
            "messages": [
                {"role": "prompter", "text": q},
                {"role": "assistant", "text": ans_text}
            ]
        })

    # Build Held-Out Evaluation Examples
    heldout_examples = []
    for a, b, op, tier in heldout_triples:
        ans = a + b if op == "+" else a - b
        is_word_problem = rng.random() < 0.25
        if is_word_problem:
            wp_list = ADD_WORD_PROBLEMS if op == "+" else SUB_WORD_PROBLEMS
            q_template, a_template = rng.choice(wp_list)
            q = q_template.format(a=a, b=b)
            ans_text = a_template.format(a=a, b=b, ans=ans)
        else:
            style = rng.choice(["direct", "sentence", "steps"])
            if op == "+":
                q = rng.choice(ADD_TEMPLATES_DIRECT).format(a=a, b=b)
                ans_text = make_add_answer(a, b, ans, style)
            else:
                q = rng.choice(SUB_TEMPLATES_DIRECT).format(a=a, b=b)
                ans_text = make_sub_answer(a, b, ans, style)

        heldout_examples.append({
            "category": "arithmetic_heldout",
            "tier": tier,
            "operation": op,
            "a": a,
            "b": b,
            "answer": ans,
            "messages": [
                {"role": "prompter", "text": q},
                {"role": "assistant", "text": ans_text}
            ]
        })

    return train_examples, heldout_examples


def main():
    print("=" * 70)
    print("EXPERIMENT 2: GENERATING ARITHMETIC DATASET (CATEGORY 2)")
    print("=" * 70)
    start_time = time.time()

    train_examples, heldout_examples = build_examples()

    # Rigorous Validation Checks
    print(f"Validating arithmetic answers and constraints...")
    for ex in train_examples + heldout_examples:
        a = ex["a"]
        b = ex["b"]
        op = ex["operation"]
        expected_ans = a + b if op == "+" else a - b
        assert ex["answer"] == expected_ans, f"Calculation error: {a} {op} {b} != {ex['answer']}"
        assert str(expected_ans) in ex["messages"][1]["text"], f"Answer string missing in assistant turn: {ex}"

    # Deduplication check
    train_prompts = set(ex["messages"][0]["text"] for ex in train_examples)
    heldout_prompts = set(ex["messages"][0]["text"] for ex in heldout_examples)
    prompt_overlap = train_prompts.intersection(heldout_prompts)
    print(f"Train/Heldout exact prompt overlap: {len(prompt_overlap)} (Target: 0)")
    assert len(prompt_overlap) == 0, f"Prompt overlap found: {prompt_overlap}"

    # Write files
    OUT_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(OUT_HELDOUT_FILE, "w", encoding="utf-8") as f:
        for ex in heldout_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"\nArithmetic Train Examples Generated:   {len(train_examples):,}")
    print(f"Arithmetic Held-Out Examples:          {len(heldout_examples):,}")
    print(f"Written train to:   {OUT_TRAIN_FILE}")
    print(f"Written heldout to: {OUT_HELDOUT_FILE}")
    print(f"Completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
