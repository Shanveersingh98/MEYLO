import sys
import json
import ast
import random
import time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
RAW_OUT_DIR = PROJECT_DIR / "data" / "ultimate_improvement" / "curriculum_raw"
BIN_OUT_DIR = PROJECT_DIR / "data" / "ultimate_improvement" / "training_sequences"
EVAL_DIR = PROJECT_DIR / "data" / "ultimate_improvement" / "eval_suite"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"

RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)
BIN_OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = random.Random(SEED)
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

# -------------------------------------------------------------
# 0. Load all evaluation prompts and numbers to prevent leakage
# -------------------------------------------------------------
eval_prompts = set()
eval_pairs = set()

for ep_file in EVAL_DIR.glob("*.json*"):
    with open(ep_file, "r", encoding="utf-8") as f:
        if ep_file.suffix == ".json":
            items = json.load(f)
        else:
            items = [json.loads(l) for l in f if l.strip()]
    for it in items:
        p = it.get("prompt")
        if not p and "turns" in it:
            p = it["turns"][0]["user"]
        if p:
            eval_prompts.add(p.strip().lower())
        if "a" in it and "b" in it and "operation" in it:
            eval_pairs.add((it["a"], it["b"], it["operation"]))

print(f"Loaded {len(eval_prompts)} evaluation prompts to protect from leakage.")
print(f"Loaded {len(eval_pairs)} arithmetic evaluation pairs to protect from leakage.")


# =============================================================
# 1. GENERATE CHAIN-OF-THOUGHT ARITHMETIC (1,500 items)
# =============================================================
print("\nGenerating Chain-of-Thought Arithmetic Dataset...")

cot_arithmetic_items = []

# Templates for single-step with explicit scratchpad reasoning
single_op_templates = [
    ("Calculate {a} + {b}.", "add"),
    ("What is {a} plus {b}?", "add"),
    ("Compute the sum of {a} and {b}.", "add"),
    ("Calculate {a} - {b}.", "sub"),
    ("What is {a} minus {b}?", "sub"),
    ("Subtract {b} from {a}.", "sub"),
]

# Word problem templates for multi-step arithmetic
multi_step_templates = [
    ("A store had {a} items. They sold {b} items in the morning and {c} items in the afternoon. How many items are left?",
     "{a} - {b} - {c}", lambda a, b, c: a - b - c,
     "1. Calculate items remaining after the morning sale: {a} - {b} = {mid}.\n2. Calculate items remaining after the afternoon sale: {mid} - {c} = {ans}.\nTherefore, there are {ans} items left."),
    ("You have ${a} in your wallet. You spend ${b} on groceries and earn ${c} doing chores. How much money do you have now?",
     "{a} - {b} + {c}", lambda a, b, c: a - b + c,
     "1. First, subtract the money spent: ${a} - ${b} = ${mid}.\n2. Next, add the money earned: ${mid} + ${c} = ${ans}.\nTherefore, you currently have ${ans}."),
    ("There are {a} students in the auditorium. {b} students leave to go to class, and {c} new students arrive. How many students are in the auditorium now?",
     "{a} - {b} + {c}", lambda a, b, c: a - b + c,
     "1. Calculate students remaining after departure: {a} - {b} = {mid}.\n2. Add the arriving students: {mid} + {c} = {ans}.\nTherefore, there are {ans} students in the auditorium."),
    ("A farmer picks {a} oranges on Monday and {b} oranges on Tuesday. If he packs {c} oranges into crates to sell, how many oranges remain?",
     "{a} + {b} - {c}", lambda a, b, c: a + b - c,
     "1. Find total oranges picked: {a} + {b} = {mid}.\n2. Subtract oranges packed: {mid} - {c} = {ans}.\nTherefore, {ans} oranges remain."),
    ("A cyclist travels {a} miles on Friday, {b} miles on Saturday, and {c} miles on Sunday. What is the total distance traveled over the three days?",
     "{a} + {b} + {c}", lambda a, b, c: a + b + c,
     "1. Add Friday and Saturday mileage: {a} + {b} = {mid}.\n2. Add Sunday mileage: {mid} + {c} = {ans}.\nTherefore, the total distance traveled is {ans} miles.")
]

# Generate multi-digit addition and subtraction CoT
while len(cot_arithmetic_items) < 800:
    op = rng.choice(["+", "-"])
    tier = rng.choice(["2digit", "3digit", "4digit"])
    if tier == "2digit":
        a = rng.randint(10, 99)
        b = rng.randint(5, 99)
    elif tier == "3digit":
        a = rng.randint(100, 999)
        b = rng.randint(20, 800)
    else:
        a = rng.randint(1000, 5000)
        b = rng.randint(100, 3000)

    if op == "-":
        if b > a:
            a, b = b, a
        ans = a - b
    else:
        ans = a + b

    if (a, b, op) in eval_pairs:
        continue

    tpl, name = rng.choice(single_op_templates)
    if (op == "+" and name != "add") or (op == "-" and name != "sub"):
        continue

    prompt = tpl.format(a=a, b=b)
    if prompt.lower() in eval_prompts:
        continue

    if op == "+":
        if b < 10:
            thought = f"To calculate {a} + {b} step by step:\n- Adding {b} directly to {a} gives {ans}."
        else:
            b_tens = (b // 10) * 10
            b_ones = b % 10
            mid = a + b_tens
            thought = f"To calculate {a} + {b} step by step:\n1. First add the tens: {a} + {b_tens} = {mid}.\n2. Next add the ones: {mid} + {b_ones} = {ans}."
        resp = f"{thought}\nTherefore, {a} + {b} = {ans}."
    else:
        if b < 10:
            thought = f"To calculate {a} - {b} step by step:\n- Subtracting {b} directly from {a} leaves {ans}."
        else:
            b_tens = (b // 10) * 10
            b_ones = b % 10
            mid = a - b_tens
            thought = f"To calculate {a} - {b} step by step:\n1. First subtract the tens: {a} - {b_tens} = {mid}.\n2. Next subtract the ones: {mid} - {b_ones} = {ans}."
        resp = f"{thought}\nTherefore, {a} - {b} = {ans}."

    cot_arithmetic_items.append({
        "messages": [
            {"role": "prompter", "text": prompt},
            {"role": "assistant", "text": resp}
        ]
    })

# Generate multi-step arithmetic word problems with CoT
while len(cot_arithmetic_items) < 1500:
    tpl_q, formula, fn, tpl_resp = rng.choice(multi_step_templates)
    a = rng.randint(20, 150)
    b = rng.randint(5, min(a, 60))
    c = rng.randint(5, 50)
    ans = fn(a, b, c)
    if ans < 0:
        continue

    prompt = tpl_q.format(a=a, b=b, c=c)
    if prompt.lower() in eval_prompts:
        continue

    if formula == "{a} - {b} - {c}":
        mid = a - b
    elif formula == "{a} - {b} + {c}":
        mid = a - b
    elif formula == "{a} + {b} - {c}":
        mid = a + b
    elif formula == "{a} + {b} + {c}":
        mid = a + b

    resp = tpl_resp.format(a=a, b=b, c=c, mid=mid, ans=ans)
    cot_arithmetic_items.append({
        "messages": [
            {"role": "prompter", "text": prompt},
            {"role": "assistant", "text": resp}
        ]
    })

print(f"Generated {len(cot_arithmetic_items)} verified Chain-of-Thought arithmetic samples.")


# =============================================================
# 2. GENERATE VERIFIED ALGORITHMIC PYTHON (1,200 items)
# =============================================================
print("\nGenerating Verified Algorithmic Python Dataset...")

py_seed_tasks = [
    {
        "func_name": "is_even_number",
        "doc": "Return True if integer n is even, False otherwise.",
        "params": "n",
        "body": "return n % 2 == 0",
        "tests": [((4,), True), ((7,), False), ((0,), True), ((-2,), True)]
    },
    {
        "func_name": "cube_number",
        "doc": "Return the cube of number x (x raised to power 3).",
        "params": "x",
        "body": "return x ** 3",
        "tests": [((2,), 8), ((3,), 27), ((-1,), -1), ((0,), 0)]
    },
    {
        "func_name": "sum_list_numbers",
        "doc": "Return the sum of all elements in list nums.",
        "params": "nums",
        "body": "total = 0\n    for x in nums:\n        total += x\n    return total",
        "tests": [(([1, 2, 3],), 6), (([],), 0), (([-5, 5],), 0)]
    },
    {
        "func_name": "count_occurrences",
        "doc": "Count how many times target appears in items list.",
        "params": "items, target",
        "body": "count = 0\n    for item in items:\n        if item == target:\n            count += 1\n    return count",
        "tests": [(([1, 2, 2, 3], 2), 2), ((['a', 'b'], 'c'), 0)]
    },
    {
        "func_name": "get_first_element",
        "doc": "Return the first element of lst, or None if lst is empty.",
        "params": "lst",
        "body": "if not lst:\n        return None\n    return lst[0]",
        "tests": [(([10, 20],), 10), (([],), None)]
    },
    {
        "func_name": "get_last_element",
        "doc": "Return the last element of lst, or None if lst is empty.",
        "params": "lst",
        "body": "if not lst:\n        return None\n    return lst[-1]",
        "tests": [(([10, 20],), 20), (([],), None)]
    },
    {
        "func_name": "square_all",
        "doc": "Return a new list containing squares of all numbers in nums.",
        "params": "nums",
        "body": "return [x * x for x in nums]",
        "tests": [(([1, 2, 3],), [1, 4, 9]), (([],), [])]
    },
    {
        "func_name": "filter_even_numbers",
        "doc": "Return a list of all even numbers from the given list nums.",
        "params": "nums",
        "body": "return [x for x in nums if x % 2 == 0]",
        "tests": [(([1, 2, 3, 4],), [2, 4]), (([1, 3],), [])]
    },
    {
        "func_name": "maximum_of_two",
        "doc": "Return the larger of two numbers a and b.",
        "params": "a, b",
        "body": "if a >= b:\n        return a\n    return b",
        "tests": [((10, 20), 20), ((5, 5), 5), ((-1, -10), -1)]
    },
    {
        "func_name": "minimum_of_two",
        "doc": "Return the smaller of two numbers a and b.",
        "params": "a, b",
        "body": "if a <= b:\n        return a\n    return b",
        "tests": [((10, 20), 10), ((5, 5), 5), ((-1, -10), -10)]
    },
    {
        "func_name": "string_length",
        "doc": "Return the number of characters in string s.",
        "params": "s",
        "body": "return len(s)",
        "tests": [(("hello",), 5), (("",), 0)]
    },
    {
        "func_name": "to_uppercase",
        "doc": "Return the string s converted to uppercase.",
        "params": "s",
        "body": "return s.upper()",
        "tests": [(("abc",), "ABC"), (("Python",), "PYTHON")]
    },
    {
        "func_name": "to_lowercase",
        "doc": "Return the string s converted to lowercase.",
        "params": "s",
        "body": "return s.lower()",
        "tests": [(("HELLO",), "hello"), (("World",), "world")]
    },
    {
        "func_name": "is_palindrome_simple",
        "doc": "Return True if string s reads the same backward as forward.",
        "params": "s",
        "body": "clean = s.lower().replace(' ', '')\n    return clean == clean[::-1]",
        "tests": [(("radar",), True), (("hello",), False), (("Racecar",), True)]
    },
    {
        "func_name": "join_words",
        "doc": "Join a list of words with spaces into a single string.",
        "params": "words",
        "body": "return ' '.join(words)",
        "tests": [((["Hello", "world"],), "Hello world"), (([],), "")]
    },
    {
        "func_name": "absolute_value",
        "doc": "Return the absolute value of number n without using built-in abs.",
        "params": "n",
        "body": "if n < 0:\n        return -n\n    return n",
        "tests": [((-5,), 5), ((12,), 12), ((0,), 0)]
    },
    {
        "func_name": "is_negative_number",
        "doc": "Return True if n is strictly less than zero, False otherwise.",
        "params": "n",
        "body": "return n < 0",
        "tests": [((-3,), True), ((5,), False), ((0,), False)]
    },
    {
        "func_name": "sum_of_first_n_integers",
        "doc": "Return the sum of positive integers from 1 to n.",
        "params": "n",
        "body": "if n <= 0:\n        return 0\n    return (n * (n + 1)) // 2",
        "tests": [((5,), 15), ((10,), 55), ((0,), 0)]
    },
    {
        "func_name": "contains_element",
        "doc": "Return True if item is present in items list, False otherwise.",
        "params": "items, item",
        "body": "return item in items",
        "tests": [(([1, 2, 3], 2), True), (([1, 2, 3], 9), False)]
    },
    {
        "func_name": "calculate_perimeter_rectangle",
        "doc": "Calculate the perimeter of a rectangle with length and width.",
        "params": "length, width",
        "body": "return 2 * (length + width)",
        "tests": [((5, 10), 30), ((3, 7), 20)]
    }
]

# Verify all seed functions with AST and sandbox execution
for task in py_seed_tasks:
    code = f"def {task['func_name']}({task['params']}):\n    \"\"\"{task['doc']}\"\"\"\n    {task['body']}"
    ast.parse(code)
    scope = {}
    exec(code, scope)
    fn = scope[task["func_name"]]
    for args, exp in task["tests"]:
        res = fn(*args)
        assert res == exp, f"Function {task['func_name']} failed test {args}: {res} != {exp}"

print(f"Verified {len(py_seed_tasks)} core Python functions with 100% AST parse and unit test execution.")

py_prompt_variations = [
    "Write a Python function named {func_name}({params}) that {doc_lower}",
    "Create a Python function called {func_name}({params}) to {doc_lower}",
    "Please implement a Python function {func_name}({params}): {doc_lower}",
    "How do you write a Python function named {func_name} taking ({params}) that {doc_lower}?",
    "Define a Python function named {func_name}({params}) with docstring that {doc_lower}"
]

python_train_items = []
while len(python_train_items) < 1200:
    task = rng.choice(py_seed_tasks)
    doc_lower = task["doc"][0].lower() + task["doc"][1:].rstrip(".")
    tpl = rng.choice(py_prompt_variations)
    prompt = tpl.format(func_name=task["func_name"], params=task["params"], doc_lower=doc_lower)
    if prompt.lower() in eval_prompts:
        continue

    code_block = (
        f"Here is the implementation in Python:\n\n"
        f"```python\n"
        f"def {task['func_name']}({task['params']}):\n"
        f"    \"\"\"\n"
        f"    {task['doc']}\n"
        f"    \"\"\"\n"
        f"    {task['body']}\n"
        f"```"
    )

    python_train_items.append({
        "messages": [
            {"role": "prompter", "text": prompt},
            {"role": "assistant", "text": code_block}
        ]
    })

print(f"Generated {len(python_train_items)} verified Python algorithmic training samples.")


# =============================================================
# 3. GENERATE INSTRUCTION FOLLOWING & STRUCTURED DATA (600 items)
# =============================================================
print("\nGenerating Instruction Following & Structured Formatting Dataset...")

instruction_structured_items = []

instruction_seeds = [
    ("Reply with only YES or NO: Is 10 greater than 5?", "YES."),
    ("Reply with only YES or NO: Is ice hot?", "NO."),
    ("Reply with only YES or NO: Does water freeze at 0 degrees Celsius?", "YES."),
    ("Reply with only YES or NO: Can dogs fly naturally?", "NO."),
    ("Reply with only TRUE or FALSE: The Earth revolves around the Sun.", "TRUE."),
    ("Reply with only TRUE or FALSE: Gold is lighter than plastic.", "FALSE."),
    ("Reply with only TRUE or FALSE: Python is an interpreted language.", "TRUE."),
    ("State only the number: What is 10 plus 10?", "20"),
    ("State only the number: What is 50 minus 25?", "25"),
    ("State only the number: How many days are in a week?", "7"),
    ("State only the number: How many hours are in a day?", "24"),
    ("State only the number: What is the square of 9?", "81"),
    ("Answer in exactly three words: What color is grass?", "Grass is green."),
    ("Answer in exactly three words: What do birds have?", "Birds have wings."),
    ("Answer in exactly three words: What is snow like?", "Snow is cold."),
    ("Answer completely in UPPERCASE letters: What is the capital of Spain?", "MADRID"),
    ("Answer completely in UPPERCASE letters: What planet do humans live on?", "EARTH"),
    ("Provide two benefits of regular reading as bullet points starting with '-':",
     "- Improves vocabulary and language comprehension.\n- Enhances focus, memory, and cognitive ability."),
    ("List 3 primary colors as a numbered list from 1 to 3:",
     "1. Red\n2. Blue\n3. Yellow"),
    ("List 3 states of matter as a numbered list from 1 to 3:",
     "1. Solid\n2. Liquid\n3. Gas"),
    ("Do not use the letter 'e' in your response. Name a common fruit:",
     "Banana"),
    ("Do not mention the word 'computer' in your response. What is a smartphone?",
     "A smartphone is a handheld mobile device that makes phone calls, browses the web, and runs software applications."),
    ("Provide a JSON object with keys 'status' (value 'success') and 'message' (value 'complete'):",
     '{"status": "success", "message": "complete"}'),
    ("Output a JSON object with keys 'item' ('laptop'), 'price' (999), and 'in_stock' (true):",
     '{"item": "laptop", "price": 999, "in_stock": true}'),
    ("Output a JSON array containing the strings 'red', 'green', and 'blue':",
     '["red", "green", "blue"]'),
    ("Create a Markdown table with columns 'Planet' and 'Type' listing Earth (Terrestrial) and Jupiter (Gas Giant):",
     "| Planet | Type |\n| --- | --- |\n| Earth | Terrestrial |\n| Jupiter | Gas Giant |"),
    ("Create a Markdown table with columns 'Language' and 'Creator' listing Python (Guido van Rossum):",
     "| Language | Creator |\n| --- | --- |\n| Python | Guido van Rossum |")
]

while len(instruction_structured_items) < 600:
    q, ans = rng.choice(instruction_seeds)
    if q.lower() in eval_prompts:
        continue
    instruction_structured_items.append({
        "messages": [
            {"role": "prompter", "text": q},
            {"role": "assistant", "text": ans}
        ]
    })

print(f"Generated {len(instruction_structured_items)} instruction and formatting samples.")


# =============================================================
# 4. GENERATE FACTUAL QA & MULTI-TURN DIALOGUE (400 items)
# =============================================================
print("\nGenerating Factual QA & Multi-Turn Dialogue Dataset...")

factual_dialogue_items = []

factual_seeds = [
    ("What is the capital of Italy?", "The capital of Italy is Rome."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of the United Kingdom?", "The capital of the United Kingdom is London."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest and deepest ocean on Earth."),
    ("What is the hardest naturally occurring mineral on Earth?", "Diamond is the hardest known natural mineral on Earth."),
    ("What gas is most abundant in Earth's atmosphere?", "Nitrogen is the most abundant gas in Earth's atmosphere, making up approximately 78% of dry air."),
    ("Who invented the World Wide Web in 1989?", "Tim Berners-Lee invented the World Wide Web in 1989."),
    ("In physics, what does the letter 'c' represent in Einstein's equation E = mc^2?", "In the equation E = mc^2, 'c' represents the speed of light in a vacuum."),
    ("What is the basic unit of life in all living organisms?", "The cell is the basic structural and functional unit of all living organisms."),
    ("Which gas is produced by humans during respiration and exhaled?", "Carbon dioxide (CO2) is produced by cellular respiration and exhaled by humans.")
]

for q, a in factual_seeds:
    if q.lower() in eval_prompts:
        continue
    factual_dialogue_items.append({
        "messages": [
            {"role": "prompter", "text": q},
            {"role": "assistant", "text": a}
        ]
    })

# Add multi-turn dialogue memory sequences
names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Sam", "Chris"]
colors = ["blue", "green", "purple", "orange", "yellow", "red", "cyan"]
cities = ["Tokyo", "London", "Sydney", "Toronto", "Berlin", "San Francisco"]

for _ in range(300):
    nm = rng.choice(names)
    col = rng.choice(colors)
    city = rng.choice(cities)
    dialogue_type = rng.choice(["name_and_city", "color_preference"])
    if dialogue_type == "name_and_city":
        factual_dialogue_items.append({
            "messages": [
                {"role": "prompter", "text": f"Hello! My name is {nm} and I live in {city}."},
                {"role": "assistant", "text": f"Hello {nm}! It's a pleasure to meet you. How is life in {city} today? How can I help you?"},
                {"role": "prompter", "text": "Can you remind me what my name is and where I live?"},
                {"role": "assistant", "text": f"Your name is {nm}, and you live in {city}."}
            ]
        })
    else:
        factual_dialogue_items.append({
            "messages": [
                {"role": "prompter", "text": f"My favorite color is {col}."},
                {"role": "assistant", "text": f"{col.capitalize()} is a great color! What do you like about it?"},
                {"role": "prompter", "text": "What is my favorite color?"},
                {"role": "assistant", "text": f"Your favorite color is {col}."}
            ]
        })

print(f"Generated {len(factual_dialogue_items)} factual and multi-turn dialogue samples.")


# =============================================================
# 5. ASSEMBLE COMBINED TRAINING & HELD-OUT CURRICULUM
# =============================================================
print("\nAssembling Combined Ultimate Curriculum...")

# Load cleaned OASST1
with open(PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category1_oasst1_cleaned.jsonl", "r", encoding="utf-8") as f:
    oasst1_items = [json.loads(l) for l in f if l.strip()]

# Take 2,500 OASST1 sequences to anchor conversational fluency
rng.shuffle(oasst1_items)
selected_oasst1 = oasst1_items[:2500]

# Combine all targeted additions:
# 1500 CoT arithmetic + 1200 Python + 600 Instruction/Format + 400 Facts/Dialogue = 3,700 items
targeted_items = cot_arithmetic_items + python_train_items + instruction_structured_items + factual_dialogue_items
rng.shuffle(targeted_items)

# Partition 3,500 to train and 200 to in-curriculum validation heldout
train_targeted = targeted_items[:3500]
heldout_targeted = targeted_items[3500:]

# Final Training Set: 2,500 OASST1 + 3,500 Targeted = 6,000 sequences
combined_train = selected_oasst1 + train_targeted
rng.shuffle(combined_train)

# Final Heldout Set: 200 Targeted Heldout sequences
combined_heldout = heldout_targeted

print(f"Final Combined Training Set: {len(combined_train)} sequences.")
print(f"Final In-Curriculum Held-Out Set: {len(combined_heldout)} sequences.")

train_jsonl = RAW_OUT_DIR / "ultimate_train.jsonl"
heldout_jsonl = RAW_OUT_DIR / "ultimate_heldout.jsonl"

with open(train_jsonl, "w", encoding="utf-8") as f:
    for it in combined_train:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

with open(heldout_jsonl, "w", encoding="utf-8") as f:
    for it in combined_heldout:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")

print(f"Saved {train_jsonl}")
print(f"Saved {heldout_jsonl}")


# =============================================================
# 6. ENCODE TO BINARY TOKENS & MASKS (EOS V2 FORMAT)
# =============================================================
print("\nEncoding Datasets to Binary Memory-Mapped Tokens & Masks...")

CONTEXT_LENGTH = 1024
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5


def create_loss_mask_v2(token_ids: list) -> np.ndarray:
    mask = np.zeros(len(token_ids), dtype=np.uint8)
    assistant_active = False
    for i, token_id in enumerate(token_ids):
        if token_id == ASSISTANT_ID:
            assistant_active = True
            mask[i] = 1
        elif token_id == EOS_ID:
            if assistant_active:
                mask[i] = 1
            assistant_active = False
        elif token_id == USER_ID:
            assistant_active = False
            mask[i] = 0
        elif assistant_active:
            mask[i] = 1
    return mask


def encode_dataset_to_bin(items, tokens_out: Path, mask_out: Path, name: str):
    token_sequences = []
    mask_sequences = []
    pad_tokens = 0
    active_loss_tokens = 0
    truncated = 0

    for idx, item in enumerate(items):
        messages = item["messages"]
        parts = ["[BOS]"]
        for msg in messages:
            role = msg["role"]
            text = msg["text"].strip()
            if role in ["prompter", "user"]:
                parts.append("<|user|>")
                parts.append(text)
            elif role == "assistant":
                parts.append("<|assistant|>")
                parts.append(text)
                parts.append("[EOS]")

        formatted_text = "\n".join(parts)
        encoded = tokenizer.encode(formatted_text)
        token_ids = encoded.ids
        raw_len = len(token_ids)
        raw_mask = create_loss_mask_v2(token_ids)

        if raw_len <= CONTEXT_LENGTH:
            pad_amt = CONTEXT_LENGTH - raw_len
            seq_tokens = np.pad(token_ids, (0, pad_amt), constant_values=PAD_ID).astype(np.uint16)
            seq_mask = np.pad(raw_mask, (0, pad_amt), constant_values=0).astype(np.uint8)
            pad_tokens += pad_amt
        else:
            truncated += 1
            seq_tokens = np.array(token_ids[:CONTEXT_LENGTH], dtype=np.uint16)
            seq_mask = raw_mask[:CONTEXT_LENGTH].astype(np.uint8)

        active_loss_tokens += int(seq_mask.sum())
        token_sequences.append(seq_tokens)
        mask_sequences.append(seq_mask)

    token_array = np.vstack(token_sequences).astype(np.uint16)
    mask_array = np.vstack(mask_sequences).astype(np.uint8)

    tokens_out.parent.mkdir(parents=True, exist_ok=True)
    token_array.tofile(tokens_out)
    mask_array.tofile(mask_out)

    total_cap = len(items) * CONTEXT_LENGTH
    print(f"[{name}] Wrote {tokens_out.name} ({tokens_out.stat().st_size:,} bytes) | Sequences: {len(items):,} | Truncated: {truncated} ({(truncated/len(items))*100:.2f}%) | Active Loss Tokens: {active_loss_tokens:,} ({(active_loss_tokens/total_cap)*100:.1f}%)")


encode_dataset_to_bin(
    combined_train,
    BIN_OUT_DIR / "train_tokens.bin",
    BIN_OUT_DIR / "train_mask.bin",
    "Ultimate Training Curriculum"
)

encode_dataset_to_bin(
    combined_heldout,
    BIN_OUT_DIR / "heldout_tokens.bin",
    BIN_OUT_DIR / "heldout_mask.bin",
    "Ultimate Held-Out Validation"
)

print("\n" + "=" * 70)
print("ULTIMATE CURRICULUM PREPARATION COMPLETE!")
print("=" * 70)
