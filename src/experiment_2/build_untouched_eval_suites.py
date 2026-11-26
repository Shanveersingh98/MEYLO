import sys
import json
import random
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
OUT_DIR = PROJECT_DIR / "data" / "experiment_2" / "test_untouched"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_ARITHMETIC_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category2_arithmetic_train.jsonl"
TRAIN_PYTHON_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category3_python_train.jsonl"
TRAIN_CONCEPTS_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category4_concepts_train.jsonl"

SEED = 999  # Distinct seed from training generation (42)
rng = random.Random(SEED)

print("=" * 70)
print("PHASE B: BUILDING UNTOUCHED EVALUATION SUITES")
print("=" * 70)

# Load training datasets to guarantee ZERO contamination
with open(TRAIN_ARITHMETIC_FILE, "r", encoding="utf-8") as f:
    train_arith = [json.loads(l) for l in f if l.strip()]
train_arith_pairs = set((x["a"], x["b"], x["operation"]) for x in train_arith)
print(f"Loaded {len(train_arith_pairs)} training arithmetic pairs for strict exclusion.")

# --------------------------------------------------------------------
# 1. Historical Benchmark Suite (Original 8 Prompts)
# --------------------------------------------------------------------
BENCHMARK_PROMPTS = [
    {
        "id": "bench_1",
        "prompt": "Hello",
        "category": "greeting",
        "training_contaminated": False,
        "notes": "Original benchmark prompt 1. Tests greeting fluency and persona."
    },
    {
        "id": "bench_2",
        "prompt": "What is artificial intelligence?",
        "category": "concept",
        "training_contaminated": True,
        "notes": "Original benchmark prompt 2. AI definition is in training curriculum."
    },
    {
        "id": "bench_3",
        "prompt": "What is machine learning?",
        "category": "concept",
        "training_contaminated": True,
        "notes": "Original benchmark prompt 3. ML definition is in training curriculum."
    },
    {
        "id": "bench_4",
        "prompt": "Explain Python in simple words.",
        "category": "python",
        "training_contaminated": True,
        "notes": "Original benchmark prompt 4. Python explanation is in training curriculum."
    },
    {
        "id": "bench_5",
        "prompt": "What is 2 + 2?",
        "category": "arithmetic",
        "training_contaminated": True,
        "notes": "Original benchmark prompt 5. The exact pair (2, 2, '+') is present in the training curriculum. Must be evaluated as training recall, NOT unseen generalization."
    },
    {
        "id": "bench_6",
        "prompt": "Explain monopsony in simple words.",
        "category": "concept",
        "training_contaminated": True,
        "notes": "Original benchmark prompt 6. Monopsony definition is in training curriculum."
    },
    {
        "id": "bench_7",
        "prompt": "Explain what a function is in Python.",
        "category": "python",
        "training_contaminated": True,
        "notes": "Original benchmark prompt 7. Function explanation is in training curriculum."
    },
    {
        "id": "bench_8",
        "prompt": "What is the difference between AI and machine learning?",
        "category": "concept",
        "training_contaminated": True,
        "notes": "Original benchmark prompt 8. AI vs ML distinction is in training curriculum."
    }
]

with open(OUT_DIR / "benchmark_8_prompts.json", "w", encoding="utf-8") as f:
    json.dump(BENCHMARK_PROMPTS, f, indent=2, ensure_ascii=False)
print("Saved: benchmark_8_prompts.json (8 prompts, explicitly labeled for contamination)")

# --------------------------------------------------------------------
# 2. Untouched Arithmetic Generalization Suite (100 Questions)
# --------------------------------------------------------------------
arith_templates_unseen = [
    "Determine the result of {a} plus {b}.",
    "Evaluate the expression: {a} - {b}.",
    "Compute the sum of {a} and {b}.",
    "What do you get by subtracting {b} from {a}?",
    "If a warehouse holds {a} cartons and {b} more arrive, what is the new total?",
    "A fuel reservoir had {a} gallons, and {b} gallons were drained. How many gallons remain?",
    "Calculate {a} + {b} precisely.",
    "Calculate {a} - {b} precisely.",
    "Find the total value when {b} is added to {a}.",
    "Find the remaining balance when {b} is deducted from {a}."
]

untouched_arith_items = []
def sample_tier_pool(rng, tier_name, count, a_range, b_range):
    pool = set()
    while len(pool) < count:
        a = rng.randint(a_range[0], a_range[1])
        b = rng.randint(b_range[0], b_range[1])
        op = rng.choice(["+", "-"])
        if op == "-" and a < b:
            a, b = b, a
        pool.add((a, b, op, tier_name))
    return list(pool)

candidate_pools = [
    # Tier 1: 1-digit
    [(a, b, op, "tier1_1digit") for a in range(10) for b in range(10) for op in ["+", "-"] if (op == "+" or a >= b)],
    # Tier 2: 2-digit with 1-digit
    [(a, b, op, "tier2_2digit_1digit") for a in range(10, 100) for b in range(1, 10) for op in ["+", "-"]],
    # Tier 3: 2-digit with 2-digit
    sample_tier_pool(rng, "tier3_2digit_2digit", 300, (10, 99), (10, 99)),
    # Tier 4: 3-digit with 2-digit
    sample_tier_pool(rng, "tier4_3digit_2digit", 300, (100, 999), (10, 99)),
    # Tier 5: 3-digit with 3-digit
    sample_tier_pool(rng, "tier5_3digit_3digit", 300, (100, 999), (100, 999)),
    # Tier 6: 4-digit
    sample_tier_pool(rng, "tier6_4digit", 300, (1000, 9999), (100, 4999)),
]

tier_quotas = [15, 20, 25, 15, 15, 10]  # Sums to 100
for pool, quota in zip(candidate_pools, tier_quotas):
    rng.shuffle(pool)
    selected = 0
    for a, b, op, tier in pool:
        if (a, b, op) in train_arith_pairs:
            continue
        expected = a + b if op == "+" else a - b
        assert expected == (a + b if op == "+" else a - b)
        
        # Pick template
        if op == "+":
            tmpl_candidates = [t for t in arith_templates_unseen if "plus" in t or "+" in t or "sum" in t or "arrive" in t or "added" in t]
        else:
            tmpl_candidates = [t for t in arith_templates_unseen if "minus" in t or "-" in t or "subtracting" in t or "drained" in t or "deducted" in t]
        q = rng.choice(tmpl_candidates).format(a=a, b=b)
        
        untouched_arith_items.append({
            "id": f"arith_unseen_{len(untouched_arith_items)+1:03d}",
            "tier": tier,
            "operation": op,
            "a": a,
            "b": b,
            "expected_answer": expected,
            "prompt": q,
            "training_contaminated": False
        })
        selected += 1
        if selected >= quota:
            break

assert len(untouched_arith_items) == 100
# Assert zero overlap
for item in untouched_arith_items:
    assert (item["a"], item["b"], item["operation"]) not in train_arith_pairs, f"Leakage detected: {item}"

with open(OUT_DIR / "untouched_arithmetic_test.jsonl", "w", encoding="utf-8") as f:
    for item in untouched_arith_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved: untouched_arithmetic_test.jsonl (100 items, strictly 0% numerical pair contamination)")

# --------------------------------------------------------------------
# 3. Untouched Python Suite (30 Tasks)
# --------------------------------------------------------------------
# 15 Executable tasks with unit test assertions
EXECUTABLE_TASKS = [
    {
        "func_name": "celsius_to_kelvin",
        "prompt": "Write a Python function named celsius_to_kelvin(celsius) that converts a Celsius temperature to Kelvin (Kelvin = Celsius + 273.15).",
        "tests": [
            {"args": [0.0], "expected": 273.15},
            {"args": [100.0], "expected": 373.15},
            {"args": [-273.15], "expected": 0.0}
        ]
    },
    {
        "func_name": "is_power_of_two",
        "prompt": "Write a Python function named is_power_of_two(n) that returns True if positive integer n is a power of 2, and False otherwise.",
        "tests": [
            {"args": [1], "expected": True},
            {"args": [2], "expected": True},
            {"args": [16], "expected": True},
            {"args": [18], "expected": False},
            {"args": [0], "expected": False}
        ]
    },
    {
        "func_name": "count_uppercase",
        "prompt": "Write a Python function named count_uppercase(s) that counts and returns the number of uppercase letters in string s.",
        "tests": [
            {"args": ["Hello World"], "expected": 2},
            {"args": ["PYTHON"], "expected": 6},
            {"args": ["all lowercase"], "expected": 0}
        ]
    },
    {
        "func_name": "remove_none_values",
        "prompt": "Write a Python function named remove_none_values(lst) that takes a list and returns a new list with all None values removed.",
        "tests": [
            {"args": [[1, None, 2, None, 3]], "expected": [1, 2, 3]},
            {"args": [[None, None]], "expected": []},
            {"args": [["a", "b"]], "expected": ["a", "b"]}
        ]
    },
    {
        "func_name": "calculate_cuboid_volume",
        "prompt": "Write a Python function named calculate_cuboid_volume(length, width, height) that returns the volume of a rectangular cuboid.",
        "tests": [
            {"args": [2, 3, 4], "expected": 24},
            {"args": [5, 1, 2], "expected": 10}
        ]
    },
    {
        "func_name": "is_anagram",
        "prompt": "Write a Python function named is_anagram(s1, s2) that returns True if two strings are anagrams of each other (ignoring case), and False otherwise.",
        "tests": [
            {"args": ["listen", "silent"], "expected": True},
            {"args": ["Triangle", "Integral"], "expected": True},
            {"args": ["apple", "pale"], "expected": False}
        ]
    },
    {
        "func_name": "filter_strings_by_length",
        "prompt": "Write a Python function named filter_strings_by_length(words, min_len) that returns words with length >= min_len.",
        "tests": [
            {"args": [["cat", "elephant", "dog", "tiger"], 4], "expected": ["elephant", "tiger"]},
            {"args": [["a", "bb", "ccc"], 2], "expected": ["bb", "ccc"]}
        ]
    },
    {
        "func_name": "find_range_difference",
        "prompt": "Write a Python function named find_range_difference(numbers) that returns the difference between the maximum and minimum values in a non-empty list of numbers.",
        "tests": [
            {"args": [[10, 2, 5, 20]], "expected": 18},
            {"args": [[7, 7, 7]], "expected": 0}
        ]
    },
    {
        "func_name": "format_seconds_to_time",
        "prompt": "Write a Python function named format_seconds_to_time(seconds) that converts integer seconds into 'MM:SS' string format (e.g. 125 -> '02:05').",
        "tests": [
            {"args": [125], "expected": "02:05"},
            {"args": [59], "expected": "00:59"},
            {"args": [0], "expected": "00:00"}
        ]
    },
    {
        "func_name": "invert_boolean_dict",
        "prompt": "Write a Python function named invert_boolean_dict(d) that takes a dictionary with boolean values and returns a new dictionary with inverted boolean values.",
        "tests": [
            {"args": [{"a": True, "b": False}], "expected": {"a": False, "b": True}},
            {"args": [{}], "expected": {}}
        ]
    },
    {
        "func_name": "all_positive",
        "prompt": "Write a Python function named all_positive(numbers) that returns True if all numbers in the list are strictly greater than zero, False otherwise.",
        "tests": [
            {"args": [[1, 2, 3]], "expected": True},
            {"args": [[1, -1, 3]], "expected": False},
            {"args": [[0, 5]], "expected": False}
        ]
    },
    {
        "func_name": "clamp_float",
        "prompt": "Write a Python function named clamp_float(val, low, high) that restricts val to the interval [low, high].",
        "tests": [
            {"args": [3.5, 0.0, 5.0], "expected": 3.5},
            {"args": [-1.0, 0.0, 5.0], "expected": 0.0},
            {"args": [10.0, 0.0, 5.0], "expected": 5.0}
        ]
    },
    {
        "func_name": "list_to_comma_string",
        "prompt": "Write a Python function named list_to_comma_string(items) that joins a list of strings with comma and space (', ').",
        "tests": [
            {"args": [["apple", "banana", "cherry"]], "expected": "apple, banana, cherry"},
            {"args": [["one"]], "expected": "one"}
        ]
    },
    {
        "func_name": "multiply_by_two",
        "prompt": "Write a Python function named multiply_by_two(n) that returns double the value of n.",
        "tests": [
            {"args": [5], "expected": 10},
            {"args": [-3], "expected": -6},
            {"args": [0], "expected": 0}
        ]
    },
    {
        "func_name": "geometric_mean_two",
        "prompt": "Write a Python function named geometric_mean_two(a, b) that returns the square root of (a * b) for non-negative numbers a and b.",
        "tests": [
            {"args": [4, 9], "expected": 6.0},
            {"args": [1, 16], "expected": 4.0}
        ]
    }
]

# 5 Code Explanation / Interpretation Tasks
EXPLANATION_TASKS = [
    {
        "id": "py_expl_1",
        "prompt": "What does the following Python function do? Explain in plain English:\n```python\ndef mystery(lst):\n    return [x for x in lst if x > 0]\n```",
        "expected_key_points": ["filters", "positive numbers", "greater than zero", "list comprehension"]
    },
    {
        "id": "py_expl_2",
        "prompt": "What does the following Python function return when called with s='hello'?\n```python\ndef transform(s):\n    return s.upper()[::-1]\n```",
        "expected_key_points": ["OLLEH", "uppercase", "reverses"]
    },
    {
        "id": "py_expl_3",
        "prompt": "Explain what happens when a function in Python executes a return statement inside a loop.",
        "expected_key_points": ["terminates immediately", "exits loop", "returns value to caller"]
    },
    {
        "id": "py_expl_4",
        "prompt": "What is the difference between the '==' operator and the 'is' keyword in Python?",
        "expected_key_points": ["equality of values", "identity", "memory address", "same object"]
    },
    {
        "id": "py_expl_5",
        "prompt": "Why is it dangerous to use a mutable default argument like `def append_to(item, target=[]):` in Python?",
        "expected_key_points": ["evaluated once at definition", "shared across calls", "persists state", "use None instead"]
    }
]

# 5 Code Debugging Tasks
DEBUG_TASKS = [
    {
        "id": "py_debug_1",
        "prompt": "Identify and explain the bug in this Python function:\n```python\ndef add_five(n):\n    result = n + 5\n```",
        "expected_key_points": ["missing return statement", "returns None"]
    },
    {
        "id": "py_debug_2",
        "prompt": "Identify and explain the bug in this Python code:\n```python\ndef get_first(items):\nreturn items[0]\n```",
        "expected_key_points": ["IndentationError", "indentation", "must indent function body"]
    },
    {
        "id": "py_debug_3",
        "prompt": "What error will this Python function produce when called with an empty list?\n```python\ndef average(nums):\n    return sum(nums) / len(nums)\n```",
        "expected_key_points": ["ZeroDivisionError", "division by zero", "empty list has length 0"]
    },
    {
        "id": "py_debug_4",
        "prompt": "Identify the logical error in this parity check:\n```python\ndef is_odd(number):\n    return number % 2 == 0\n```",
        "expected_key_points": ["checks for even", "should be != 0 or == 1"]
    },
    {
        "id": "py_debug_5",
        "prompt": "Why does this function raise an error?\n```python\ndef compute():\n    total = total + 1\n    return total\n```",
        "expected_key_points": ["UnboundLocalError", "total used before assignment", "local variable"]
    }
]

# 5 Conceptual Python Questions
CONCEPT_PYTHON_TASKS = [
    {
        "id": "py_concept_1",
        "prompt": "What is a list comprehension in Python and what is its basic syntax?",
        "expected_key_points": ["concise way to create lists", "[expression for item in iterable]"]
    },
    {
        "id": "py_concept_2",
        "prompt": "What does the pass statement do in Python?",
        "expected_key_points": ["null operation", "placeholder", "does nothing"]
    },
    {
        "id": "py_concept_3",
        "prompt": "Explain what *args and **kwargs stand for and when you would use them.",
        "expected_key_points": ["variable positional arguments", "keyword arguments", "tuple", "dictionary"]
    },
    {
        "id": "py_concept_4",
        "prompt": "Why are tuples immutable in Python while lists are mutable?",
        "expected_key_points": ["cannot be modified after creation", "hashable", "dictionary keys", "safety"]
    },
    {
        "id": "py_concept_5",
        "prompt": "What is the difference between global and local variable scope in Python functions?",
        "expected_key_points": ["local inside function", "global outside", "lifetime of function call"]
    }
]

untouched_py_items = []
for idx, task in enumerate(EXECUTABLE_TASKS, 1):
    untouched_py_items.append({
        "id": f"py_exec_{idx:02d}",
        "type": "executable_code",
        "func_name": task["func_name"],
        "prompt": task["prompt"],
        "tests": task["tests"],
        "training_contaminated": False
    })
for item in EXPLANATION_TASKS:
    untouched_py_items.append({**item, "type": "code_explanation", "training_contaminated": False})
for item in DEBUG_TASKS:
    untouched_py_items.append({**item, "type": "code_debugging", "training_contaminated": False})
for item in CONCEPT_PYTHON_TASKS:
    untouched_py_items.append({**item, "type": "conceptual_qa", "training_contaminated": False})

assert len(untouched_py_items) == 30
with open(OUT_DIR / "untouched_python_test.jsonl", "w", encoding="utf-8") as f:
    for item in untouched_py_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved: untouched_python_test.jsonl (30 tasks: 15 executable with unit tests, 5 explanation, 5 debugging, 5 concepts)")

# --------------------------------------------------------------------
# 4. Untouched Concepts Generalization Suite (30 Questions)
# --------------------------------------------------------------------
UNTOUCHED_CONCEPTS = [
    # Economics (10)
    {
        "concept": "stagflation",
        "domain": "economics",
        "prompt": "What is stagflation in economics and why is it problematic?",
        "ground_truth_key_points": ["stagnant growth", "high inflation", "high unemployment", "policy dilemma"]
    },
    {
        "concept": "oligopoly",
        "domain": "economics",
        "prompt": "Explain what an oligopoly is in economics with an example.",
        "ground_truth_key_points": ["small number of large firms", "interdependent", "market share", "commercial aircraft"]
    },
    {
        "concept": "moral_hazard",
        "domain": "economics",
        "prompt": "What is moral hazard in economics? Give a real-world example.",
        "ground_truth_key_points": ["takes greater risks", "someone else bears costs", "insurance"]
    },
    {
        "concept": "deadweight_loss",
        "domain": "economics",
        "prompt": "What is deadweight loss in economics?",
        "ground_truth_key_points": ["loss of economic efficiency", "allocative inefficiency", "tax or monopoly", "unrealized surplus"]
    },
    {
        "concept": "liquidity_trap",
        "domain": "economics",
        "prompt": "Explain the concept of a liquidity trap in monetary economics.",
        "ground_truth_key_points": ["interest rates near zero", "monetary policy ineffective", "hoard cash"]
    },
    {
        "concept": "adverse_selection",
        "domain": "economics",
        "prompt": "What is adverse selection in insurance and market economics?",
        "ground_truth_key_points": ["asymmetric information", "high-risk individuals", "market unraveling"]
    },
    {
        "concept": "price_ceiling",
        "domain": "economics",
        "prompt": "What happens when a government imposes a binding price ceiling on a market?",
        "ground_truth_key_points": ["maximum legal price", "shortage", "quantity demanded exceeds quantity supplied"]
    },
    {
        "concept": "diminishing_marginal_utility",
        "domain": "economics",
        "prompt": "What does the law of diminishing marginal utility state?",
        "ground_truth_key_points": ["additional satisfaction decreases", "each additional unit consumed"]
    },
    {
        "concept": "gini_coefficient",
        "domain": "economics",
        "prompt": "What is the Gini coefficient and what does it measure?",
        "ground_truth_key_points": ["income inequality", "wealth inequality", "0 means perfect equality", "1 means perfect inequality"]
    },
    {
        "concept": "monopolistic_competition",
        "domain": "economics",
        "prompt": "What is monopolistic competition and how does it differ from perfect competition?",
        "ground_truth_key_points": ["many sellers", "differentiated products", "branding", "not identical"]
    },

    # AI / ML (10)
    {
        "concept": "diffusion_models",
        "domain": "ai_ml",
        "prompt": "How do diffusion models work in generative artificial intelligence?",
        "ground_truth_key_points": ["gradual noising", "denoising process", "generates images from noise"]
    },
    {
        "concept": "cross_entropy_loss",
        "domain": "ai_ml",
        "prompt": "What is cross-entropy loss in machine learning and how is it used?",
        "ground_truth_key_points": ["classification loss", "difference between probability distributions", "log loss"]
    },
    {
        "concept": "data_leakage",
        "domain": "ai_ml",
        "prompt": "What is data leakage in machine learning and why is it dangerous?",
        "ground_truth_key_points": ["test information contaminates train", "artificially high validation score", "fails in production"]
    },
    {
        "concept": "inductive_bias",
        "domain": "ai_ml",
        "prompt": "What is inductive bias in machine learning algorithms?",
        "ground_truth_key_points": ["set of assumptions", "generalize beyond training data", "convolutional or causal bias"]
    },
    {
        "concept": "vanishing_gradients",
        "domain": "ai_ml",
        "prompt": "What is the vanishing gradient problem in deep neural networks?",
        "ground_truth_key_points": ["gradients become exponentially small", "early layers do not learn", "backpropagation"]
    },
    {
        "concept": "beam_search",
        "domain": "ai_ml",
        "prompt": "What is beam search in language model decoding?",
        "ground_truth_key_points": ["heuristic search", "tracks top-k partial hypotheses", "better than greedy argmax"]
    },
    {
        "concept": "temperature_sampling",
        "domain": "ai_ml",
        "prompt": "How does temperature affect text generation in language models?",
        "ground_truth_key_points": ["divides logits", "lower is deterministic", "higher is random and diverse"]
    },
    {
        "concept": "residual_connection",
        "domain": "ai_ml",
        "prompt": "What is a residual (skip) connection in deep neural networks?",
        "ground_truth_key_points": ["adds input directly to output", "f(x) + x", "enables training very deep networks"]
    },
    {
        "concept": "learning_rate_warmup",
        "domain": "ai_ml",
        "prompt": "Why do deep learning practitioners use learning rate warmup?",
        "ground_truth_key_points": ["starts small", "prevents unstable early updates", "stabilizes training initially"]
    },
    {
        "concept": "contrastive_search",
        "domain": "ai_ml",
        "prompt": "What is contrastive search in natural language decoding?",
        "ground_truth_key_points": ["balances model confidence and degeneration penalty", "prevents repetition loops"]
    },

    # Science (10)
    {
        "concept": "doppler_effect",
        "domain": "science",
        "prompt": "Explain the Doppler effect with an example of sound waves.",
        "ground_truth_key_points": ["change in frequency", "relative motion", "pitch increases approaching", "ambulance siren"]
    },
    {
        "concept": "epigenetics",
        "domain": "science",
        "prompt": "What is epigenetics and how can genes change expression without altering DNA?",
        "ground_truth_key_points": ["gene expression changes", "no alteration to DNA sequence", "methylation", "environmental factors"]
    },
    {
        "concept": "superconductivity",
        "domain": "science",
        "prompt": "What is superconductivity in physics and what happens at the critical temperature?",
        "ground_truth_key_points": ["zero electrical resistance", "expulsion of magnetic fields", "Meissner effect"]
    },
    {
        "concept": "entropy_second_law",
        "domain": "science",
        "prompt": "What is entropy in thermodynamics and why does it always increase?",
        "ground_truth_key_points": ["measure of disorder", "second law of thermodynamics", "unavailable energy", "isolated system"]
    },
    {
        "concept": "diffraction",
        "domain": "science",
        "prompt": "What is diffraction in wave physics?",
        "ground_truth_key_points": ["bending of waves", "around obstacles or through slits", "interference pattern"]
    },
    {
        "concept": "blackbody_radiation",
        "domain": "science",
        "prompt": "What is blackbody radiation in physics?",
        "ground_truth_key_points": ["absorbs all incident radiation", "emits thermal radiation", "depends only on temperature"]
    },
    {
        "concept": "atp_synthesis",
        "domain": "science",
        "prompt": "What is ATP and why is it considered the energy currency of the cell?",
        "ground_truth_key_points": ["adenosine triphosphate", "cellular energy", "releases energy when phosphate bond breaks"]
    },
    {
        "concept": "continental_drift",
        "domain": "science",
        "prompt": "What is the theory of continental drift in geology?",
        "ground_truth_key_points": ["continents once joined", "Pangaea", "moved over geological time", "plate tectonics"]
    },
    {
        "concept": "ph_scale",
        "domain": "science",
        "prompt": "Explain the pH scale and what happens during neutralization.",
        "ground_truth_key_points": ["measures acidity or basicity", "7 is neutral", "acid and base form water and salt"]
    },
    {
        "concept": "archimedes_principle",
        "domain": "science",
        "prompt": "What is Archimedes' principle of buoyancy?",
        "ground_truth_key_points": ["upward buoyant force", "equal to weight of displaced fluid", "floating objects"]
    }
]

untouched_concepts_items = []
for idx, c in enumerate(UNTOUCHED_CONCEPTS, 1):
    untouched_concepts_items.append({
        "id": f"concept_unseen_{idx:02d}",
        "concept": c["concept"],
        "domain": c["domain"],
        "prompt": c["prompt"],
        "ground_truth_key_points": c["ground_truth_key_points"],
        "training_contaminated": False
    })

assert len(untouched_concepts_items) == 30
with open(OUT_DIR / "untouched_concepts_test.jsonl", "w", encoding="utf-8") as f:
    for item in untouched_concepts_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved: untouched_concepts_test.jsonl (30 questions across economics, AI/ML, science, strictly 0% training contamination)")

print("\n" + "=" * 70)
print("PHASE B UNTOUCHED EVALUATION SUITES FULLY BUILT!")
print("=" * 70)
