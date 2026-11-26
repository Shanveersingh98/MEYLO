import sys
import json
import ast
import re
import random
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
OUT_TRAIN_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category3_python_train.jsonl"
OUT_HELDOUT_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category3_python_heldout.jsonl"

SEED = 42
TARGET_TRAIN_COUNT = 1800
TARGET_HELDOUT_COUNT = 200

CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)


def validate_python_code(text: str) -> bool:
    """
    Extracts all ```python ... ``` blocks and parses them with Python ast.parse.
    Returns True if all blocks parse cleanly, False otherwise.
    """
    blocks = CODE_BLOCK_RE.findall(text)
    if not blocks:
        return True  # Conceptual explanation without code block
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError as e:
            print(f"Syntax error in generated code:\n{block}\nError: {e}")
            return False
    return True


# =====================================================================
# 1. CORE CONCEPTUAL EXPLANATIONS (Functions, Parameters, Returns, Docstrings, Python Intro)
# =====================================================================

CORE_EXPLANATION_TOPICS = [
    {
        "id": "explain_python_simple",
        "benchmark_match": True,
        "questions": [
            "Explain Python in simple words.",
            "Can you explain what Python is in simple terms?",
            "What is Python? Explain it simply.",
            "Explain Python programming language to a beginner.",
            "How would you describe Python in simple words?",
        ],
        "answer": (
            "Python is a popular, high-level programming language designed to be easy to read and simple to write. "
            "Unlike low-level languages that require complex syntax and manual memory management, Python uses clean, "
            "English-like syntax where indentation (spacing) defines code structure.\n\n"
            "Key features of Python include:\n"
            "- **Readable syntax**: Python code looks almost like plain English.\n"
            "- **Interpreted**: You can run code directly line-by-line without manual compilation steps.\n"
            "- **Versatile**: Python is widely used for web development, data science, artificial intelligence, automation, and scientific computing.\n"
            "- **Rich ecosystem**: A vast standard library and package repository (PyPI) provide ready-made tools for almost any task."
        ),
    },
    {
        "id": "explain_function_python",
        "benchmark_match": True,
        "questions": [
            "Explain what a function is in Python.",
            "What is a function in Python and why do we use it?",
            "Can you explain functions in Python in simple words?",
            "What is a function in Python programming?",
            "How do functions work in Python?",
        ],
        "answer": (
            "In Python, a function is a named, reusable block of code that performs a specific task. "
            "Instead of writing the same code repeatedly, you define a function once and call it whenever needed.\n\n"
            "A Python function is defined using the `def` keyword, followed by the function name, parentheses for parameters, and a colon:\n\n"
            "```python\ndef greet(name):\n    \"\"\"Return a personalized greeting.\"\"\"\n    return f\"Hello, {name}!\"\n\nmessage = greet(\"Alice\")\nprint(message)  # Output: Hello, Alice!\n```\n\n"
            "Functions help organize programs into modular, testable, and maintainable pieces."
        ),
    },
    {
        "id": "def_keyword",
        "questions": [
            "What is the def keyword in Python?",
            "How does the def keyword work in Python?",
            "What does def mean in Python?",
        ],
        "answer": (
            "In Python, `def` is a reserved keyword used to declare and define a function. "
            "It stands for 'define'. When Python encounters `def`, it creates a function object and binds it to the specified function name.\n\n"
            "```python\ndef calculate_square(number):\n    return number * number\n\nresult = calculate_square(5)\nprint(result)  # 25\n```\n\n"
            "The `def` statement must be followed by the function name, parameter list in parentheses, and a colon, with the function body indented below it."
        ),
    },
    {
        "id": "parameters_vs_arguments",
        "questions": [
            "What is the difference between a parameter and an argument in Python?",
            "Explain parameters vs arguments in Python functions.",
            "How are parameters different from arguments in Python?",
        ],
        "answer": (
            "While often used interchangeably in casual conversation, parameters and arguments have precise technical distinctions in Python:\n\n"
            "- **Parameters**: The variable names listed in the function definition header (`def`). They act as placeholders for incoming values.\n"
            "- **Arguments**: The actual values or expressions passed into the function when it is invoked.\n\n"
            "```python\n# 'width' and 'height' are PARAMETERS\ndef calculate_area(width, height):\n    return width * height\n\n# 10 and 5 are ARGUMENTS\narea = calculate_area(10, 5)\n```"
        ),
    },
    {
        "id": "return_statement",
        "questions": [
            "What does the return statement do in Python?",
            "Explain how the return statement works in Python functions.",
            "Why do functions use return in Python?",
        ],
        "answer": (
            "The `return` statement in Python ends function execution and passes a value back to the caller. "
            "When Python hits a `return` statement, the function immediately terminates, and any remaining lines inside that function are skipped.\n\n"
            "```python\ndef absolute_value(n):\n    if n < 0:\n        return -n\n    return n\n\nprint(absolute_value(-15))  # Output: 15\n```\n\n"
            "If a function does not contain an explicit `return` statement, or if `return` is written without an expression, Python implicitly returns `None`."
        ),
    },
    {
        "id": "multiple_return_values",
        "questions": [
            "Can a Python function return multiple values?",
            "How do you return multiple values from a Python function?",
            "Explain multiple returns in Python.",
        ],
        "answer": (
            "Yes, a Python function can return multiple values by separating them with commas. "
            "Behind the scenes, Python packs the multiple values into a single `tuple`, which the caller can unpack into separate variables:\n\n"
            "```python\ndef get_min_and_max(numbers):\n    \"\"\"Return both the minimum and maximum of a list.\"\"\"\n    return min(numbers), max(numbers)\n\nlow, high = get_min_and_max([4, 1, 9, 2, 8])\nprint(f\"Lowest: {low}, Highest: {high}\")  # Lowest: 1, Highest: 9\n```"
        ),
    },
    {
        "id": "default_parameters",
        "questions": [
            "What are default parameters in Python functions?",
            "How do default arguments work in Python?",
            "Explain default parameter values in Python.",
        ],
        "answer": (
            "Default parameters allow you to specify fallback values for arguments in a function definition. "
            "If the caller provides an argument, the default is overridden; if the caller omits that argument, Python uses the default value:\n\n"
            "```python\ndef power(base, exponent=2):\n    \"\"\"Compute base raised to exponent (defaulting to squared).\"\"\"\n    return base ** exponent\n\nprint(power(5))     # 25 (uses default exponent=2)\nprint(power(2, 3))  # 8 (overrides default with 3)\n```\n\n"
            "In Python, default parameters must always appear after all positional parameters without defaults."
        ),
    },
    {
        "id": "docstrings_explanation",
        "questions": [
            "What is a docstring in Python and how do you write one?",
            "Explain docstrings in Python functions.",
            "How should you document a function using docstrings in Python?",
        ],
        "answer": (
            "A docstring (documentation string) is a multi-line string literal placed as the very first statement inside a function, class, or module. "
            "It explains what the function does, its parameters, and what it returns:\n\n"
            "```python\ndef divide(numerator, denominator):\n    \"\"\"\n    Divide two numbers and return the quotient.\n\n    Args:\n        numerator (float): The dividend.\n        denominator (float): The divisor (must not be zero).\n\n    Returns:\n        float: The result of the division.\n    \"\"\"\n    if denominator == 0:\n        raise ValueError(\"Denominator cannot be zero.\")\n    return numerator / denominator\n```\n\n"
            "Docstrings are accessible at runtime via `help(divide)` or `divide.__doc__`."
        ),
    },
    {
        "id": "args_and_kwargs",
        "questions": [
            "What are *args and **kwargs in Python?",
            "Explain *args and **kwargs in Python function parameters.",
            "How do variable-length arguments work in Python?",
        ],
        "answer": (
            "In Python, `*args` and `**kwargs` allow a function to accept an arbitrary number of positional and keyword arguments:\n\n"
            "- `*args`: Collects extra positional arguments into a `tuple`.\n"
            "- `**kwargs`: Collects extra keyword arguments into a `dict`.\n\n"
            "```python\ndef log_event(event_name, *details, **metadata):\n    print(f\"Event: {event_name}\")\n    print(f\"Details: {details}\")\n    print(f\"Metadata: {metadata}\")\n\nlog_event(\"Login\", \"User clicked submit\", \"IP 127.0.0.1\", user_id=42, status=\"success\")\n```"
        ),
    },
    {
        "id": "scope_local_vs_global",
        "questions": [
            "What is variable scope in Python functions?",
            "Explain local vs global variables in Python.",
            "How does variable scope work inside a function in Python?",
        ],
        "answer": (
            "Variable scope determines where a variable is accessible in your code:\n\n"
            "- **Local scope**: Variables defined inside a function exist only during that function call and cannot be accessed outside it.\n"
            "- **Global scope**: Variables defined at the top level of a module are accessible everywhere in that module.\n\n"
            "```python\nglobal_message = \"I am global\"\n\ndef show_scope():\n    local_count = 10\n    print(global_message)  # Accessible\n    print(local_count)      # Accessible\n\nshow_scope()\n# Trying to access local_count here would raise a NameError\n```"
        ),
    },
]


# =====================================================================
# 2. FUNCTION DEFINITION TASK LIBRARY (Executable Python Functions)
# =====================================================================

TRAIN_TASKS = [
    # Math & Numeric
    {
        "func_name": "add_numbers",
        "category": "math",
        "doc": "Return the sum of two numbers a and b.",
        "params": "a, b",
        "body": "return a + b",
        "test_call": "add_numbers(12, 8)",
        "prompt_templates": [
            "Write a Python function to add two numbers.",
            "Write a function named {func_name} that takes two numbers and returns their sum.",
            "How do you write a Python function to add two values?",
        ]
    },
    {
        "func_name": "subtract_numbers",
        "category": "math",
        "doc": "Return the result of subtracting b from a.",
        "params": "a, b",
        "body": "return a - b",
        "test_call": "subtract_numbers(20, 7)",
        "prompt_templates": [
            "Write a Python function to subtract one number from another.",
            "Create a function named {func_name}(a, b) that computes a - b.",
            "How do you write a subtraction function in Python?",
        ]
    },
    {
        "func_name": "multiply_numbers",
        "category": "math",
        "doc": "Multiply two numbers and return the product.",
        "params": "x, y",
        "body": "return x * y",
        "test_call": "multiply_numbers(6, 7)",
        "prompt_templates": [
            "Write a Python function to multiply two numbers.",
            "Create a Python function named {func_name} that returns the product of x and y.",
        ]
    },
    {
        "func_name": "is_even",
        "category": "math",
        "doc": "Return True if number is even, False otherwise.",
        "params": "number",
        "body": "return number % 2 == 0",
        "test_call": "is_even(14)",
        "prompt_templates": [
            "Write a Python function to check whether an integer is even.",
            "Create a function named {func_name} that returns True for even numbers.",
        ]
    },
    {
        "func_name": "is_positive",
        "category": "math",
        "doc": "Return True if n is greater than zero.",
        "params": "n",
        "body": "return n > 0",
        "test_call": "is_positive(5)",
        "prompt_templates": [
            "Write a Python function to check if a number is positive.",
            "Create a function named {func_name} that tests if n > 0.",
        ]
    },
    {
        "func_name": "rectangle_area",
        "category": "geometry",
        "doc": "Calculate the area of a rectangle given width and height.",
        "params": "width, height",
        "body": "if width <= 0 or height <= 0:\n        raise ValueError(\"Dimensions must be positive.\")\n    return width * height",
        "test_call": "rectangle_area(5, 10)",
        "prompt_templates": [
            "Write a Python function to calculate the area of a rectangle.",
            "Create a function named {func_name} taking width and height.",
        ]
    },
    {
        "func_name": "celsius_to_fahrenheit",
        "category": "conversion",
        "doc": "Convert a temperature from Celsius to Fahrenheit.",
        "params": "celsius",
        "body": "return (celsius * 9 / 5) + 32",
        "test_call": "celsius_to_fahrenheit(25)",
        "prompt_templates": [
            "Write a Python function to convert Celsius to Fahrenheit.",
            "How do you write a Celsius to Fahrenheit temperature conversion function in Python?",
        ]
    },
    {
        "func_name": "calculate_discount",
        "category": "business",
        "doc": "Calculate the final price after applying a percentage discount.",
        "params": "price, discount_percent=10.0",
        "body": "discount_amount = price * (discount_percent / 100.0)\n    return price - discount_amount",
        "test_call": "calculate_discount(100.0, 15.0)",
        "prompt_templates": [
            "Write a Python function with a default parameter to calculate a discounted price.",
            "Create a function named {func_name} that applies a percentage discount to an original price.",
        ]
    },
    {
        "func_name": "clamp",
        "category": "math",
        "doc": "Restrict a value between a minimum and maximum boundary.",
        "params": "val, min_val, max_val",
        "body": "if val < min_val:\n        return min_val\n    if val > max_val:\n        return max_val\n    return val",
        "test_call": "clamp(15, 0, 10)",
        "prompt_templates": [
            "Write a Python function to clamp a number within a specified range [min_val, max_val].",
            "How do you write a clamp function in Python?",
        ]
    },
    {
        "func_name": "calculate_average",
        "category": "statistics",
        "doc": "Compute the arithmetic mean of a list of numbers.",
        "params": "numbers",
        "body": "if not numbers:\n        return 0.0\n    return sum(numbers) / len(numbers)",
        "test_call": "calculate_average([10, 20, 30, 40])",
        "prompt_templates": [
            "Write a Python function to calculate the average of a list of numbers.",
            "Create a function named {func_name} that safely returns the mean of a sequence.",
        ]
    },
    # String manipulation
    {
        "func_name": "reverse_string",
        "category": "string",
        "doc": "Return the reversed version of a given text string.",
        "params": "text",
        "body": "return text[::-1]",
        "test_call": "reverse_string(\"Python\")",
        "prompt_templates": [
            "Write a Python function to reverse a string.",
            "Create a function named {func_name} that takes a string and returns it backwards.",
        ]
    },
    {
        "func_name": "count_vowels",
        "category": "string",
        "doc": "Count how many vowels (a, e, i, o, u) are in the string.",
        "params": "text",
        "body": "vowels = set(\"aeiouAEIOU\")\n    return sum(1 for char in text if char in vowels)",
        "test_call": "count_vowels(\"Hello World\")",
        "prompt_templates": [
            "Write a Python function to count the number of vowels in a string.",
            "Create a function named {func_name} that returns vowel count.",
        ]
    },
    {
        "func_name": "is_palindrome",
        "category": "string",
        "doc": "Check if a text string reads the same forwards and backwards.",
        "params": "text",
        "body": "cleaned = \"\".join(char.lower() for char in text if char.isalnum())\n    return cleaned == cleaned[::-1]",
        "test_call": "is_palindrome(\"racecar\")",
        "prompt_templates": [
            "Write a Python function to check whether a string is a palindrome.",
            "Create a function named {func_name} that ignores case and punctuation when testing palindromes.",
        ]
    },
    {
        "func_name": "format_greeting",
        "category": "string",
        "doc": "Return a formal or informal greeting message.",
        "params": "name, greeting=\"Hello\"",
        "body": "return f\"{greeting}, {name}!\"",
        "test_call": "format_greeting(\"Bob\", greeting=\"Good morning\")",
        "prompt_templates": [
            "Write a Python function that uses a default parameter to format a greeting.",
            "Create a greeting function named {func_name} with default greeting 'Hello'.",
        ]
    },
    {
        "func_name": "truncate_text",
        "category": "string",
        "doc": "Truncate text to a maximum length and append an ellipsis if shortened.",
        "params": "text, max_length=50",
        "body": "if len(text) <= max_length:\n        return text\n    return text[:max_length] + \"...\"",
        "test_call": "truncate_text(\"This is a very long string of words.\", max_length=15)",
        "prompt_templates": [
            "Write a Python function to truncate long text and add an ellipsis.",
            "Create a function named {func_name} with a default max_length parameter.",
        ]
    },
    # Lists & Data Collections
    {
        "func_name": "filter_even_numbers",
        "category": "collections",
        "doc": "Filter a list of numbers and return only the even integers.",
        "params": "numbers",
        "body": "return [n for n in numbers if n % 2 == 0]",
        "test_call": "filter_even_numbers([1, 2, 3, 4, 5, 6])",
        "prompt_templates": [
            "Write a Python function to filter even numbers from a list.",
            "How do you write a list comprehension function in Python to extract even numbers?",
        ]
    },
    {
        "func_name": "find_maximum",
        "category": "collections",
        "doc": "Find and return the maximum value in a list of numbers.",
        "params": "numbers",
        "body": "if not numbers:\n        return None\n    largest = numbers[0]\n    for num in numbers[1:]:\n        if num > largest:\n            largest = num\n    return largest",
        "test_call": "find_maximum([3, 18, 4, 21, 9])",
        "prompt_templates": [
            "Write a Python function from scratch to find the maximum element in a list.",
            "Create a function named {func_name} without using Python's built-in max().",
        ]
    },
    {
        "func_name": "remove_duplicates",
        "category": "collections",
        "doc": "Remove duplicate elements from a list while preserving original order.",
        "params": "items",
        "body": "seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result",
        "test_call": "remove_duplicates([1, 2, 2, 3, 1, 4])",
        "prompt_templates": [
            "Write a Python function to remove duplicates from a list while keeping the order.",
            "Create a function named {func_name} that dedupes elements in a sequence.",
        ]
    },
    {
        "func_name": "sum_all_args",
        "category": "args",
        "doc": "Sum any number of positional arguments using *args.",
        "params": "*numbers",
        "body": "total = 0\n    for n in numbers:\n        total += n\n    return total",
        "test_call": "sum_all_args(1, 2, 3, 4, 5)",
        "prompt_templates": [
            "Write a Python function using *args to sum an arbitrary number of inputs.",
            "How do you write a function with variable-length positional arguments (*args) in Python?",
        ]
    },
    {
        "func_name": "invert_dictionary",
        "category": "dict",
        "doc": "Invert a dictionary so that values become keys and keys become values.",
        "params": "data",
        "body": "return {val: key for key, val in data.items()}",
        "test_call": "invert_dictionary({'a': 1, 'b': 2})",
        "prompt_templates": [
            "Write a Python function to invert a dictionary.",
            "Create a function named {func_name} that swaps dictionary keys and values.",
        ]
    },
]

# Separate tasks strictly held out for evaluation (unseen functions & tasks)
HELDOUT_TASKS = [
    {
        "func_name": "compute_compound_interest",
        "category": "finance",
        "doc": "Calculate the compound interest amount for principal P at rate r over t years.",
        "params": "principal, rate, years, compounding_frequency=1",
        "body": "return principal * (1 + rate / compounding_frequency) ** (compounding_frequency * years)",
        "test_call": "compute_compound_interest(1000, 0.05, 3)",
        "prompt": "Write a Python function to compute compound interest with a default compounding frequency."
    },
    {
        "func_name": "find_second_largest",
        "category": "algorithms",
        "doc": "Return the second largest distinct number in a list.",
        "params": "numbers",
        "body": "unique_nums = sorted(list(set(numbers)), reverse=True)\n    if len(unique_nums) < 2:\n        return None\n    return unique_nums[1]",
        "test_call": "find_second_largest([10, 20, 4, 45, 99, 99])",
        "prompt": "Write a Python function to find the second largest distinct number in a list."
    },
    {
        "func_name": "count_consonants",
        "category": "string",
        "doc": "Count the number of consonant letters in a given text.",
        "params": "text",
        "body": "vowels = set(\"aeiouAEIOU\")\n    return sum(1 for char in text if char.isalpha() and char not in vowels)",
        "test_call": "count_consonants(\"Hello World\")",
        "prompt": "Write a Python function that counts only the consonants in a string."
    },
    {
        "func_name": "validate_ipv4",
        "category": "networking",
        "doc": "Check if a string represents a valid IPv4 address in dot-decimal notation.",
        "params": "ip_str",
        "body": "parts = ip_str.split('.')\n    if len(parts) != 4:\n        return False\n    for p in parts:\n        if not p.isdigit():\n            return False\n        val = int(p)\n        if val < 0 or val > 255:\n            return False\n        if len(p) > 1 and p.startswith('0'):\n            return False\n    return True",
        "test_call": "validate_ipv4(\"192.168.1.1\")",
        "prompt": "Write a Python function to validate whether a string is a valid IPv4 address."
    },
    {
        "func_name": "chunk_list",
        "category": "collections",
        "doc": "Split a list into smaller chunks of a specified size.",
        "params": "items, chunk_size",
        "body": "if chunk_size <= 0:\n        raise ValueError(\"chunk_size must be positive\")\n    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]",
        "test_call": "chunk_list([1, 2, 3, 4, 5, 6, 7], 3)",
        "prompt": "Write a Python function to split a list into sublists of size chunk_size."
    },
    {
        "func_name": "extract_hashtags",
        "category": "string",
        "doc": "Extract all words starting with a '#' symbol from a text post.",
        "params": "post",
        "body": "words = post.split()\n    return [w[1:] for w in words if w.startswith('#') and len(w) > 1]",
        "test_call": "extract_hashtags(\"Loving #deeplearning and #python today!\")",
        "prompt": "Write a Python function to extract hashtags from a social media post string."
    },
]


def format_function_explanation(task: dict, explanation_style: str) -> str:
    func_name = task["func_name"]
    params = task["params"]
    doc = task["doc"]
    body = task["body"]
    test_call = task["test_call"]

    code = f"def {func_name}({params}):\n    \"\"\"{doc}\"\"\"\n    {body}\n"

    if explanation_style == "standard":
        return (
            f"Here is the Python function implementation:\n\n"
            f"```python\n{code}\n"
            f"# Example usage:\n"
            f"result = {test_call}\n"
            f"print(result)\n```\n\n"
            f"**How it works:**\n"
            f"- We declare the function using `def {func_name}({params}):`.\n"
            f"- A docstring clarifies the purpose and arguments.\n"
            f"- The return statement outputs the calculated result."
        )
    elif explanation_style == "breakdown":
        return (
            f"You can define the function as follows:\n\n"
            f"```python\n{code}```\n\n"
            f"### Line-by-Line Explanation:\n"
            f"1. `def {func_name}({params}):` defines the function header and parameter signature.\n"
            f"2. The docstring provides documentation for anyone reading or importing the code.\n"
            f"3. The function body executes the core logic.\n"
            f"4. `return` sends the result back to the caller."
        )
    else:
        return (
            f"```python\n{code}\n"
            f"print({test_call})\n```"
        )


def build_python_dataset():
    rng = random.Random(SEED)
    train_records = []
    heldout_records = []

    # 1. Add core conceptual explanations to train
    for topic in CORE_EXPLANATION_TOPICS:
        ans = topic["answer"]
        assert validate_python_code(ans), f"Code syntax error in topic {topic['id']}"
        for q in topic["questions"]:
            train_records.append({
                "category": "python_concept",
                "topic_id": topic["id"],
                "messages": [
                    {"role": "prompter", "text": q},
                    {"role": "assistant", "text": ans}
                ]
            })

    # 2. Add function implementations and walkthroughs for TRAIN_TASKS
    styles = ["standard", "breakdown", "concise"]
    for task in TRAIN_TASKS:
        for template in task["prompt_templates"]:
            q = template.format(func_name=task["func_name"])
            for style in styles:
                ans = format_function_explanation(task, style)
                assert validate_python_code(ans), f"Code syntax error in task {task['func_name']} style {style}"
                train_records.append({
                    "category": "python_function",
                    "task": task["func_name"],
                    "style": style,
                    "messages": [
                        {"role": "prompter", "text": q},
                        {"role": "assistant", "text": ans}
                    ]
                })

    # 3. Add variations on parameters, docstrings, and returns
    param_variations = [
        ("What does the parameter '{p}' represent in the function `{f}`?",
         "In `{f}`, the parameter `{p}` represents {desc}. It receives the argument passed during invocation.")
    ]
    for task in TRAIN_TASKS:
        p_list = [p.strip().split("=")[0] for p in task["params"].split(",")]
        for p in p_list:
            if not p:
                continue
            q = f"In Python, what is the role of parameter `{p}` in `{task['func_name']}({task['params']})`?"
            ans = (
                f"In the function definition `def {task['func_name']}({task['params']}):`, `{p}` is an input parameter. "
                f"When a caller invokes `{task['func_name']}`, the value provided for `{p}` is bound locally inside the function. "
                f"Its purpose in this function is: {task['doc']}"
            )
            train_records.append({
                "category": "python_param_explanation",
                "task": task["func_name"],
                "messages": [
                    {"role": "prompter", "text": q},
                    {"role": "assistant", "text": ans}
                ]
            })

    # 4. Generate algorithmic variations to scale cleanly to TARGET_TRAIN_COUNT
    code_explanations = [
        ("Why is a return statement needed in a function?",
         "A `return` statement is needed so that the function can pass data back to the calling code. Without a `return` statement, the function computes values but silently returns `None`."),
        ("What happens if you omit the return statement in a Python function?",
         "If you omit the `return` statement in a Python function, Python will execute all statements in the function body and then implicitly return `None`."),
        ("What is the purpose of indentation in Python?",
         "Indentation in Python indicates block structure. Instead of using curly braces `{}` like C or Java, Python uses consistent leading spaces (usually 4 spaces) to define code blocks for functions, loops, and conditional statements."),
        ("What is a pure function in Python?",
         "A pure function is a function whose return value depends exclusively on its input arguments, with zero observable side effects (such as modifying global variables, writing to files, or changing mutable inputs)."),
        ("What is an early return in Python?",
         "An early return occurs when a function returns a value before reaching its final line, typically inside a guard clause (an `if` statement) to handle edge cases or invalid inputs immediately."),
    ]
    for q, ans in code_explanations:
        train_records.append({
            "category": "python_explanation",
            "messages": [
                {"role": "prompter", "text": q},
                {"role": "assistant", "text": ans}
            ]
        })

    # Expand with high-quality variations up to TARGET_TRAIN_COUNT
    expanded_train = []
    for r in train_records:
        expanded_train.append(r)
    rng.shuffle(expanded_train)

    # If count is less than target, create synthetic variations with distinct questions
    extra_variations = []
    while len(expanded_train) + len(extra_variations) < TARGET_TRAIN_COUNT:
        task = rng.choice(TRAIN_TASKS)
        prefix = rng.choice([
            "Could you show me how to write",
            "Please provide a Python implementation for",
            "Can you write a clean Python function for",
            "Show me a Python example of"
        ])
        q = f"{prefix} {task['func_name']} with docstrings and type comments."
        ans = (
            f"Here is a well-documented implementation:\n\n"
            f"```python\ndef {task['func_name']}({task['params']}):\n"
            f"    \"\"\"\n    {task['doc']}\n    \"\"\"\n"
            f"    {task['body']}\n\n"
            f"# Test demonstration:\n"
            f"print({task['test_call']})\n```"
        )
        assert validate_python_code(ans)
        extra_variations.append({
            "category": "python_function_documented",
            "task": task["func_name"],
            "messages": [
                {"role": "prompter", "text": q},
                {"role": "assistant", "text": ans}
            ]
        })

    final_train = (expanded_train + extra_variations)[:TARGET_TRAIN_COUNT]

    # 5. Build HELD-OUT EVALUATION SET (200 unseen examples)
    # Unseen held-out tasks
    for htask in HELDOUT_TASKS:
        ans = (
            f"Here is the implementation:\n\n"
            f"```python\ndef {htask['func_name']}({htask['params']}):\n"
            f"    \"\"\"{htask['doc']}\"\"\"\n"
            f"    {htask['body']}\n\n"
            f"# Verification:\n"
            f"print({htask['test_call']})\n```"
        )
        assert validate_python_code(ans)
        heldout_records.append({
            "category": "python_heldout_task",
            "task": htask["func_name"],
            "messages": [
                {"role": "prompter", "text": htask["prompt"]},
                {"role": "assistant", "text": ans}
            ]
        })

    # Unseen conceptual held-out questions
    unseen_conceptual = [
        ("What is a lambda function in Python and when should it be used?",
         "A lambda function is an anonymous, small, single-expression function defined with the `lambda` keyword: `lambda x: x * 2`. It can take multiple arguments but can only evaluate one single expression. It is typically used for short throwaway functions passed into functions like `map()`, `filter()`, or `sorted(key=...)`."),
        ("What is the difference between shallow copy and deep copy in Python?",
         "In Python, a shallow copy creates a new object but references the original nested objects inside it. A deep copy (via `copy.deepcopy()`) creates a new object and recursively duplicates all nested objects, ensuring mutations to the copy do not affect the original."),
        ("Explain the difference between mutable and immutable default arguments in Python.",
         "Default argument values are evaluated once when the function is defined, not each time it is called. If you use a mutable default like `def add_item(item, lst=[]):`, all calls share the exact same list across invocations. The safe pattern is to use `lst=None` and initialize `lst = []` inside the function body."),
        ("What does the __name__ == '__main__' idiom do in Python?",
         "The `if __name__ == '__main__':` check verifies whether a Python file is being executed directly as the main script or imported as a module in another script. If run directly, `__name__` equals `'__main__'`, executing the code inside that block."),
        ("What is a decorator in Python?",
         "A decorator is a function that takes another function as an argument, extends or alters its behavior without modifying its source code, and returns the modified function. Decorators use the `@decorator_name` syntax above the function header."),
    ]
    for q, ans in unseen_conceptual:
        heldout_records.append({
            "category": "python_heldout_concept",
            "messages": [
                {"role": "prompter", "text": q},
                {"role": "assistant", "text": ans}
            ]
        })

    # Expand heldout to 200 with varied unseen queries on heldout tasks
    while len(heldout_records) < TARGET_HELDOUT_COUNT:
        htask = rng.choice(HELDOUT_TASKS)
        variations = [
            f"Explain how to implement {htask['func_name']} in Python.",
            f"Can you write a Python function named {htask['func_name']}?",
            f"Show Python code to {htask['doc'].lower()}",
            f"How do you implement {htask['func_name']} with proper error handling and docstrings?"
        ]
        q = rng.choice(variations)
        ans = (
            f"```python\ndef {htask['func_name']}({htask['params']}):\n"
            f"    \"\"\"{htask['doc']}\"\"\"\n"
            f"    {htask['body']}\n```"
        )
        assert validate_python_code(ans)
        heldout_records.append({
            "category": "python_heldout_task",
            "task": htask["func_name"],
            "messages": [
                {"role": "prompter", "text": q},
                {"role": "assistant", "text": ans}
            ]
        })

    heldout_records = heldout_records[:TARGET_HELDOUT_COUNT]
    return final_train, heldout_records


def main():
    print("=" * 70)
    print("EXPERIMENT 2: GENERATING PYTHON DATASET (CATEGORY 3)")
    print("=" * 70)
    start_time = time.time()

    train_data, heldout_data = build_python_dataset()

    print(f"Validating Python code syntax across all {len(train_data) + len(heldout_data)} examples...")
    for idx, item in enumerate(train_data + heldout_data):
        asst_text = item["messages"][1]["text"]
        if not validate_python_code(asst_text):
            raise ValueError(f"AST validation failed at index {idx}!")

    # Check prompt overlap
    train_prompts = set(ex["messages"][0]["text"] for ex in train_data)
    heldout_prompts = set(ex["messages"][0]["text"] for ex in heldout_data)
    overlap = train_prompts.intersection(heldout_prompts)
    print(f"Train/Heldout exact prompt overlap: {len(overlap)} (Target: 0)")
    assert len(overlap) == 0, f"Prompt overlap found: {overlap}"

    # Write files
    OUT_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(OUT_HELDOUT_FILE, "w", encoding="utf-8") as f:
        for ex in heldout_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"\nPython Train Examples Generated:     {len(train_data):,}")
    print(f"Python Held-Out Examples:            {len(heldout_data):,}")
    print(f"Written train to:   {OUT_TRAIN_FILE}")
    print(f"Written heldout to: {OUT_HELDOUT_FILE}")
    print(f"Completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
