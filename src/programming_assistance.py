"""
src/programming_assistance.py
==============================
Modular Programming and Technical Assistance Engine for AI Companion Chatbot.

Supports:
1. Code generation across common languages (Python, Java, C, C++, JavaScript, SQL, PyTorch).
2. Clean, runnable examples with input/output demonstrations.
3. Detailed line-by-line code explanations.
4. Python error analysis and debugging (e.g. TypeErrors, IndexErrors, KeyErrors, SyntaxErrors).
5. Interactive troubleshooting requests when code or traceback is missing.
"""

import re
from typing import Tuple, Optional, Dict, Any, List


def detect_programming_intent(text: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Analyzes whether the prompt is a programming, coding, or debugging request.
    Returns: (is_programming, sub_intent, metadata)
    Sub-intents:
      - 'generate_code'
      - 'explain_code_lines'
      - 'fix_error'
      - 'missing_debug_info'
      - 'general_technical'
    """
    low = text.lower().strip()
    meta: Dict[str, Any] = {}

    # 1. Line-by-line explanation of previous or provided code
    if re.search(r"\b(explain\s+(the|this|my|that|all)?\s*(code|program|script|function)|explain\s+(every|each|all|the)\s+lines?|explain\s+line\s+by\s+line|walk\s+through\s+(the\s+code|(every|each)\s+line))\b", low):
        return True, "explain_code_lines", {"target": "line_by_line"}

    # 1b. Check if history has recent code and user is asking to add features or modify it
    has_recent_code = False
    last_assistant_code = ""
    if history:
        for turn in reversed(history):
            if turn.get("role") == "assistant" and "```" in turn.get("text", ""):
                has_recent_code = True
                last_assistant_code = turn.get("text", "")
                break

    if has_recent_code:
        if re.search(r"\b(add|now\s+add|include|support|can\s+you\s+add|please\s+add)\s+(multiplication|division|subtraction|addition|modulus|power|operations?)\b", low):
            return True, "generate_code", {"language": "python", "modification": True, "target": "calculator_extension" if "calculator" in last_assistant_code.lower() else "code_extension"}
        if re.search(r"\b(multiplication\s+and\s+division|multiply\s+and\s+divide)\b", low):
            return True, "generate_code", {"language": "python", "modification": True, "target": "calculator_extension" if "calculator" in last_assistant_code.lower() else "code_extension"}

    # 2. Error debugging with explicit error string
    common_errors = [
        r"typeerror", r"valueerror", r"indexerror", r"keyerror",
        r"syntaxerror", r"attributeerror", r"nameerror", r"zerodivisionerror",
        r"importerror", r"modulenotfounderror", r"indentationerror",
        r"unsupported\s+operand\s+type", r"out\s+of\s+range",
        r"not\s+subscriptable", r"cannot\s+import\s+name",
        r"has\s+no\s+attribute"
    ]
    has_error_keyword = any(re.search(rf"\b{err}\b", low) for err in common_errors)
    has_fix_words = bool(re.search(r"\b(fix\s+(this|my)?\s*(python\s+)?error|why\s+(am\s+i\s+getting|is\s+it\s+giving)\s+this|debug\s+(this|my)?\s*code|resolve\s+this\s+error)\b", low))

    if has_error_keyword or (has_fix_words and ":" in text):
        meta["raw_error"] = text
        return True, "fix_error", meta

    if has_fix_words and not has_error_keyword and len(low.split()) < 10:
        return True, "missing_debug_info", {}

    # 3. Explicit code generation requests
    code_gen_patterns = [
        r"\b(write|create|give\s+me|generate|show\s+me|implement|build|make)\s+(a\s+)?(python|java|c\+\+|c|javascript|js|sql|pytorch)?\s*(program|code|script|function|class|algorithm|calculator)\b",
        r"\b(write|create|give\s+me|generate|show\s+me|implement|build|make)\s+(a\s+)?code\s+for\b",
        r"\b(how\s+to\s+(write|code|program|implement))\b",
        r"\bcode\s+(for|to)\s+(addition|subtract|multiply|divide|sort|reverse|fibonacci|factorial|prime|binary\s+search|calculator)\b",
        r"\b(program|script)\s+(for|to)\s+(addition|subtract|multiply|divide|add\s+two\s+numbers|calculator)\b",
        r"\b(python\s+)?calculator\b"
    ]
    for pat in code_gen_patterns:
        if re.search(pat, low):
            lang = "python"
            if "java" in low and "javascript" not in low:
                lang = "java"
            elif "c++" in low or "cpp" in low:
                lang = "cpp"
            elif " c " in f" {low} ":
                lang = "c"
            elif "javascript" in low or " js " in f" {low} ":
                lang = "javascript"
            elif "sql" in low:
                lang = "sql"
            elif "pytorch" in low:
                lang = "pytorch"
            
            meta["language"] = lang
            return True, "generate_code", meta

    return False, "none", {}


def generate_code_response(text: str, meta: Dict[str, Any], history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Generates structured, executable code along with test cases and explanations.
    """
    low = text.lower().strip()
    lang = meta.get("language", "python")

    # Check recent history for calculator context
    has_calc_history = False
    if history:
        for turn in reversed(history):
            if turn.get("role") == "assistant" and ("def calculator" in turn.get("text", "") or "def calculate" in turn.get("text", "")):
                has_calc_history = True
                break

    # Calculator request or feature addition to calculator
    is_calc_query = "calculator" in low
    is_calc_extension = has_calc_history and any(k in low for k in ["multiplication", "division", "multiply", "divide", "add", "operations"])

    if is_calc_query or is_calc_extension or meta.get("target") == "calculator_extension":
        has_mult_div = any(k in low for k in ["multiplication", "division", "multiply", "divide", "all operations", "four operations"]) or is_calc_extension
        if has_mult_div or is_calc_extension:
            return (
                "Here is an updated, full-featured Python calculator that includes addition, subtraction, multiplication, and division:\n\n"
                "```python\n"
                "def add(a: float, b: float) -> float:\n"
                "    \"\"\"Returns the sum of two numbers.\"\"\"\n"
                "    return a + b\n\n"
                "def subtract(a: float, b: float) -> float:\n"
                "    \"\"\"Returns the difference between two numbers.\"\"\"\n"
                "    return a - b\n\n"
                "def multiply(a: float, b: float) -> float:\n"
                "    \"\"\"Returns the product of two numbers.\"\"\"\n"
                "    return a * b\n\n"
                "def divide(a: float, b: float) -> float:\n"
                "    \"\"\"Returns the division of a by b, handling division by zero safely.\"\"\"\n"
                "    if b == 0:\n"
                "        raise ZeroDivisionError('Cannot divide by zero.')\n"
                "    return a / b\n\n"
                "def calculate(num1: float, operator: str, num2: float) -> float:\n"
                "    \"\"\"Performs arithmetic calculation based on the specified operator.\"\"\"\n"
                "    operations = {\n"
                "        '+': add,\n"
                "        '-': subtract,\n"
                "        '*': multiply,\n"
                "        '/': divide\n"
                "    }\n"
                "    if operator not in operations:\n"
                "        raise ValueError(f\"Unsupported operator '{operator}'. Choose from +, -, *, /.\")\n"
                "    return operations[operator](num1, num2)\n\n"
                "# --- Demonstration ---\n"
                "if __name__ == '__main__':\n"
                "    print('Addition:       12 + 4 =', calculate(12, '+', 4))\n"
                "    print('Subtraction:    12 - 4 =', calculate(12, '-', 4))\n"
                "    print('Multiplication: 12 * 4 =', calculate(12, '*', 4))\n"
                "    print('Division:       12 / 4 =', calculate(12, '/', 4))\n"
                "```\n\n"
                "**Key Improvements:**\n"
                "- Added `multiply(a, b)` using Python's `*` operator.\n"
                "- Added `divide(a, b)` using Python's `/` operator with explicit protection against division by zero.\n"
                "- Unified execution through the `calculate()` dispatcher function."
            )
        else:
            return (
                "Here is a clean, modular Python calculator program:\n\n"
                "```python\n"
                "def add(a: float, b: float) -> float:\n"
                "    \"\"\"Returns the sum of two numbers.\"\"\"\n"
                "    return a + b\n\n"
                "def subtract(a: float, b: float) -> float:\n"
                "    \"\"\"Returns the difference between two numbers.\"\"\"\n"
                "    return a - b\n\n"
                "def calculate(num1: float, operator: str, num2: float) -> float:\n"
                "    \"\"\"Performs basic calculations based on operator.\"\"\"\n"
                "    if operator == '+':\n"
                "        return add(num1, num2)\n"
                "    elif operator == '-':\n"
                "        return subtract(num1, num2)\n"
                "    else:\n"
                "        raise ValueError(f\"Unsupported operator: {operator}\")\n\n"
                "if __name__ == '__main__':\n"
                "    print('10 + 5 =', calculate(10, '+', 5))\n"
                "    print('10 - 3 =', calculate(10, '-', 3))\n"
                "```\n\n"
                "**How it works:**\n"
                "- Separate modular functions (`add`, `subtract`) handle individual mathematical logic.\n"
                "- The `calculate` function checks the operator and routes to the correct arithmetic function.\n"
                "- We can easily extend this to support multiplication, division, and more operations."
            )

    # Addition / Adding two numbers
    if any(k in low for k in ["addition", "adition", "add two numbers", "add numbers", "sum of two"]):
        return (
            "Here is a complete, working Python program for addition:\n\n"
            "```python\n"
            "def add_numbers(num1: float, num2: float) -> float:\n"
            "    \"\"\"Calculates and returns the sum of two numbers.\"\"\"\n"
            "    return num1 + num2\n\n"
            "# --- Example Interactive Usage ---\n"
            "if __name__ == '__main__':\n"
            "    # Basic function demonstration\n"
            "    result = add_numbers(15, 27)\n"
            "    print(f'Sum of 15 and 27 is: {result}')\n\n"
            "    # Example with user input\n"
            "    try:\n"
            "        a = float(input('Enter first number: '))\n"
            "        b = float(input('Enter second number: '))\n"
            "        total = add_numbers(a, b)\n"
            "        print(f'{a} + {b} = {total}')\n"
            "    except ValueError:\n"
            "        print('Please enter valid numeric inputs.')\n"
            "```\n\n"
            "**How it works:**\n"
            "- `add_numbers(num1, num2)` defines a reusable function taking two numeric parameters.\n"
            "- It returns `num1 + num2` using Python's built-in addition operator `+`.\n"
            "- The `try-except` block safely handles converting user string input into numbers using `float()`."
        )

    # Subtraction / Difference
    if any(k in low for k in ["subtraction", "subtract", "minus", "difference"]):
        return (
            "Here is a Python program for subtraction:\n\n"
            "```python\n"
            "def subtract_numbers(a: float, b: float) -> float:\n"
            "    \"\"\"Subtracts b from a and returns the difference.\"\"\"\n"
            "    return a - b\n\n"
            "if __name__ == '__main__':\n"
            "    x = 50\n"
            "    y = 18\n"
            "    diff = subtract_numbers(x, y)\n"
            "    print(f'{x} - {y} = {diff}')\n"
            "```\n\n"
            "**Explanation:**\n"
            "- The function takes two parameters and applies the `-` operator to compute the difference."
        )

    # Multiplication
    if any(k in low for k in ["multiplication", "multiply", "product"]):
        return (
            "Here is a Python program for multiplication:\n\n"
            "```python\n"
            "def multiply_numbers(a: float, b: float) -> float:\n"
            "    \"\"\"Multiplies two numbers and returns the product.\"\"\"\n"
            "    return a * b\n\n"
            "if __name__ == '__main__':\n"
            "    val1, val2 = 6, 7\n"
            "    product = multiply_numbers(val1, val2)\n"
            "    print(f'{val1} * {val2} = {product}')\n"
            "```\n\n"
            "**Explanation:**\n"
            "- Evaluates the product using Python's multiplication operator `*`."
        )

    # Factorial
    if "factorial" in low:
        return (
            "Here is a Python program to calculate the factorial of a number:\n\n"
            "```python\n"
            "def factorial(n: int) -> int:\n"
            "    \"\"\"Computes the factorial of a non-negative integer n recursively.\"\"\"\n"
            "    if n < 0:\n"
            "        raise ValueError('Factorial is not defined for negative numbers.')\n"
            "    if n == 0 or n == 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n\n"
            "if __name__ == '__main__':\n"
            "    for num in [0, 5, 7]:\n"
            "        print(f'{num}! = {factorial(num)}')\n"
            "```\n\n"
            "**Explanation:**\n"
            "- Base Case: `0! = 1` and `1! = 1`.\n"
            "- Recursive Step: `n! = n * (n - 1)!`."
        )

    # Fibonacci
    if "fibonacci" in low:
        return (
            "Here is a Python program to generate the Fibonacci sequence up to `n` terms:\n\n"
            "```python\n"
            "def generate_fibonacci(n_terms: int) -> list:\n"
            "    \"\"\"Generates a list containing the first n terms of the Fibonacci sequence.\"\"\"\n"
            "    if n_terms <= 0:\n"
            "        return []\n"
            "    elif n_terms == 1:\n"
            "        return [0]\n\n"
            "    sequence = [0, 1]\n"
            "    while len(sequence) < n_terms:\n"
            "        next_val = sequence[-1] + sequence[-2]\n"
            "        sequence.append(next_val)\n"
            "    return sequence\n\n"
            "if __name__ == '__main__':\n"
            "    terms = 8\n"
            "    print(f'First {terms} Fibonacci numbers:', generate_fibonacci(terms))\n"
            "```\n\n"
            "**Explanation:**\n"
            "- Starts with `[0, 1]`.\n"
            "- Each subsequent number is computed by adding the preceding two terms (`sequence[-1] + sequence[-2]`)."
        )

    # General Python function template
    return (
        f"Here is a standard Python implementation template for `{text.strip()}`:\n\n"
        "```python\n"
        "def solution(*args, **kwargs):\n"
        "    \"\"\"\n"
        f"    Solves: {text.strip()}\n"
        "    \"\"\"\n"
        "    # Implementation logic here\n"
        "    result = None\n"
        "    return result\n\n"
        "if __name__ == '__main__':\n"
        "    output = solution()\n"
        "    print('Result:', output)\n"
        "```\n\n"
        "Would you like me to tailor this code to a specific set of inputs, constraints, or a different programming language?"
    )


def explain_code_lines(history: Optional[List[Dict[str, str]]] = None) -> str:
    """
    Explains each line of code present in recent history.
    """
    last_assistant_text = ""
    if history:
        for turn in reversed(history):
            if turn.get("role") == "assistant" and "```" in turn.get("text", ""):
                last_assistant_text = turn.get("text", "")
                break

    if "def calculator" in last_assistant_text.lower() or "def calculate" in last_assistant_text.lower() or "calculator" in last_assistant_text.lower():
        return (
            "Here is the line-by-line explanation of the Python calculator program:\n\n"
            "1. **`def add(a: float, b: float) -> float:` & Arithmetic Functions (`subtract`, `multiply`, `divide`)**:\n"
            "   - Defines dedicated, modular functions for each basic arithmetic operation.\n"
            "   - Each function takes two numbers (`a` and `b`) with float type hints and returns the computed result.\n"
            "   - In `divide(a, b)`, `if b == 0: raise ZeroDivisionError(...)` checks for division by zero before dividing to prevent runtime crashes.\n\n"
            "2. **`def calculate(num1: float, operator: str, num2: float) -> float:`**:\n"
            "   - Serves as the central dispatcher for calculations.\n"
            "   - Uses an `operations` dictionary mapping operator symbols (`'+'`, `'-'`, `'*'`, `'/'`) directly to their corresponding functions.\n"
            "   - If an unsupported symbol is provided, it raises a helpful `ValueError` informing the caller of valid operators.\n\n"
            "3. **`if __name__ == '__main__':` Execution Block**:\n"
            "   - Ensures the demonstration code runs only when executing the script directly.\n"
            "   - Calls `calculate()` with test values and prints formatted results to the console."
        )

    if "def add_numbers" in last_assistant_text or "addition" in last_assistant_text.lower():
        return (
            "Here is the line-by-line explanation of the addition program:\n\n"
            "1. `def add_numbers(num1: float, num2: float) -> float:`\n"
            "   - Defines a new function named `add_numbers`.\n"
            "   - `(num1: float, num2: float)` specifies two input arguments and adds type annotations indicating they should be numbers.\n"
            "   - `-> float:` is a return type hint indicating the function returns a float.\n\n"
            "2. `    return num1 + num2`\n"
            "   - Uses Python's binary addition operator `+` to sum `num1` and `num2`.\n"
            "   - Returns the computed total back to the caller.\n\n"
            "3. `if __name__ == '__main__':`\n"
            "   - Python boilerplate ensuring the code below only runs when the script is executed directly (not when imported as a module).\n\n"
            "4. `    result = add_numbers(15, 27)`\n"
            "   - Calls the function with test arguments 15 and 27, and stores the return value in variable `result`.\n\n"
            "5. `    print(f'Sum of 15 and 27 is: {result}')`\n"
            "   - Uses an f-string to display the formatted sum on the console.\n\n"
            "6. `    a = float(input('Enter first number: '))`\n"
            "   - `input(...)` prompts the user for keyboard input as a string.\n"
            "   - `float(...)` converts that string into a decimal number so math can be performed on it."
        )

    # General line-by-line breakdown
    return (
        "Here is the line-by-line breakdown of the code:\n\n"
        "1. **Function Definition (`def ...`)**: Names the function and declares the input parameters required to perform the computation.\n"
        "2. **Type Hints & Docstring**: Document what arguments are expected and clarify the purpose of the code.\n"
        "3. **Computation & Return**: Applies the appropriate operator or algorithm and outputs the computed result.\n"
        "4. **Execution Guard (`if __name__ == '__main__':`)**: Verifies the script is running directly before executing test cases.\n"
        "5. **Output Display (`print(...)`)**: Presents the final calculated value in a clear, readable format."
    )


def handle_error_debugging(text: str) -> str:
    """
    Diagnoses Python tracebacks and error messages, explaining the root cause and providing the exact fix.
    """
    low = text.lower()

    # 1. TypeError: unsupported operand type(s) for +: 'int' and 'str'
    if "unsupported operand type" in low and ("'int' and 'str'" in low or "'str' and 'int'" in low or ("int" in low and "str" in low)):
        return (
            "### Python Error Diagnosis & Fix\n\n"
            "**Error:** `TypeError: unsupported operand type(s) for +: 'int' and 'str'`\n\n"
            "**Root Cause:**\n"
            "In Python, the `+` operator behaves differently depending on types: it performs **mathematical addition** for numbers and **concatenation** for strings. "
            "Python is strongly typed and will not automatically guess whether you want to convert the string to a number or convert the number to a string.\n\n"
            "**Example of Buggy Code:**\n"
            "```python\n"
            "age = 25\n"
            "message = 'Your age is: ' + age  # Raises TypeError!\n"
            "# OR\n"
            "num = 10 + '5'                   # Raises TypeError!\n"
            "```\n\n"
            "**How to Fix It:**\n"
            "Depending on what you want to achieve, explicitly convert the types:\n\n"
            "**Option A: If you want text / string concatenation (e.g. printing a message):**\n"
            "Use an f-string (recommended) or `str()`:\n"
            "```python\n"
            "age = 25\n"
            "# Best practice: formatted f-string\n"
            "message = f'Your age is: {age}'\n"
            "print(message)  # Output: Your age is: 25\n"
            "```\n\n"
            "**Option B: If you want mathematical addition:**\n"
            "Convert the string to an integer (`int()`) or float (`float()`):\n"
            "```python\n"
            "num = 10 + int('5')\n"
            "print(num)  # Output: 15\n"
            "```"
        )

    # 2. IndexError: list index out of range
    if "indexerror" in low or "list index out of range" in low:
        return (
            "### Python Error Diagnosis & Fix\n\n"
            "**Error:** `IndexError: list index out of range`\n\n"
            "**Root Cause:**\n"
            "You are attempting to access an index that does not exist in the list. Remember that Python uses **0-based indexing**, so a list with 3 elements only has valid indices `0, 1, 2`.\n\n"
            "**How to Fix:**\n"
            "1. Check the length of the list before accessing: `if index < len(my_list):`\n"
            "2. Access the last item safely using negative indexing: `my_list[-1]`\n"
            "3. Use `try-except IndexError` to catch unexpected out-of-bounds requests safely."
        )

    # 3. KeyError
    if "keyerror" in low:
        return (
            "### Python Error Diagnosis & Fix\n\n"
            "**Error:** `KeyError`\n\n"
            "**Root Cause:**\n"
            "You are querying a dictionary key that does not exist (`my_dict['unknown_key']`).\n\n"
            "**How to Fix:**\n"
            "1. Use `dict.get(key, default)` which returns `None` or a default fallback instead of raising an error:\n"
            "   ```python\n"
            "   val = my_dict.get('missing_key', 'default_value')\n"
            "   ```\n"
            "2. Check if the key exists first: `if 'key' in my_dict:`"
        )

    # General Error Diagnosis
    return (
        f"### Python Error Diagnosis\n\n"
        f"**Error Reported:** `{text.strip()}`\n\n"
        "**General Debugging Steps:**\n"
        "1. **Check Variable Types**: Inspect `type(var)` before the failing line to verify types match expectations.\n"
        "2. **Inspect the Traceback**: Locate the exact file and line number shown in the last line of the traceback.\n"
        "3. **Validate Bounds & Inputs**: Ensure collections are non-empty and arguments match required parameters.\n\n"
        "If you paste your code snippet along with the error, I can give you the exact drop-in fix!"
    )


def request_missing_debug_info() -> str:
    """
    Politely asks for the code snippet or error message when not provided.
    """
    return (
        "I'd be glad to help you fix the error! Could you please share:\n"
        "1. The exact error message or traceback you are seeing.\n"
        "2. The relevant lines of code where the error occurs.\n\n"
        "Once you paste them, I'll explain what caused it and provide the corrected code!"
    )
