# Comprehensive Baseline Failure Analysis: Experiment 2 Reference Checkpoint

**Checkpoint Evaluated:** `checkpoints/experiment_2/best.pt`  
**Verified SHA-256:** `f717e5efff9dc6d58b0fc1d4872e74b0b03bd734029715a3ae14aa549b5b4ab1`  
**Tokenizer SHA-256:** `0f00fae1c3852879cb0bb9a807b45aad93d7700ca4e3f903b1c607ac8c565b08`  
**Evaluation Script:** `src/ultimate_improvement/evaluate_comprehensive.py`  
**Evaluation Date:** 2026-10-27  
**Raw Metrics Data:** `logs/ultimate_improvement/baseline_metrics.json`  

---

## 1. Executive Summary of Baseline Performance

Across 248 systematically evaluated test cases spanning 12 distinct evaluation suites and 14 functional dimensions, the reference checkpoint demonstrates clear strengths in specialized domain concepts, elementary single-digit subtraction, and clean EOS termination on standard benchmarks, but reveals severe structural failure modes in general reasoning, multi-step problem solving, instruction compliance, and multi-turn dialogue.

### Baseline Summary Metrics:

| Evaluation Dimension | Test Suite | Total Cases | Successes | Failures | Baseline Success Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Historical Benchmarks** | Suite 01 (8 Benchmarks) | 8 | 8 clean EOS | 0 loops | **100.0% EOS (0% loops)** |
| **Elementary Arithmetic** | Suite 02 (Single-step) | 100 | 6 | 94 | **6.00%** |
| **Multi-Step Arithmetic** | Suite 03 (Multi-step) | 20 | 0 | 20 | **0.00%** |
| **Python Code Generation** | Suite 04 (Executable code) | 15 | 1 syntax / 0 func | 15 func | **0.00% functional (6.67% syntax)** |
| **Concept Explanations** | Suite 05 (Untouched concepts) | 97 key points | 29 matched | 68 missing | **29.90% key-point recall** |
| **Basic Factual QA** | Suite 06 (General facts) | 20 | 2 | 18 | **10.00%** |
| **Instruction Following** | Suite 07 (Constraints) | 15 | 1 | 14 | **6.67%** |
| **Structured Formatting** | Suite 08 (JSON / Tables) | 10 | 0 | 10 | **0.00%** |
| **Multi-Turn Consistency**| Suite 09 (Dialogue memory) | 10 | 3 | 7 | **30.00%** |
| **Refusal & Uncertainty** | Suite 10 (Impossibility QA) | 10 | 1 | 9 | **10.00%** |
| **Input Robustness** | Suite 11 (Malformed inputs) | 10 | 10 handled | 0 crashes | **100.00% (No crashes)** |
| **Context Length Boundary**| Suite 12 (Long context) | 5 | 0 | 5 | **0.00%** |
| **Overall Behavior** | All Suites Combined | 248 | 133 EOS (53.6%) | 46 loops (18.5%) | **53.63% EOS / 18.55% loops** |

---

## 2. In-Depth Failure Mode Analysis by Category

### Failure Mode 1: Subword Bigram Degeneration Traps in Code Generation
* **Empirical Evidence:**
  - Prompt: *"Write a Python function named celsius_to_kelvin(celsius) that converts a Celsius temperature to Kelvin..."*
  - Generated Output:
    ```python
    def celsius_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_to_...
    ```
  - Prompt: *"Write a Python function named is_power_of_two(n)..."*
  - Generated Output:
    ```python
    def in_of_of_of_of_of_of_of_of_of_of_of_of_of_of_of_of_of_of_of_of_of_...
    ```
* **Root Cause Diagnosis:**
  1. Tokenizer BPE representation: The identifier `celsius_to_to` is split into subwords `_` (ID 68) and `to` (ID 1271).
  2. In greedy decoding, the model's highest logit after `_` is `to`, and after `to` is `_`.
  3. The standard token repetition penalty (`repetition_penalty = 1.15`) penalizes tokens in `set(generated_tokens)` by dividing positive logits by 1.15. When raw logits exceed +15.0, scaling down to +13.0 still leaves them significantly higher than all other 7,998 tokens.
  4. The model was trained on only 1,800 Python snippets where function names followed standard names (`add_numbers`, `rectangle_area`), but never learned to copy arbitrary unseen identifier names from prompts.

### Failure Mode 2: Multi-Digit & Multi-Step Arithmetic Breakdown
* **Empirical Evidence:**
  - Single-digit arithmetic: Achieves **33.3% accuracy** on unseen 1-digit subtractions (`8 - 5 = 3`, `7 - 0 = 7`, `8 - 0 = 8`, `7 - 6 = 1`, `4 - 1 = 3`).
  - Multi-digit arithmetic: **0.0% accuracy** across 65 problems (Tier 3 through Tier 6).
  - Multi-step arithmetic: **0.0% accuracy** (0/20).
    - Prompt: *"If Sarah has 15 apples, buys 10 more, and then gives 7 apples to Bob, how many apples does Sarah have left?"*
    - Model Output: *"If you have 15 apples, you can use the following formula: 15 apples x 10 apples = 150 apples."* (Hallucinates multiplication of apples).
* **Root Cause Diagnosis:**
  1. Direct zero-shot prediction of final numerical answers requires the Transformer to perform multi-column mental addition, carry-over, and multi-step tracking inside fixed-depth feed-forward layers ($L=6, D=384$).
  2. A 14.1M parameter model does not possess sufficient circuit depth to compute multi-column carry arithmetic directly in a single forward pass without intermediate scratchpad reasoning tokens.
  3. In Experiment 2 training, arithmetic answers were mixed across direct format (`a + b = c`), wordy format, and step format. The model frequently defaulted to emitting direct numbers rather than working through intermediate steps.

### Failure Mode 3: Instruction & Format Blindness (Semantic Drift)
* **Empirical Evidence:**
  - Prompt: *"Answer this question in exactly three words: What color is a clear daytime sky?"*
  - Model Output: *"Sure! Here's a few examples of the colors that can be used to make the sky..."* (49 words, completely ignores the 3-word instruction).
  - Prompt: *"Reply with only the word YES or NO: Is Python a programming language?"*
  - Model Output: *"Here is a well-documented implementation:\n\n```python\ndef is_void_void_void(text)..."* (Sees the word "Python" and triggers its code generator instead of answering YES).
  - Prompt: *"Output a JSON object with keys 'status' (value 'success') and 'code' (value 200). Output only valid JSON."*
  - Model Output: English prose describing a Python dictionary, zero valid JSON tokens emitted.
* **Root Cause Diagnosis:**
  1. The training dataset consists almost exclusively of conversational OpenAssistant turns and unconstrained question-answering examples.
  2. The model has **zero exposure** to meta-instructions such as word limits, strict negative constraints ("do not use the letter e"), or structured output formats (raw JSON, markdown tables).
  3. Consequently, strong keyword triggers (e.g. "Python") dominate attention, bypassing instruction tokens entirely.

### Failure Mode 4: Tautological Degeneration on Unfamiliar Factual Queries
* **Empirical Evidence:**
  - Prompt: *"What is the capital of France?"*
  - Model Output: *"The capital of the capital of the capital of the capital is the capital of the c..."*
  - Prompt: *"What is the largest planet in our solar system?"*
  - Model Output: *"The largest solar system is the solar system in the solar system, which is the s..."*
* **Root Cause Diagnosis:**
  1. When queried on entities absent from the 7,422 curriculum sequences (e.g. world capitals, astronomy), the model has low epistemic confidence.
  2. Instead of terminating or stating uncertainty, greedy decoding falls into the high-probability ngram loop of repeating the prompt's head noun ("capital", "solar system").

### Failure Mode 5: Conversational Memory Loss in Multi-Turn Dialogue
* **Empirical Evidence:**
  - Turn 1: *"Hello! My name is Shanveer."* $\rightarrow$ Model: *"I'm sorry, but I am not sure you have an..."*
  - Turn 2: *"What is my name?"* $\rightarrow$ Model: *"I'm sorry, but I am not sure you have an..."*
* **Root Cause Diagnosis:**
  1. The training curriculum contained 2,972 OASST1 conversations, but short introductory greetings ("My name is X") were filtered out by the minimum length filter ($\ge 18$ words, $\ge 60$ characters).
  2. The model's training on personal dialogue was dominated by canned assistant disclaimers or philosophical essays, creating a strong negative bias against brief personal interaction.

---

## 3. Verified Hypotheses & Strategic Recommendations

Based on these empirical failure modes, the path to maximizing the model's accuracy and reliability requires addressing four specific pillars:

1. **Decoding Enhancement (Immediate Correctness Fix):**
   - Introduce n-gram repetition blocking (e.g., `no_repeat_ngram_size = 3`) during inference. This mathematically guarantees that loops like `to_to_to` and `capital of the capital of` cannot occur.
2. **Scratchpad (Chain-of-Thought) Reasoning Curriculum:**
   - For multi-step arithmetic and numerical problem solving, train the model to output explicit intermediate `<thought>` steps (e.g. `Step 1: 15 + 10 = 25. Step 2: 25 - 7 = 18. Final Answer: 18.`) before stating the result.
3. **Targeted Algorithmic & Instruction Curriculum:**
   - Augment training data with:
     - Verified algorithmic Python functions covering string processing, numerical logic, and list manipulation with matching unit tests.
     - Instruction-following examples (word count limits, YES/NO responses, JSON format, markdown tables).
     - Factual knowledge QA (geography, basic science, computing facts).
     - Conversational turn-taking and persona consistency (remembering user name and facts).
4. **Deterministic Tool Grounding (Optional Hybrid Assistance):**
   - For deterministic calculations and Python execution where a 14M language model alone is fundamentally limited, provide a clean, clearly labeled tool-assisted mode.
