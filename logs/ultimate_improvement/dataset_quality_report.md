# Comprehensive Dataset Quality Report: Pre-Training & Curriculum Audits

**Project Root:** `D:\DEEP-LEARNING-PROJECT`  
**Audited Datasets:** `data/experiment_2/curriculum_raw/`, `data/experiment_2/training_sequences/`, `data/processed/`  
**Tokenizer:** Byte-Pair Encoding (`tokenizer/tokenizer.json`, 8,000 vocabulary)  
**Audit Date:** 2026-10-27  
**Raw Audit Data:** `logs/ultimate_improvement/dataset_audit_stats.json`  

---

## 1. Executive Summary & Inventory

This report details the forensic audit of all datasets currently utilized by the project, including token distributions, duplicate density, conversation tree topology, sequence formatting, and potential evaluation leakage across train, validation, and untouched test splits.

### Summary Inventory Table:

| Dataset Category | Source File | Records | Unique Prompts | Duplicate Prompts | Mean Seq Tokens | Min / Max Tokens | Truncation (>1024) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cleaned OASST1** | `category1_oasst1_cleaned.jsonl` | 2,972 | 2,972 | 0 (0.0%) | 315.2 | 50 / 942 | 0 (0.00%) |
| **Arithmetic Train** | `category2_arithmetic_train.jsonl` | 1,800 | 1,800 | 0 (0.0%) | 42.4 | 18 / 77 | 0 (0.00%) |
| **Python Train** | `category3_python_train.jsonl` | 1,800 | 190 | **1,610 (89.4%)** | 144.3 | 42 / 271 | 0 (0.00%) |
| **Concepts Train** | `category4_concepts_train.jsonl` | 850 | 313 | **537 (63.2%)** | 128.2 | 68 / 200 | 0 (0.00%) |
| **Arithmetic Heldout**| `category2_arithmetic_heldout.jsonl` | 200 | 200 | 0 (0.0%) | 45.1 | 18 / 75 | 0 (0.00%) |
| **Python Heldout** | `category3_python_heldout.jsonl` | 200 | 35 | **165 (82.5%)** | 143.5 | 45 / 287 | 0 (0.00%) |
| **Concepts Heldout** | `category4_concepts_heldout.jsonl` | 150 | 80 | **70 (46.7%)** | 110.1 | 62 / 139 | 0 (0.00%) |
| **Combined Curriculum**| `curriculum_train_all.jsonl` | **7,422** | **5,275** | **2,147 (28.9%)** | **172.8** | **18 / 942** | **0 (0.00%)** |

---

## 2. Token-Length Distribution & Sequence Capacity Analysis

### A. Sequence Length Characteristics
1. **OASST1 Conversations (`category1_oasst1_cleaned.jsonl`):**
   - Mean length: **315.2 tokens** (standard deviation: 184.6 tokens).
   - Median length: 278 tokens.
   - Filtering guarantees: All sequences with $< 50$ tokens or $> 950$ tokens were filtered out during preparation. Consequently, **truncation rate is strictly 0.00%**.
2. **Arithmetic Sequences (`category2_arithmetic_train.jsonl`):**
   - Mean length: **42.4 tokens** (compact, focused numerical QA).
   - 77.7% of the 1024-token sequence is padded with `[PAD] = 0`.
3. **Python Programming Sequences (`category3_python_train.jsonl`):**
   - Mean length: **144.3 tokens**. Covers function definition, docstring, body, and test invocation.
4. **Concepts Sequences (`category4_concepts_train.jsonl`):**
   - Mean length: **128.2 tokens**. Standard textbook definitions with structural examples.

### B. Binary Memory-Mapped Packaging Statistics
* `data/experiment_2/training_sequences/train_tokens.bin`:
  - 7,422 sequences $\times$ 1,024 context = **7,600,128 tokens** total allocation (15,200,256 bytes, uint16).
  - Real active tokens: **1,694,584 (22.3%)**.
  - Padding tokens: **5,905,544 (77.7%)**.
* `data/experiment_2/training_sequences/train_mask.bin`:
  - Supervised active loss tokens (`mask == 1`): **1,425,125** (84.1% of all non-padding tokens).
  - Unsupervised prompt tokens (`mask == 0`): **269,459** (15.9% of non-padding tokens).
  - Unsupervised padding tokens (`mask == 0`): **5,905,544** (100% excluded from loss).

---

## 3. Structural Findings, Flaws & Duplicate Density

### Finding 1: Extreme Template Repetition in Python and Concept Splits
* **Audit Metric:** 89.4% of samples in `category3_python_train.jsonl` are repeated variations of just 190 core prompts. In `category4_concepts_train.jsonl`, 63.2% of samples are repeated variations of 313 prompts.
* **Impact on Model Weights:**
  - The model received high gradient volume reinforcing a tiny cluster of function names (`is_positive`, `add_numbers`, `rectangle_area`) and concepts (`monopsony`, `artificial intelligence`).
  - This explains why the model can recite the definition of *monopsony* perfectly from memory, but failed on unseen functions (falling into bigram loops) and general factual questions.

### Finding 2: Absence of Scratchpad (Chain-of-Thought) Reasoning
* **Audit Metric:** In arithmetic training, 75% of examples provided direct answers (`12 + 15 = 27.` or `The sum is 27.`), while only 25% included informal mental breakdown steps.
* **Impact on Reasoning:**
  - Direct autoregressive prediction of multi-digit arithmetic forces the model to memorize outputs rather than execute a generalized calculation procedure.
  - As proven by our baseline evaluation, zero-shot arithmetic on 2-digit to 4-digit numbers yielded **0/65 (0.0%)** accuracy.

### Finding 3: Complete Absence of Instruction & Formatting Examples
* **Audit Metric:** Out of 7,422 training sequences:
  - Sequences with strict length constraints ("answer in three words"): **0 (0.0%)**.
  - Sequences with raw JSON responses: **0 (0.0%)**.
  - Sequences with Markdown table outputs: **0 (0.0%)**.
  - Sequences with strict YES/NO or binary constraints: **0 (0.0%)**.
* **Impact on Instruction Following:**
  - The model never learned to constrain its output length, suppress disallowed words, or format tables/JSON.

### Finding 4: Multi-Turn Personality & Short Dialogue Sparsity
* **Audit Metric:** In `category1_oasst1_cleaned.jsonl`, a minimum length threshold ($\ge 18$ words, $\ge 60$ characters) was enforced to remove canned filler.
* **Unintended Side Effect:**
  - Brief personal dialogue turns (e.g. "Hello! My name is Shanveer", "I live in New York") were completely purged.
  - When the model encounters a simple personal user turn, it has no relevant training data and defaults to canned disclaimers (`"I'm sorry, but I am not sure you have an..."`).

---

## 4. Evaluation Leakage & Tree Isolation Verification

To guarantee that improvements are genuine and not artifacts of data contamination, an automated leakage scan was performed against all 12 evaluation suites:

| Evaluation Suite | Items | Exact Prompt Matches in Training | Status |
| :--- | :---: | :---: | :--- |
| `suite_01_benchmarks.json` | 8 | 7 | **Historical Benchmark (Expected Contamination documented)** |
| `suite_02_arithmetic_single_step.jsonl` | 100 | **0** | **100% UNTOUCHED (Disjoint numerical pairs)** |
| `suite_03_arithmetic_multi_step.jsonl` | 20 | **0** | **100% UNTOUCHED** |
| `suite_04_python_tasks.jsonl` | 30 | **0** | **100% UNTOUCHED** |
| `suite_05_concepts.jsonl` | 30 | **0** | **100% UNTOUCHED** |
| `suite_06_factual_qa.jsonl` | 20 | 1 | 1 match from OASST1 ("What is the capital of France?") |
| `suite_07_instruction_following.jsonl` | 15 | **0** | **100% UNTOUCHED** |
| `suite_08_structured_answers.jsonl` | 10 | **0** | **100% UNTOUCHED** |
| `suite_09_multiturn.jsonl` | 10 | **0** | **100% UNTOUCHED** |
| `suite_10_refusal_uncertainty.jsonl` | 10 | **0** | **100% UNTOUCHED** |
| `suite_11_robustness_malformed.jsonl` | 10 | **0** | **100% UNTOUCHED** |
| `suite_12_context_boundary.jsonl` | 5 | **0** | **100% UNTOUCHED** |

**Conclusion:** All new evaluation suites (Suites 2 through 5, 7 through 12) have **0.0% leakage** into the training curriculum.

---

## 5. Strategic Data Expansion Specification for Next Experiments

To solve the diagnosed root causes while maintaining strict isolation from the evaluation test suites, we define the **Ultimate Improvement Curriculum (Experiment 3 & 4)**:

1. **Chain-of-Thought (Scratchpad) Arithmetic Dataset:**
   - 1,200 verified multi-digit and multi-step problems with explicit `<thought>` tags detailing intermediate addition/subtraction steps, carries, and subtotals before providing the final answer.
   - All numerical pairs strictly disjoint from untouched evaluation suites.
2. **Algorithmic Python with AST & Unit-Test Verification:**
   - 1,000 verified Python problem-solution pairs spanning string manipulation, numerical loops, condition clamping, list comprehensions, and error handling.
   - 100% validated via Python AST parsing and deterministic execution against unit tests.
3. **Instruction Following & Structured Formatting:**
   - 600 examples demonstrating strict word count constraints, single-number outputs, exact YES/NO choices, bullet lists, JSON dictionaries, and Markdown tables.
4. **General Factual Knowledge & Multi-Turn Dialogue:**
   - 800 educational QA pairs (geography, science, world facts, computing) and multi-turn conversational sequences reinforcing personal memory and dialogue etiquette.
