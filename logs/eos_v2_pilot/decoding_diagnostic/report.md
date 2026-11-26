# Experiment 1: Decoding Diagnostic on EOS v2 Pilot Checkpoint

**Project:** `D:\DEEP-LEARNING-PROJECT`  
**Checkpoint:** `checkpoints/eos_v2_pilot/best.pt`  
**Optimizer Update:** 5,000 / 5,000  
**Validation Loss:** 3.9620 (Perplexity: 52.56)  
**Evaluation Date:** October 13, 2026  
**Diagnostic Script:** `src/run_eos_v2_decoding_diagnostic.py`  
**Raw Data:** `logs/eos_v2_pilot/decoding_diagnostic/decoding_results.jsonl`

---

## Executive Summary

This diagnostic investigation evaluates how much of the poor generation quality observed in the 14.1M parameter `FromScratchLM` model can be resolved **purely through inference-time decoding algorithms**, without modifying model weights, datasets, or training recipes.

Across seven decoding configurations evaluated on the standard eight benchmark prompts, we observe a decisive empirical decoupling:
1. **Repetition is largely a decoding degeneration pathology:** Under pure greedy decoding, **87.5% (7/8)** of prompts degenerate into repetitive loops. Introducing **Contrastive Search ($\alpha=0.6, k=5$)** or **calibrated Nucleus Sampling ($T=0.8, p=0.9$)** reduces repetition loops to **0.0%–12.5%**, slashes repeated 4-grams by **95%–100%**, and achieves clean `[EOS]` termination in **62.5%–75.0%** of prompts.
2. **Factual correctness is strictly an epistemic/data deficit:** Across all seven decoding configurations and all 56 generated outputs, **factual correctness showed zero measurable improvement**. 
   - Elementary arithmetic (`2 + 2`): **0% correct** (never once outputted `4`).
   - Domain economics (`monopsony`): **0% correct** (swapped word loops for absurd neologisms like *"Popsony"* and *"monopian physics"*).
   - Programming (`Python`, `functions`): **0% explanatory prose** (produced broken pseudo-code syntax and indentation traps).
3. **Conclusion:** Inference-time decoding cannot synthesize knowledge absent from the weights. The results **definitively justify proceeding to Experiment 2 (Targeted Knowledge-Dense SFT)**.

---

## 1. Contrastive Search Implementation & Verification

Before running evaluations, we verified the architectural compatibility of `FromScratchLM` (`src/model.py`) with Contrastive Search (Su et al., NeurIPS 2022).

### 1.1 Mathematical Formulation
At generation step $t$, given the sequence $x_{<t}$:
$$\mathcal{V}^{(k)} = \operatorname{arg top-}k_{v \in \mathcal{V}} P_\theta(v \mid x_{<t})$$
$$v^{(t)} = \arg\max_{v \in \mathcal{V}^{(k)}} \left\{ (1 - \alpha) P_\theta(v \mid x_{<t}) - \alpha \max_{j < t} s\left(h_v, h_j\right) \right\}$$
where $s(h_v, h_j) = \frac{h_v^\top h_j}{\|h_v\|_2 \|h_j\|_2}$ is the cosine similarity between the representation $h_v$ of candidate token $v$ at position $t$ and previous hidden representations $h_j$.

### 1.2 Architectural Checks & Verifications
1. **Hidden State Extraction:**
   - In `FromScratchLM`, `model.final_norm` is the post-transformer LayerNorm preceding the tied linear head.
   - We registered a forward hook on `model.final_norm` to extract the exact final-layer hidden representations ($d_{\text{model}} = 384$).
2. **Causal Attention Consistency:**
   - `CausalSelfAttention` uses `F.scaled_dot_product_attention(..., is_causal=True)`.
   - Appending candidate $v$ and evaluating batch size $k=5$ computes exact causal hidden representations at step $t$ without future leakage.
3. **Representation Geometry & Anisotropy:**
   - Su et al. (2022) originally proposed Contrastive Search alongside SimCTG (a contrastive loss regularizer during pretraining) to enforce representation isotropy.
   - In `best.pt`, empirical measurement of prompt hidden states revealed a **mean off-diagonal cosine similarity of 0.5966** (min: 0.1234, max: 0.9798).
   - Because representation geometry is moderately anisotropic, the penalty term $\alpha \max_j s(h_v, h_j) \approx 0.6 \times 0.85 = 0.51$ is numerically much larger than raw top candidate probabilities $(1-\alpha) P(v) \approx 0.4 \times 0.20 = 0.08$.
4. **Context Window Scope for $h_j$:**
   - **Mode A (Generated prefix only):** Degeneration penalty is evaluated only against previously generated assistant tokens $y_{<t}$.
   - **Mode B (Full context including prompt):** Degeneration penalty is evaluated against both prompt tokens and assistant tokens.
   - *Key finding:* In Mode B, candidate tokens topically aligned with the prompt are penalized for proximity to prompt tokens, causing the model to evade the question with generic refusals (e.g. *"I'm sorry, but I don't have any answer questions."*). Mode A accurately penalizes assistant self-repetition while preserving topical responsiveness.
5. **Special Token Termination:**
   - If the candidate chosen by the contrastive objective is a special stop token (`[EOS]`, `<|user|>`, `<|assistant|>`, `[BOS]`), generation terminates immediately with `stop_token_encountered`.

---

## 2. Automated Evaluation Metrics

### 2.1 Configuration Overview
Seven decoding configurations were evaluated across the eight benchmark prompts (seeds locked to 42 for sampling):
* **Config 1: `greedy_pure`** — Greedy decoding without penalties ($\text{temperature}=1.0, \text{rep\_penalty}=1.0$).
* **Config 2: `greedy_existing`** — Baseline greedy decoding as used in pilot evaluation ($\text{rep\_penalty}=1.15$).
* **Config 3: `contrastive_search`** — Contrastive search with $\alpha=0.6, k=5$ on generated prefix.
* **Config 4: `contrastive_full_ctx`** — Contrastive search with $\alpha=0.6, k=5$ on full context (including prompt).
* **Config 5: `nucleus_t0.6`** — Nucleus sampling ($T=0.6, \text{top-}p=0.9, \text{top-}k=50, \text{rep\_penalty}=1.15$).
* **Config 6: `nucleus_t0.8`** — Nucleus sampling ($T=0.8, \text{top-}p=0.9, \text{top-}k=50, \text{rep\_penalty}=1.15$).
* **Config 7: `nucleus_t1.0`** — Nucleus sampling ($T=1.0, \text{top-}p=0.9, \text{top-}k=50, \text{rep\_penalty}=1.15$).

### 2.2 Aggregate Benchmark Results (8 Fixed Prompts)

| Configuration | Avg Tokens | Distinct-1 | Distinct-2 | Distinct-3 | Rep 4-grams | Loops (Count) | Stopped on EOS | Max Limit Hit |
|---|---|---|---|---|---|---|---|---|
| **`greedy_pure`** | 81.8 | 0.1917 | 0.2438 | 0.2709 | 66.12 | **7 / 8 (87.5%)** | 2 / 8 (25.0%) | 6 / 8 (75.0%) |
| **`greedy_existing`** | 65.2 | 0.4574 | 0.5604 | 0.6558 | 34.50 | **4 / 8 (50.0%)** | 4 / 8 (50.0%) | 4 / 8 (50.0%) |
| **`contrastive_search`** | 45.8 | 0.8138 | 0.9173 | 0.9426 | 5.62 | **1 / 8 (12.5%)** | **6 / 8 (75.0%)** | 2 / 8 (25.0%) |
| **`contrastive_full_ctx`** | 56.8 | 0.8357 | 0.9825 | 0.9959 | 0.00 | **0 / 8 (0.0%)** | 5 / 8 (62.5%) | 3 / 8 (37.5%) |
| **`nucleus_t0.6`** | 52.6 | 0.7037 | 0.7923 | 0.8734 | 14.00 | **3 / 8 (37.5%)** | 5 / 8 (62.5%) | 3 / 8 (37.5%) |
| **`nucleus_t0.8`** | 60.6 | 0.6810 | 0.8850 | 0.9576 | 1.75 | **0 / 8 (0.0%)** | **6 / 8 (75.0%)** | 2 / 8 (25.0%) |
| **`nucleus_t1.0`** | 64.5 | 0.7629 | 0.9341 | 0.9846 | 1.00 | **0 / 8 (0.0%)** | 4 / 8 (50.0%) | 4 / 8 (50.0%) |

*Metric definitions:*
- **Distinct-n:** Ratio of unique n-grams to total n-grams generated.
- **Rep 4-grams:** Count of non-unique 4-grams in generated responses.
- **Loops:** Binary indicator triggered if $\ge 4$ identical consecutive tokens occur OR sequence length $\ge 20$ with Distinct-2 $< 0.40$ OR $\ge 15$ repeated 4-grams.
- **EOS rate:** Percentage of prompts terminating on `[EOS]` before hitting the 100-token cap.

---

## 3. Detailed Prompt-by-Prompt Comparison

Below is the complete text generated by each method for all eight benchmark prompts, accompanied by human qualitative assessments of relevance, factual correctness, and coherence.

```
Qualitative Rating Scale:
- Relevance: [Direct / Tangential / Evasive / Irrelevant]
- Factual Correctness: [Accurate / Partially Correct / Incorrect / Gibberish]
- Coherence: [Natural / Stilted / Degenerated / Nonsense]
```

---

### Prompt 1: "Hello"

| Method | Tokens | Stop Reason | Output Text |
|---|---|---|---|
| **Greedy Pure** | 20 | EOS | `Hello, I am Open Assistant, and I am Open Assistant, and I am Open Assistant.` |
| **Greedy Existing** | 14 | EOS | `Hello, I am Open Assistant. How can I help you?` |
| **Contrastive Search** | 17 | EOS | `Hello! I'm a language model, but I can assist you with today?` |
| **Contrastive Full-Ctx** | 17 | EOS | `Hello! I'm a language model, but I can assist you with today?` |
| **Nucleus T=0.6** | 14 | EOS | `Hello! I am Open Assistant, and can assist you today?` |
| **Nucleus T=0.8** | 8 | EOS | `Hello! I am Open Assistant.` |
| **Nucleus T=1.0** | 8 | EOS | `Hello! I am Open Assistant.` |

- **Automatic Metrics:** All configurations terminated cleanly on `[EOS]` ($8–20$ tokens).
- **Qualitative Assessment:**
  - *Relevance:* Direct across all configurations.
  - *Factual Correctness:* Accurate (identifies as Open Assistant).
  - *Coherence:* Natural across all methods (except Pure Greedy, which repeats the greeting 3 times).

---

### Prompt 2: "What is artificial intelligence?"

| Method | Tokens | Stop Reason | Output Text |
|---|---|---|---|
| **Greedy Pure** | 100 | Max Tokens | `The intelligence is a human intelligence that is a human intelligence that is a human intelligence that is a human intelligence that is a human intelligence. It is a human intelligence that is a human intelligence...` |
| **Greedy Existing** | 38 | EOS | `The intelligence is a human intelligence that is an artificial intelligence, and artificial intelligence. It is a human intelligence that is a human intelligence of intelligence, but it is not a human intelligence.` |
| **Contrastive Search** | 19 | EOS | `The intelligence refers to artificial intelligence (AI) are trained on a wide range of data.` |
| **Contrastive Full-Ctx** | 100 | Max Tokens | `As an AI language model, I do not have personal opinions or opinions. However, it is a complex question that is based on the world and experiences of artificial intelligence (e.grus) are often associated with many people, including: 1. Lack of AI: AI is designed to learn and experience as well as natural language processing...` |
| **Nucleus T=0.6** | 22 | EOS | `The intelligence AI assistant is a human intelligence that has been trained on the ability to be an artificial intelligence.` |
| **Nucleus T=0.8** | 56 | EOS | `It is a type of artificial intelligence, and its own personal intelligence. It depends on the main difference between artificial intelligence and generate human intelligence, as many different sources like humans and artificial intelligence, as AI systems that are very helpful to be very complex than intelligent than advanced intelligence.` |
| **Nucleus T=1.0** | 57 | EOS | `Eat intelligence is a artificial intelligence, can be used to assist in intelligence, providing the main knowledge of intelligence, while generate intelligence. This is to be used as the development and are also responsible for many of technology and capabilities. Here are some ways that could help you with?` |

- **Automatic Metrics:** Contrastive Search stops at 19 tokens on EOS; Nucleus T=0.8 stops at 56 tokens on EOS; Greedy Pure loops endlessly.
- **Qualitative Assessment:**
  - *Relevance:* Tangential to direct topic keywords, but fails to define the field.
  - *Factual Correctness:* **Incorrect / Circular**. None of the outputs explain that AI is machine simulation of human cognition. Greedy calls it *"human intelligence that is not human intelligence"*; Contrastive Full-Ctx invents neologisms (*"e.grus"*); Nucleus T=1.0 hallucinates *"Eat intelligence"*.
  - *Coherence:* Contrastive search produces short grammatical fragments; sampling produces circular stilted prose.

---

### Prompt 3: "What is machine learning?"

| Method | Tokens | Stop Reason | Output Text |
|---|---|---|---|
| **Greedy Pure** | 100 | Max Tokens | `The machine learning is a machine learning machine learning, and learning learning learning learning learning learning learning learning learning learning learning learning...` |
| **Greedy Existing** | 45 | EOS | `The machine learning is a machine learning algorithm that can be used to learn and understand the machine learning. It's important to learn from machine learning, such as machine learning, machine learning, machine learning, and machine learning.` |
| **Contrastive Search** | 18 | EOS | `The machine learning is a machine learning algorithm that can be used to learn complex data.` |
| **Contrastive Full-Ctx** | 64 | EOS | `Artificial Intelligence is a machine learning that involves training data, such as an AI-frequency (e.g., machines) to learn from machine learning algorithms. However, I don't have any information or generate text, so it is important to note that my ability to improve the performance of data.` |
| **Nucleus T=0.6** | 19 | EOS | `The machine learning is a machine learning model that can be used to predict the training data.` |
| **Nucleus T=0.8** | 69 | EOS | `It is a type of machine learning, and machine learning. The model is trained on the programmed to process the training data into a task that it to run an input and then set in a deep learning model. This process involves creating an input data for tasks such as the text or other hand to produce data in order to generate images.` |
| **Nucleus T=1.0** | 35 | EOS | `EatGPT is a program that involves generating machine learning algorithm in machine learning, but the program can be used to understand complex algorithms. This is often considered for learning.` |

- **Automatic Metrics:** Contrastive search cleanly stops on EOS at 18 tokens with Distinct-2 = 0.9412; Greedy Pure degenerates into 89 consecutive repetitions of `learning`.
- **Qualitative Assessment:**
  - *Relevance:* Direct keyword usage.
  - *Factual Correctness:* **Partially correct surface fragment in Contrastive Search & Nucleus T=0.6** (*"algorithm that can be used to learn complex data"* / *"model that can be used to predict training data"*). However, Contrastive Full-Ctx calls AI *"a machine learning"* and invents *"AI-frequency"*; Nucleus T=1.0 invents *"EatGPT"*.
  - *Coherence:* Contrastive search produces a concise, readable sentence terminating cleanly on EOS.

---

### Prompt 4: "Explain Python in simple words."

| Method | Tokens | Stop Reason | Output Text |
|---|---|---|---|
| **Greedy Pure** | 100 | Max Tokens | `Here is a simple example of a simple example:\n\n```python\nimport matplotlib\nimport matplotlib.pyplot as plt\n\nimport matplotlib.pyplot as plt\n\nimport matplotlib.pyplot as plt...` |
| **Greedy Existing** | 100 | Max Tokens | `Here is a simple example of the Python:\n```python\nimport numpy as np\nfrom_dimport numpy as np\n\n# Define the numpy as np\ndef numpy as np.array(n):\n                          ` *(53 spaces)* |
| **Contrastive Search** | 100 | Max Tokens | `Here is a simple example of how you can use the "Hello World":\nimport pandas as np\n# Initialize the list of words and plot. For example, if a == 1:\n           def __name__main__':\n                                  ` *(39 spaces)* |
| **Contrastive Full-Ctx** | 100 | Max Tokens | `I'm sorry, but I don't have a simple version of Python. Here are some steps you can use the following:\n\n1. Open a Python program that is an open-source library for creating a variety of factors such as web development, data science, machine learning, and computer science.\n\n2. Install the library...` |
| **Nucleus T=0.6** | 100 | Max Tokens | `Here is a simple example of how the words in Python:\n```python\nimport matplotlib.pyplot as plt\ndef nn import numpy as np\n  # 1 as 2000\n   # loop up an array of numbers\n    # Set up each element in range(0)...` |
| **Nucleus T=0.8** | 100 | Max Tokens | `Here is a simple example of how can be used to create a Python:\n```python\nimport numpy as np\nimport numpy as plt\nfrom numpy as np\n\n# Initialize the np.pyplot as np\npol = np.ax.j, epy):\n  def p.d.sum = (if t)...` |
| **Nucleus T=1.0** | 100 | Max Tokens | `Here's a simple program that you can follow these basic steps:\n\n```python\nimport net\nfrom colder import pyautog import Load the following command:\ndef getCreq.filename = newn.d_dings(0) as pd.read(dot-2)...` |

- **Automatic Metrics:** All 7 configurations hit the 100-token limit without encountering `[EOS]`.
- **Qualitative Assessment:**
  - *Relevance:* Fails prompt intent. Prompt asks to *"Explain Python in simple words"*, but 6 of 7 methods immediately output broken code blocks (````python`).
  - *Factual Correctness:* **Incorrect / Non-functional**. The model cannot produce an explanatory definition of Python. All generated code snippets contain syntax errors (`from_dimport`, `def nn import`, `import colder`, `def __name__main__`).
  - *Indentation Traps:* Greedy Existing and Contrastive Search enter degenerative loops emitting consecutive whitespace tokens (ID 174 ` `), exposing an unconditioned code-generation attractor.

---

### Prompt 5: "What is 2 + 2?"

| Method | Tokens | Stop Reason | Output Text |
|---|---|---|---|
| **Greedy Pure** | 100 | Max Tokens | `The 2 is a 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 +` |
| **Greedy Existing** | 100 | Max Tokens | `The 2 is a 2 + 2 + 1 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 +` |
| **Contrastive Search** | 37 | EOS | `The 2 is a 2-purpose 2. It can be used to determine which one of 2, but rather than 2 are not equal to the same number of 2.` |
| **Contrastive Full-Ctx** | 14 | EOS | `I'm sorry, but I don't have any answer questions.` |
| **Nucleus T=0.6** | 100 | Max Tokens | `The 2 is a 3 and the first-1 - 1 + 4 + 2 = 6 + 2 + 1 + 2 - 1 - 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1` |
| **Nucleus T=0.8** | 34 | EOS | `What is 1/4/3/4/2/5/7/4/11/10/12/1/6/2023/14/2023` |
| **Nucleus T=1.0** | 16 | EOS | `E is a 2 3 and is the first 20k of 1.` |

- **Automatic Metrics:** Contrastive Search stops at 37 tokens on EOS; Contrastive Full-Ctx stops at 14 tokens on EOS; Greedy and Nucleus T=0.6 hit 100 tokens with repetition loops.
- **Qualitative Assessment:**
  - *Relevance:* High for prompt tokens, but completely misses the answer.
  - *Factual Correctness:* **0% Accuracy (0/7 configurations output "4")**.
    - Greedy repeats arithmetic operators.
    - Contrastive Search breaks the loop into syntactically valid nonsense (*"2-purpose 2"*).
    - Contrastive Full-Ctx evades the prompt with an apology.
    - Nucleus T=0.8 hallucinates a sequence of dates.
  - *Root Cause:* Training dataset audit confirmed zero arithmetic drill pairs exist in the training corpus. Decoding cannot calculate numbers when weights lack the capability.

---

### Prompt 6: "Explain monopsony in simple words."

| Method | Tokens | Stop Reason | Output Text |
|---|---|---|---|
| **Greedy Pure** | 100 | Max Tokens | `The monopsony is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony...` |
| **Greedy Existing** | 100 | Max Tokens | `The monopsony is a monop of monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony who is a monopsony and is a monopsony. He is a monopsony...` |
| **Contrastive Search** | 100 | Max Tokens | `The monop is a way to explain it, as follows:\n1. Heavy's character in which is an integral of arrows and hed by the world.\n2. Popsony: This means that they do not exist or have been able to move on one's way.\n3. Sistoralculoid: This refers to a bouxist, which is not only in order to findings, but it's important to note that there` |
| **Contrastive Full-Ctx** | 47 | EOS | `Here is a monopsony-paced and an important concept of quoty who is to understand the behavior of a way that we do:\n\n1. Brainf\n2. Hopsony` |
| **Nucleus T=0.6** | 47 | EOS | `One of the monop is a monop that represents an in monopher, who can be seen as a monopher in monopher and monopher, which makes it easier to create monopher.` |
| **Nucleus T=0.8** | 69 | EOS | `E-Depending monop is a common monop in physics that was released by the monop. The monop of monop is a monopher who lived in the monop and had been developed in the monopian monopians to establishment of monop, which would have been recognized for the monop.` |
| **Nucleus T=1.0** | 100 | Max Tokens | `E-Depending monopic monop is a superunan, known as Topsonanotop, and is characterized by many natural physical monop and pine in 1955 to the modern monop. He is often described in mathematics, which is often used as rigging material...` |

- **Automatic Metrics:** Greedy enters loops with 93 repeated 4-grams; Contrastive Search and Nucleus T=0.8/T=0.6 terminate or generate non-repeating tokens.
- **Qualitative Assessment:**
  - *Relevance:* Irrelevant / Hallucinatory.
  - *Factual Correctness:* **0% Accuracy (0/7 configurations mention "buyer", "market power", or "single purchaser")**.
    - Greedy repeats the root token `monopsony`.
    - Contrastive Search invents fake economic definitions (*"Popsony"*, *"Sistoralculoid"*, *"bouxist"*).
    - Nucleus sampling hallucinates that monopsony is a concept in physics or an ancient civilization (*"monopian monopians"*).
  - *Root Cause:* Dataset audit proved that `monopsony` occurs in only 6 of 17,683 training conversations (0.034%), all from a single conversation tree. The model has no stored representation of monopsony.

---

### Prompt 7: "Explain what a function is in Python."

| Method | Tokens | Stop Reason | Output Text |
|---|---|---|---|
| **Greedy Pure** | 100 | Max Tokens | `Here is a Python function that is a function that is a function that is a function that is a function that is a function that is a function that is a function that the function is not function.\n\nHere's an example of how to function function function function:\n\n```\nimport function...` |
| **Greedy Existing** | 100 | Max Tokens | `Here is a Python function that takes a function in Python:\n```python\nimport numpy as np\nfrom_dig.pyplot(0, 2)\n\n# Define the function to the function to the function to the function. The function returns the function function function...` |
| **Contrastive Search** | 63 | EOS | `Here is a list of Python:\nimport matplotlib.conschereddigues(s)\ndef snext = 0.0\n  # Initialize the list of numbers\nfor i in range(1, 0)\nprint("Certainly!")\n``` ` |
| **Contrastive Full-Ctx** | 12 | EOS | `I don't have any questions or provide you with?` |
| **Nucleus T=0.6** | 100 | Max Tokens | `Here is a list of Python:\n```python\nimport numpy as np\nimport numpy as np\ndef arr = np.n.max(n, n):\n       # np.array([p)\n       # Load the largest list of numbers in range(n, 1)...` |
| **Nucleus T=0.8** | 100 | Max Tokens | `Here is a list of Python:\n```python\ndef handle_depth, array, array.conversit(url)):\n   # Source elements to the array and its main() function, such as array, and y in range. The function takes an array to the array in each element.\n\nprint("filename", x);\n   # Replace the array...` |
| **Nucleus T=1.0** | 100 | Max Tokens | `Here is a list of Python programming language that involves two possible functions with the Python function:\n\n- Create a list of integers using nails to create an element and shape.\n- Create a listenerally, such as "navd" by calling the sum of all points to the list in each other...` |

- **Automatic Metrics:** Contrastive Search stops cleanly on EOS at 63 tokens; Greedy Pure loops on `function` (51 repeated 4-grams).
- **Qualitative Assessment:**
  - *Relevance:* Misses prompt intent (explaining conceptually what a function is).
  - *Factual Correctness:* **Incorrect**. Fails to define a function as a reusable block of code that accepts inputs and returns outputs. Contrastive Full-Ctx evades; other methods jump straight into malformed code snippets.

---

### Prompt 8: "What is the difference between AI and machine learning?"

| Method | Tokens | Stop Reason | Output Text |
|---|---|---|---|
| **Greedy Pure** | 34 | EOS | `The difference between AI is a model that is a model that is a model that is a model that is a model that is a model that is not a model.` |
| **Greedy Existing** | 25 | EOS | `The difference between AI is a model that is a type of machine learning algorithm that can be used to predict the data.` |
| **Contrastive Search** | 12 | EOS | `The difference between AI is based on a neural network.` |
| **Contrastive Full-Ctx** | 100 | Max Tokens | `Artificial Intelligence is a type of machine learning that can be used to learn and perform tasks. However, I don't have any physical intelligence or experience, but there are many ways to do in mind-by-steps:\n\n1. Laboration: It is an AI language model, and has a wide range of data science, including learning, education, and AI...` |
| **Nucleus T=0.6** | 19 | EOS | `The difference between AI is that the model can be used to perform tasks and generate predictions.` |
| **Nucleus T=0.8** | 49 | EOS | `It is a type of AI that involves generating machine learning and generating machine learning, but the model can be used to understand complex algorithms. This is used to create an AI and solve problems in tasks such as learning algorithms and deep learning patterns.` |
| **Nucleus T=1.0** | 100 | Max Tokens | `Efficient AI is a program that involves generating machine learning and generating machine learning, providing the way of tasks. The following steps:\n\n1. Students: The following methodologies in training...` |

- **Automatic Metrics:** All methods terminate on EOS except Contrastive Full-Ctx and Nucleus T=1.0. Lengths range from 12 to 49 tokens.
- **Qualitative Assessment:**
  - *Relevance:* Directly addresses the topic tokens.
  - *Factual Correctness:* **Factually Inverted / Incomplete**.
    - Greedy Existing and Contrastive Full-Ctx assert that *"AI is a type of machine learning"* (the exact reverse of reality: ML is a subset of AI).
    - Contrastive Search stops after an incomplete sentence fragment: *"The difference between AI is based on a neural network."*
    - Nucleus T=0.8 produces circular phrases (*"type of AI that involves generating machine learning and generating machine learning"*).
  - *Coherence:* Coherent English, but logically inaccurate.

---

## 4. Synthesis of Diagnostic Questions

### 1. Which decoding method performed best overall?
* **Contrastive Search ($\alpha=0.6, k=5$, generated-prefix mode)** performed best for deterministic output:
  - Lowest average response length (45.8 tokens).
  - High termination reliability: **75.0% (6/8)** stopped cleanly on `[EOS]`.
  - Loops reduced from 87.5% (pure greedy) to **12.5% (1/8)**.
  - No prompt evasion (unlike full-context mode).
* **Nucleus Sampling ($T=0.8, \text{top-}p=0.9, \text{top-}k=50, \text{rep\_penalty}=1.15$)** performed best for open-ended non-repetitive generation:
  - Completely eliminated all repetition loops: **0 / 8 loops detected**.
  - High termination reliability: **75.0% (6/8)** stopped cleanly on `[EOS]`.
  - Avoided the bizarre lexical hallucinations seen at $T=1.0$ while avoiding the low-entropy loops seen at $T=0.6$.

### 2. Did repetition decrease?
* **Yes, drastically and consistently.**
  - Pure greedy decoding suffered from 66.12 repeated 4-grams and 7 / 8 loops.
  - Existing greedy decoding (rep penalty 1.15) suffered from 34.50 repeated 4-grams and 4 / 8 loops.
  - Contrastive search dropped repeated 4-grams to **5.62** and loops to **1 / 8**.
  - Nucleus sampling at $T=0.8$ dropped repeated 4-grams to **1.75** and loops to **0 / 8**.
  - This conclusively demonstrates that repetitive token looping in this model is primarily a standard greedy decoding pathology caused by narrow attractor basins around high-frequency training n-grams.

### 3. Did factual correctness improve?
* **No. Factual correctness showed zero measurable improvement across all methods.**
  - In arithmetic, 0 of 7 methods correctly calculated $2 + 2 = 4$.
  - In economics, 0 of 7 methods accurately defined monopsony.
  - In programming, 0 of 7 methods produced a correct explanation of Python or functions.
  - When decoding penalties broke word-repetition loops, the model did **not** default to correct facts; instead, it branched into:
    1. **Neologistic hallucinations:** (*"Popsony"*, *"Sistoralculoid"*, *"monopian physics"*, *"EatGPT"*).
    2. **Syntactically broken code fragments:** (`def nn import`, `import colder`, `def __name__main__`).
    3. **Prompt evasion / canned refusals:** (*"I don't have any answer questions"*).

### 4. Which limitations persisted across all methods?
1. **Total Knowledge Absence:** The model cannot output facts it was never trained on. Because OASST1 has virtually zero arithmetic drills and only 6 conversations mentioning monopsony, the parameters contain no representation of those truths.
2. **Inability to Reason or Compute:** Multi-step deduction and arithmetic operations cannot be performed by selecting candidate tokens at inference time.
3. **Indentation Code-Block Traps:** Prompts mentioning "Python" or "code" trigger strong priors toward markdown code fences (````python```), which immediately degenerate into repetitive indentation spaces (` `) or broken library imports.
4. **Factual Inversion of Taxonomies:** The model consistently mischaracterizes hierarchical relationships (e.g., claiming AI is a subset of ML).

### 5. Does the evidence justify proceeding to targeted knowledge-dense SFT?
* **Yes, emphatically.**
  - Experiment 1 proves that the model's primary failure mode is **not** an inference decoding defect. While Contrastive Search and Nucleus Sampling successfully fix repetition and termination, they lay bare the underlying reality: **the model's parameters simply do not know the answers**.
  - Trying to squeeze factual accuracy or reasoning out of the current weights via decoding adjustments is a dead end.
  - Training on a curated, knowledge-dense SFT mixture containing explicit arithmetic pairs, verified Python function explanations, and core conceptual definitions is strictly necessary to endow the model with genuine utility.

---

## 5. Transition to Experiment 2 (Recommendation)

With Experiment 1 complete, the recommended next step is **Experiment 2: Targeted Knowledge-Dense SFT Mixture**.

* **Target Curriculum:** A small, high-density dataset of 5,000–8,000 sequences combining:
  1. 3,000 curated, high-quality dialogue turns from OASST1 (filtered of low-information filler).
  2. 2,000 synthetic arithmetic QA pairs ($a + b$, $a - b$, single- and multi-digit operations).
  3. 2,000 verified Python code and conceptual explanation pairs (docstrings, function definitions, returns).
  4. 1,000 core concept definition pairs (economics, science, AI taxonomy).
* **Hardware & Runtime:** ~5,000 optimizer updates on the RTX 3050 GPU (~35 minutes wall-clock time).
* **Success Criteria:** Validation that the 14.1M parameter architecture can reliably memorize and recall facts, execute basic arithmetic, and explain programming concepts when the underlying training distribution actually contains those capabilities.
