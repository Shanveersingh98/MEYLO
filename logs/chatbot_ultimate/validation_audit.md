# Forensic Audit: Training and Validation Pipeline Audit

**Project Directory:** `D:\DEEP-LEARNING-PROJECT`  
**Audit Target:** Validation loss drop to 0.3721 in `ultimate_improvement`, train/validation separation, and data leakage.  
**Auditor:** Senior Deep Learning Researcher & Test Engineer  
**Date:** 2026-10-27  

---

## 1. Executive Summary

A previous experiment (`ultimate_improvement`) reported a dramatic drop in curriculum validation loss from **3.2623 (Perplexity 26.1) down to 0.3721 (Perplexity 1.45)**, while the general conversational validation loss on OASST1 barely moved (from **4.0219 to 4.0206**).

This audit conducted an independent, empirical investigation into the exact dataset files, split mechanics, loss calculations, sequence construction, and evaluation scripts.

### Core Verdict:
The reported curriculum validation loss of 0.3721 **does NOT represent generalizable learning or mastery of educational reasoning.** It is the artifact of **massive train-validation data leakage and template memorization**:
1. **51.0% (52 out of 102) of the unique held-out validation sequences were exact identical copies of sequences in the training set.**
2. The remaining heldout sequences were drawn from the exact same synthetic generation templates (sharing the same prompt prefixes, structural phrasing, and answer formats).
3. The training set itself was heavily over-saturated with repeated synthetic examples (e.g. identical prompts repeated up to 34 times).
4. On the true unseen distribution (OASST1 validation split across 1,989 sequences), validation loss remained stagnant at **4.0206 (Perplexity 55.7)**.
5. The composite validation loss metric (`0.5 * curriculum_val + 0.5 * oasst1_val = 0.5 * 0.3721 + 0.5 * 4.0206 = 2.1963`) gave a false impression of rapid overall model improvement by averaging genuine test loss with memorized template loss.

---

## 2. Investigation of Suspected Flaws

### A. Train/Validation Leakage & Duplicate Examples
- **File Checked:** `data/ultimate_improvement/curriculum_raw/ultimate_train.jsonl` (6,000 items) vs `ultimate_heldout.jsonl` (110 items).
- **Audit Tool:** `scratch/audit_curriculum_leakage.py`
- **Empirical Findings:**
  - `ultimate_train.jsonl`: 6,000 total sequences, but only 4,138 unique prompts.
  - `ultimate_heldout.jsonl`: 110 total sequences, only 102 unique prompts.
  - **Prompt Overlap:** Exactly 52 prompts appeared in both training and validation (51.0% of held-out prompts).
  - **Exact Pair Overlap:** Exactly 52 `(prompt, response)` pairs were 100% byte-for-byte identical between training and validation.
  - Examples of leaked pairs:
    - `"Answer completely in UPPERCASE letters: What planet do humans live on?"` -> `"EARTH"`
    - `"State only the number: What is 10 plus 10?"` -> `"20"`
    - `"Provide two benefits of regular reading as bullet points starting with '-':"` -> `"- Improves vocabulary and language comprehension.\n- Enhances focus, memory, and cognitive ability."`
    - `"Write a Python function named is_even_number(n)..."`
    - `"Please implement a Python function minimum_of_two(a, b)..."`

### B. Overfitting to Repeated Templated Questions
Within the 6,000-sequence training set, synthetic items were generated with minimal variation and repeated dozens of times:
- `"Reply with only YES or NO: Is ice hot?"` appeared **34 times**.
- `"State only the number: What is 10 plus 10?"` appeared **29 times**.
- `"Answer in exactly three words: What do birds have?"` appeared **27 times**.
- `"List 3 states of matter as a numbered list from 1 to 3:"` appeared **26 times**.
- `"Provide two benefits of regular reading as bullet points starting with '-':"` appeared **25 times**.
- `"My favorite color is red. What is my favorite color?"` appeared **25 times**.

Because the Transformer has 14.1 million parameters and the synthetic vocabulary in these templates was small, the model quickly allocated capacity to memorizing these exact strings, pushing cross-entropy loss on these tokens below 0.4.

### C. Evaluation on the Wrong Dataset & Metric Distortion
- In `train_ultimate.py`, model checkpoint selection was driven by:
  $$\mathcal{L}_{\text{composite}} = 0.5 \mathcal{L}_{\text{curriculum}} + 0.5 \mathcal{L}_{\text{OASST1}}$$
- At Step 0: $\mathcal{L}_{\text{curriculum}} = 3.2623$, $\mathcal{L}_{\text{OASST1}} = 4.0219$, $\mathcal{L}_{\text{composite}} = 3.6421$.
- At Step 1,500: $\mathcal{L}_{\text{curriculum}} = 0.3721$, $\mathcal{L}_{\text{OASST1}} = 4.0206$, $\mathcal{L}_{\text{composite}} = 2.1963$.
- **Interpretation:** The model did not become 40% smarter overall. The composite metric was dragged down entirely by the leaked in-curriculum validation split, which masked the fact that conversational competence did not improve.

### D. Input-Target Shifting and Loss Normalization
- **Implementation in `train_ultimate.py`:**
  - `input_ids = tokens[:, :-1]`, `target_ids = tokens[:, 1:]`, `loss_mask = mask[:, 1:]`.
  - Loss calculation:
    ```python
    loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1), reduction='none')
    loss = (loss * loss_mask.view(-1)).sum()
    ```
  - Gradient accumulation: microbatch losses are summed without premature division, unscaled, and then normalized by the total active supervised tokens across the 16-accumulation window (`accum_active_tokens`).
  - **Verdict:** The autoregressive alignment and loss normalization mathematics are correct with zero off-by-one errors. The flaw was not in the PyTorch code, but in the data generation and split methodology.

### E. Special Token and EOS Handling
- The sequence format uses EOS v2:
  `[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n{response}\n[EOS]`
- The loss mask correctly attributes 0 to `[BOS]`, `<|user|>`, prompt tokens, and trailing `[PAD]`, and attributes 1 to `<|assistant|>`, response tokens, and terminating `[EOS]`.
- All sequences in `training_sequences` strictly begin with `[BOS]` (ID 2) and assistant turns terminate with `[EOS]` (ID 3).

---

## 3. Comparison with Experiment 2

In `experiment_2` (`data/experiment_2/curriculum_raw/`):
- `curriculum_train_all.jsonl` had 7,422 items.
- `curriculum_heldout_eval.jsonl` had 550 items.
- **Audit Finding:** **0.0% prompt overlap and 0.0% pair overlap.**
- In `experiment_2`, held-out curriculum loss was **3.2623**, not 0.3721.
- `experiment_2` also included 851 concept definitions (`category4_concepts`), which were omitted in `ultimate_improvement`.

---

## 4. Recommendations for Next-Stage Training

1. **Strict Train-Val Isolation:** Generate synthetic datasets using disjoint seeds, disjoint entity pools, and strict set-intersection assertions before binary packaging.
2. **Deduplication:** Enforce prompt uniqueness in training data to prevent the model from memorizing repeated prompt templates at the expense of general language modeling.
3. **Primary Metric Grounding:** Evaluate models primarily on held-out open conversational benchmarks (such as OASST1 or real educational QA benchmarks) rather than synthetic in-curriculum splits.
4. **Preserve Conceptual Diversity:** Retain educational concepts, definitions, and domain-specific vocabulary in the curriculum rather than training solely on arithmetic or coding stubs.
