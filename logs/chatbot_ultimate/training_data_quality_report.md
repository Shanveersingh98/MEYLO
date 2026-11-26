# Training Data Quality and Integrity Report

**Project Directory:** `D:\DEEP-LEARNING-PROJECT`  
**Datasets Inspected:**
- Legacy: `data/processed/` (OASST1 trees, conversation paths)
- Experiment 2: `data/experiment_2/curriculum_raw/` & `training_sequences/`
- Ultimate Improvement: `data/ultimate_improvement/curriculum_raw/` & `training_sequences/`
**Date:** 2026-10-27  

---

## 1. Overview of Data Assets in Repository

The repository contains three generations of training data:

```
data/
├── processed/                           (Legacy, Pre-EOS v2)
│   ├── conversation_paths/              (Raw multi-turn OASST1 trees)
│   ├── training_sequences_masked/       (Flawed legacy: missing [EOS] on 57.9% of turns)
│   └── training_sequences_eos_v2/       (Clean OASST1: 7,422 train seqs, 1,989 val seqs)
├── experiment_2/                        (Curriculum 1)
│   ├── curriculum_raw/                  (OASST1 cleaned + Arithmetic + Python + Concepts)
│   └── training_sequences/              (7,422 train sequences, 550 heldout sequences)
└── ultimate_improvement/                (Curriculum 2)
    ├── curriculum_raw/                  (OASST1 cleaned + CoT Arithmetic + Python + Stubs)
    └── training_sequences/              (6,000 train sequences, 110 heldout sequences)
```

---

## 2. In-Depth Quality Assessment by Curriculum

### A. Legacy Data (`data/processed/`)
- **Origin:** OpenAssistant Conversations (OASST1) English trees.
- **Defect in Legacy Masking:** In `training_sequences_masked/`, turns ended without an explicit `[EOS]` token in 57.9% of sequences. This produced checkpoints (such as `checkpoints/masked_best.pt`) that suffered from runaway generation because the model never learned when a conversation turn was complete.
- **Correction in EOS v2:** The creation of `training_sequences_eos_v2/` fixed this issue by appending `[EOS]` to every assistant turn.

### B. Experiment 2 Data (`data/experiment_2/`)
- **Composition:**
  - Category 1 (OASST1 Cleaned): 4,000 sequences
  - Category 2 (Arithmetic QA): 1,500 sequences
  - Category 3 (Python QA): 1,071 sequences
  - Category 4 (Concepts QA): 851 sequences (AI, ML, Deep Learning, Economics, Science)
- **Train/Val Split Integrity:**
  - Train: 7,422 sequences.
  - Held-out: 550 sequences.
  - **Empirical Overlap:** 0.0% prompt overlap; 0.0% exact pair overlap.
- **Strengths:** Contained rich educational concept definitions for AI, Machine Learning, and science.
- **Weaknesses:** Arithmetic was mostly direct zero-shot calculations without scratchpad tokens, leading to 0% multi-digit arithmetic accuracy. Python examples were mostly signature stubs with type hints rather than functional logic.

### C. Ultimate Improvement Data (`data/ultimate_improvement/`)
- **Composition:**
  - Selected OASST1: 2,500 sequences
  - CoT Arithmetic: 1,500 sequences
  - Python AST QA: 1,200 sequences
  - Instruction / Format Constraints: 600 sequences
  - Dialogue / Facts: 400 sequences
  - **Omission:** Did NOT include the 851 concept definitions from `category4_concepts`!
- **Data Quality & Duplication Analysis:**
  - Total sequences: 6,000
  - Unique prompts: 4,138 (1,862 duplicated sequences, 31.0% duplication rate)
  - Synthetic items were heavily repeated:
    - `"Reply with only YES or NO: Is ice hot?"` repeated 34 times.
    - `"State only the number: What is 10 plus 10?"` repeated 29 times.
    - `"Provide two benefits of regular reading as bullet points starting with '-':"` repeated 25 times.
- **Data Leakage into Held-Out Set:**
  - Heldout sequences: 110 sequences (102 unique).
  - Exact duplicates in training: **52 sequences (51.0% leakage)**.
  - Consequences: Model rapidly overfitted to memorizing the exact prompt strings, driving in-curriculum validation loss to 0.3721 while conversational loss remained high (~4.02).

---

## 3. Sequence Formatting and Masking Integrity

All binary sequences in `data/ultimate_improvement/training_sequences/` and `data/experiment_2/training_sequences/` follow the strict EOS v2 format:

$$\text{Format: } \texttt{[BOS]}\backslash\text{n}\langle|\text{user}|\rangle\backslash\text{n}\{\text{Prompt}\}\backslash\text{n}\langle|\text{assistant}|\rangle\backslash\text{n}\{\text{Response}\}\backslash\text{n}\texttt{[EOS]}$$

### Masking Logic:
- Token IDs: `[PAD]=0`, `[UNK]=1`, `[BOS]=2`, `[EOS]=3`, `<|user|>=4`, `<|assistant|>=5`.
- Loss Mask:
  - `0` for `[BOS]`, `<|user|>`, prompt tokens, and trailing `[PAD]`.
  - `1` for `<|assistant|>`, response tokens, and terminating `[EOS]`.
- Alignment: For input sequence $S[0 \dots T-1]$, target sequence is $S[1 \dots T]$, and loss mask is $M[1 \dots T]$.
- The alignment between input, target, and mask is mathematically verified: no off-by-one errors exist in the dataset preparation scripts.

---

## 4. Fundamental Model Capacity vs. Data Scale

The language model has **14,120,768 parameters**.
- An 8,000-vocabulary tied-embedding layer accounts for:
  $$8,000 \times 384 = 3,072,000 \text{ parameters (21.8\% of total)}.$$
- 6 Transformer layers account for:
  $$6 \times \left(4 \times 384^2 + 2 \times 384 \times 1536 + \text{norms/biases}\right) \approx 11,048,768 \text{ parameters}.$$
- Total: 14.12M parameters.

### Factual Knowledge Storage Limit:
In modern neural scaling research (e.g. Kaplan et al., Chinchilla, LLaMA), storing dense factual knowledge across multiple domains requires hundreds of millions or billions of parameters exposed to hundreds of gigabytes of text.
A 14.1M parameter model trained on ~6,000 to 7,400 sequences (~6 million tokens total) simply **cannot store general world encyclopedic facts in its weights**.
When asked open-ended questions outside its narrow training paths, it produces hallucinations or tangles related words together.

### Conclusion & Architectural Solution:
Rather than attempting to force a 14M model to memorize the internet through endless synthetic data training, the scientifically rigorous solution is:
1. Preserve the trained model for linguistic syntax and instruction comprehension.
2. Provide **real runtime web retrieval and evidence extraction** to feed factual context into the model's prompt when answering information-seeking queries.
