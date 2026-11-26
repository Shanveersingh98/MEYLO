# TRAINING DATA QUALITY AUDIT
**Scope**: Complete Forensic Audit of All Training, Validation, and Evaluation Datasets  
**Datasets Covered**: OASST1 English, Experiment 2 Curriculum, Ultimate Improvement Curriculum  
**Auditor**: Senior NLP Data Auditor & Curriculum Design Specialist  
**Status**: Completed & Verified  

---

## 1. Inventory & Statistics Across All Dataset Splits

| Dataset Category | File Identifier | Format | Record Count | File Size | Usable Records | Active Token % | Padding Token % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **OASST1 Raw Paths** | `train_paths.jsonl`<br>`validation_paths.jsonl` | JSONL<br>JSONL | 17,683<br>1,989 | 33.18 MB<br>3.71 MB | 17,683<br>1,989 | —<br>— | —<br>— |
| **OASST1 EOS v2 Binary**| `train_tokens.bin`<br>`validation_tokens.bin` | uint16 memmap<br>uint16 memmap | 17,683 seqs<br>1,989 seqs | 36.21 MB<br>4.07 MB | 17,683<br>1,989 | 32.65%<br>32.40% | 60.88%<br>61.19% |
| **Experiment 2 (Cleaned OASST1)**| `category1_oasst1_cleaned.jsonl` | JSONL | 2,972 | 5.54 MB | 2,972 | — | — |
| **Experiment 2 (Arithmetic)**| `category2_arithmetic_train.jsonl`<br>`category2_arithmetic_heldout.jsonl` | JSONL<br>JSONL | 1,800<br>200 | 473 KB<br>56 KB | 1,800<br>200 | —<br>— | —<br>— |
| **Experiment 2 (Python)**| `category3_python_train.jsonl`<br>`category3_python_heldout.jsonl` | JSONL<br>JSONL | 1,800<br>200 | 1.01 MB<br>101 KB | 1,800<br>200 | —<br>— | —<br>— |
| **Experiment 2 (Concepts)**| `category4_concepts_train.jsonl`<br>`category4_concepts_heldout.jsonl` | JSONL<br>JSONL | 850<br>150 | 620 KB<br>98 KB | 850<br>150 | —<br>— | —<br>— |
| **Experiment 2 Binary**| `train_tokens.bin`<br>`heldout_tokens.bin` | uint16 memmap<br>uint16 memmap | 7,422 seqs<br>550 seqs | 15.20 MB<br>1.13 MB | 7,422<br>550 | 25.12%<br>24.80% | 71.40%<br>72.10% |
| **Ultimate Curriculum Raw**| `ultimate_train.jsonl`<br>`ultimate_heldout.jsonl` | JSONL<br>JSONL | 6,000<br>110 | 5.72 MB<br>36.7 KB | 6,000<br>110 | —<br>— | —<br>— |
| **Ultimate Curriculum Binary**| `train_tokens.bin`<br>`heldout_tokens.bin` | uint16 memmap<br>uint16 memmap | 6,000 seqs<br>110 seqs | 12.29 MB<br>225 KB | 6,000<br>110 | 18.11%<br>6.02% | **77.74%**<br>**90.71%** |

---

## 2. In-Depth Analysis of Individual Datasets

### 2.1 OASST1 (OpenAssistant Conversations)
- **Source**: Filtered English subset of OpenAssistant Conversations (OASST1), extracted from tree structures to linear conversation paths.
- **Language**: English (`oasst1_en`).
- **Characteristics**:
  - High linguistic diversity and authentic human conversational flow.
  - Mean assistant turn length: **788.6 characters**.
  - Multi-turn depth: Ranges from 1 to 7 turns.
  - Inherent Challenges: Contains noisy crowd-sourced instructions, conflicting stylistic preferences, complex coding prompts far beyond a 14M model's capacity, and varying levels of factual accuracy.
- **Suitability for 14M LM**: Good for general linguistic scaffolding and basic conversational tone, but overly dense and structurally heterogeneous for teaching concise instruction following.

### 2.2 Experiment 2 Curriculum Data
- **Filtering Approach**: Extracted 2,972 high-quality conversational paths from OASST1, and augmented them with 4,450 synthetic examples (1,800 arithmetic, 1,800 Python, 850 concepts).
- **Quality Observations**:
  - `category2_arithmetic_train.jsonl`: Highly structured, synthetic arithmetic step-by-step additions and subtractions (mean length: 44.9 characters).
  - `category3_python_train.jsonl`: Short Python syntax functions and explanations (mean length: 303.0 characters).
  - `category4_concepts_train.jsonl`: Hand-crafted or template-generated concept definitions (mean length: 453.0 characters).
- **Split Cleanliness**: Verified **0 exact duplicates** between training and heldout splits.

### 2.3 Ultimate Improvement Curriculum Data
- **Composition**: 6,000 training records blending filtered conversational turns, synthetic educational QA, reasoning traces, and refusal templates.
- **Quality & Methodological Defects**:
  1. **Extreme Padding**: Average active sequence length in `ultimate_train.bin` is only **197.7 tokens** out of a 1024 context window. **77.74%** of the binary dataset is pure padding (`[PAD]`). In `ultimate_heldout.bin`, active tokens average only **61.7 tokens** (90.71% padding).
  2. **Severe Duplicate Leakage**: As demonstrated in the Validation Integrity Report, 60 out of 110 validation sequences in `ultimate_heldout` (54.55%) are identical to training sequences.
  3. **Template Monotony**: Synthetic generation used repetitive boilerplate templates (e.g., *"Artificial Intelligence (AI) is a broad machine learning..."*). This caused the model to over-index on generic boilerplate openings regardless of the prompt.
  4. **Factual Inconsistencies**: Synthetic templates contained slight factual inversions and tautologies (e.g., defining machine learning as "a machine learning algorithm that learns patterns").

---

## 3. Topic & Instruction Distribution Assessment

We evaluated how well the existing training data covers the target behaviors:

| Capability Target | Training Data Representation | Observed Empirical Behavior | Audit Finding |
| :--- | :--- | :--- | :--- |
| **General Chit-Chat** | High in OASST1 (~60%) | Adequate greetings ("Hello! I am a pleasure to help you.") | Fluent but occasionally awkward grammar. |
| **Educational Explanations** | Moderate (~15%) | Tautological definitions ("A neural network is a neural network of neural networks") | Severe circular phrasing; memorized opening clauses. |
| **Instruction Following** (formatting, length constraints) | Extremely low (< 3%) | Fails sentence count constraints; outputs garbage lists when asked for numbered differences. | Model lacks supervised examples teaching negative constraints (e.g., "in exactly two sentences"). |
| **Basic Reasoning / Math** | Present only as synthetic fixed-template arithmetic (~12%) | Fails 5 - 2 = 3 ("If you have 4, there are 5 items left.") | Template memorization without learning arithmetic operations. |
| **Multi-Turn Context Tracking** | Present in OASST1 | Model repeats prior turn's output 3 to 4 times in a loop. | Model fails multi-turn state tracking without conversational memory. |

---

## 4. Key Takeaways & Recommendations

1. **Do Not Blame Model Architecture for Data Quality Failures**: The model's circular definitions and inability to follow formatting constraints directly reflect the training distribution. The model was never given diverse, high-density instruction-following datasets (such as Dolly-15k, Alpaca-cleaned, or Open-Instruct subsets).
2. **Eliminate Synthetic Template Generation**: Synthetic data generated via rigid string interpolation creates severe mode collapse and repetitiveness. Training data must be derived from curated, linguistically rich, and human-verified instruction corpora.
3. **Sequence Packing is Essential**: Cease the practice of padding every single turn to 1024 tokens. Pack multiple short conversations into each 1024-token sequence separated by `[EOS]` tokens. This will immediately increase token throughput and gradient density by **$4\times$ to $5\times$**.
