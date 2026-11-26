# Comprehensive Initial Forensic Audit: From-Scratch AI Chatbot & Retrieval System

**Project Directory:** `D:\DEEP-LEARNING-PROJECT`  
**Python Environment:** `D:\DEEP-LEARNING-PROJECT\.venv\Scripts\python.exe` (Python 3.12.10, PyTorch 2.14.0+cu130)  
**Hardware Accelerators:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM, CUDA Available: True)  
**Audit Date:** 2026-10-27  
**Artifact Directory:** `logs/chatbot_ultimate/`  

---

## 1. Executive Summary & Purpose

This forensic audit investigates the real operational, architectural, and data-pipeline state of the from-scratch language model chatbot at `D:\DEEP-LEARNING-PROJECT`. 

The primary mission mandate is to:
1. Diagnose why the chatbot produces incoherent, repetitive, or nonsensical answers (e.g. definitions that claim machine learning "minimizes physical learning and cognitive ability" or invent non-existent concepts like "Growthrowing" and "Ancientizing Mary Learning").
2. Independently verify the training and validation pipelines, specifically auditing the report of a curriculum validation loss drop to 0.3721 while conversational loss was stagnant.
3. Establish an evidence-based repair loop to fix decoding and prompt formatting pathologies.
4. Architect and integrate an autonomous, free, secure web-retrieval and source-grounded question-answering subsystem into the actual chatbot application (`src/chat.py`).
5. Execute end-to-end regression and verification test suites and provide a turnkey Windows launcher (`RUN_CHATBOT.bat`).

---

## 2. Real System Architecture & Execution Path

The complete flow from user input to generated response was inspected and empirically traced across active and historical scripts:

### A. Model Architecture (`src/model.py`)
- **Class:** `FromScratchLM`
- **Type:** Decoder-only Transformer implementing Pre-LayerNorm architecture.
- **Parameters:** Exactly **14,120,768** learnable parameters.
- **Hyperparameters:**
  - `vocab_size`: 8,000 subwords
  - `context_length`: 1,024 tokens
  - `d_model`: 384
  - `num_heads`: 6 (head dimension: 64)
  - `num_layers`: 6
  - `d_ff`: 1,536 (4 * d_model)
  - `dropout`: 0.1 during training, 0.0 at inference
- **Attention Implementation:** PyTorch `scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)`. Verified to enforce strict causality without future token leakage.
- **Weight Tying:** Tied embeddings between `token_embedding.weight` and language model output projection, with a dedicated learnable output bias vector of size 8,000.

### B. Tokenizer (`tokenizer/tokenizer.json`)
- **Type:** Byte-Level BPE (Byte-Pair Encoding) trained via HuggingFace `tokenizers`.
- **Vocabulary Size:** Exactly 8,000 subwords.
- **Special Token ID Mapping:**
  - `[PAD]` = 0
  - `[UNK]` = 1
  - `[BOS]` = 2
  - `[EOS]` = 3
  - `<|user|>` = 4
  - `<|assistant|>` = 5
- **Encoding Characteristics:** Spaces are encoded as byte-level marker `Ġ` (token 174) and newlines as `Ċ` (token 171). Special tokens `<|user|>`, `<|assistant|>`, `[EOS]`, `[BOS]` are preserved as atomic single IDs.

### C. Checkpoints In Repository
The repository contains 28 checkpoint files across several experimental generations:
1. `checkpoints/best.pt` (Optimizer step 29,300, val_loss 3.7745) - Root legacy checkpoint.
2. `checkpoints/phase3_best.pt` (Step 50,000, val_loss 3.0692) - Early conversational run suffering from severe loop degeneration.
3. `checkpoints/clean_50k_best.pt` (Step 50,000, val_loss 3.8162) - Early clean OASST1 run with subword cycle pathologies (`text-like-like-like`).
4. `checkpoints/experiment_2/best.pt` (Optimizer step 1,750, composite val_loss 3.6421, SHA-256 `f717e5efff9dc6d58b0fc1d4872e74b0b03bd734029715a3ae14aa549b5b4ab1`) - Corrected assistant-only loss masking and clean EOS v2 format.
5. `checkpoints/ultimate_improvement/best.pt` (Optimizer step 1,500, composite val_loss 2.1963, curriculum val_loss 0.3721) - Fine-tuned on synthetic templates.

---

## 3. Confirmed Defects and Deficiencies

| Defect ID | Severity | Category | Description | Root Cause / Evidence |
| :--- | :---: | :--- | :--- | :--- |
| **DEF-01** | **CRITICAL** | Data Leakage & Evaluation Illusion | Curriculum validation loss reported 0.3721, but conversational validation loss was unchanged (~4.0206). | 51.0% (52/102) of unique heldout sequences in `ultimate_improvement` were exact duplicates of sequences in `ultimate_train.jsonl`. Synthetic prompts were replicated up to 34 times. The model memorized repeated prompt templates rather than learning general reasoning. |
| **DEF-02** | **CRITICAL** | Model Forgetting & Concept Erasure | Model produces bizarre hallucinations on basic concepts (e.g. defining machine learning as "Growthrowing" or "Ancientizing Mary Learning"). | The `ultimate_improvement` curriculum omitted the 851 concept definition sequences from `category4_concepts`, training for 1,500 steps exclusively on math and Python templates. The model suffered catastrophic forgetting of basic definitions. |
| **DEF-03** | **HIGH** | Inference Decoding Pathology | Extreme hallucinations and incoherent answers under default generation parameters. | `src/generate.py` and `src/chat.py` used `temperature=0.8` and `top_k=50`. With an 8,000-token vocabulary and high model perplexity (~55), top_k=50 frequently samples noisy tail tokens. Calibrating decoding to `temperature=0.35`, `top_p=0.85`, and `top_k=15` restores coherent English syntax. |
| **DEF-04** | **HIGH** | Missing Live Retrieval Capability | Chatbot possesses no mechanism to retrieve real-world facts, current information, or verify claims. | A 14.1M parameter model cannot memorize encyclopedic world knowledge in its weights. The application had no external search or webpage scraping pipeline. |
| **DEF-05** | **MEDIUM** | Security / SSRF Vulnerability | Potential for arbitrary network requests if the chatbot or tools follow user-specified URLs. | Absence of IP validation (allowing access to `localhost`, `127.0.0.1`, `192.168.x.x`, or cloud metadata endpoints `169.254.169.254`). |
| **DEF-06** | **MEDIUM** | Prompt Injection Vulnerability | Raw user inputs or scraped webpages could inject special tokens (`<|assistant|>`, `<|user|>`). | Special token strings in untrusted web text or user input must be sanitized and isolated into data boundaries. |

---

## 4. Baseline Behavior & Reproduction of Incoherent Answers

Empirical testing on both active checkpoints reproduced the exact failure modes:

### Prompt: *"What is machine learning?"*
- **`ultimate_improvement/best.pt` (Default Decoding):**
  > *"Artificial Intelligence (AI), and Machine learning is a great goal of machine learning that involves the algorithm of a language of machine neural networks. It involves using machine learning algorithms, including data analysis, and mathematical tasks."*
- **`ultimate_improvement/best.pt` (Sampling T=0.8, TopK=50):**
  > Generated hallucinated steps including *"1. Growthrowing: This involves the model with multiple layers..."* and *"2. Create an image editor: You can use visualization for machine learning (HRNOU)..."*
- **`experiment_2/best.pt` (Default Decoding):**
  > *"Deep Learning and Machine Learning is a machine learning paradigm for machine learning that explains a programmerse but can interact with data through data or decisions without being explicitly data. A"*

### Prompt: *"Give three practical examples of machine learning."*
- **Output:**
  > *"1. Ancientizing Mary Learning (PCA) is a set of machine learning that uses a deep learning algorithm to get a good and challenging task..."*

The root cause of these fabrications is the combination of model capacity limits (14.1M parameters cannot hold all factual concepts), omission of educational concepts from recent training curricula, and excessive sampling entropy (top_k=50 at temperature 0.8).

---

## 5. Audit Conclusions & Strategic Plan

1. **Model Preservation:** `checkpoints/experiment_2/best.pt` and `checkpoints/ultimate_improvement/best.pt` are preserved as immutable reference points.
2. **Decoding Calibration:** Replace raw high-temperature top-k sampling with calibrated nucleus decoding (`temperature=0.35`, `top_p=0.85`, `top_k=15`, 3-gram repetition blocking).
3. **Autonomous Web Retrieval Subsystem:** Implement a zero-cost, API-key-free web-search provider abstraction with URL validation, SSRF protection, redirect limits, and clean HTML text extraction.
4. **Chatbot Application Integration:** Upgrade `src/chat.py` with multi-turn conversation memory, transparent `/web on|off` toggles, evidence grounding, and source citation formatting.
5. **Turnkey Launcher:** Provide a robust `RUN_CHATBOT.bat` script using the project's dedicated virtual environment.
