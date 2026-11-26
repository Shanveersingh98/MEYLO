# Complete Controlled Experiment Log: Ultimate Chatbot Improvement

**Project Root:** `D:\DEEP-LEARNING-PROJECT`  
**Starting Checkpoint:** `checkpoints/experiment_2/best.pt` (SHA-256: `f717e5efff9dc6d58b0fc1d4872e74b0b03bd734029715a3ae14aa549b5b4ab1`)  
**Final Checkpoint:** `checkpoints/ultimate_improvement/best.pt` (SHA-256: `cecba4946a8b5a0404e42e9e1790c0d780228da45e07b1910d190d89cbadf2fc`)  
**Date:** 2026-10-27  

---

## Experiment Index & Trajectory Overview

```
[Experiment 2 Baseline]
  │   - Checkpoint: `checkpoints/experiment_2/best.pt`
  │   - Benchmark EOS: 100%, Composite Val Loss: 3.6421
  │   - Flaw: Repetition loops in 18.55% of test suite; 0/15 Python; 6/100 Arithmetic
  ▼
[Experiment A: Decoding Correctness]
  │   - Change: Added 3-gram repetition blocking (no_repeat_ngram_size=3) to inference
  │   - Model Weights: UNCHANGED (Reference checkpoint)
  │   - Outcome: Repetition loops dropped from 18.55% down to EXACTLY 0.00%
  │   - Concept recall rose from 29.9% to 36.1%
  ▼
[Experiment B & C: Targeted Chain-of-Thought & Curriculum Expansion]
  │   - Change: Synthesized 6,000 sequence curriculum with explicit step-by-step CoT arithmetic,
  │     AST-validated Python functions, instruction-following constraints, and multi-turn dialogue.
  │   - Training: 1,500 updates on RTX 3050 GPU (15.15 minutes, composite val loss: 2.1963)
  │   - Outcome: Instruction following improved from 6.67% to 33.33% (+400% relative).
  │     Multi-turn consistency improved from 30.0% to 50.0%.
  │     Zero catastrophic forgetting on OASST1 (val loss 4.0206 vs 4.0219 baseline).
  ▼
[Experiment D: Tool Grounding & Hybrid Assistant Mode]
      - Change: Implemented optional, transparent deterministic Python calculator tool.
      - Outcome: Single-step arithmetic accuracy jumped from 6.0% (neural alone) to 68.0% (with tool).
      - Multi-step arithmetic improved from 0.0% to 10.0%.
```

---

## Detailed Experiment Logs

### Experiment A: Decoding Correctness & Loop Elimination

* **Hypothesis:** Under pure greedy decoding, the small 14.1M parameter model gets trapped in alternating subword cycles (e.g. `_to_to_to...`, `capital of the capital of...`) because scalar repetition penalty (1.15) cannot overcome large positive logits on short cycles. Implementing n-gram repetition blocking (`no_repeat_ngram_size = 3`) will eliminate degenerate loops without retraining.
* **Independent Variable:** Inference decoding algorithm (standard scalar repetition penalty 1.15 vs 1.15 + 3-gram repetition blocking).
* **Model Checkpoint:** `checkpoints/experiment_2/best.pt` (SHA-256: `f717e5efff9dc6d58b0fc1d4872e74b0b03bd734029715a3ae14aa549b5b4ab1`).
* **Controlled Evaluation Suite:** 248 test items across 12 suites.
* **Results:**
  - **Repetition Loops:** Dropped from **46/248 (18.55%) down to 0/248 (0.00%)** — complete elimination of loops.
  - **Clean EOS Terminations:** Rose from **133/248 (53.63%) to 139/248 (56.05%)**.
  - **Concept Key-Point Recall:** Rose from **29/97 (29.90%) to 35/97 (36.08%)** (+20.7% relative gain).
  - **Factual QA:** Rose from **2/20 (10.0%) to 3/20 (15.0%)**.
  - **Arithmetic & Python:** Remained unchanged (6/100 and 0/15).
* **Conclusion:** Hypothesis confirmed. Decoding fixes successfully eradicate degenerate looping and improve explanatory recall, but cannot impart missing arithmetic or coding knowledge.

---

### Experiment B, C & E: Curriculum Expansion, CoT Reasoning & Fine-Tuning Optimization

* **Hypothesis:** Fine-tuning from `experiment_2/best.pt` on a balanced curriculum combining filtered OASST1 dialogue with explicit Chain-of-Thought (CoT) arithmetic, AST-validated Python algorithms, instruction following, and multi-turn persona memory will improve instruction adherence, multi-turn consistency, and structured formatting without degrading conversational fluency.
* **Dataset Composition (6,000 sequences):**
  - Cleaned OASST1 Conversations: 2,500 sequences (41.7%).
  - Chain-of-Thought Arithmetic: 1,500 sequences (25.0%) with explicit step-by-step scratchpad calculations.
  - Verified Python Algorithms: 1,200 sequences (20.0%) with docstrings and unit test assertions.
  - Instruction Following & Structured Formatting: 600 sequences (10.0%) covering word limits, YES/NO, JSON, and tables.
  - Conversational & Factual Dialogue: 310 sequences (5.2%) reinforcing dialogue memory and entity facts.
  - Zero leakage with test suites: Verified 0.0% prompt and numerical overlap.
* **Training Configuration:**
  - Base Checkpoint: `checkpoints/experiment_2/best.pt`.
  - Microbatch: 1 sequence $\times$ 16 gradient accumulation steps = 16 sequences per update.
  - Learning Rate: Conservative $6 \times 10^{-5}$ peak with 100-step warmup and cosine decay to $6 \times 10^{-6}$.
  - Weight Decay: 0.05, Gradient Clip: 1.0.
  - Precision: FP16 AMP.
  - Max Updates: 1,500 updates (4.0 epochs).
  - Checkpoint Selection: Composite loss $0.5 \mathcal{L}_{\text{curr}} + 0.5 \mathcal{L}_{\text{oasst1}}$.
* **Training Progress:**
  - Update 0: Curriculum Val Loss: 2.4913 (PPL: 12.08) | OASST1 Val Loss: 4.0219 (PPL: 55.81) | Composite: 3.2566
  - Update 150: Curriculum Val Loss: 0.8784 (PPL: 2.41) | OASST1: 4.0463 | Composite: 2.4624 (New Best)
  - Update 300: Curriculum Val Loss: 0.5925 (PPL: 1.81) | OASST1: 4.0415 | Composite: 2.3170 (New Best)
  - Update 600: Curriculum Val Loss: 0.4584 (PPL: 1.58) | OASST1: 4.0388 | Composite: 2.2486 (New Best)
  - Update 1050: Curriculum Val Loss: 0.3885 (PPL: 1.47) | OASST1: 4.0265 | Composite: 2.2075 (New Best)
  - Update 1500: Curriculum Val Loss: **0.3721 (PPL: 1.45)** | OASST1: **4.0206 (PPL: 55.73)** | Composite: **2.1963 (OPTIMAL CHECKPOINT)**
  - Total Training Time: 15.15 minutes on NVIDIA GeForce RTX 3050 Laptop GPU.
* **Evaluation on Untouched Test Suites:**
  - **Instruction Following:** Rose from **6.67% to 33.33%** (+400% relative increase). The model successfully complied with word-count limits, single-number outputs, and binary YES/NO constraints.
  - **Multi-Turn Consistency:** Rose from **30.0% to 50.0%** (+66.7% relative increase).
  - **Structured Formatting:** Rose from **0.0% to 10.0%**.
  - **Repetition Loops:** **0.0%** across all 248 tests.
  - **Conversational Retention:** OASST1 validation loss was strictly maintained at 4.0206 (better than baseline 4.0219), demonstrating zero catastrophic forgetting.
* **Limitations Identified:**
  - Zero-shot algorithmic execution on unseen multi-condition unit tests remained at 0/15. A 14.1M parameter model trained from scratch without large pretraining cannot synthesize novel algorithms zero-shot.

---

### Experiment D: Tool Grounding & Hybrid Assistant Mode (Part 9)

* **Hypothesis:** Elementary and multi-step arithmetic, as well as code execution, are fundamentally deterministic tasks that should be served by transparent, local, offline tools (a safe Python math calculator and sandboxed execution environment) rather than forcing a 14M language model to hallucinate multi-digit mental arithmetic.
* **Implementation:**
  - Integrated `safe_calculator` inside `src/ultimate_improvement/evaluate_tool_assisted.py` and `src/chat.py`.
  - When enabled, the calculator intercepts arithmetic queries, evaluates them deterministically, and clearly prefixes the output with `[Tool: Calculator]`.
  - The model alone and tool-assisted modes are evaluated and reported separately.
* **Results:**
  - **Single-Step Arithmetic:** Jumped from **6.0% (neural alone) to 68.0% (tool-assisted)**.
  - **Multi-Step Arithmetic:** Jumped from **0.0% (neural alone) to 10.0% (tool-assisted)**.
* **Conclusion:** Transparent tool grounding provides immediate, verifiable accuracy gains on deterministic tasks while keeping the neural model as the core language generator.
