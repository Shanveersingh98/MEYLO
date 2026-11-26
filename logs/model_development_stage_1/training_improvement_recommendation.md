# TRAINING IMPROVEMENT RECOMMENDATION & PILOT EXPERIMENT PLAN
**Stage**: Phase Seven — Scientifically Justified Model Improvement Strategy  
**Hardware Environment**: NVIDIA RTX 3050 Laptop GPU (6 GB VRAM), ~8 GB System RAM, Windows 11  
**Auditor**: Senior Deep Learning Research Scientist & PyTorch Systems Architect  
**Status**: Proposal Ready for Review (No Training Executed in this Mission)  

---

## 1. Executive Summary

Based on the empirical evidence gathered during Stage 1, the model's primary limitations are **data quality, severe padding inefficiency (77.7%), synthetic template overfitting, duplicate validation contamination, and lack of diverse instruction-following supervision**—not an inherent architectural defect in PyTorch code.

We outline a concrete, mathematically grounded strategy to upgrade the neural model within the 6 GB VRAM budget. We specify curated data sources, sequence-packing algorithms, training hyperparameters, and a controlled pilot experiment design.

---

## 2. Training Data Improvement Strategy

### 2.1 Target Data Distribution
To transform the 14M model into a reliable conversational assistant, the training corpus must be restructured into four balanced tiers:

```
┌────────────────────────────────────────────────────────┐
│ Target Training Distribution (10,000 Packed Sequences) │
├─────────────────────────┬──────────────────────────────┤
│ 1. Conversational Flow  │ 35% Clean Filtered OASST1    │
│ 2. Instruction Following│ 35% Dolly-15k / Alpaca-Clean │
│ 3. Core Explanations    │ 20% Verified Concept QA      │
│ 4. Structured Reasoning │ 10% Chain-of-Thought Math/Py │
└─────────────────────────┴──────────────────────────────┘
```

### 2.2 Recommended Open Data Sources & Licensing

1. **Databricks Dolly-15k** (`databricks/databricks-dolly-15k`):
   - **License**: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) — commercially permissive.
   - **Suitability**: Authored by thousands of Databricks employees. Covers open Q&A, closed Q&A, summarization, brainstorming, and classification with strict constraint adherence.
2. **Cleaned Stanford Alpaca** (`yahma/alpaca-cleaned`):
   - **License**: Apache 2.0 / Permissive research.
   - **Suitability**: Curated subset with hallucinated facts, empty outputs, and formatting errors stripped.
3. **Filtered OASST1 Conversations** (English only):
   - Filtered for quality score > 0.5, length between 100 and 600 characters, no profanity or adversarial prompts.

---

## 3. Dataset Construction & Engineering Innovations

### 3.1 Innovation 1: Sequence Packing (Eliminating 77.7% Padding)
Instead of allocating a full 1024-token sequence to a single 150-token conversation, implement **sequence packing**:
- Concatenate multiple short dialogues into one 1024-token block, separated by `[EOS]`:
  ```
  [BOS] <|user|> Q1 <|assistant|> A1 [EOS] <|user|> Q2 <|assistant|> A2 [EOS] ...
  ```
- Assistant tokens in both dialogues are assigned `mask = 1`.
- User and prompt tokens in both dialogues are assigned `mask = 0`.
- Padding only occurs at the very end of the packed 1024 block if no further dialogue fits.
- **Expected Impact**: Reduces padding from **77.74% down to < 5%**. Each training update will process **$\sim 4.3\times$ more supervised tokens**, drastically improving learning efficiency and token diversity per compute hour.

### 3.2 Innovation 2: Global Deduplication & Clean Split Guarantee
- Use MinHash LSH and exact SHA-256 sentence hashing across the combined dataset.
- Enforce strict split partitioning: **Train (90%)**, **Validation (10%)**.
- Zero overlap between train and validation prompts.
- Preserve a completely untouchable benchmark suite (e.g., our `independent_model_benchmark.json`).

---

## 4. Architecture Considerations: Retain vs. Scale

### 4.1 Recommendation: Retain Current Architecture for Stage 2
We recommend **retaining the current `FromScratchLM` architecture** (`d_model=384`, `num_heads=6`, `num_layers=6`, `d_ff=1536`, `vocab=8000`) for the immediate next training experiment:

- **Scientific Justification**: Before changing architecture variables (layer count, RoPE, head dimensions), we must isolate the impact of data quality and sequence packing. If we change both data and architecture simultaneously, we will not know which factor drove performance changes.
- **Hardware Profile**: Retaining 14.1M parameters keeps total training VRAM usage at **~1.8 GB to 2.2 GB**, leaving ample headroom on the 6 GB RTX 3050 and guaranteeing zero CUDA out-of-memory errors even with larger microbatches.

### 4.2 Potential Stage 3 Scale-Up (Optional Future Consideration)
If the data-clean pilot achieves low perplexity and clean instruction following, the architecture can be cleanly scaled to **28.4M parameters** ($d_{\text{model}}=512$, $8$ layers, $8$ heads, $d_{\text{ff}}=2048$) within **~3.2 GB VRAM**, well inside the 6 GB budget.

---

## 5. Recommended Hyperparameters for Next Training Experiment

| Hyperparameter | Current Value (`train_ultimate.py`) | Recommended Value for Next Run | Scientific Justification |
| :--- | :--- | :--- | :--- |
| **Data Format** | 1 dialogue / seq (77.7% pad) | **Sequence Packed (< 5% pad)** | $4.3\times$ increase in token throughput. |
| **Microbatch Size** | `1` | `2` or `4` | Improves GPU tensor core utilization. |
| **Gradient Accumulation** | `16` steps | `8` steps (with microbatch 2) | Maintains effective batch size of 16 sequences ($16,384$ tokens). |
| **Peak Learning Rate** | `6e-5` | `1.5e-4` (warmup 150 steps) | Optimal rate for training on dense packed data. |
| **Min Learning Rate** | `6e-6` | `1.5e-5` (10% of peak) | Cosine annealing floor. |
| **Weight Decay** | `0.05` | `0.05` | Retained (proven stable). |
| **Gradient Clip** | `1.0` | `1.0` | Retained (prevents gradient explosion). |
| **Optimizer** | AdamW $(\beta_1=0.9, \beta_2=0.95)$ | Retained | Retained (modern LLM standard). |
| **Precision** | PyTorch AMP `float16` | PyTorch AMP `float16` | Retained (stable and fast on RTX 3050). |

---

## 6. Controlled Pilot Experiment Plan (Readiness Blueprint)

### 6.1 Pilot Specifications
- **Objective**: Train a clean checkpoint from `checkpoints/experiment_2/best.pt` on 3,000 packed sequences of clean instruction data to verify that sequence packing and instruction data eliminate circular repetition loops without catastrophic forgetting.
- **Training Dataset**: `data/pilot_stage_2/packed_train_tokens.bin` (3,000 packed sequences $\approx 2.8\text{M}$ active tokens).
- **Validation Dataset**: `data/pilot_stage_2/packed_val_tokens.bin` (300 packed sequences, 0% duplicate leakage).
- **Control Validation**: `data/processed/training_sequences_eos_v2/validation_tokens.bin` (OASST1 hold-out to monitor conversational retention).
- **Expected Training Duration**: ~25 to 35 minutes on RTX 3050 GPU.
- **Checkpoint Location**: `checkpoints/pilot_stage_2/best.pt`.
- **Target Metrics**:
  - Validation Loss on clean instruction split: $< 3.20$.
  - Validation Loss on OASST1 conversational split: $\le 3.65$ (no regression from Experiment 2).
  - Repetition loops on `independent_model_benchmark.json`: Reduced from 4 loops down to 0.

### 6.2 Rollback & Safety Strategy
- `checkpoints/experiment_2/best.pt` and `checkpoints/ultimate_improvement/best.pt` remain write-protected and verified by SHA-256 hash.
- All pilot artifacts will write strictly to dedicated directories (`data/pilot_stage_2/`, `checkpoints/pilot_stage_2/`, `logs/pilot_stage_2/`).
- If validation loss does not improve within 300 optimizer updates, early stopping will abort the run automatically.
