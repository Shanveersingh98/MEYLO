# Comprehensive Initial Forensic Audit: From-Scratch AI Chatbot

**Project Root:** `D:\DEEP-LEARNING-PROJECT`  
**Protected Reference Checkpoint:** `checkpoints\experiment_2\best.pt`  
**Verified SHA-256:** `f717e5efff9dc6d58b0fc1d4872e74b0b03bd734029715a3ae14aa549b5b4ab1`  
**Tokenizer Path:** `tokenizer\tokenizer.json`  
**Verified Tokenizer SHA-256:** `0f00fae1c3852879cb0bb9a807b45aad93d7700ca4e3f903b1c607ac8c565b08`  
**Audit Date:** 2026-10-27  
**Status:** COMPLETE — All Evidence Empirically Verified Against Local Code and Artifacts  

---

## 1. Executive Summary & Audit Mandate

This forensic audit investigates the architecture, datasets, tokenization, training mechanics, decoding behavior, and operational interfaces of the from-scratch educational chatbot project at `D:\DEEP-LEARNING-PROJECT`.

The project centers on `FromScratchLM`, a 14,120,768-parameter decoder-only Transformer built from first principles in PyTorch. The primary goal is to determine the exact root causes of its performance bottlenecks (such as failures in arithmetic, algorithmic Python generation, multi-turn conversation, and decoding repetition) and evaluate the codebase against strict standards of correctness, reliability, and reproducibility before implementing enhancements.

---

## 2. Dependency and Data-Flow Map

The complete lifecycle from raw conversational corpus to inference output is mapped below:

```
[Raw Data Sources]
  ├── OASST1 Raw English Conversations (`data/processed/conversation_paths/train_paths.jsonl`)
  ├── Arithmetic QA Generator (`src/experiment_2/generate_arithmetic.py`)
  ├── Python QA & AST Generator (`src/experiment_2/generate_python.py`)
  └── Concepts Definition Generator (`src/experiment_2/generate_concepts.py`)
          │
          ▼
[Curriculum Packaging (`src/experiment_2/build_experiment_2_sequences.py`)]
  ├── Formatted as EOS v2: [BOS]\n<|user|>\n{prompt}\n<|assistant|>\n{response}\n[EOS]
  ├── Padded to 1024 context tokens with [PAD]=0
  └── Binary memmap serialization:
        ├── `train_tokens.bin` (uint16, 7422 seqs x 1024)
        └── `train_mask.bin`   (uint8, assistant-only loss mask: 1=active, 0=ignored)
          │
          ▼
[Model Optimization (`src/experiment_2/train_experiment_2.py`)]
  ├── Architecture: `FromScratchLM` (`src/model.py`, 6 layers, 6 heads, d_model=384, d_ff=1536)
  ├── Base weights loaded: `checkpoints/eos_v2_pilot/best.pt`
  ├── Microbatch 1 x 16 gradient accumulation = 16 sequences effective batch
  ├── Exact token-averaged loss normalization across accumulation window
  ├── Mixed-precision (FP16 via torch.amp.GradScaler)
  └── Best checkpoint selected by composite loss (0.5 * Curriculum_Val + 0.5 * OASST1_Val)
          │
          ▼
[Model Artifacts]
  ├── `checkpoints/experiment_2/best.pt` (Optimizer step 1,750, SHA-256: `f717e5eff...`)
  └── `checkpoints/experiment_2/latest.pt` (Optimizer step 2,000)
          │
          ▼
[Inference & Evaluation Pipelines]
  ├── Disconnected Legacy Script: `src/generate.py` (STILL loads old `checkpoints/masked_best.pt`!)
  ├── Diagnostic Evaluator: `src/experiment_2/run_full_functional_evaluation.py`
  └── Heldout Validator: `src/experiment_2/evaluate_experiment_2_heldout.py`
```

---

## 3. Forensic Investigation by Category

### A. Model Correctness (`src/model.py`)

1. **Causal Attention Masking:**
   - Implementation: Uses `torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)`.
   - Verification: A mathematical perturbation test was executed. Random input IDs of length 10 were passed to `FromScratchLM`, then future tokens $[5..9]$ were perturbed. The maximum absolute difference in past logits $[0..4]$ was exactly `0.00000000`. Future logits $[5..9]$ changed by `0.87638950`. This proves strict causal attention without future leakage.
2. **Weight Tying:**
   - Implementation: `logits = F.linear(x, self.token_embedding.weight, self.output_bias)`.
   - Analysis: The language model head correctly shares the 8,000 x 384 embedding weight tensor with `token_embedding.weight`, but retains its own dedicated learnable output bias vector (`torch.zeros(vocab_size)`), matching the standard tied-embedding design.
3. **Normalization & Residual Stream:**
   - Pre-LayerNorm architecture: `x = x + attention(norm1(x))` followed by `x = x + feed_forward(norm2(x))`, terminated with `final_norm(x)`. Pre-LN guarantees stable gradient flow during deep transformer optimization.
4. **Positional Encoding:**
   - Learned positional embedding: `self.position_embedding = nn.Embedding(context_length, d_model)`. Positions are created dynamically with `torch.arange(sequence_length, device=input_ids.device)`.
   - Length Bounds Check: If `sequence_length > self.context_length`, a `ValueError` is explicitly raised.
5. **Padding Semantics in Attention:**
   - `FromScratchLM.forward(input_ids)` takes only `input_ids`. No key padding mask is supplied to SDPA.
   - Impact: In training, sequences have trailing padding tokens. Because attention is causal, real tokens precede padding tokens and therefore *cannot* attend to padding tokens. Padding tokens do attend to prior tokens, but their loss is masked out to 0. In autoregressive inference, generation grows token by token without padding.
6. **Mixed-Precision & Numerical Stability:**
   - Verified under `torch.autocast('cuda', dtype=torch.float16)`. Parameter count is exactly 14,120,768. Forward and backward passes produce finite values without NaNs or Infinities.
7. **Checkpoint Serialization & Restoration:**
   - Round-trip loading of `checkpoints\experiment_2\best.pt` was executed. All state dict keys matched with 0 missing and 0 unexpected keys.

### B. Tokenizer Correctness (`tokenizer/tokenizer.json`)

1. **Vocabulary & Special Tokens:**
   - Vocabulary size: Exactly 8,000 subwords.
   - Special Token IDs:
     - `[PAD]` = 0
     - `[UNK]` = 1
     - `[BOS]` = 2
     - `[EOS]` = 3
     - `<|user|>` = 4
     - `<|assistant|>` = 5
2. **Special Token Preservation:**
   - Tested encoding: `<|user|>` encodes strictly to `[4]`, `<|assistant|>` to `[5]`, `[EOS]` to `[3]`, `[BOS]` to `[2]`. They are preserved as atomic tokens.
3. **Byte-Level BPE Space & Newline Mapping:**
   - Due to byte-level BPE, spaces are represented by `Ġ` (`\u0120`, token 174) and newlines by `Ċ` (`\u010a`, token 171). The string `\n` encodes to `['Ġ', 'Ċ']` (`[174, 171]`).
4. **Prompt Injection Risk:**
   - If user input contains the raw string `<|assistant|>` or `<|user|>`, the tokenizer converts it to token 5 or 4, which could spoof conversation boundaries unless inputs are sanitized.

### C. Conversation Construction & Masking (`src/experiment_2/build_experiment_2_sequences.py`)

1. **Sequence Format (EOS v2):**
   - Format: `[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n{response}\n[EOS]`
   - Every assistant turn explicitly terminates with `[EOS]`.
2. **Loss Masking Logic:**
   - `loss_mask` is 1 for `<|assistant|>`, all assistant response tokens, and the terminating `[EOS]`.
   - `loss_mask` is 0 for `[BOS]`, `<|user|>`, user prompt tokens, and padding `[PAD]`.
3. **Autoregressive Alignment in Dataset:**
   - `input_ids = seq[:-1]`, `target_ids = seq[1:]`, `loss_mask = m[1:]`.
   - Mathematical check: For input token $S[j]$, the target is $S[j+1]$ and the loss mask is $M[j+1]$. At the last prompt token, the target is `<|assistant|>` and $M=1$. At the last assistant token, the target is `[EOS]` and $M=1$. At `[EOS]`, the target is `[PAD]` and $M=0$. Verified: zero off-by-one errors.
4. **Data Isolation & Previous Contamination:**
   - Legacy directory `data\processed\training_sequences_masked\` was previously contaminated by missing turn-level `[EOS]` in 57.9% of turns.
   - Active directory `data\experiment_2\training_sequences\` contains clean EOS v2 sequences with 0% truncation, zero tree overlap with validation, and 100% active loss tokens correctly attributed.

### D. Training Correctness (`src/experiment_2/train_experiment_2.py`)

1. **Gradient Accumulation & Loss Normalization:**
   - Accumulation: 16 microbatches of size 1 sequence.
   - Mathematical formulation: Microbatch losses are summed without per-batch scaling (`scaler.scale(mb_loss_sum).backward()`), unscaled via `scaler.unscale_(optimizer)`, and divided by the exact sum of active supervised tokens across the 16-microbatch window (`p.grad.data.mul_(1.0 / accum_active_tokens)`). This is mathematically identical to a true batch size of 16.
2. **Optimizer & Regularization:**
   - AdamW with $\beta_1=0.9, \beta_2=0.95, \text{weight\_decay}=0.05, \text{eps}=10^{-8}$.
   - Gradient clipping at $\text{norm} = 1.0$.
3. **Learning Rate Schedule:**
   - 150 updates linear warmup from 0 to $8 \times 10^{-5}$, followed by cosine decay down to $8 \times 10^{-6}$ at update 2,500.
4. **Validation Separation & Early Stopping:**
   - Model selection is governed by composite validation loss: $0.5 \mathcal{L}_{\text{curriculum}} + 0.5 \mathcal{L}_{\text{oasst1}}$.
   - Optimum reached at update 1,750 ($\mathcal{L}_{\text{composite}} = 3.6421$). Checkpoint safely saved as `best.pt`.

### E. Inference, Decoding & Application Audit

1. **CRITICAL FINDING: Main Inference Script Disconnected:**
   - File: `src/generate.py`.
   - Observation: Line 33 loads `"checkpoints/masked_best.pt"` (the legacy flawed checkpoint) instead of `"checkpoints/experiment_2/best.pt"`.
   - Impact: Anyone running `python src/generate.py` executes the old broken model that hallucinates and loops, completely unaware that Experiment 2 improved the model.
2. **Inference Stop Token Defect in `src/generate.py`:**
   - Line 151 stops *only* on `next_token.item() == eos_id`.
   - If the model emits `<|user|>` (token 4), `generate.py` does not stop, drifting into runaway generation. In contrast, `run_full_functional_evaluation.py` correctly stops on `{0, 2, 3, 4, 5}`.
3. **Repetition Penalty Vulnerability on Subword Bigram Cycles:**
   - While token-level repetition penalty (1.15) penalizes individual tokens, it fails when the model enters an alternating subword loop such as `_` (68) and `to` (1271) in `def celsius_to_to_to_to...`. Because the raw logits are overwhelmingly positive, scaling them by $1/1.15$ leaves them higher than any alternative vocabulary token.
4. **Absence of Interactive Chat Application:**
   - The repository contains no interactive CLI or GUI script for continuous multi-turn dialogue with conversation history management.

---

## 4. Comprehensive Audit Findings Table

| ID | Category | Severity | Description & Evidence | Likely Impact | Proposed Correction | Verification Method |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **F-01** | Application | **CRITICAL** | `src/generate.py` loads legacy `checkpoints/masked_best.pt` instead of reference `checkpoints/experiment_2/best.pt`. | Users/evaluators run the old corrupted model rather than the improved Experiment 2 checkpoint. | Update `src/generate.py` to point to the verified reference checkpoint or provide an interactive chat CLI. | Run `python src/generate.py` and inspect loaded checkpoint step and loss. |
| **F-02** | Decoding | **HIGH** | `src/generate.py` stop condition only checks for `[EOS]` (token 3), ignoring `<|user|>` (4) and `<|assistant|>` (5). | Runaway generation when turn boundary is emitted. | Update stop tokens to `{0, 2, 3, 4, 5}`. | Verify termination when role token is emitted. |
| **F-03** | Decoding | **HIGH** | Scalar token repetition penalty (1.15) fails on 2-token alternating subword loops (e.g., `_to_to_to_`). | In Python function generation, model gets stuck in repeated subword cycles in function headers. | Implement n-gram repetition blocking or calibrated repetition penalty prior to argmax. | Evaluate on `celsius_to_kelvin` prompt under greedy decoding. |
| **F-04** | Security / Robustness | **MEDIUM** | Raw user input containing special token strings (`<|user|>`, `<|assistant|>`) is encoded directly to special IDs. | Potential prompt injection or boundary corruption from raw inputs. | Sanitize or escape special token literals in user prompt before formatting. | Test prompt containing `<|assistant|>` and verify it does not trigger premature assistant mode. |
| **F-05** | Modeling / Attention | **LOW** | `FromScratchLM` does not pass key padding mask to SDPA. | Safe in causal training with trailing pads, but trailing tokens attend to padding tokens. | Keep as-is to preserve exact weights compatibility with `best.pt`. In generation, sequence is unpadded. | Forward pass verification with variable context length. |
| **F-06** | Reasoning / Data | **HIGH** | Multi-digit arithmetic accuracy is 0% due to zero-shot direct autoregressive prediction without intermediate calculation tokens. | 14.1M model cannot perform multi-digit carry/borrow without scratchpad tokens. | Introduce structured step-by-step reasoning tokens (Chain of Thought scratchpads) or deterministic tool assistance. | Benchmark 100 untouched arithmetic problems comparing zero-shot vs CoT/tool. |
| **F-07** | Coding / Data | **HIGH** | Python algorithmic functional correctness is 0/15 on unseen tasks because training data mostly contains syntax headers and explanations. | Model generates valid syntax headers but fails complex logical problem solving. | Augment training curriculum with verified algorithmic problem/solution pairs and unit test execution. | Execute unit test suite on generated Python functions. |

---

## 5. Audit Conclusion & Next Actions

The model architecture (`FromScratchLM`), tokenizer (8k BPE), and training pipeline in `src/experiment_2` are verified to be mathematically sound, strictly causal, and free of gradient accumulation or off-by-one target bugs.

The primary system-level flaw in the project was that **the default inference script was pointing to an obsolete, flawed checkpoint**, while the primary capability bottlenecks are **lack of scratchpad tokens for multi-step reasoning** and **insufficient algorithmic problem-solving coverage in the curriculum**.

We are now ready to establish the baseline evaluation suite as mandated in Part 4.
