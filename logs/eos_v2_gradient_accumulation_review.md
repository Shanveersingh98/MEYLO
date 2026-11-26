# Read-Only Review of Gradient Accumulation and Loss Normalization in `src/train_eos_v2_pilot.py`

**Date:** 2026-10-13  
**Review Target:** `src/train_eos_v2_pilot.py`  
**Reference Baseline:** `src/train_clean_50k.py`  
**Dataset:** `data/processed/training_sequences_eos_v2/`  
**Review Type:** Read-Only Audit (No training executed, no code modified)

---

## Executive Summary & Verdict

### Final Verdict: **CORRECTION REQUIRED**

A mathematical and empirical audit of `src/train_eos_v2_pilot.py` reveals that the current gradient accumulation implementation normalizes loss **per sequence** rather than **per supervised token** across the 16-sequence effective batch. 

Because sequences in the EOS v2 dataset have highly variable assistant response lengths (ranging from 8 to 997 active supervised tokens per sequence, an intra-batch variance of up to 92.6x), dividing each microbatch's mean loss by 16 weights tokens in short sequences **up to 92.6 times higher** than tokens in long sequences. Furthermore, this creates an objective mismatch between training (which optimizes sequence-averaged loss) and validation (which evaluates token-averaged loss).

The gradient unscaling, gradient clipping, and learning-rate scheduler update cadence are verified to be correct. A minimal, isolated correction is specified below and awaits approval before any file modifications or training execution.

---

## Detailed Audit Findings

### 1. Averaging Convention: Per-Sequence vs. Per-Supervised-Token

In `src/train_eos_v2_pilot.py`:

```python
# calculate_masked_loss (lines 124-138)
def calculate_masked_loss(logits, target_ids, loss_mask):
    ...
    masked_losses = token_losses[loss_mask]
    if masked_losses.numel() == 0:
        return token_losses.sum() * 0.0
    return masked_losses.mean()

# Training loop (lines 528-532)
with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
    logits = model(input_ids)
    loss = calculate_masked_loss(logits, target_ids, loss_mask)
    scaled_loss = loss / GRAD_ACCUM_STEPS  # GRAD_ACCUM_STEPS = 16

scaler.scale(scaled_loss).backward()
```

* **Sequence-Level Normalization:** For each microbatch $m \in \{1, \dots, 16\}$, `calculate_masked_loss` calculates `masked_losses.mean()`, which divides the sum of token losses $S_m$ by that sequence's active token count $N_m$:
  $$\mathcal{L}_m = \frac{S_m}{N_m} = \frac{1}{N_m} \sum_{t \in \mathcal{M}_m} \ell_{m, t}$$
* **Accumulated Objective:** When scaled by $\frac{1}{16}$ and accumulated over 16 microbatches:
  $$\mathcal{L}_{\text{accum}} = \frac{1}{16} \sum_{m=1}^{16} \mathcal{L}_m = \frac{1}{16} \sum_{m=1}^{16} \left(\frac{1}{N_m} \sum_{t \in \mathcal{M}_m} \ell_{m, t}\right)$$
* **Finding:** The loss is averaged **per sequence**, NOT per supervised token across the accumulated batch.

---

### 2. Normalization Distortion Across Variable Active Token Counts

An empirical measurement of active tokens ($N_m$) was conducted across sequences in `data/processed/training_sequences_eos_v2/train_mask.bin`:

| Metric | Measured Value |
| :--- | :--- |
| **Minimum Supervised Tokens in Sequence** | **8 tokens** |
| **Maximum Supervised Tokens in Sequence** | **997 tokens** |
| **Mean Supervised Tokens** | **376.2 tokens** |
| **Median Supervised Tokens** | **330.5 tokens** |
| **5th Percentile ($P_5$)** | **45.0 tokens** |
| **95th Percentile ($P_{95}$)** | **901.1 tokens** |
| **Average Max/Min Token Ratio within 16-Sequence Batches** | **21.88x** |
| **Peak Max/Min Token Ratio within 16-Sequence Batches** | **92.60x** |

#### Consequence for Token Gradients:
Under sequence-averaged accumulation, each token in microbatch $m$ contributes a gradient weight of:
$$w_{m, t} = \frac{1}{16 \times N_m}$$

* In a short sequence with $N_1 = 8$ tokens (e.g., a brief confirmation or short greeting):
  $$w_{1, t} = \frac{1}{16 \times 8} = \frac{1}{128} \approx 7.81 \times 10^{-3}$$
* In a long sequence with $N_2 = 740$ tokens (e.g., a detailed code explanation):
  $$w_{2, t} = \frac{1}{16 \times 740} = \frac{1}{11840} \approx 8.45 \times 10^{-5}$$

**Gradient Weight Discrepancy:** Every token in the 8-token response receives **92.5 times larger gradient magnitude** than a token in the 740-token explanation. In particular, the supervised `[EOS]` token terminating a short turn receives 92.5x more gradient influence than the `[EOS]` token terminating an extensive turn.

---

### 3. Mathematical Evaluation of Dividing by 16

In standard deep learning and distributed language model pretraining/finetuning, gradient accumulation over $K$ microbatches is defined to **reproduce the exact gradient of a single physical batch of size $K$**.

If a batch of 16 sequences was processed concurrently in a tensor of shape `[16, 1024]`, standard masked cross-entropy computes:
$$\mathcal{L}_{\text{physical\_batch}} = \frac{1}{\sum_{m=1}^{16} N_m} \sum_{m=1}^{16} \sum_{t \in \mathcal{M}_m} \ell_{m, t} = \frac{S_{\text{total}}}{N_{\text{total}}}$$

Comparing the two mathematical objectives:
$$\mathcal{L}_{\text{seq-avg}} = \frac{1}{16} \sum_{m=1}^{16} \frac{S_m}{N_m} \quad \ne \quad \frac{\sum_{m=1}^{16} S_m}{\sum_{m=1}^{16} N_m} = \mathcal{L}_{\text{token-avg}}$$

The two expressions are mathematically identical **only if** $N_1 = N_2 = \dots = N_{16}$ (fixed constant length for all sequences). Because active tokens vary by up to 92.6x across sequences, dividing every microbatch loss by 16 does **not** compute the intended global token-averaged cross-entropy of an effective batch of 16 sequences.

---

### 4. Comparison with `src/train_clean_50k.py`

In the baseline script `src/train_clean_50k.py`:
* `BATCH_SIZE = 1`, and `GRAD_ACCUM_STEPS` was not present (1 microbatch per optimizer update).
* For every sequence $i$, it evaluated `calculate_masked_loss` (which normalized by $N_i$) and executed an immediate optimizer step:
  $$\Delta \theta_i = -\eta \cdot \text{AdamW}\left(\frac{1}{N_i} \sum_{t} \nabla \ell_{i, t}\right)$$
* At batch size 1, sequence-averaging and token-averaging are trivially identical ($K = 1$).
* When extending to effective batch size 16 via gradient accumulation, naively inserting `loss / 16` causes the optimizer to step on the unweighted average of 16 sequence gradients rather than the true batch-level token-averaged gradient.
* **Validation Metric Mismatch:**
  * In `src/train_clean_50k.py`, validation averaged the losses across 20 individual sequences (`mean_val_loss = total_loss / batches`, sequence-averaged).
  * In `src/train_eos_v2_pilot.py`, `evaluate_validation` was implemented as:
    $$\text{mean\_val\_loss} = \frac{\sum \text{losses}}{\sum \text{active tokens}}$$
    This is **token-averaged**.
  * Consequently, the pilot's training loss ($\mathcal{L}_{\text{seq-avg}}$) and validation loss ($\mathcal{L}_{\text{token-avg}}$) currently measure two fundamentally different loss functions.

---

### 5. Verification of Gradient Clipping & Unscaling Order

The execution sequence in `src/train_eos_v2_pilot.py` (lines 537–544) was inspected:

```python
# Unscale, Clip, Step Optimizer, Update Scaler & Scheduler
scaler.unscale_(optimizer)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
scaler.step(optimizer)
scaler.update()
scheduler.step()
```

* **Step 1:** All 16 microbatches execute `scaler.scale(scaled_loss).backward()`. Gradients accumulate in parameter `.grad` attributes.
* **Step 2:** `scaler.unscale_(optimizer)` unscales parameter gradients from FP16 scaling back to full-precision floats.
* **Step 3:** `torch.nn.utils.clip_grad_norm_` operates on unscaled, accumulated gradients, correctly clipping global $L_2$ norm to `1.0`.
* **Step 4:** `scaler.step(optimizer)` updates weights using the clipped gradients.
* **Step 5:** `scaler.update()` adjusts scale factor for subsequent updates.
* **Verdict on Ordering:** **PASS**. Gradient unscaling and clipping occur strictly after accumulation and strictly before the optimizer step.

---

### 6. Verification of Learning-Rate Scheduler Cadence

The scheduler invocation in `src/train_eos_v2_pilot.py` was inspected:

* `scheduler.step()` is invoked on line 542, located strictly **outside** the 16-microbatch inner loop.
* It is called exactly once per optimizer update (5,000 times across the run).
* In the pre-flight run, the initial LR was `0.0`, and after 1 optimizer update (16 microbatches), the LR advanced to exactly `0.00000120` ($3.0 \times 10^{-4} / 250$).
* **Verdict on Cadence:** **PASS**. The scheduler advances per optimizer update, not per microbatch.

---

## 7. Recommended Isolated Correction

To align the pilot training objective with true token-averaged cross-entropy (reproducing a physical batch of 16 sequences and matching `evaluate_validation`):

### Proposed Design
Buffer the 16 microbatches for the current update (16 sequences in CPU RAM = 32 KB), calculate the exact sum of supervised tokens across the window:
$$N_{\text{total}} = \sum_{m=1}^{16} N_m$$
Then scale each microbatch by $N_{\text{total}}$:
$$\text{scaled\_loss}_m = \frac{\sum_{t \in \mathcal{M}_m} \ell_{m, t}}{N_{\text{total}}}$$

This guarantees that:
$$\sum_{m=1}^{16} \text{scaled\_loss}_m = \frac{\sum_{m=1}^{16} S_m}{N_{\text{total}}} = \mathcal{L}_{\text{token-avg}}$$
Every token in the 16-sequence window receives identical gradient weight $\frac{1}{N_{\text{total}}}$.

### Proposed Code Diff (for review only; not applied):

```python
# In train_eos_v2_pilot.py (Training Loop):

-       for _ in range(GRAD_ACCUM_STEPS):
-           try:
-               input_ids, target_ids, loss_mask = next(train_iter)
-           except StopIteration:
-               train_iter = iter(train_loader)
-               input_ids, target_ids, loss_mask = next(train_iter)
-
-           input_ids = input_ids.to(device)
-           target_ids = target_ids.to(device)
-           loss_mask = loss_mask.to(device)
-
-           with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
-               logits = model(input_ids)
-               loss = calculate_masked_loss(logits, target_ids, loss_mask)
-               scaled_loss = loss / GRAD_ACCUM_STEPS
-
-           scaler.scale(scaled_loss).backward()
-           accumulated_loss += loss.item()
-           microbatch_count += 1
+       # 1. Collect 16 microbatches and count total active tokens
+       microbatches = []
+       total_active_tokens = 0
+       for _ in range(GRAD_ACCUM_STEPS):
+           try:
+               batch = next(train_iter)
+           except StopIteration:
+               train_iter = iter(train_loader)
+               batch = next(train_iter)
+           microbatches.append(batch)
+           total_active_tokens += batch[2].sum().item()
+
+       total_active_tokens = max(1, total_active_tokens)
+       accumulated_sum_loss = 0.0
+
+       # 2. Accumulate backward passes normalized by total_active_tokens
+       for input_ids, target_ids, loss_mask in microbatches:
+           input_ids = input_ids.to(device)
+           target_ids = target_ids.to(device)
+           loss_mask = loss_mask.to(device)
+
+           with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
+               logits = model(input_ids)
+               token_losses = criterion(
+                   logits.reshape(-1, VOCAB_SIZE),
+                   target_ids.reshape(-1)
+               ).reshape(target_ids.shape)
+               sum_loss = token_losses[loss_mask].sum()
+               scaled_loss = sum_loss / total_active_tokens
+
+           scaler.scale(scaled_loss).backward()
+           accumulated_sum_loss += sum_loss.item()
+           microbatch_count += 1
+
+       avg_train_loss = accumulated_sum_loss / total_active_tokens
```

---

## Summary of Checklist Items

| # | Inspection Item | Status | Finding |
| :---: | :--- | :---: | :--- |
| 1 | Loss averaged per sequence or per token | **PER SEQUENCE** | Each microbatch divided by $N_m$, then by 16. |
| 2 | Loss normalization across variable active tokens | **SKEWED** | Short sequences weighted up to 92.6x higher per token than long sequences. |
| 3 | Dividing every microbatch loss by 16 | **NON-EQUIVALENT** | Does not equal true token-averaged loss of an effective batch of 16. |
| 4 | Comparison with `train_clean_50k.py` | **DIVERGENT** | 50k baseline used batch size 1 (1 update/seq); accumulation introduces sequence weighting. Validation was sequence-averaged in 50k, but token-averaged in pilot. |
| 5 | Gradient clipping ordering | **VERIFIED** | Occurs after unscaling, after accumulation, before optimizer step. |
| 6 | Learning-rate scheduler step cadence | **VERIFIED** | Advances strictly once per optimizer update (5,000 steps). |
| 7 | Recommended isolated correction | **SPECIFIED** | Normalize microbatch loss sums by total active tokens across the 16-sequence window. |

**Current State:** No files have been modified. Standing by for review and approval.
