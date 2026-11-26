# Verification Report: Token-Averaged Gradient Accumulation Correction in `src/train_eos_v2_pilot.py`

**Date:** 2026-10-13  
**Target Script:** `src/train_eos_v2_pilot.py`  
**Dataset:** `data/processed/training_sequences_eos_v2/`  
**Verification Type:** Pre-Flight Verification & Mathematical Equivalence Proof  
**Final Verdict:** **READY TO TRAIN**

---

## 1. Executive Summary

The mathematical correction approved for `src/train_eos_v2_pilot.py` has been implemented and verified. The training objective is now strictly token-averaged across each 16-microbatch accumulation window, matching the exact loss and gradient of an effective physical batch of 16 sequences, and aligning 100% with the token-averaged validation loss evaluation.

All pre-flight checks, deterministic equivalence tests, and epoch-boundary handling verifications have passed without error. Baseline assets remain completely isolated and untouched.

---

## 2. Summary of Applied Corrections

The following isolated changes were applied to `src/train_eos_v2_pilot.py`:

1. **Unreduced Loss Summation:** Created `calculate_masked_token_loss_sum(logits, target_ids, loss_mask)`, which computes unreduced cross-entropy and sums only the positions selected by `loss_mask`:
   $$\text{sum\_loss}_m = \sum_{t \in \mathcal{M}_m} \ell(x_{m, t}, y_{m, t})$$
   This explicitly preserves the existing loss-mask alignment and supervised `[EOS]` turn boundaries.
2. **Batch Window Collection & Token Counting:** Each optimizer update collects exactly 16 microbatches and calculates the exact active supervised token count across the window:
   $$N_{\text{total}} = \sum_{m=1}^{16} N_m = \sum_{m=1}^{16} \sum_{t} M_{m, t}$$
3. **Exact Token-Averaged Scaling:** Each microbatch's summed loss is scaled by $N_{\text{total}}$ before backpropagation:
   $$\text{scaled\_loss}_m = \frac{\text{sum\_loss}_m}{N_{\text{total}}}$$
   Accumulating these gradients across the 16 microbatches yields the exact batch token-averaged gradient:
   $$\sum_{m=1}^{16} \nabla \text{scaled\_loss}_m = \frac{1}{N_{\text{total}}} \sum_{m=1}^{16} \nabla \text{sum\_loss}_m = \nabla \mathcal{L}_{\text{batch}}$$
4. **Consistent Loss Reporting:** Training loss is logged as:
   $$\mathcal{L}_{\text{train}} = \frac{\sum_{m=1}^{16} \text{sum\_loss}_m}{N_{\text{total}}}$$
   This is directly comparable to `evaluate_validation` (which calculates $\frac{\sum \text{val losses}}{\sum \text{val active tokens}}$).
5. **Seamless Epoch Boundary Provider (`ContinuousDataLoader`):** Replaced one-off epoch iterators with a continuous iterator that transitions seamlessly across epoch boundaries.

---

## 3. Epoch Boundary & Accumulation Handling

### The Problem Addressed
The training dataset contains 17,683 sequences. With an effective batch size of 16 sequences:
$$\frac{17,683}{16} = 1,105 \text{ updates} + 3 \text{ leftover sequences}$$
If epoch boundaries were handled naively (e.g. discarding incomplete batches or restarting without accumulating across the boundary), trailing microbatches would either be silently dropped or divided by an incorrect token count.

### The Implementation in `ContinuousDataLoader`
```python
class ContinuousDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.epoch = 1

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.iterator = iter(self.dataloader)
            return next(self.iterator)
```

### Verification of Boundary Behavior:
1. **Zero Discarded Sequences:** On update 1,106, the window draws sequences 17,681, 17,682, and 17,683 from Epoch 1. When `StopIteration` occurs, `ContinuousDataLoader` increments the epoch counter, creates a fresh iterator for Epoch 2, and draws sequences 1 through 13. All 16 sequences are collected into the accumulation window.
2. **Correct Token Normalization:** $N_{\text{total}}$ is computed over all 16 microbatches in that window ($3 \text{ from Epoch 1} + 13 \text{ from Epoch 2}$). Every microbatch is scaled by that exact sum.
3. **No Incomplete Updates:** Across the entire 5,000-update run (80,000 microbatches $\approx 4.524$ epochs), every single optimizer update contains exactly 16 microbatches and is normalized by the exact active token count.

---

## 4. Pre-Flight Test Results

The pre-flight verification routine was executed via `python src/train_eos_v2_pilot.py --preflight`.

### A. Deterministic Mathematical Equivalence Test
A controlled deterministic benchmark comparing the accumulated microbatch gradient against a single physical batch was executed on the GPU with 4 sequences of differing active token counts ($[5, 20, 40, 60]$ tokens):

| Metric | Physical Batch (Concurrently) | Accumulated Microbatches (Proposed) | Discrepancy | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Loss Value** | `9.027241` | `9.027240` | $9.16 \times 10^{-7}$ | **PASS** |
| **Max Absolute Parameter Gradient Diff** | — | — | **$8.94 \times 10^{-8}$** | **PASS** ($< 10^{-5}$) |

*Note for comparison: The old sequence-averaged method produced a gradient discrepancy of **$1.91 \times 10^{-1}$** on the same test.* The proposed token-averaged accumulation is mathematically equivalent to the physical batch down to float precision.

### B. Single Optimizer Update Verification (16 Microbatches)
* **Active Tokens in Test Update:** 5,704 supervised tokens across 16 microbatches.
* **Token-Averaged Training Loss:** **9.0606** (Finite: PASS).
* **Unscaled Gradient Norm:** **3.2669** (Finite: PASS).
* **Optimizer & Scaler Step:** Weights updated cleanly with zero NaN/Inf.
* **Learning-Rate Cadence:** LR at step 0 = `0.0`, LR after update 1 = `0.00000120` (exact match to schedule: $3.0 \times 10^{-4} / 250$).

### C. Pre-Training Validation Baseline (Step 0)
* **Validation Sequences Evaluated:** 1,989 sequences (659,864 active tokens).
* **Evaluation Time:** 8.00 s.
* **Initial Validation Loss:** **9.0592** (Perplexity: 8,597.64).
* **Theoretical Alignment:** Matches $-\ln(1/8000) = \ln(8000) = 8.987$.

### D. Resource & Runtime Benchmarking
* **Measured Time per Update:** 0.406 s / update (16 microbatches).
* **Projected 5,000-Update Training Time:** 2,029 s (~33.8 minutes).
* **Validation Time (11 full split evals):** 88 s (~1.5 minutes).
* **Total Estimated Pilot Runtime:** **~35.3 minutes**.
* **Peak GPU VRAM Allocated:** 1,282.9 MB (20.9% of 6,144 MB capacity).
* **Baseline File Integrity:** [checkpoints/clean_50k_best.pt](file:///d:/DEEP-LEARNING-PROJECT/checkpoints/clean_50k_best.pt) timestamp and content confirmed untouched.

---

## 5. Exact Code Changes Applied to `src/train_eos_v2_pilot.py`

### Change 1: Added `calculate_masked_token_loss_sum` and `ContinuousDataLoader`
```python
# Lines 140-165:
class ContinuousDataLoader:
    """
    Continuous sequence provider across epoch boundaries.
    Guarantees no microbatch is silently discarded at epoch boundaries,
    and all accumulation windows have exact active token counts.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.epoch = 1

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

criterion = nn.CrossEntropyLoss(reduction="none")

def calculate_masked_token_loss_sum(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor):
    """
    Computes unreduced cross-entropy losses and sums only the positions
    selected by the existing loss mask, preserving supervised [EOS] turn boundaries.
    Returns (sum_loss, active_token_count).
    """
    vocab_size = logits.size(-1)
    token_losses = criterion(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1)
    ).reshape(target_ids.shape)

    active_losses = token_losses[loss_mask]
    if active_losses.numel() == 0:
        return token_losses.sum() * 0.0, 0
    return active_losses.sum(), active_losses.numel()

def calculate_masked_loss(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    sum_loss, count = calculate_masked_token_loss_sum(logits, target_ids, loss_mask)
    if count == 0:
        return sum_loss
    return sum_loss / count
```

### Change 2: Deterministic Equivalence Test in Pre-Flight
```python
# Lines 314-375: Added verify_deterministic_physical_batch_equivalence(device)
# comparing physical batch vs accumulated microbatches and asserting max_grad_diff < 1e-5.
```

### Change 3: Token-Averaged Accumulation in Main Training Loop
```python
# Lines 773-820 in run_pilot_training:
    training_start = time.time()
    data_stream = ContinuousDataLoader(train_loader)

    milestone_updates = {2500, 5000}
    model.train()

    while update_step < MAX_OPTIMIZER_UPDATES:
        optimizer.zero_grad(set_to_none=True)

        # 1. Collect exactly 16 microbatches (seamless across epoch boundaries)
        microbatches = []
        total_active_tokens = 0
        for _ in range(GRAD_ACCUM_STEPS):
            b_inp, b_tgt, b_msk = data_stream.next()
            microbatches.append((b_inp, b_tgt, b_msk))
            total_active_tokens += b_msk.sum().item()

        total_active_tokens = max(1, total_active_tokens)
        accumulated_sum_loss = 0.0

        # 2. Accumulate backward passes normalized by total active tokens across the 16 microbatches
        for input_ids, target_ids, loss_mask in microbatches:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids)
                sum_loss, active_count = calculate_masked_token_loss_sum(logits, target_ids, loss_mask)
                scaled_loss = sum_loss / total_active_tokens

            scaler.scale(scaled_loss).backward()
            accumulated_sum_loss += sum_loss.item()
            microbatch_count += 1

        # Unscale, Clip, Step Optimizer, Update Scaler & Scheduler
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        update_step += 1
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - training_start
        gpu_mb = (torch.cuda.memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
        avg_train_loss = accumulated_sum_loss / total_active_tokens
```

---

## 6. Audit Verdict

**READY TO TRAIN**

All requirements are verified, all mathematical equivalence tests pass, all edge cases at epoch boundaries are handled cleanly, and all baseline assets remain untouched.
