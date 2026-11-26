from pathlib import Path
import sys
import json
import time
import math
import argparse

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

# ============================================================
# 1. PATH CONFIGURATION (EOS v2 Pilot Isolation)
# ============================================================

DATA_DIR = PROJECT_DIR / "data" / "processed" / "training_sequences_eos_v2"
TRAIN_TOKENS_FILE = DATA_DIR / "train_tokens.bin"
TRAIN_MASK_FILE = DATA_DIR / "train_mask.bin"
VAL_TOKENS_FILE = DATA_DIR / "validation_tokens.bin"
VAL_MASK_FILE = DATA_DIR / "validation_mask.bin"
METADATA_FILE = DATA_DIR / "dataset_metadata.json"

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"

# Checkpoint directory - isolated in eos_v2_pilot
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "eos_v2_pilot"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

# Baseline checkpoint to protect
BASELINE_BEST_CHECKPOINT = PROJECT_DIR / "checkpoints" / "clean_50k_best.pt"
BASELINE_LATEST_CHECKPOINT = PROJECT_DIR / "checkpoints" / "clean_50k_latest.pt"

# Log directory - isolated in eos_v2_pilot
LOG_DIR = PROJECT_DIR / "logs" / "eos_v2_pilot"
LOG_FILE = LOG_DIR / "training_eos_v2_pilot.jsonl"
VAL_LOG_FILE = LOG_DIR / "validation_eos_v2_pilot.jsonl"

# ============================================================
# 2. EXPERIMENT HYPERPARAMETERS
# ============================================================

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024

# Architecture (identical to FromScratchLM baseline, 14.1M parameters)
D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
D_FF = 1536
DROPOUT = 0.1

# Batch and accumulation
MICROBATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
EFFECTIVE_BATCH_SIZE = MICROBATCH_SIZE * GRAD_ACCUM_STEPS  # 16 sequences

# Optimization
PEAK_LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0

# Schedule
MAX_OPTIMIZER_UPDATES = 5000
WARMUP_UPDATES = 250
VALIDATION_INTERVAL = 500  # Evaluate every 500 optimizer updates

# Special Tokens
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5

# ============================================================
# 3. DATASET CLASS
# ============================================================

class EOSv2TokenDataset(Dataset):
    """
    Memory-mapped dataset for EOS v2 tokenized sequences.
    Ensures input_ids, target_ids, and assistant loss_masks
    are properly shifted and formatted.
    """
    def __init__(self, token_file_path: Path, mask_file_path: Path):
        self.token_file_path = Path(token_file_path)
        self.mask_file_path = Path(mask_file_path)

        if not self.token_file_path.exists():
            raise FileNotFoundError(f"Tokens binary not found: {self.token_file_path}")
        if not self.mask_file_path.exists():
            raise FileNotFoundError(f"Mask binary not found: {self.mask_file_path}")

        self.tokens = np.memmap(self.token_file_path, dtype=np.uint16, mode="r")
        self.loss_mask = np.memmap(self.mask_file_path, dtype=np.uint8, mode="r")

        if len(self.tokens) != len(self.loss_mask):
            raise ValueError(
                f"Tokens length ({len(self.tokens)}) != Mask length ({len(self.loss_mask)})"
            )

        if len(self.tokens) % CONTEXT_LENGTH != 0:
            raise ValueError(
                f"Dataset length ({len(self.tokens)}) not divisible by {CONTEXT_LENGTH}"
            )

        self.sequence_count = len(self.tokens) // CONTEXT_LENGTH

    def __len__(self):
        return self.sequence_count

    def __getitem__(self, index):
        start = index * CONTEXT_LENGTH
        end = start + CONTEXT_LENGTH

        sequence = np.asarray(self.tokens[start:end], dtype=np.int64)
        mask = np.asarray(self.loss_mask[start:end], dtype=np.uint8)

        # Shifted input, target, and loss mask
        # input_ids: tokens[0 : 1023]
        # target_ids: tokens[1 : 1024]
        # loss_mask: mask[1 : 1024]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[1:], dtype=torch.bool)

        return input_ids, target_ids, loss_mask

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

# ============================================================
# 4. LOSS FUNCTIONS
# ============================================================

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
    """
    Computes sequence-level mean masked cross-entropy loss over assistant-only tokens.
    """
    sum_loss, count = calculate_masked_token_loss_sum(logits, target_ids, loss_mask)
    if count == 0:
        return sum_loss
    return sum_loss / count

# ============================================================
# 5. LEARNING RATE SCHEDULE
# ============================================================

def get_cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    """
    Learning rate scheduler aligned with optimizer updates:
    - Linear warmup from 0 to 1.0 * peak_lr over warmup_steps.
    - Cosine decay from 1.0 to min_lr_ratio * peak_lr over (total_steps - warmup_steps).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# ============================================================
# 6. EVALUATION FUNCTIONS
# ============================================================

def evaluate_validation(model, val_loader, device: torch.device, use_amp: bool):
    """
    Evaluates loss on the validation split.
    Uses microbatch_size=16 for evaluation to maximize GPU throughput.
    """
    model.eval()
    total_loss = 0.0
    total_active_tokens = 0
    total_sequences = 0

    eval_crit = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for input_ids, target_ids, loss_mask in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids)
                tok_losses = eval_crit(
                    logits.reshape(-1, VOCAB_SIZE),
                    target_ids.reshape(-1)
                ).reshape(target_ids.shape)
                active_losses = tok_losses[loss_mask]
                total_loss += active_losses.sum().item()
                total_active_tokens += active_losses.numel()
                total_sequences += input_ids.size(0)

    model.train()
    mean_val_loss = (total_loss / total_active_tokens) if total_active_tokens > 0 else float("inf")
    val_ppl = float(np.exp(min(mean_val_loss, 20.0)))
    return mean_val_loss, val_ppl, total_sequences, total_active_tokens

# ============================================================
# 7. CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(
    path: Path,
    update_step: int,
    microbatch_count: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    val_loss: float,
    best_loss: float,
    extra_metadata: dict = None
):
    """
    Saves complete training checkpoint with strict path isolation.
    """
    # Safety assertion: Never overwrite baseline checkpoints
    if "clean_50k" in str(path).lower():
        raise RuntimeError(f"CRITICAL SAFETY VIOLATION: Attempted to write to baseline path: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "optimizer_update": update_step,
        "microbatch_count": microbatch_count,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "validation_loss": val_loss,
        "best_validation_loss": best_loss,
        "config": {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "microbatch_size": MICROBATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "peak_learning_rate": PEAK_LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "warmup_updates": WARMUP_UPDATES,
            "max_optimizer_updates": MAX_OPTIMIZER_UPDATES,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "dataset": "training_sequences_eos_v2",
            "experiment": "eos_v2_pilot"
        },
        "extra_metadata": extra_metadata or {}
    }
    torch.save(ckpt, path)
    file_mb = path.stat().st_size / (1024 ** 2)
    print(f"Checkpoint saved: {path.name} ({file_mb:.1f} MB) at update {update_step:,}")

# ============================================================
# 8. PRE-FLIGHT CHECKS
# ============================================================

def verify_deterministic_physical_batch_equivalence(device):
    """
    Validates that the token-averaged accumulated gradient is mathematically
    equivalent to the physical-batch gradient on a controlled deterministic test.
    """
    print("\n[VERIFY] Testing Mathematical Equivalence: Accumulated vs Physical Batch...")
    vocab = 8000
    ctx = 64
    d_m = 128
    heads = 4
    layers = 2
    ff = 256

    m_phys = FromScratchLM(vocab_size=vocab, context_length=ctx, d_model=d_m, num_heads=heads, num_layers=layers, d_ff=ff, dropout=0.0).to(device)
    m_acc = FromScratchLM(vocab_size=vocab, context_length=ctx, d_model=d_m, num_heads=heads, num_layers=layers, d_ff=ff, dropout=0.0).to(device)
    m_acc.load_state_dict(m_phys.state_dict())

    b_size = 4
    s_len = ctx - 1
    torch.manual_seed(123)
    t_inp = torch.randint(2, vocab, (b_size, s_len), device=device)
    t_tgt = torch.randint(2, vocab, (b_size, s_len), device=device)
    t_msk = torch.zeros((b_size, s_len), dtype=torch.bool, device=device)
    t_msk[0, :5] = True
    t_msk[1, :20] = True
    t_msk[2, :40] = True
    t_msk[3, :60] = True

    tot_tokens = t_msk.sum().item()
    crit_test = nn.CrossEntropyLoss(reduction="none")

    # 1. Physical batch (all 4 sequences concurrently)
    m_phys.zero_grad()
    log_p = m_phys(t_inp)
    loss_p = crit_test(log_p.reshape(-1, vocab), t_tgt.reshape(-1)).reshape(t_tgt.shape)
    phys_loss = loss_p[t_msk].sum() / tot_tokens
    phys_loss.backward()

    # 2. Accumulated microbatches (1 sequence per pass)
    m_acc.zero_grad()
    tot_acc_loss = 0.0
    for i in range(b_size):
        log_i = m_acc(t_inp[i:i+1])
        loss_i = crit_test(log_i.reshape(-1, vocab), t_tgt[i:i+1].reshape(-1)).reshape(1, -1)
        sum_i = loss_i[t_msk[i:i+1]].sum()
        scaled_i = sum_i / tot_tokens
        scaled_i.backward()
        tot_acc_loss += sum_i.item()

    acc_mean_loss = tot_acc_loss / tot_tokens
    loss_diff = abs(phys_loss.item() - acc_mean_loss)

    max_grad_diff = 0.0
    for (n1, p1), (n2, p2) in zip(m_phys.named_parameters(), m_acc.named_parameters()):
        d = torch.max(torch.abs(p1.grad - p2.grad)).item()
        max_grad_diff = max(max_grad_diff, d)

    print(f"  Physical Batch Loss:      {phys_loss.item():.6f}")
    print(f"  Accumulated Loss:         {acc_mean_loss:.6f}")
    print(f"  Loss Difference:          {loss_diff:.8e}")
    print(f"  Max Gradient Diff:        {max_grad_diff:.8e} (Tolerance: < 1e-5: PASS)")
    assert max_grad_diff < 1e-5, f"Gradient equivalence check failed: diff={max_grad_diff}"
    print("  Deterministic Equivalence: CONFIRMED (Exact mathematical match with physical batch)")

def run_preflight_checks():
    """
    Comprehensive pre-flight verification required before pilot training.
    Halts execution if any check fails.
    """
    print("=" * 80)
    print("EOS V2 PILOT: COMPREHENSIVE PRE-FLIGHT VERIFICATION")
    print("=" * 80)

    results = {}

    # 1. Hardware & Environment Verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    gpu_mem_total = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if cuda_avail else 0.0

    print(f"[HW] CUDA Available:       {cuda_avail}")
    print(f"[HW] GPU Name:             {gpu_name}")
    print(f"[HW] Total VRAM:           {gpu_mem_total:.2f} GB")
    print(f"[HW] PyTorch Version:      {torch.__version__}")

    if not cuda_avail or device.type != "cuda":
        raise RuntimeError("PRE-FLIGHT FAILED: CUDA GPU is required for training.")
    results["hardware"] = {"cuda": cuda_avail, "gpu": gpu_name, "vram_gb": gpu_mem_total}

    # 2. Baseline Checkpoint Isolation Verification
    print("\n[SAFETY] Verifying Baseline Checkpoint Protection...")
    assert BASELINE_BEST_CHECKPOINT.exists(), f"Baseline checkpoint missing: {BASELINE_BEST_CHECKPOINT}"
    assert CHECKPOINT_DIR != BASELINE_BEST_CHECKPOINT.parent or CHECKPOINT_DIR.name == "eos_v2_pilot", \
        "Checkpoint directory collision!"
    assert str(CHECKPOINT_DIR).endswith("eos_v2_pilot"), f"Invalid checkpoint directory: {CHECKPOINT_DIR}"
    assert str(LOG_DIR).endswith("eos_v2_pilot"), f"Invalid log directory: {LOG_DIR}"

    baseline_best_mtime = BASELINE_BEST_CHECKPOINT.stat().st_mtime
    print(f"  Baseline best checkpoint: {BASELINE_BEST_CHECKPOINT} (size: {BASELINE_BEST_CHECKPOINT.stat().st_size / (1024**2):.1f} MB)")
    print(f"  Pilot checkpoint dir:     {CHECKPOINT_DIR}")
    print(f"  Pilot log dir:            {LOG_DIR}")
    print("  Baseline Isolation:       PASS (No overlap with baseline paths)")
    results["baseline_protection"] = "PASS"

    # 3. Dataset Verification
    print("\n[DATA] Verifying EOS v2 Dataset Files and Structure...")
    assert METADATA_FILE.exists(), f"Metadata missing: {METADATA_FILE}"
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["format_version"] == "EOS_v2", f"Unexpected dataset format: {meta.get('format_version')}"
    assert meta["train_statistics"]["output_sequences"] == 17683
    assert meta["validation_statistics"]["output_sequences"] == 1989
    assert meta["train_statistics"]["tokens_file_bytes"] == 36214784
    assert meta["train_statistics"]["mask_file_bytes"] == 18107392
    assert meta["validation_statistics"]["tokens_file_bytes"] == 4073472
    assert meta["validation_statistics"]["mask_file_bytes"] == 2036736

    train_ds = EOSv2TokenDataset(TRAIN_TOKENS_FILE, TRAIN_MASK_FILE)
    val_ds = EOSv2TokenDataset(VAL_TOKENS_FILE, VAL_MASK_FILE)

    assert len(train_ds) == 17683, f"Expected 17,683 train sequences, got {len(train_ds)}"
    assert len(val_ds) == 1989, f"Expected 1,989 val sequences, got {len(val_ds)}"

    # Sample batch shapes and types
    s_inp, s_tgt, s_msk = train_ds[0]
    assert s_inp.shape == torch.Size([1023]) and s_inp.dtype == torch.long
    assert s_tgt.shape == torch.Size([1023]) and s_tgt.dtype == torch.long
    assert s_msk.shape == torch.Size([1023]) and s_msk.dtype == torch.bool
    assert s_inp[0].item() == BOS_ID, f"First token should be BOS ({BOS_ID}), got {s_inp[0].item()}"
    assert s_msk.sum().item() > 0, "Loss mask has 0 active tokens!"

    # Verify that EOS is supervised under the loss mask
    masked_targets = s_tgt[s_msk]
    has_supervised_eos = (masked_targets == EOS_ID).any().item()
    print(f"  Train sequences:          {len(train_ds):,}")
    print(f"  Validation sequences:     {len(val_ds):,}")
    print(f"  Sequence context length:  {CONTEXT_LENGTH}")
    print(f"  Input / Target / Mask:    {tuple(s_inp.shape)} | {tuple(s_tgt.shape)} | {tuple(s_msk.shape)}")
    print(f"  Sample 0 Masked Tokens:   {s_msk.sum().item():,}")
    print(f"  Supervised [EOS] Present: {has_supervised_eos} (PASS)")
    assert has_supervised_eos, "Pre-flight verification failed: Sample sequence assistant turn did not include supervised [EOS]"
    results["dataset"] = {
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
        "supervised_eos_confirmed": True
    }

    # 4. Model Fresh Initialization Verification
    print("\n[MODEL] Verifying Fresh Model Initialization...")
    torch.manual_seed(42)
    fresh_model = FromScratchLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in fresh_model.parameters())
    print(f"  Parameter Count:          {total_params:,}")
    assert total_params == 14120768, f"Expected 14,120,768 params, got {total_params}"

    # Verify no NaN / Inf in fresh weights
    for name, p in fresh_model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"Fresh model parameter '{name}' contains NaN or Inf!")

    # Compare against baseline checkpoint weights to guarantee fresh initialization
    baseline_ckpt = torch.load(BASELINE_BEST_CHECKPOINT, map_location="cpu", weights_only=False)
    baseline_embed = baseline_ckpt["model_state_dict"]["token_embedding.weight"].to(device)
    diff = torch.norm(fresh_model.token_embedding.weight - baseline_embed).item()
    print(f"  L2 diff with baseline emb:{diff:.4f} (Must be > 0.0: PASS)")
    assert diff > 1.0, f"Model appears to be loaded from baseline! Embedding diff: {diff}"
    del baseline_ckpt, baseline_embed
    results["model_init"] = {"parameters": total_params, "verified_fresh": True}

    # 5a. Deterministic Mathematical Equivalence Test against Physical Batch
    verify_deterministic_physical_batch_equivalence(device)

    # 5b. Optimizer, Scheduler & Token-Averaged Accumulation Verification
    print("\n[OPTIM] Verifying Optimizer, Scheduler, and Token-Averaged Accumulation...")
    opt = torch.optim.AdamW(fresh_model.parameters(), lr=PEAK_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    min_ratio = MIN_LEARNING_RATE / PEAK_LEARNING_RATE
    sched = get_cosine_warmup_scheduler(opt, WARMUP_UPDATES, MAX_OPTIMIZER_UPDATES, min_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Initial LR check
    init_lr = opt.param_groups[0]["lr"]
    print(f"  Initial LR at step 0:     {init_lr:.8f}")
    assert init_lr == 0.0, f"Expected step 0 LR = 0.0, got {init_lr}"

    # Simulate 1 full optimizer update (16 microbatches) with exact token-averaging
    opt.zero_grad(set_to_none=True)

    print("  Executing 16 microbatches of token-averaged gradient accumulation...")
    train_loader = DataLoader(train_ds, batch_size=MICROBATCH_SIZE, shuffle=False)
    data_stream = ContinuousDataLoader(train_loader)

    # Collect 16 microbatches
    microbatches = []
    total_supervised_tokens = 0
    for _ in range(GRAD_ACCUM_STEPS):
        b_inp, b_tgt, b_msk = data_stream.next()
        microbatches.append((b_inp, b_tgt, b_msk))
        total_supervised_tokens += b_msk.sum().item()

    assert total_supervised_tokens > 0, "No active supervised tokens found in 16 microbatches!"
    accum_loss_sum = 0.0

    for mb_i, (inp, tgt, msk) in enumerate(microbatches):
        inp, tgt, msk = inp.to(device), tgt.to(device), msk.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = fresh_model(inp)
            sum_loss, active_count = calculate_masked_token_loss_sum(logits, tgt, msk)
            scaled_loss = sum_loss / total_supervised_tokens

        assert not torch.isnan(sum_loss) and not torch.isinf(sum_loss), f"Microbatch {mb_i} loss is NaN/Inf"
        scaler.scale(scaled_loss).backward()
        accum_loss_sum += sum_loss.item()

    mean_step_loss = accum_loss_sum / total_supervised_tokens
    print(f"  Active Tokens in Update:  {total_supervised_tokens:,}")
    print(f"  Token-Averaged Loss:      {mean_step_loss:.4f} (Finite: PASS)")

    # Unscale and clip gradients
    scaler.unscale_(opt)
    grad_norm = torch.nn.utils.clip_grad_norm_(fresh_model.parameters(), GRADIENT_CLIP)
    print(f"  Grad Norm (unscaled):     {grad_norm.item():.4f} (Finite: PASS)")
    assert not torch.isnan(grad_norm) and not torch.isinf(grad_norm), "Grad norm is NaN/Inf"

    # Step optimizer, scaler, and scheduler
    scaler.step(opt)
    scaler.update()
    sched.step()

    updated_lr = opt.param_groups[0]["lr"]
    expected_lr_1 = PEAK_LEARNING_RATE * (1.0 / WARMUP_UPDATES)
    print(f"  LR after update 1:        {updated_lr:.8f} (Expected: {expected_lr_1:.8f})")
    assert abs(updated_lr - expected_lr_1) < 1e-9, f"LR mismatch: {updated_lr} vs {expected_lr_1}"

    # Verify parameters updated without NaN/Inf
    for name, p in fresh_model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"Parameter '{name}' contains NaN/Inf after step 1!")
    print("  Optimizer Step & Update:  PASS (Weights updated cleanly, scheduler advanced)")
    results["optimization"] = {
        "mean_loss": mean_step_loss,
        "grad_norm": grad_norm.item(),
        "lr_step_1": updated_lr,
        "total_supervised_tokens": total_supervised_tokens
    }

    # 6. Evaluation Verification
    print("\n[EVAL] Verifying Pre-Training Validation Evaluation on Split...")
    val_loader_eval = DataLoader(val_ds, batch_size=16, shuffle=False)
    t0 = time.time()
    val_loss, val_ppl, n_val_seqs, n_val_tokens = evaluate_validation(
        fresh_model, val_loader_eval, device, use_amp=True
    )
    val_time = time.time() - t0
    print(f"  Val sequences evaluated:  {n_val_seqs:,} ({n_val_tokens:,} active tokens)")
    print(f"  Val evaluation time:      {val_time:.2f} s")
    print(f"  Initial Val Loss:         {val_loss:.4f} (Perplexity: {val_ppl:.2f})")
    assert 8.0 < val_loss < 10.0, f"Expected initial random loss near ln(8000)=8.99, got {val_loss}"
    results["initial_validation"] = {
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "eval_seconds": val_time
    }

    # 7. Resource & Runtime Projection
    print("\n[BENCHMARK] Timing & Resource Projections (Token-Averaged Accumulation)...")
    # Measure 5 updates
    t0 = time.time()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        bench_mbs = []
        tot_tok = 0
        for _ in range(GRAD_ACCUM_STEPS):
            b_inp, b_tgt, b_msk = data_stream.next()
            bench_mbs.append((b_inp, b_tgt, b_msk))
            tot_tok += b_msk.sum().item()
        tot_tok = max(1, tot_tok)
        for inp, tgt, msk in bench_mbs:
            inp, tgt, msk = inp.to(device), tgt.to(device), msk.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = fresh_model(inp)
                sum_l, _ = calculate_masked_token_loss_sum(logits, tgt, msk)
                scaled_l = sum_l / tot_tok
            scaler.scale(scaled_l).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(fresh_model.parameters(), GRADIENT_CLIP)
        scaler.step(opt)
        scaler.update()
        sched.step()
    torch.cuda.synchronize()
    benchmark_time = time.time() - t0
    sec_per_update = benchmark_time / 5.0
    est_train_time_sec = sec_per_update * MAX_OPTIMIZER_UPDATES
    est_train_time_min = est_train_time_sec / 60.0

    # 11 validation checkpoints (update 0 + 10 intervals)
    est_val_time_sec = 11 * val_time
    est_total_time_min = (est_train_time_sec + est_val_time_sec) / 60.0

    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Checkpoint disk space estimate
    # Model: ~56.5 MB, Opt: ~113 MB -> ~169.5 MB per checkpoint
    # Saved: latest.pt, best.pt, step_2500.pt, step_5000.pt -> ~680 MB
    est_disk_mb = 170.0 * 4

    print(f"  Measured sec / update:    {sec_per_update:.3f} s")
    print(f"  Estimated training time:  {est_train_time_min:.1f} minutes ({est_train_time_sec:.0f} s)")
    print(f"  Estimated validation time:{est_val_time_sec / 60:.1f} minutes (11 full split evals)")
    print(f"  Estimated total runtime:  {est_total_time_min:.1f} minutes")
    print(f"  Peak GPU VRAM allocated:  {peak_gpu_mb:.1f} MB (Capacity: {gpu_mem_total*1024:.0f} MB)")
    print(f"  Expected checkpoint disk: ~{est_disk_mb:.0f} MB (Free: {shutil_free_gb():.2f} GB)")

    results["projections"] = {
        "sec_per_update": sec_per_update,
        "est_total_minutes": est_total_time_min,
        "peak_gpu_mb": peak_gpu_mb,
        "est_disk_mb": est_disk_mb
    }

    # Verify baseline checkpoint timestamp did NOT change
    assert BASELINE_BEST_CHECKPOINT.stat().st_mtime == baseline_best_mtime, \
        "CRITICAL ERROR: Baseline checkpoint was modified during pre-flight!"
    print("  Baseline File Integrity:  CONFIRMED (Untouched)")

    print("\n" + "=" * 80)
    print("ALL PRE-FLIGHT CHECKS PASSED SUCCESSFULLY.")
    print("READY FOR PILOT TRAINING UPON USER APPROVAL.")
    print("=" * 80)

    # Cleanup memory
    del fresh_model, opt, sched, scaler
    torch.cuda.empty_cache()

    return results

def shutil_free_gb():
    import shutil
    _, _, free = shutil.disk_usage(str(PROJECT_DIR))
    return free / (1024 ** 3)

# ============================================================
# 9. MAIN TRAINING LOOP (5,000 Optimizer Updates)
# ============================================================

def run_pilot_training():
    """
    Executes the 5,000-optimizer-update controlled training pilot.
    Only executed after explicit user confirmation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing logs for a clean pilot run
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    if VAL_LOG_FILE.exists():
        VAL_LOG_FILE.unlink()

    print("=" * 80)
    print("STARTING 5,000-UPDATE EOS V2 CONTROLLED TRAINING PILOT")
    print(f"Target Updates:        {MAX_OPTIMIZER_UPDATES:,}")
    print(f"Microbatch Size:       {MICROBATCH_SIZE}")
    print(f"Grad Accumulation:     {GRAD_ACCUM_STEPS} microbatches / update")
    print(f"Effective Batch Size:  {EFFECTIVE_BATCH_SIZE} sequences")
    print(f"Peak LR:               {PEAK_LEARNING_RATE} (Warmup: {WARMUP_UPDATES} updates)")
    print(f"Min LR:                {MIN_LEARNING_RATE}")
    print(f"Checkpoints:           {CHECKPOINT_DIR}")
    print(f"Logs:                  {LOG_DIR}")
    print("=" * 80)

    # 1. Datasets
    train_dataset = EOSv2TokenDataset(TRAIN_TOKENS_FILE, TRAIN_MASK_FILE)
    val_dataset = EOSv2TokenDataset(VAL_TOKENS_FILE, VAL_MASK_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICROBATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # 2. Fresh Model Initialization
    torch.manual_seed(1337)
    model = FromScratchLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    min_ratio = MIN_LEARNING_RATE / PEAK_LEARNING_RATE
    scheduler = get_cosine_warmup_scheduler(
        optimizer,
        warmup_steps=WARMUP_UPDATES,
        total_steps=MAX_OPTIMIZER_UPDATES,
        min_lr_ratio=min_ratio
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 4. Pre-training Validation (Step 0)
    print("\nEvaluating initial validation baseline (Step 0)...")
    init_val_loss, init_val_ppl, _, _ = evaluate_validation(model, val_loader, device, use_amp)
    print(f"Initial Validation Loss (Step 0): {init_val_loss:.4f} | PPL: {init_val_ppl:.2f}\n")

    val_record_0 = {
        "optimizer_update": 0,
        "microbatch_count": 0,
        "validation_loss": init_val_loss,
        "perplexity": init_val_ppl,
        "elapsed_seconds": 0.0
    }
    with open(VAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(val_record_0) + "\n")

    best_val_loss = init_val_loss
    best_update = 0

    # Save initial latest checkpoint
    save_checkpoint(
        LATEST_CHECKPOINT, 0, 0, model, optimizer, scheduler, scaler,
        init_val_loss, best_val_loss
    )

    # 5. Training Loop
    update_step = 0
    microbatch_count = 0
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

        val_loss_at_step = None
        val_ppl_at_step = None
        ckpt_saved_path = None

        # Validation check every 500 updates
        if update_step % VALIDATION_INTERVAL == 0:
            val_loss_at_step, val_ppl_at_step, _, _ = evaluate_validation(
                model, val_loader, device, use_amp
            )
            print("\n" + "=" * 80)
            print(f"VALIDATION UPDATE {update_step:,} / {MAX_OPTIMIZER_UPDATES:,}")
            print(f"Validation Loss:  {val_loss_at_step:.4f} (Perplexity: {val_ppl_at_step:.2f})")
            print(f"Training Loss:    {avg_train_loss:.4f}")
            print(f"Learning Rate:    {current_lr:.7f}")
            print(f"Elapsed Time:     {elapsed:.1f}s ({elapsed / 60:.2f} min)")
            print(f"Peak GPU VRAM:    {gpu_mb:.1f} MB")
            print("=" * 80 + "\n")

            # Record in validation log
            val_record = {
                "optimizer_update": update_step,
                "microbatch_count": microbatch_count,
                "validation_loss": val_loss_at_step,
                "perplexity": val_ppl_at_step,
                "training_loss": avg_train_loss,
                "learning_rate": current_lr,
                "elapsed_seconds": elapsed
            }
            with open(VAL_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(val_record) + "\n")

            # Check for best checkpoint
            if val_loss_at_step < best_val_loss:
                best_val_loss = val_loss_at_step
                best_update = update_step
                save_checkpoint(
                    BEST_CHECKPOINT, update_step, microbatch_count,
                    model, optimizer, scheduler, scaler,
                    val_loss_at_step, best_val_loss
                )
                ckpt_saved_path = str(BEST_CHECKPOINT)
                print(f">>> New BEST checkpoint recorded at update {update_step:,} (Val Loss: {best_val_loss:.4f}) <<<\n")

            # Save latest checkpoint
            save_checkpoint(
                LATEST_CHECKPOINT, update_step, microbatch_count,
                model, optimizer, scheduler, scaler,
                val_loss_at_step, best_val_loss
            )

        # Milestone Checkpoints (2500, 5000)
        if update_step in milestone_updates:
            milestone_path = CHECKPOINT_DIR / f"step_{update_step}.pt"
            save_checkpoint(
                milestone_path, update_step, microbatch_count,
                model, optimizer, scheduler, scaler,
                val_loss_at_step if val_loss_at_step is not None else -1.0,
                best_val_loss
            )
            ckpt_saved_path = str(milestone_path)

        # Periodic save latest checkpoint every 1,000 updates if not already saved
        elif update_step % 1000 == 0:
            save_checkpoint(
                LATEST_CHECKPOINT, update_step, microbatch_count,
                model, optimizer, scheduler, scaler,
                val_loss_at_step if val_loss_at_step is not None else -1.0,
                best_val_loss
            )

        # Log training step
        record = {
            "optimizer_update": update_step,
            "microbatch_count": microbatch_count,
            "training_loss": avg_train_loss,
            "validation_loss": val_loss_at_step,
            "learning_rate": current_lr,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "elapsed_seconds": elapsed,
            "gpu_memory_mb": gpu_mb,
            "checkpoint_saved": ckpt_saved_path
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Console logging
        if update_step == 1 or update_step % 50 == 0 or update_step == MAX_OPTIMIZER_UPDATES:
            val_str = f" | Val: {val_loss_at_step:.4f}" if val_loss_at_step is not None else ""
            print(
                f"Update {update_step:04d}/{MAX_OPTIMIZER_UPDATES} | "
                f"MB: {microbatch_count:05d} | "
                f"Loss: {avg_train_loss:.4f} | "
                f"LR: {current_lr:.7f} | "
                f"Grad: {grad_norm.item():.2f} | "
                f"GPU: {gpu_mb:.1f}MB | "
                f"Time: {elapsed:.1f}s"
                f"{val_str}"
            )

    total_time = time.time() - training_start
    print("\n" + "=" * 80)
    print("EOS V2 PILOT TRAINING COMPLETE (5,000 UPDATES)")
    print("=" * 80)
    print(f"Total Optimizer Updates: {update_step:,}")
    print(f"Total Microbatches:      {microbatch_count:,}")
    print(f"Total Wall Time:         {total_time:.1f}s ({total_time / 60:.2f} min)")
    print(f"Best Validation Loss:    {best_val_loss:.4f} at Update {best_update:,}")
    print(f"Best Checkpoint:         {BEST_CHECKPOINT}")
    print(f"Latest Checkpoint:       {LATEST_CHECKPOINT}")
    print(f"Training Log:            {LOG_FILE}")
    print(f"Validation Log:          {VAL_LOG_FILE}")

# ============================================================
# 10. ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EOS v2 Pilot Training Script")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run comprehensive pre-flight verification only"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Execute the 5,000 optimizer update training run"
    )
    args = parser.parse_args()

    if args.preflight:
        run_preflight_checks()
    elif args.train:
        # Before starting training, always execute pre-flight
        run_preflight_checks()
        run_pilot_training()
    else:
        # Default: run preflight checks
        print("No mode specified. Running pre-flight verification checks by default.")
        run_preflight_checks()

if __name__ == "__main__":
    main()
