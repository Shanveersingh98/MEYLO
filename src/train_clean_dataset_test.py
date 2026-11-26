from pathlib import Path
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

# ============================================================
# 1. FILE PATHS (Isolated Test Environment)
# ============================================================

DATA_DIR = PROJECT_DIR / "data" / "processed" / "training_sequences_clean"
TRAIN_FILE = DATA_DIR / "train_tokens.bin"
TRAIN_MASK_FILE = DATA_DIR / "train_mask.bin"
VALIDATION_FILE = DATA_DIR / "validation_tokens.bin"
VALIDATION_MASK_FILE = DATA_DIR / "validation_mask.bin"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "clean_dataset_test_latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "clean_dataset_test_best.pt"
LOG_FILE = LOG_DIR / "clean_dataset_test.jsonl"

# ============================================================
# 2. CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
D_FF = 1536
DROPOUT = 0.1

BATCH_SIZE = 1
LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0

MAX_STEPS = 1000
VALIDATION_INTERVAL = 250
VALIDATION_BATCHES = 20

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5

# ============================================================
# 3. DATASET CLASS
# ============================================================

class CleanTokenDataset(Dataset):
    def __init__(self, token_file_path, mask_file_path):
        self.tokens = np.memmap(token_file_path, dtype=np.uint16, mode="r")
        self.loss_mask = np.memmap(mask_file_path, dtype=np.uint8, mode="r")

        if len(self.tokens) != len(self.loss_mask):
            raise ValueError("Token and mask dataset sizes do not match.")

        if len(self.tokens) % CONTEXT_LENGTH != 0:
            raise ValueError("Dataset size is not divisible by context length.")

        self.sequence_count = len(self.tokens) // CONTEXT_LENGTH

    def __len__(self):
        return self.sequence_count

    def __getitem__(self, index):
        start = index * CONTEXT_LENGTH
        end = start + CONTEXT_LENGTH

        sequence = np.asarray(self.tokens[start:end], dtype=np.int64)
        mask = np.asarray(self.loss_mask[start:end], dtype=np.uint8)

        # Model input: sequence[:-1], target: sequence[1:], mask: mask[1:]
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[1:], dtype=torch.bool)

        return input_ids, target_ids, loss_mask

# ============================================================
# 4. PRE-TRAINING SANITY CHECK
# ============================================================

def run_pre_training_sanity_check(train_ds, val_ds):
    print("=" * 70)
    print("RUNNING PRE-TRAINING DATASET SANITY CHECKS")
    print("=" * 70)

    # Check sample indices
    sample_indices = [0, 1, 10, 50, 500, len(train_ds) - 1]
    for idx in sample_indices:
        inp, tgt, msk = train_ds[idx]

        # Check shapes
        assert inp.shape == torch.Size([1023]), f"Input shape mismatch: {inp.shape}"
        assert tgt.shape == torch.Size([1023]), f"Target shape mismatch: {tgt.shape}"
        assert msk.shape == torch.Size([1023]), f"Mask shape mismatch: {msk.shape}"

        # Check position 0 is BOS ID 2
        assert inp[0].item() == BOS_ID, f"Seq {idx}: Position 0 is not BOS! Got {inp[0].item()}"

        # Check no internal BOS
        assert (inp[1:] == BOS_ID).sum().item() == 0, f"Seq {idx}: Found internal BOS in input!"
        assert (tgt[1:] == BOS_ID).sum().item() == 0, f"Seq {idx}: Found internal BOS in target!"

        # Check token ID bounds
        assert inp.min().item() >= 0 and inp.max().item() < VOCAB_SIZE, "Token IDs out of range [0, 7999]!"
        assert tgt.min().item() >= 0 and tgt.max().item() < VOCAB_SIZE, "Target IDs out of range [0, 7999]!"

        # Check PAD tokens have mask 0
        pad_positions_in_tgt = (tgt == PAD_ID)
        pad_loss_tokens = (pad_positions_in_tgt & msk).sum().item()
        assert pad_loss_tokens == 0, f"Seq {idx}: Found {pad_loss_tokens} PAD tokens with mask == 1!"

    print(f"  [PASS] Sample sequences ({len(sample_indices)}) verified.")
    print(f"  [PASS] All position 0 tokens are BOS (ID {BOS_ID}).")
    print(f"  [PASS] No internal BOS tokens found.")
    print(f"  [PASS] All token IDs within [0, {VOCAB_SIZE - 1}].")
    print(f"  [PASS] PAD tokens (ID 0) have mask == 0 (zero loss on padding).")
    print(f"  [PASS] Tensor shapes confirmed: input (1023), target (1023), mask (1023).")
    print("SANITY CHECKS PASSED. Ready to train.\n")

# ============================================================
# 5. DEVICE & MODEL INITIALIZATION
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("CONTROLLED 1,000-STEP CLEAN DATASET DIAGNOSTIC TEST")
print("=" * 70)
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

train_dataset = CleanTokenDataset(TRAIN_FILE, TRAIN_MASK_FILE)
validation_dataset = CleanTokenDataset(VALIDATION_FILE, VALIDATION_MASK_FILE)

print(f"Training sequences:   {len(train_dataset):,}")
print(f"Validation sequences: {len(validation_dataset):,}")

# Run sanity check before proceeding
run_pre_training_sanity_check(train_dataset, validation_dataset)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Freshly initialized model
print("Creating fresh randomly initialized FromScratchLM...")
model = FromScratchLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
).to(device)

param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# Cosine LR scheduler over exactly 1,000 steps
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_STEPS,
    eta_min=MIN_LEARNING_RATE
)

criterion = nn.CrossEntropyLoss(reduction="none")

def calculate_masked_loss(logits, target_ids, loss_mask):
    token_losses = criterion(
        logits.reshape(-1, VOCAB_SIZE),
        target_ids.reshape(-1)
    ).reshape(target_ids.shape)

    masked_losses = token_losses[loss_mask]
    if masked_losses.numel() == 0:
        return token_losses.sum() * 0.0
    return masked_losses.mean()

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate():
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for input_ids, target_ids, loss_mask in validation_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids)
                loss = calculate_masked_loss(logits, target_ids, loss_mask)

            total_loss += loss.item()
            batches += 1
            if batches >= VALIDATION_BATCHES:
                break

    model.train()
    if batches == 0:
        return float("inf")
    return total_loss / batches

# ============================================================
# 7. CHECKPOINTING & LOGGING
# ============================================================

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if LOG_FILE.exists():
    LOG_FILE.unlink()

def save_checkpoint(path, step_num, val_loss, best_loss):
    ckpt = {
        "step": step_num,
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
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_STEPS,
            "dataset": "training_sequences_clean (Design A)",
            "experiment": "clean_dataset_test"
        }
    }
    torch.save(ckpt, path)

def write_log(step_num, tr_loss, v_loss, lr_val, elapsed, gpu_mb):
    record = {
        "step": step_num,
        "training_loss": tr_loss,
        "validation_loss": v_loss,
        "learning_rate": lr_val,
        "elapsed_seconds": elapsed,
        "gpu_memory_mb": gpu_mb,
        "dataset": "training_sequences_clean"
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

# ============================================================
# 8. TRAINING LOOP (Exactly 1,000 steps)
# ============================================================

step = 0
best_val_loss = float("inf")
start_time = time.time()
model.train()

print("=" * 70)
print(f"STARTING TRAINING: {MAX_STEPS} STEPS ON CLEAN DESIGN A DATASET")
print("=" * 70)

val_results = {}

while step < MAX_STEPS:
    for input_ids, target_ids, loss_mask in train_loader:
        if step >= MAX_STEPS:
            break

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids)
            loss = calculate_masked_loss(logits, target_ids, loss_mask)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        gpu_mb = (torch.cuda.memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0

        val_loss = None
        # Periodic validation at steps 250, 500, 750, 1000
        if step % VALIDATION_INTERVAL == 0:
            val_loss = evaluate()
            val_results[step] = val_loss

            print(
                f"\n>>> Step {step:04d} | Validation Loss: {val_loss:.4f} "
                f"(PPL: {np.exp(min(val_loss, 20.0)):.2f}) <<<\n"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(BEST_CHECKPOINT, step, val_loss, best_val_loss)
                print(f"New best clean test checkpoint saved to {BEST_CHECKPOINT.name}")

            save_checkpoint(LATEST_CHECKPOINT, step, val_loss, best_val_loss)

        write_log(step, loss.item(), val_loss, current_lr, elapsed, gpu_mb)

        # Periodic console logging
        if step == 1 or step % 50 == 0:
            val_str = f" | Val: {val_loss:.4f}" if val_loss is not None else ""
            print(
                f"Step {step:04d}/{MAX_STEPS} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.7f} | "
                f"Grad: {grad_norm.item():.2f} | "
                f"GPU: {gpu_mb:.1f}MB | "
                f"Time: {elapsed:.1f}s"
                f"{val_str}"
            )

# Final checkpoint
save_checkpoint(LATEST_CHECKPOINT, step, val_results.get(1000), best_val_loss)

total_elapsed = time.time() - start_time
print("\n" + "=" * 70)
print("1,000-STEP CONTROLLED DIAGNOSTIC TEST COMPLETE")
print("=" * 70)
print(f"Total steps:         {step}")
print(f"Total time:          {total_elapsed:.2f} seconds ({total_elapsed / step:.3f} s/step)")
print(f"Best validation loss:{best_val_loss:.4f}")
print("Validation checkpoints:")
for s, vl in val_results.items():
    print(f"  Step {s:04d}: Val Loss = {vl:.4f} (Perplexity = {np.exp(vl):.2f})")
print(f"\nBest Checkpoint:   {BEST_CHECKPOINT}")
print(f"Latest Checkpoint: {LATEST_CHECKPOINT}")
print(f"Log file:          {LOG_FILE}")
