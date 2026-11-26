from pathlib import Path
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

# ============================================================
# DIAGNOSTIC PATHS (Completely Isolated)
# ============================================================

DATA_DIR = PROJECT_DIR / "data" / "diagnostic_overfit"
TRAIN_TOKENS_FILE = DATA_DIR / "diagnostic_tokens.bin"
TRAIN_MASK_FILE = DATA_DIR / "diagnostic_mask.bin"

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "diagnostic_overfit_latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "diagnostic_overfit_best.pt"
LOG_FILE = LOG_DIR / "diagnostic_overfit.jsonl"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"

# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024
D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
D_FF = 1536
DROPOUT = 0.0  # 0.0 for pure diagnostic memorization test

BATCH_SIZE = 1
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0  # 0.0 so L2 penalty does not fight memorization
GRADIENT_CLIP = 1.0

MAX_STEPS = 600
EVAL_INTERVAL = 25

# ============================================================
# DATASET
# ============================================================

class DiagnosticTokenDataset(Dataset):
    def __init__(self, token_file_path, mask_file_path):
        self.tokens = np.fromfile(token_file_path, dtype=np.uint16)
        self.loss_mask = np.fromfile(mask_file_path, dtype=np.uint8)

        if len(self.tokens) != len(self.loss_mask):
            raise ValueError("Token and mask sizes do not match.")

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

        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[1:], dtype=torch.bool)

        return input_ids, target_ids, loss_mask

# ============================================================
# DEVICE SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("TINY-DATASET OVERFIT DIAGNOSTIC EXPERIMENT")
print("=" * 70)
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD DATA & CREATE MODEL
# ============================================================

dataset = DiagnosticTokenDataset(TRAIN_TOKENS_FILE, TRAIN_MASK_FILE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Diagnostic sequences: {len(dataset)}")
print(f"Total tokens: {len(dataset) * CONTEXT_LENGTH:,}")
total_loss_tokens = sum((mask[1:] == 1).sum() for _, _, mask in eval_loader)
print(f"Total masked target tokens: {total_loss_tokens:,}")

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

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
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
# EVALUATION FUNCTION (Full diagnostic dataset metrics)
# ============================================================

def evaluate_full():
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked_tokens = 0
    seq_losses = []

    with torch.no_grad():
        for seq_idx, (inp, tgt, msk) in enumerate(eval_loader):
            inp, tgt, msk = inp.to(device), tgt.to(device), msk.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inp)
                loss = calculate_masked_loss(logits, tgt, msk)

            seq_loss = loss.item()
            seq_losses.append(seq_loss)
            total_loss += seq_loss

            # Calculate accuracy on masked tokens
            pred_tokens = logits.argmax(dim=-1)
            correct = (pred_tokens[msk] == tgt[msk]).sum().item()
            masked_count = msk.sum().item()

            total_correct += correct
            total_masked_tokens += masked_count

    model.train()
    mean_loss = total_loss / len(eval_loader)
    acc = total_correct / total_masked_tokens if total_masked_tokens > 0 else 0.0
    ppl = float(np.exp(min(mean_loss, 20.0)))
    return mean_loss, acc, ppl, seq_losses

# ============================================================
# LOGGING & CHECKPOINTS
# ============================================================

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if LOG_FILE.exists():
    LOG_FILE.unlink()

def save_diagnostic_checkpoint(path, step_num, loss_val, acc_val):
    ckpt = {
        "step": step_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss_val,
        "accuracy": acc_val,
        "config": {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "experiment": "diagnostic_overfit"
        }
    }
    torch.save(ckpt, path)

# Initial baseline evaluation
init_loss, init_acc, init_ppl, _ = evaluate_full()
print(f"Step 000 | Baseline Loss: {init_loss:.4f} | Masked Token Acc: {init_acc * 100:.2f}% | PPL: {init_ppl:.2f}")

# ============================================================
# TRAINING LOOP
# ============================================================

step = 0
best_loss = float("inf")
start_time = time.time()

print("\nStarting diagnostic overfitting run...")

while step < MAX_STEPS:
    for inp, tgt, msk in loader:
        if step >= MAX_STEPS:
            break

        inp, tgt, msk = inp.to(device), tgt.to(device), msk.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(inp)
            loss = calculate_masked_loss(logits, tgt, msk)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        num_masked = msk.sum().item()

        # Log entry
        log_entry = {
            "step": step,
            "loss": loss.item(),
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "masked_tokens": num_masked,
            "lr": LEARNING_RATE,
            "elapsed_seconds": time.time() - start_time
        }

        # Periodic evaluation
        if step % EVAL_INTERVAL == 0 or step == 1 or step == MAX_STEPS:
            eval_loss, eval_acc, eval_ppl, _ = evaluate_full()
            log_entry["eval_loss"] = eval_loss
            log_entry["eval_acc"] = eval_acc
            log_entry["eval_ppl"] = eval_ppl

            print(
                f"Step {step:03d}/{MAX_STEPS} | "
                f"Train Step Loss: {loss.item():.4f} | "
                f"Full Dataset Loss: {eval_loss:.4f} | "
                f"Acc: {eval_acc * 100:.2f}% | "
                f"PPL: {eval_ppl:.2f} | "
                f"Grad: {log_entry['grad_norm']:.2f}"
            )

            if eval_loss < best_loss:
                best_loss = eval_loss
                save_diagnostic_checkpoint(BEST_CHECKPOINT, step, eval_loss, eval_acc)

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Early termination if memorization is basically complete (<0.01 loss)
        if step >= 100 and eval_loss < 0.01:
            print(f"\nTarget memorization threshold reached (eval_loss < 0.01) at step {step}!")
            break

# Final evaluation and checkpoint
final_loss, final_acc, final_ppl, final_seq_losses = evaluate_full()
save_diagnostic_checkpoint(LATEST_CHECKPOINT, step, final_loss, final_acc)

print("\n" + "=" * 70)
print("DIAGNOSTIC TRAINING COMPLETE")
print("=" * 70)
print(f"Total steps: {step}")
print(f"Initial loss: {init_loss:.4f}")
print(f"Best full loss: {best_loss:.4f}")
print(f"Final full loss: {final_loss:.4f}")
print(f"Final masked token accuracy: {final_acc * 100:.2f}%")
print(f"Final perplexity: {final_ppl:.2f}")
print("\nPer-sequence losses:")
for s_i, sl in enumerate(final_seq_losses):
    print(f"  Sequence {s_i}: loss = {sl:.4f}")

print(f"\nBest checkpoint: {BEST_CHECKPOINT}")
print(f"Latest checkpoint: {LATEST_CHECKPOINT}")
print(f"Log file: {LOG_FILE}")
