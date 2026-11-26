import sys
import json
import time
import math
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

# ============================================================
# 1. PATH CONFIGURATION
# ============================================================
DATA_DIR = PROJECT_DIR / "data" / "ultimate_improvement" / "training_sequences"
TRAIN_TOKENS_FILE = DATA_DIR / "train_tokens.bin"
TRAIN_MASK_FILE = DATA_DIR / "train_mask.bin"
CURRICULUM_VAL_TOKENS_FILE = DATA_DIR / "heldout_tokens.bin"
CURRICULUM_VAL_MASK_FILE = DATA_DIR / "heldout_mask.bin"

# Original OASST1 validation split (for catastrophic forgetting monitoring)
OASST1_VAL_TOKENS_FILE = PROJECT_DIR / "data" / "processed" / "training_sequences_eos_v2" / "validation_tokens.bin"
OASST1_VAL_MASK_FILE = PROJECT_DIR / "data" / "processed" / "training_sequences_eos_v2" / "validation_mask.bin"

# Reference checkpoint to fine-tune from (Experiment 2 best.pt)
REFERENCE_CHECKPOINT = PROJECT_DIR / "checkpoints" / "experiment_2" / "best.pt"

# Ultimate Improvement output directories
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "ultimate_improvement"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

LOG_DIR = PROJECT_DIR / "logs" / "ultimate_improvement"
LOG_FILE = LOG_DIR / "training_log.jsonl"
VAL_LOG_FILE = LOG_DIR / "validation_log.jsonl"

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"

# ============================================================
# 2. HYPERPARAMETERS
# ============================================================
VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
D_FF = 1536
DROPOUT = 0.1

MICROBATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
EFFECTIVE_BATCH_SIZE = MICROBATCH_SIZE * GRAD_ACCUM_STEPS  # 16 sequences per update

PEAK_LEARNING_RATE = 6e-5    # Gentle fine-tuning rate
MIN_LEARNING_RATE = 6e-6     # 10% of peak
WEIGHT_DECAY = 0.05
GRADIENT_CLIP = 1.0

MAX_OPTIMIZER_UPDATES = 1500  # 4.0 epochs over 6,000 sequences
WARMUP_UPDATES = 100          # 100-step linear warmup
VALIDATION_INTERVAL = 150     # Validate every 150 updates (~0.4 epoch)
EARLY_STOPPING_PATIENCE = 4   # Stop if no improvement for 4 evaluations (600 updates)

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
USER_ID = 4
ASSISTANT_ID = 5


# ============================================================
# 3. DATASET CLASS & CONTINUOUS LOADER
# ============================================================
class BinaryDataset(Dataset):
    def __init__(self, token_path: Path, mask_path: Path):
        self.token_path = Path(token_path)
        self.mask_path = Path(mask_path)
        if not self.token_path.exists():
            raise FileNotFoundError(f"Tokens binary not found: {self.token_path}")
        if not self.mask_path.exists():
            raise FileNotFoundError(f"Mask binary not found: {self.mask_path}")

        self.tokens = np.memmap(self.token_path, dtype=np.uint16, mode="r")
        self.mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r")

        if len(self.tokens) != len(self.mask):
            raise ValueError(f"Length mismatch: {len(self.tokens)} != {len(self.mask)}")
        if len(self.tokens) % CONTEXT_LENGTH != 0:
            raise ValueError(f"Tokens length {len(self.tokens)} not divisible by {CONTEXT_LENGTH}")

        self.seq_count = len(self.tokens) // CONTEXT_LENGTH

    def __len__(self):
        return self.seq_count

    def __getitem__(self, index):
        start = index * CONTEXT_LENGTH
        end = start + CONTEXT_LENGTH
        seq = np.asarray(self.tokens[start:end], dtype=np.int64)
        m = np.asarray(self.mask[start:end], dtype=np.uint8)

        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        target_ids = torch.tensor(seq[1:], dtype=torch.long)
        loss_mask = torch.tensor(m[1:], dtype=torch.bool)
        return input_ids, target_ids, loss_mask


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


# ============================================================
# 4. LOSS CALCULATION & SCHEDULER
# ============================================================
criterion = nn.CrossEntropyLoss(reduction="none")


def calculate_token_loss_sum(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor):
    vsize = logits.size(-1)
    tok_loss = criterion(
        logits.reshape(-1, vsize),
        target_ids.reshape(-1)
    ).reshape(target_ids.shape)
    active = tok_loss[loss_mask]
    if active.numel() == 0:
        return tok_loss.sum() * 0.0, 0
    return active.sum(), active.numel()


def get_cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 5. VALIDATION EVALUATION
# ============================================================
def evaluate_split(model, dataset: Dataset, device: torch.device, batch_size: int = 16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_active_tokens = 0
    total_sequences = 0

    with torch.no_grad():
        for input_ids, target_ids, loss_mask in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            loss_mask = loss_mask.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(input_ids)
                tok_loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    target_ids.reshape(-1)
                ).reshape(target_ids.shape)
                active = tok_loss[loss_mask]
                total_loss += active.sum().item()
                total_active_tokens += active.numel()
                total_sequences += input_ids.size(0)

    model.train()
    mean_loss = (total_loss / total_active_tokens) if total_active_tokens > 0 else float("inf")
    ppl = float(np.exp(min(mean_loss, 20.0)))
    return mean_loss, ppl, total_sequences, total_active_tokens


# ============================================================
# 6. CHECKPOINT MANAGEMENT & SAFETY
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
    # Strict safety protection against overwriting protected reference checkpoints
    p_str = str(path).lower()
    if "experiment_2" in p_str and "checkpoints\\ultimate_improvement" not in p_str and "checkpoints/ultimate_improvement" not in p_str:
        raise RuntimeError(f"CRITICAL SAFETY VIOLATION: Attempted write to protected reference checkpoint: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "experiment": "ultimate_improvement",
        "optimizer_update": update_step,
        "microbatch_count": microbatch_count,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "validation_loss": val_loss,
        "best_loss": best_loss,
        "timestamp": time.time(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "d_model": D_MODEL,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "peak_lr": PEAK_LEARNING_RATE,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE
        },
        "extra_metadata": extra_metadata or {}
    }
    torch.save(ckpt, path)


def load_reference_model(device: torch.device):
    print(f"Loading reference model from: {REFERENCE_CHECKPOINT}")
    if not REFERENCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Reference checkpoint not found at: {REFERENCE_CHECKPOINT}")

    ckpt = torch.load(REFERENCE_CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    print(f"  Reference update: {ckpt.get('optimizer_update')}, Val Loss: {ckpt.get('validation_loss', 'N/A')}")

    model = FromScratchLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded successfully. Total parameters: {param_count:,}")
    return model, ckpt


# ============================================================
# 7. MANDATORY PREFLIGHT VERIFICATION
# ============================================================
def run_preflight_tests(model, device: torch.device, train_dataset: Dataset):
    print("\n" + "=" * 70)
    print("RUNNING MANDATORY PREFLIGHT TESTS")
    print("=" * 70)

    inp, tgt, msk = train_dataset[0]
    inp = inp.unsqueeze(0).to(device)
    tgt = tgt.unsqueeze(0).to(device)
    msk = msk.unsqueeze(0).to(device)

    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
        logits = model(inp)
    assert logits.shape == (1, CONTEXT_LENGTH - 1, VOCAB_SIZE)
    l_sum, count = calculate_token_loss_sum(logits, tgt, msk)
    assert torch.isfinite(l_sum) and count > 0
    print(f"[PASS] Forward pass & loss mask: Loss Sum = {l_sum.item():.4f}, Active Tokens = {count}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = l_sum / count
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
    opt.zero_grad(set_to_none=True)
    print("[PASS] Backward pass: All gradients computed and finite.")

    test_path = CHECKPOINT_DIR / "preflight_test.pt"
    save_checkpoint(test_path, 0, 0, model, opt, None, None, 1.0, 1.0)
    assert test_path.exists()
    test_path.unlink()
    print("[PASS] Isolated checkpoint save & unlink verified.")
    assert REFERENCE_CHECKPOINT.exists()
    print(f"[PASS] Reference checkpoint verified safe at: {REFERENCE_CHECKPOINT}")
    print("=" * 70)
    print("ALL PREFLIGHT TESTS PASSED! READY FOR TRAINING.")
    print("=" * 70 + "\n")


# ============================================================
# 8. MAIN TRAINING LOOP
# ============================================================
def train(max_updates: int = MAX_OPTIMIZER_UPDATES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_dataset = BinaryDataset(TRAIN_TOKENS_FILE, TRAIN_MASK_FILE)
    curriculum_val_dataset = BinaryDataset(CURRICULUM_VAL_TOKENS_FILE, CURRICULUM_VAL_MASK_FILE)
    oasst1_val_dataset = BinaryDataset(OASST1_VAL_TOKENS_FILE, OASST1_VAL_MASK_FILE)

    print(f"  Training sequences:       {len(train_dataset):,}")
    print(f"  Curriculum val sequences: {len(curriculum_val_dataset):,}")
    print(f"  OASST1 val sequences:     {len(oasst1_val_dataset):,}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=MICROBATCH_SIZE,
        shuffle=True,
        drop_last=False
    )
    continuous_loader = ContinuousDataLoader(train_loader)

    model, ref_ckpt = load_reference_model(device)
    run_preflight_tests(model, device, train_dataset)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PEAK_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
        eps=1e-8
    )

    scheduler = get_cosine_warmup_scheduler(
        optimizer,
        warmup_steps=WARMUP_UPDATES,
        total_steps=max_updates,
        min_lr_ratio=MIN_LEARNING_RATE / PEAK_LEARNING_RATE
    )

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    print("Conducting pre-training baseline validation evaluations...")
    curr_loss, curr_ppl, _, _ = evaluate_split(model, curriculum_val_dataset, device)
    oasst_loss, oasst_ppl, _, _ = evaluate_split(model, oasst1_val_dataset, device)
    initial_composite_loss = 0.5 * curr_loss + 0.5 * oasst_loss

    print(f"  Baseline Curriculum Val: Loss = {curr_loss:.4f}, PPL = {curr_ppl:.2f}")
    print(f"  Baseline OASST1 Val:     Loss = {oasst_loss:.4f}, PPL = {oasst_ppl:.2f}")
    print(f"  Baseline Composite Val:  Loss = {initial_composite_loss:.4f}")

    best_composite_loss = initial_composite_loss
    patience_counter = 0

    with open(VAL_LOG_FILE, "a", encoding="utf-8") as vf:
        vf.write(json.dumps({
            "step": 0,
            "curriculum_val_loss": round(curr_loss, 4),
            "curriculum_val_ppl": round(curr_ppl, 2),
            "oasst1_val_loss": round(oasst_loss, 4),
            "oasst1_val_ppl": round(oasst_ppl, 2),
            "composite_loss": round(initial_composite_loss, 4),
            "is_best": True,
            "timestamp": time.time()
        }) + "\n")

    print("\n" + "=" * 70)
    print(f"STARTING ULTIMATE IMPROVEMENT TRAINING (Max Updates = {max_updates:,})")
    print("=" * 70)

    model.train()
    optimizer.zero_grad(set_to_none=True)

    update_step = 0
    microbatch_count = 0
    start_time = time.time()

    accum_unreduced_loss = 0.0
    accum_active_tokens = 0
    accum_microbatches = 0

    while update_step < max_updates:
        input_ids, target_ids, loss_mask = continuous_loader.next()
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(input_ids)
            mb_loss_sum, mb_active_tokens = calculate_token_loss_sum(logits, target_ids, loss_mask)

        if mb_active_tokens > 0:
            scaler.scale(mb_loss_sum).backward()
            accum_unreduced_loss += mb_loss_sum.item()
            accum_active_tokens += mb_active_tokens

        accum_microbatches += 1
        microbatch_count += 1

        if accum_microbatches == GRAD_ACCUM_STEPS:
            if accum_active_tokens > 0:
                scaler.unscale_(optimizer)
                scale_correction = 1.0 / accum_active_tokens
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(scale_correction)

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_step += 1

            train_step_loss = (accum_unreduced_loss / accum_active_tokens) if accum_active_tokens > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0]

            accum_unreduced_loss = 0.0
            accum_active_tokens = 0
            accum_microbatches = 0

            if update_step % 50 == 0 or update_step == 1:
                elapsed = time.time() - start_time
                ms_per_update = (elapsed / update_step) * 1000
                print(
                    f"Update {update_step:5d}/{max_updates} | "
                    f"Epoch {continuous_loader.epoch:2d} | "
                    f"Train Loss: {train_step_loss:6.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {ms_per_update:5.1f} ms/upd"
                )
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps({
                        "step": update_step,
                        "epoch": continuous_loader.epoch,
                        "train_loss": round(train_step_loss, 4),
                        "lr": current_lr,
                        "timestamp": time.time()
                    }) + "\n")

            if update_step % VALIDATION_INTERVAL == 0 or update_step == max_updates:
                val_start = time.time()
                print("\n--- Running Periodic Validation ---")
                c_loss, c_ppl, _, _ = evaluate_split(model, curriculum_val_dataset, device)
                o_loss, o_ppl, _, _ = evaluate_split(model, oasst1_val_dataset, device)
                composite = 0.5 * c_loss + 0.5 * o_loss
                val_dur = time.time() - val_start

                is_best = composite < best_composite_loss
                if is_best:
                    delta = best_composite_loss - composite
                    best_composite_loss = composite
                    patience_counter = 0
                    print(f"  >>> NEW BEST COMPOSITE LOSS: {composite:.4f} (improved by {delta:.4f})! Saving best.pt...")
                    save_checkpoint(
                        BEST_CHECKPOINT,
                        update_step,
                        microbatch_count,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        val_loss=composite,
                        best_loss=best_composite_loss,
                        extra_metadata={
                            "curriculum_val_loss": c_loss,
                            "curriculum_val_ppl": c_ppl,
                            "oasst1_val_loss": o_loss,
                            "oasst1_val_ppl": o_ppl
                        }
                    )
                else:
                    patience_counter += 1
                    print(f"  No composite improvement ({composite:.4f} >= {best_composite_loss:.4f}). Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

                save_checkpoint(
                    LATEST_CHECKPOINT,
                    update_step,
                    microbatch_count,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    val_loss=composite,
                    best_loss=best_composite_loss
                )

                print(
                    f"Update {update_step:5d} | "
                    f"Curr Loss: {c_loss:6.4f} (PPL: {c_ppl:5.2f}) | "
                    f"OASST1 Loss: {o_loss:6.4f} (PPL: {o_ppl:5.2f}) | "
                    f"Composite: {composite:6.4f} | "
                    f"Val Time: {val_dur:.1f}s\n"
                )

                with open(VAL_LOG_FILE, "a", encoding="utf-8") as vf:
                    vf.write(json.dumps({
                        "step": update_step,
                        "epoch": continuous_loader.epoch,
                        "curriculum_val_loss": round(c_loss, 4),
                        "curriculum_val_ppl": round(c_ppl, 2),
                        "oasst1_val_loss": round(o_loss, 4),
                        "oasst1_val_ppl": round(o_ppl, 2),
                        "composite_loss": round(composite, 4),
                        "is_best": is_best,
                        "timestamp": time.time()
                    }) + "\n")

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered at update {update_step}. Best composite loss: {best_composite_loss:.4f}")
                    break

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE! Total time: {total_time/60:.2f} minutes.")
    print(f"Best Checkpoint: {BEST_CHECKPOINT} (Composite Loss: {best_composite_loss:.4f})")
    print(f"Latest Checkpoint: {LATEST_CHECKPOINT}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_updates", type=int, default=1500)
    args = parser.parse_args()
    train(max_updates=args.max_updates)
