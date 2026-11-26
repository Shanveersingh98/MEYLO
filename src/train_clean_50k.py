from pathlib import Path
import sys
import json
import time

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM
from generate_clean_50k import generate_text, TEST_PROMPTS

# ============================================================
# 1. PATH CONFIGURATION (Isolated 50k Experiment)
# ============================================================

DATA_DIR = PROJECT_DIR / "data" / "processed" / "training_sequences_clean"
TRAIN_TOKENS_FILE = DATA_DIR / "train_tokens.bin"
TRAIN_MASK_FILE = DATA_DIR / "train_mask.bin"
VAL_TOKENS_FILE = DATA_DIR / "validation_tokens.bin"
VAL_MASK_FILE = DATA_DIR / "validation_mask.bin"

TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "clean_50k_latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "clean_50k_best.pt"

LOG_FILE = LOG_DIR / "training_clean_50k.jsonl"
GEN_EVAL_FILE = LOG_DIR / "clean_50k_generation_eval.jsonl"

# ============================================================
# 2. EXPERIMENT HYPERPARAMETERS
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

MAX_STEPS = 50000
VALIDATION_BATCHES = 20

# Milestones for validation and generation evaluation
MILESTONES = [5000, 10000, 20000, 30000, 40000, 50000]

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
            raise ValueError("Tokens and mask dataset lengths do not match.")

        if len(self.tokens) % CONTEXT_LENGTH != 0:
            raise ValueError("Dataset length is not divisible by context length.")

        self.sequence_count = len(self.tokens) // CONTEXT_LENGTH

    def __len__(self):
        return self.sequence_count

    def __getitem__(self, index):
        start = index * CONTEXT_LENGTH
        end = start + CONTEXT_LENGTH

        sequence = np.asarray(self.tokens[start:end], dtype=np.int64)
        mask = np.asarray(self.loss_mask[start:end], dtype=np.uint8)

        # Shifted input, target, and loss mask
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)
        loss_mask = torch.tensor(mask[1:], dtype=torch.bool)

        return input_ids, target_ids, loss_mask

# ============================================================
# 4. PRE-TRAINING SANITY CHECK
# ============================================================

def perform_pre_training_sanity_check(device, train_dataset, val_dataset):
    print("=" * 70)
    print("PRE-TRAINING COMPLETE SANITY CHECK")
    print("=" * 70)

    # 1. Environment & Hardware
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    print(f"CUDA availability:                 {cuda_avail}")
    print(f"GPU name:                        {gpu_name}")
    print(f"Device:                          {device}")
    print(f"initial learning rate:           {LEARNING_RATE}")
    print(f"minimum learning rate:           {MIN_LEARNING_RATE}")
    print(f"target max steps:                {MAX_STEPS:,}")

    if not cuda_avail or device.type != "cuda":
        raise RuntimeError("CUDA is not available or device is not CUDA! Training must run on GPU.")

    # 2. Dataset metrics
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    print(f"dataset train sequence count:    {n_train:,}")
    print(f"validation sequence count:       {n_val:,}")
    print(f"sequence length:                 {CONTEXT_LENGTH}")
    print(f"token dtype:                     uint16")
    print(f"mask dtype:                      uint8")

    assert n_train == 17683, f"Expected 17,683 train sequences, got {n_train}"
    assert n_val == 1989, f"Expected 1,989 val sequences, got {n_val}"

    # Sample batch shapes
    sample_inp, sample_tgt, sample_msk = train_dataset[0]
    print(f"batch shape:                     (1, {CONTEXT_LENGTH})")
    print(f"input shape:                     (1, {sample_inp.shape[0]})")
    print(f"target shape:                    (1, {sample_tgt.shape[0]})")
    print(f"mask shape:                      (1, {sample_msk.shape[0]})")
    print(f"number of masked assistant tokens: {sample_msk.sum().item()}")

    assert sample_inp.shape == torch.Size([1023])
    assert sample_tgt.shape == torch.Size([1023])
    assert sample_msk.shape == torch.Size([1023])
    assert sample_inp[0].item() == BOS_ID

    # 3. Model parameters check
    temp_model = FromScratchLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    total_params = sum(p.numel() for p in temp_model.parameters())
    print(f"model parameter count:           {total_params:,}")
    assert total_params == 14120768, f"Expected 14,120,768 parameters, got {total_params}"

    # Check for NaN / Inf in initial weights
    for name, p in temp_model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"Initial parameter {name} contains NaN or Inf!")

    # 4. Perform ONE forward/backward/update step
    print("\nExecuting 1 forward/backward/update sanity step...")
    temp_opt = torch.optim.AdamW(temp_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    temp_scaler = torch.amp.GradScaler("cuda", enabled=True)
    crit = nn.CrossEntropyLoss(reduction="none")

    inp_batch = sample_inp.unsqueeze(0).to(device)
    tgt_batch = sample_tgt.unsqueeze(0).to(device)
    msk_batch = sample_msk.unsqueeze(0).to(device)

    temp_opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        logits = temp_model(inp_batch)
        tok_losses = crit(logits.reshape(-1, VOCAB_SIZE), tgt_batch.reshape(-1)).reshape(tgt_batch.shape)
        loss = tok_losses[msk_batch].mean()

    assert not torch.isnan(loss) and not torch.isinf(loss), "Sanity step loss is NaN or Inf!"
    print(f"  Sanity step loss:       {loss.item():.4f} (Finite: PASS)")

    temp_scaler.scale(loss).backward()
    temp_scaler.unscale_(temp_opt)
    norm = torch.nn.utils.clip_grad_norm_(temp_model.parameters(), GRADIENT_CLIP)
    assert not torch.isnan(norm) and not torch.isinf(norm), "Sanity step grad norm is NaN or Inf!"
    print(f"  Sanity step grad norm:  {norm.item():.4f} (Finite: PASS)")

    temp_scaler.step(temp_opt)
    temp_scaler.update()

    for name, p in temp_model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise RuntimeError(f"Parameter {name} contains NaN or Inf after sanity step!")
    print("  Sanity optimizer step:  SUCCESS (No NaN/Inf in updated parameters)")

    del temp_model, temp_opt, temp_scaler
    torch.cuda.empty_cache()

    print("\nALL PRE-TRAINING SANITY CHECKS PASSED. STARTING 50K RUN.\n")

# ============================================================
# 5. LOSS FUNCTION
# ============================================================

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

# ============================================================
# 6. EVALUATION FUNCTIONS
# ============================================================

def evaluate_validation(model, val_loader, device, use_amp):
    model.eval()
    total_loss = 0.0
    batches = 0

    with torch.no_grad():
        for input_ids, target_ids, loss_mask in val_loader:
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
    mean_val_loss = total_loss / batches if batches > 0 else float("inf")
    val_ppl = float(np.exp(min(mean_val_loss, 20.0)))
    return mean_val_loss, val_ppl

def run_milestone_generation(model, tokenizer, current_step, val_loss):
    print(f"\n{'='*70}\nGENERATION EVALUATION AT MILESTONE STEP {current_step:,}\n{'='*70}")
    model.eval()
    eval_record = {
        "step": current_step,
        "validation_loss": val_loss,
        "evaluations": []
    }

    for idx, prompt in enumerate(TEST_PROMPTS, start=1):
        # 1. Greedy response
        greedy_resp = generate_text(model, tokenizer, prompt, do_sample=False)
        # 2. Sampled response
        sample_resp = generate_text(model, tokenizer, prompt, do_sample=True, temperature=0.8, top_k=50, repetition_penalty=1.15)

        print(f"\n--- PROMPT {idx}/8: '{prompt}' ---")
        print(f"[GREEDY]:  {greedy_resp if greedy_resp else '[EMPTY]'}")
        print(f"[SAMPLED]: {sample_resp if sample_resp else '[EMPTY]'}")

        eval_record["evaluations"].append({
            "prompt": prompt,
            "greedy": greedy_resp,
            "sampled": sample_resp
        })

    model.train()

    with open(GEN_EVAL_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(eval_record) + "\n")

# ============================================================
# 7. CHECKPOINT MANAGEMENT
# ============================================================

def save_checkpoint(path, step_num, model, optimizer, scheduler, scaler, val_loss, best_loss):
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
            "experiment": "clean_50k"
        }
    }
    torch.save(ckpt, path)
    print(f"Checkpoint saved: {path.name}")

# ============================================================
# 8. MAIN TRAINING LOOP
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if LOG_FILE.exists():
        LOG_FILE.unlink()
    if GEN_EVAL_FILE.exists():
        GEN_EVAL_FILE.unlink()

    # Load datasets
    train_dataset = CleanTokenDataset(TRAIN_TOKENS_FILE, TRAIN_MASK_FILE)
    val_dataset = CleanTokenDataset(VAL_TOKENS_FILE, VAL_MASK_FILE)

    # Sanity checks
    perform_pre_training_sanity_check(device, train_dataset, val_dataset)

    # Tokenizer for milestone generations
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize FRESH Model
    print("Initializing fresh FromScratchLM from random weights...")
    model = FromScratchLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_STEPS,
        eta_min=MIN_LEARNING_RATE
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    step = 0
    best_val_loss = float("inf")
    best_step = 0
    training_start = time.time()
    model.train()

    print("=" * 70)
    print(f"STARTING 50,000-STEP EXPERIMENT: CLEAN DESIGN A DATASET")
    print("=" * 70)

    val_records = {}

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
            elapsed = time.time() - training_start
            gpu_mb = (torch.cuda.memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0

            val_loss = None

            # MILESTONE EVALUATION & CHECKPOINTING
            if step in MILESTONES:
                val_loss, val_ppl = evaluate_validation(model, val_loader, device, use_amp)
                val_records[step] = (val_loss, val_ppl)

                print("\n" + "=" * 70)
                print(f"MILESTONE REACHED: STEP {step:,} / {MAX_STEPS:,}")
                print(f"Validation Loss:       {val_loss:.4f} (Perplexity: {val_ppl:.2f})")
                print(f"Training Loss (Step):  {loss.item():.4f}")
                print(f"Learning Rate:         {current_lr:.7f}")
                print(f"Elapsed Time:          {elapsed:.1f}s ({elapsed / 60:.2f} min)")
                print(f"Peak GPU VRAM:         {gpu_mb:.1f} MB")
                print("=" * 70)

                # Save milestone checkpoint
                milestone_ckpt_path = CHECKPOINT_DIR / f"clean_50k_step_{step}.pt"
                save_checkpoint(milestone_ckpt_path, step, model, optimizer, scheduler, scaler, val_loss, best_val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step
                    save_checkpoint(BEST_CHECKPOINT, step, model, optimizer, scheduler, scaler, val_loss, best_val_loss)
                    print(f">>> New BEST checkpoint recorded at step {step:,} (Val Loss: {best_val_loss:.4f}) <<<")

                save_checkpoint(LATEST_CHECKPOINT, step, model, optimizer, scheduler, scaler, val_loss, best_val_loss)

                # Run generation evaluation suite
                run_milestone_generation(model, tokenizer, step, val_loss)

            # Periodic saving of latest checkpoint every 1,000 steps
            elif step % 1000 == 0:
                save_checkpoint(LATEST_CHECKPOINT, step, model, optimizer, scheduler, scaler, val_loss, best_val_loss)

            # Write training log
            record = {
                "step": step,
                "training_loss": loss.item(),
                "validation_loss": val_loss,
                "learning_rate": current_lr,
                "elapsed_seconds": elapsed,
                "gpu_memory_mb": gpu_mb,
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
            }
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Periodic console progress
            if step == 1 or step % 100 == 0:
                val_str = f" | Val: {val_loss:.4f}" if val_loss is not None else ""
                print(
                    f"Step {step:05d}/{MAX_STEPS} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.7f} | "
                    f"Grad: {grad_norm.item():.2f} | "
                    f"GPU: {gpu_mb:.1f}MB | "
                    f"Time: {elapsed:.1f}s"
                    f"{val_str}"
                )

    # Final wrap-up
    total_time = time.time() - training_start
    print("\n" + "=" * 70)
    print("50,000-STEP TRAINING EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Total Steps:           {step:,}")
    print(f"Total Time:            {total_time:.1f}s ({total_time / 60:.2f} minutes)")
    print(f"Best Validation Loss:  {best_val_loss:.4f} at Step {best_step:,}")
    print("\nMilestone Results:")
    for m_step, (m_val, m_ppl) in val_records.items():
        print(f"  Step {m_step:5d}: Val Loss = {m_val:.4f} | PPL = {m_ppl:.2f}")

    print(f"\nBest Checkpoint:       {BEST_CHECKPOINT}")
    print(f"Latest Checkpoint:     {LATEST_CHECKPOINT}")
    print(f"Training Log:          {LOG_FILE}")
    print(f"Generation Eval Log:   {GEN_EVAL_FILE}")

if __name__ == "__main__":
    main()
