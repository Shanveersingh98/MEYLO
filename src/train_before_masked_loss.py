
from pathlib import Path
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "D:/DEEP-LEARNING-PROJECT/src")

from model import FromScratchLM


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")

TRAIN_FILE = (
    PROJECT_DIR
    / "data/processed/training_sequences/train.bin"
)

VALIDATION_FILE = (
    PROJECT_DIR
    / "data/processed/training_sequences/validation.bin"
)

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
LOG_DIR = PROJECT_DIR / "logs"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"
LOG_FILE = LOG_DIR / "training_log.jsonl"


# ============================================================
# CHECKPOINT CONFIGURATION
# ============================================================

CHECKPOINT_INTERVAL = 100
VALIDATION_INTERVAL = 100


# ============================================================
# MODEL CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
D_FF = 1536
DROPOUT = 0.1


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 1

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.1
GRADIENT_CLIP = 1.0
MAX_STEPS = 10000
VALIDATION_BATCHES = 20

PAD_ID = 0


# ============================================================
# DATASET
# ============================================================

class TokenDataset(Dataset):

    def __init__(self, file_path):

        self.data = np.memmap(
            file_path,
            dtype=np.uint16,
            mode="r"
        )

        if len(self.data) % CONTEXT_LENGTH != 0:

            raise ValueError(
                "Dataset size is not divisible by context length."
            )

        self.sequence_count = (
            len(self.data) // CONTEXT_LENGTH
        )

    def __len__(self):

        return self.sequence_count

    def __getitem__(self, index):

        start = index * CONTEXT_LENGTH
        end = start + CONTEXT_LENGTH

        sequence = np.asarray(
            self.data[start:end],
            dtype=np.int64
        )

        input_ids = torch.tensor(
            sequence[:-1],
            dtype=torch.long
        )

        target_ids = torch.tensor(
            sequence[1:],
            dtype=torch.long
        )

        return input_ids, target_ids


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("STEP 27 - REAL TRAINING SYSTEM")
print("=" * 70)

print()
print(f"Device: {device}")

if device.type == "cuda":

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CREATE DIRECTORIES
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("Loading datasets...")

train_dataset = TokenDataset(
    TRAIN_FILE
)

validation_dataset = TokenDataset(
    VALIDATION_FILE
)

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

print(
    f"Training sequences:   {len(train_dataset):,}"
)

print(
    f"Validation sequences: {len(validation_dataset):,}"
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("Creating model...")

model = FromScratchLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT,
)

model = model.to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    f"Parameters: {parameter_count:,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_STEPS,
    eta_min=MIN_LEARNING_RATE
)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# TRAINING STATE
# ============================================================

step = 0
best_validation_loss = float("inf")


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def save_checkpoint(
    path,
    current_step,
    validation_loss,
    best_loss
):

    checkpoint = {

        "step": current_step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "validation_loss":
            validation_loss,

        "best_validation_loss":
            best_loss,

        "config": {

            "vocab_size": VOCAB_SIZE,

            "context_length":
                CONTEXT_LENGTH,

            "d_model":
                D_MODEL,

            "num_heads":
                NUM_HEADS,

            "num_layers":
                NUM_LAYERS,

            "d_ff":
                D_FF,

            "dropout":
                DROPOUT,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

        }
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved: {path}"
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(path):

    global step
    global best_validation_loss

    print()
    print(
        f"Loading checkpoint: {path}"
    )

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    if "scaler_state_dict" in checkpoint:

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

    step = checkpoint["step"]

    best_validation_loss = checkpoint[
        "best_validation_loss"
    ]

    print(
        f"Resumed from step: {step}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate():

    model.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():

        for input_ids, target_ids in validation_loader:

            input_ids = input_ids.to(
                device
            )

            target_ids = target_ids.to(
                device
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                logits = model(
                    input_ids
                )

                loss = criterion(
                    logits.reshape(
                        -1,
                        VOCAB_SIZE
                    ),
                    target_ids.reshape(-1)
                )

            total_loss += loss.item()

            batches += 1

            if batches >= VALIDATION_BATCHES:

                break

    model.train()

    if batches == 0:

        return float("inf")

    return total_loss / batches


# ============================================================
# LOGGING
# ============================================================

def write_log(
    current_step,
    training_loss,
    validation_loss,
    learning_rate,
    elapsed_seconds
):

    record = {

        "step": current_step,

        "training_loss":
            training_loss,

        "validation_loss":
            validation_loss,

        "learning_rate":
            learning_rate,

        "elapsed_seconds":
            elapsed_seconds,

    }

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(record)
            + "\n"
        )


# ============================================================
# RESUME CHECK
# ============================================================

if LATEST_CHECKPOINT.exists():

    answer = input(
        "\nLatest checkpoint found. "
        "Resume training? [y/N]: "
    )

    if answer.lower() == "y":

        load_checkpoint(
            LATEST_CHECKPOINT
        )


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print()
print(f"Maximum steps: {MAX_STEPS}")
print(
    f"Checkpoint interval: "
    f"{CHECKPOINT_INTERVAL}"
)

print(
    f"Validation interval: "
    f"{VALIDATION_INTERVAL}"
)

model.train()

training_start_time = time.time()

while step < MAX_STEPS:

    for input_ids, target_ids in train_loader:

        if step >= MAX_STEPS:

            break

        input_ids = input_ids.to(
            device
        )

        target_ids = target_ids.to(
            device
        )

        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            logits = model(
                input_ids
            )

            loss = criterion(
                logits.reshape(
                    -1,
                    VOCAB_SIZE
                ),
                target_ids.reshape(-1)
            )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )

        # ----------------------------------------------------
        # Update model
        # ----------------------------------------------------

        scaler.step(
            optimizer
        )

        scaler.update()

        # ----------------------------------------------------
        # Update learning rate
        # ----------------------------------------------------

        scheduler.step()

        step += 1

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        elapsed = (
            time.time()
            - training_start_time
        )

        print(
            f"Step {step:05d} | "
            f"Loss {loss.item():.4f} | "
            f"LR {current_lr:.7f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_loss = None

        if step % VALIDATION_INTERVAL == 0:

            validation_loss = evaluate()

            print(
                f"Step {step:05d} | "
                f"Validation loss: "
                f"{validation_loss:.4f}"
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if validation_loss < best_validation_loss:

                best_validation_loss = (
                    validation_loss
                )

                save_checkpoint(
                    BEST_CHECKPOINT,
                    step,
                    validation_loss,
                    best_validation_loss
                )

                print(
                    "New best model saved."
                )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        if step % CHECKPOINT_INTERVAL == 0:

            latest_validation_loss = (
                validation_loss
                if validation_loss is not None
                else best_validation_loss
            )

            save_checkpoint(
                LATEST_CHECKPOINT,
                step,
                latest_validation_loss,
                best_validation_loss
            )

        # ----------------------------------------------------
        # Write log
        # ----------------------------------------------------

        write_log(
            step,
            loss.item(),
            validation_loss,
            current_lr,
            elapsed
        )

        if step >= MAX_STEPS:

            break


# ============================================================
# FINAL CHECKPOINT
# ============================================================

final_validation_loss = evaluate()

print()
print(
    f"Final validation loss: "
    f"{final_validation_loss:.4f}"
)

save_checkpoint(
    LATEST_CHECKPOINT,
    step,
    final_validation_loss,
    best_validation_loss
)


# ============================================================
# GPU MEMORY
# ============================================================

if device.type == "cuda":

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print()
    print(
        f"Peak GPU memory: "
        f"{peak_memory:.2f} MB"
    )


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Steps completed: {step}"
)

print(
    f"Best validation loss: "
    f"{best_validation_loss:.4f}"
)

print(
    f"Latest checkpoint: "
    f"{LATEST_CHECKPOINT}"
)

print(
    f"Best checkpoint: "
    f"{BEST_CHECKPOINT}"
)

print(
    f"Training log: "
    f"{LOG_FILE}"
)

print()
print("PASS: STEP 27 TRAINING SYSTEM COMPLETED.")
