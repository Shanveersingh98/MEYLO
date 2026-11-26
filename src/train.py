from pathlib import Path
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


sys.path.insert(
    0,
    "D:/DEEP-LEARNING-PROJECT/src"
)

from model import FromScratchLM


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT"
)

TRAIN_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "training_sequences_masked"
    / "train_tokens.bin"
)

VALIDATION_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "training_sequences_masked"
    / "validation_tokens.bin"
)

TRAIN_MASK_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "training_sequences_masked"
    / "train_mask.bin"
)

VALIDATION_MASK_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "training_sequences_masked"
    / "validation_mask.bin"
)

CHECKPOINT_DIR = (
    PROJECT_DIR / "checkpoints"
)

LOG_DIR = (
    PROJECT_DIR / "logs"
)


# IMPORTANT:
# These are separate from the old unmasked checkpoints.

LATEST_CHECKPOINT = (
    CHECKPOINT_DIR / "masked_latest.pt"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR / "masked_best.pt"
)

LOG_FILE = (
    LOG_DIR / "training_masked.jsonl"
)


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

# First masked-loss experiment.
MAX_STEPS = 10000

VALIDATION_BATCHES = 20

PAD_ID = 0


# ============================================================
# DATASET
# ============================================================

class TokenDataset(Dataset):

    def __init__(
        self,
        token_file_path,
        mask_file_path
    ):

        self.tokens = np.memmap(
            token_file_path,
            dtype=np.uint16,
            mode="r"
        )

        self.loss_mask = np.memmap(
            mask_file_path,
            dtype=np.uint8,
            mode="r"
        )

        if len(self.tokens) != len(
            self.loss_mask
        ):

            raise ValueError(
                "Token and mask dataset sizes "
                "do not match."
            )

        if len(self.tokens) % CONTEXT_LENGTH != 0:

            raise ValueError(
                "Dataset size is not divisible "
                "by context length."
            )

        self.sequence_count = (
            len(self.tokens)
            // CONTEXT_LENGTH
        )


    def __len__(self):

        return self.sequence_count


    def __getitem__(self, index):

        start = (
            index * CONTEXT_LENGTH
        )

        end = (
            start + CONTEXT_LENGTH
        )

        sequence = np.asarray(
            self.tokens[start:end],
            dtype=np.int64
        )

        mask = np.asarray(
            self.loss_mask[start:end],
            dtype=np.uint8
        )

        input_ids = torch.tensor(
            sequence[:-1],
            dtype=torch.long
        )

        target_ids = torch.tensor(
            sequence[1:],
            dtype=torch.long
        )

        loss_mask = torch.tensor(
            mask[1:],
            dtype=torch.bool
        )

        return (
            input_ids,
            target_ids,
            loss_mask
        )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("ASSISTANT-ONLY LOSS TRAINING")
print("=" * 70)

print()
print(f"Device: {device}")

if device.type == "cuda":

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
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
print("Loading masked datasets...")

train_dataset = TokenDataset(
    TRAIN_FILE,
    TRAIN_MASK_FILE
)

validation_dataset = TokenDataset(
    VALIDATION_FILE,
    VALIDATION_MASK_FILE
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
    f"Training sequences:   "
    f"{len(train_dataset):,}"
)

print(
    f"Validation sequences: "
    f"{len(validation_dataset):,}"
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
    f"Parameters: "
    f"{parameter_count:,}"
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

# IMPORTANT:
# We need one loss value per token first.
# The loss mask will then select only assistant tokens.

criterion = nn.CrossEntropyLoss(
    reduction="none"
)


def calculate_masked_loss(
    logits,
    target_ids,
    loss_mask
):

    token_losses = criterion(
        logits.reshape(
            -1,
            VOCAB_SIZE
        ),
        target_ids.reshape(-1)
    )

    token_losses = token_losses.reshape(
        target_ids.shape
    )

    masked_losses = token_losses[
        loss_mask
    ]

    if masked_losses.numel() == 0:

        return (
            token_losses.sum() * 0.0
        )

    return masked_losses.mean()


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

best_validation_loss = float(
    "inf"
)


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

            "vocab_size":
                VOCAB_SIZE,

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

            "loss_mode":
                "assistant_only",

            "max_steps":
                MAX_STEPS
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
# VALIDATION
# ============================================================

def evaluate():

    model.eval()

    total_loss = 0.0

    batches = 0

    with torch.no_grad():

        for (
            input_ids,
            target_ids,
            loss_mask
        ) in validation_loader:

            input_ids = input_ids.to(
                device
            )

            target_ids = target_ids.to(
                device
            )

            loss_mask = loss_mask.to(
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

                loss = calculate_masked_loss(
                    logits,
                    target_ids,
                    loss_mask
                )

            total_loss += loss.item()

            batches += 1

            if batches >= VALIDATION_BATCHES:

                break

    model.train()

    if batches == 0:

        return float("inf")

    return (
        total_loss / batches
    )


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

        "step":
            current_step,

        "training_loss":
            training_loss,

        "validation_loss":
            validation_loss,

        "learning_rate":
            learning_rate,

        "elapsed_seconds":
            elapsed_seconds,

        "loss_mode":
            "assistant_only"
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
# NO OLD CHECKPOINT RESUME
# ============================================================

print()
print(
    "Starting fresh masked-loss training."
)

print(
    "Old unmasked checkpoints will NOT "
    "be loaded."
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING ASSISTANT-ONLY LOSS TRAINING")
print("=" * 70)

print()
print(
    f"Maximum steps: "
    f"{MAX_STEPS}"
)

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

    for (
        input_ids,
        target_ids,
        loss_mask
    ) in train_loader:

        if step >= MAX_STEPS:

            break

        input_ids = input_ids.to(
            device
        )

        target_ids = target_ids.to(
            device
        )

        loss_mask = loss_mask.to(
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

            loss = calculate_masked_loss(
                logits,
                target_ids,
                loss_mask
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

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

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

        if (
            step
            % VALIDATION_INTERVAL
            == 0
        ):

            validation_loss = evaluate()

            print(
                f"Step {step:05d} | "
                f"Validation loss: "
                f"{validation_loss:.4f}"
            )

            # ------------------------------------------------
            # Best checkpoint
            # ------------------------------------------------

            if (
                validation_loss
                < best_validation_loss
            ):

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
                    "New best masked-loss "
                    "model saved."
                )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        if (
            step
            % CHECKPOINT_INTERVAL
            == 0
        ):

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
# FINAL VALIDATION
# ============================================================

final_validation_loss = evaluate()

print()

print(
    f"Final validation loss: "
    f"{final_validation_loss:.4f}"
)


# ============================================================
# FINAL CHECKPOINT
# ============================================================

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
print("MASKED-LOSS TRAINING COMPLETE")
print("=" * 70)

print(
    f"Steps completed: "
    f"{step}"
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

print(
    "PASS: ASSISTANT-ONLY LOSS "
    "TRAINING COMPLETED."
)
