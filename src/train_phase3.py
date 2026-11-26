import json
import time
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import FromScratchLM


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\DEEP-LEARNING-PROJECT")

TRAIN_BIN = PROJECT_ROOT / "data" / "processed" / "training_sequences" / "train.bin"
VAL_BIN = PROJECT_ROOT / "data" / "processed" / "training_sequences" / "validation.bin"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

SOURCE_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"

PHASE3_LATEST = CHECKPOINT_DIR / "phase3_latest.pt"
PHASE3_BEST = CHECKPOINT_DIR / "phase3_best.pt"

LOG_FILE = LOG_DIR / "training_phase3.jsonl"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024

D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
D_FF = 1536
DROPOUT = 0.1

BATCH_SIZE = 1

START_STEP = 30000
MAX_STEPS = 50000

PHASE3_STEPS = MAX_STEPS - START_STEP

LEARNING_RATE = 3e-4
MIN_LEARNING_RATE = 3e-5

WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

VALIDATION_INTERVAL = 100
VALIDATION_BATCHES = 20
CHECKPOINT_INTERVAL = 100


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("PHASE 3 TRAINING")
print("=" * 70)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Start step:", START_STEP)
print("Maximum step:", MAX_STEPS)
print("Additional steps:", PHASE3_STEPS)

print("=" * 70)


# ============================================================
# DATASET
# ============================================================

class BinarySequenceDataset(Dataset):

    def __init__(self, file_path):

        self.file_path = str(file_path)

        self.data = np.memmap(
            self.file_path,
            dtype=np.uint16,
            mode="r"
        )

        self.sequence_length = CONTEXT_LENGTH

        self.num_sequences = len(self.data) // self.sequence_length

    def __len__(self):

        return self.num_sequences

    def __getitem__(self, index):

        start = index * self.sequence_length
        end = start + self.sequence_length

        sequence = self.data[start:end].astype(np.int64)

        sequence = torch.from_numpy(sequence).long()

        input_ids = sequence[:-1]
        target_ids = sequence[1:]

        return input_ids, target_ids


# ============================================================
# DATASETS
# ============================================================

train_dataset = BinarySequenceDataset(TRAIN_BIN)
val_dataset = BinarySequenceDataset(VAL_BIN)

print("Training sequences:", len(train_dataset))
print("Validation sequences:", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = FromScratchLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT
).to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0
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
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# LOAD 30K CHECKPOINT
# ============================================================

print()
print("Loading source checkpoint:")
print(SOURCE_CHECKPOINT)

if not SOURCE_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Source checkpoint not found: {SOURCE_CHECKPOINT}"
    )

checkpoint = torch.load(
    SOURCE_CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

source_step = checkpoint["step"]

print("Source checkpoint step:", source_step)

if source_step != START_STEP:
    raise RuntimeError(
        f"Expected checkpoint at step {START_STEP}, "
        f"but found step {source_step}."
    )


model.load_state_dict(
    checkpoint["model_state_dict"]
)

optimizer.load_state_dict(
    checkpoint["optimizer_state_dict"]
)


# ============================================================
# RESET LEARNING RATE FOR PHASE 3
# ============================================================

for param_group in optimizer.param_groups:
    param_group["lr"] = LEARNING_RATE


# ============================================================
# LOAD SCALER STATE
# ============================================================

if checkpoint.get("scaler_state_dict") is not None:

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )


# ============================================================
# BEST VALIDATION LOSS
# ============================================================

best_validation_loss = checkpoint.get(
    "best_validation_loss",
    float("inf")
)

print("Previous best validation loss:", best_validation_loss)


# ============================================================
# FRESH PHASE 3 COSINE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=PHASE3_STEPS,
    eta_min=MIN_LEARNING_RATE
)


# ============================================================
# DIRECTORIES
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
# VALIDATION FUNCTION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    batches = 0

    for input_ids, target_ids in val_loader:

        if batches >= VALIDATION_BATCHES:
            break

        input_ids = input_ids.to(
            DEVICE,
            non_blocking=True
        )

        target_ids = target_ids.to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp
        ):

            logits = model(input_ids)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1)
            )

        total_loss += loss.item()
        batches += 1

    model.train()

    if batches == 0:
        return float("inf")

    return total_loss / batches


# ============================================================
# CHECKPOINT FUNCTION
# ============================================================

def save_checkpoint(
    path,
    step,
    validation_loss
):

    checkpoint_data = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "phase": 3,
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
            "grad_clip": GRAD_CLIP,
            "start_step": START_STEP,
            "max_steps": MAX_STEPS
        }
    }

    torch.save(
        checkpoint_data,
        path
    )

    print(
        f"Checkpoint saved: {path}"
    )


# ============================================================
# TRAINING LOOP
# ============================================================

model.train()

global_step = START_STEP

running_loss = 0.0
step_start_time = time.time()

train_iterator = iter(train_loader)

print()
print("=" * 70)
print("STARTING PHASE 3 TRAINING")
print("=" * 70)
print("30,000 -> 50,000")
print("=" * 70)


while global_step < MAX_STEPS:

    try:

        input_ids, target_ids = next(
            train_iterator
        )

    except StopIteration:

        train_iterator = iter(train_loader)

        input_ids, target_ids = next(
            train_iterator
        )

    input_ids = input_ids.to(
        DEVICE,
        non_blocking=True
    )

    target_ids = target_ids.to(
        DEVICE,
        non_blocking=True
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.amp.autocast(
        "cuda",
        enabled=use_amp
    ):

        logits = model(input_ids)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            target_ids.reshape(-1)
        )

    scaler.scale(loss).backward()

    scaler.unscale_(optimizer)

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )

    scaler.step(
        optimizer
    )

    scaler.update()

    scheduler.step()

    global_step += 1

    loss_value = loss.item()

    running_loss += loss_value

    if global_step == 1 or global_step % 5 == 0:

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Step {global_step} | "
            f"Loss {loss_value:.4f} | "
            f"LR {current_lr:.7f}"
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_loss = None

    if global_step % VALIDATION_INTERVAL == 0:

        validation_loss = validate()

        print(
            f"Step {global_step} | "
            f"Validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            save_checkpoint(
                PHASE3_BEST,
                global_step,
                validation_loss
            )

            print("New best model saved.")

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    if global_step % 5 == 0:

        log_entry = {
            "step": global_step,
            "loss": loss_value,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_loss": validation_loss,
            "time": time.time()
        }

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(log_entry)
                + "\n"
            )

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    if global_step % CHECKPOINT_INTERVAL == 0:

        save_checkpoint(
            PHASE3_LATEST,
            global_step,
            validation_loss
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()
print("=" * 70)
print("PHASE 3 TRAINING COMPLETE")
print("=" * 70)

final_validation_loss = validate()

print(
    f"Final validation loss: {final_validation_loss}"
)


# ============================================================
# FINAL CHECKPOINT
# ============================================================

save_checkpoint(
    PHASE3_LATEST,
    global_step,
    final_validation_loss
)


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print(
        f"Peak GPU memory: {peak_memory:.2f} MB"
    )


training_time = (
    time.time() - step_start_time
)

print(
    f"Training time: "
    f"{training_time / 3600:.2f} hours"
)

print("=" * 70)