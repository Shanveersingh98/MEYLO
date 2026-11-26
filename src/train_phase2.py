# Phase 2 training script
# Continues the 10,000-step model to 30,000 total steps.

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import FromScratchLM


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\DEEP-LEARNING-PROJECT")

TRAIN_BIN = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_sequences"
    / "train.bin"
)

VAL_BIN = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_sequences"
    / "validation.bin"
)

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

LOG_FILE = LOG_DIR / "training_log.jsonl"


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

MAX_STEPS = 30000

PHASE1_STEPS = 10000

PHASE2_STEPS = MAX_STEPS - PHASE1_STEPS

BATCH_SIZE = 1

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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    "Using device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
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
# DATASET
# ============================================================

class BinarySequenceDataset(Dataset):

    def __init__(self, path):

        self.path = Path(path)

        # The training .bin files store token IDs as uint16.
        self.data = np.memmap(
            self.path,
            dtype=np.uint16,
            mode="r"
        )

        # Every sequence must contain exactly
        # CONTEXT_LENGTH tokens.
        if len(self.data) % CONTEXT_LENGTH != 0:

            raise ValueError(
                "Dataset size is not divisible "
                "by context length."
            )

        self.num_sequences = (
            len(self.data)
            // CONTEXT_LENGTH
        )

    def __len__(self):

        return self.num_sequences

    def __getitem__(self, index):

        start = (
            index
            * CONTEXT_LENGTH
        )

        end = (
            start
            + CONTEXT_LENGTH
        )

        # Read one complete sequence.
        sequence = np.asarray(
            self.data[start:end],
            dtype=np.int64
        )

        # Input = tokens 0 ... 1022
        x = torch.tensor(
            sequence[:-1],
            dtype=torch.long
        )

        # Target = tokens 1 ... 1023
        y = torch.tensor(
            sequence[1:],
            dtype=torch.long
        )

        return x, y


# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading datasets...")

train_dataset = BinarySequenceDataset(
    TRAIN_BIN
)

val_dataset = BinarySequenceDataset(
    VAL_BIN
)

print(
    "Training sequences:",
    len(train_dataset)
)

print(
    "Validation sequences:",
    len(val_dataset)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)


# ============================================================
# MODEL
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
    dropout=DROPOUT
)

model = model.to(device)

print(
    "Model parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
    )
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
# PHASE 2 SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=PHASE2_STEPS,
    eta_min=MIN_LEARNING_RATE
)


# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=0
)


# ============================================================
# TRAINING STATE
# ============================================================

step = 0

best_validation_loss = float("inf")

phase2_checkpoint_loaded = False


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0

    batches = 0

    for x, y in val_loader:

        x = x.to(device)

        y = y.to(device)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            logits = model(x)

            loss = criterion(
                logits.reshape(
                    -1,
                    VOCAB_SIZE
                ),
                y.reshape(-1)
            )

        total_loss += loss.item()

        batches += 1

        if batches >= VALIDATION_BATCHES:

            break

    model.train()

    if batches == 0:

        return float("inf")

    return (
        total_loss
        / batches
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    validation_loss=None
):

    checkpoint = {

        "step":
            step,

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
            best_validation_loss,

        "phase":
            2,

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
                WEIGHT_DECAY
        }
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        "Checkpoint saved:",
        path
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(path):

    global step
    global best_validation_loss
    global phase2_checkpoint_loaded

    print()
    print(
        "Loading checkpoint:",
        path
    )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    step = checkpoint[
        "step"
    ]

    best_validation_loss = checkpoint[
        "best_validation_loss"
    ]

    checkpoint_phase = checkpoint.get(
        "phase",
        1
    )

    # --------------------------------------------------------
    # Phase 2 checkpoint
    # --------------------------------------------------------

    if checkpoint_phase == 2:

        if "scheduler_state_dict" in checkpoint:

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        phase2_checkpoint_loaded = True

        print(
            "Phase 2 checkpoint detected."
        )

        print(
            "Phase 2 scheduler state restored."
        )

    # --------------------------------------------------------
    # Phase 1 checkpoint
    # --------------------------------------------------------

    else:

        # The Phase 1 scheduler reached the end
        # of its 10,000-step schedule.
        #
        # Phase 2 needs a completely new
        # 20,000-step cosine schedule.

        for parameter_group in optimizer.param_groups:

            parameter_group["lr"] = (
                LEARNING_RATE
            )

        phase2_checkpoint_loaded = False

        print(
            "Phase 1 checkpoint detected."
        )

        print(
            "Starting a fresh Phase 2 scheduler."
        )

    if "scaler_state_dict" in checkpoint:

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    print(
        "Checkpoint step:",
        step
    )

    print(
        "Best validation loss:",
        best_validation_loss
    )


# ============================================================
# RESUME
# ============================================================

if LATEST_CHECKPOINT.exists():

    print()
    print(
        "Latest checkpoint found:"
    )

    print(
        LATEST_CHECKPOINT
    )

    load_checkpoint(
        LATEST_CHECKPOINT
    )

else:

    print()
    print(
        "No latest checkpoint found."
    )

    if BEST_CHECKPOINT.exists():

        print(
            "Using best checkpoint."
        )

        load_checkpoint(
            BEST_CHECKPOINT
        )

    else:

        raise FileNotFoundError(
            "No training checkpoint was found."
        )


# ============================================================
# SAFETY CHECK
# ============================================================

if step < PHASE1_STEPS:

    raise RuntimeError(
        f"Checkpoint step is {step}, "
        f"but Phase 2 requires at least "
        f"{PHASE1_STEPS} steps."
    )


if step >= MAX_STEPS:

    print()

    print(
        "Training is already complete."
    )

    print(
        "Current step:",
        step
    )

    print(
        "Maximum step:",
        MAX_STEPS
    )

    raise SystemExit


# ============================================================
# LOGGING
# ============================================================

def write_log(
    current_step,
    loss,
    validation_loss,
    learning_rate
):

    record = {

        "step":
            current_step,

        "loss":
            loss,

        "validation_loss":
            validation_loss,

        "learning_rate":
            learning_rate,

        "phase":
            2,

        "timestamp":
            time.time()
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
# TRAINING
# ============================================================

print()
print("=" * 60)
print("PHASE 2 TRAINING")
print("=" * 60)

print(
    "Starting step:",
    step
)

print(
    "Target step:",
    MAX_STEPS
)

print(
    "New Phase 2 steps:",
    PHASE2_STEPS
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Minimum learning rate:",
    MIN_LEARNING_RATE
)

print("=" * 60)


model.train()

train_iterator = iter(
    train_loader
)

start_time = time.time()


while step < MAX_STEPS:

    try:

        x, y = next(
            train_iterator
        )

    except StopIteration:

        train_iterator = iter(
            train_loader
        )

        x, y = next(
            train_iterator
        )

    x = x.to(device)

    y = y.to(device)

    optimizer.zero_grad(
        set_to_none=True
    )

    with torch.amp.autocast(
        device_type="cuda",
        enabled=torch.cuda.is_available()
    ):

        logits = model(x)

        loss = criterion(
            logits.reshape(
                -1,
                VOCAB_SIZE
            ),
            y.reshape(-1)
        )

    scaler.scale(
        loss
    ).backward()

    scaler.unscale_(
        optimizer
    )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )

    scaler.step(
        optimizer
    )

    scaler.update()

    scheduler.step()

    step += 1

    loss_value = loss.item()

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # --------------------------------------------------------
    # PRINT TRAINING PROGRESS
    # --------------------------------------------------------

    if step == 1 or step % 5 == 0:

        print(
            f"Step {step} | "
            f"Loss {loss_value:.4f} | "
            f"LR {current_lr:.7f}"
        )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_loss = None

    if step % VALIDATION_INTERVAL == 0:

        validation_loss = validate()

        print(
            f"Step {step} | "
            f"Validation loss: "
            f"{validation_loss:.4f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            save_checkpoint(
                BEST_CHECKPOINT,
                validation_loss
            )

            print(
                "New best model saved."
            )


    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    if step % CHECKPOINT_INTERVAL == 0:

        save_checkpoint(
            LATEST_CHECKPOINT,
            validation_loss
        )


    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    write_log(
        step,
        loss_value,
        validation_loss,
        current_lr
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print()

print("=" * 60)

print(
    "PHASE 2 TRAINING COMPLETE"
)

print("=" * 60)


final_validation_loss = validate()

print(
    "Final validation loss:",
    final_validation_loss
)


# ============================================================
# FINAL CHECKPOINT
# ============================================================

save_checkpoint(
    LATEST_CHECKPOINT,
    final_validation_loss
)


if (
    final_validation_loss
    < best_validation_loss
):

    best_validation_loss = (
        final_validation_loss
    )

    save_checkpoint(
        BEST_CHECKPOINT,
        final_validation_loss
    )

    print(
        "Final model is the new best model."
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
        f"Peak GPU memory: "
        f"{peak_memory:.2f} MB"
    )


elapsed = (
    time.time()
    - start_time
)

print(
    f"Training time: "
    f"{elapsed / 3600:.2f} hours"
)

print("=" * 60)