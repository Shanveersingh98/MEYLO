from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "D:/DEEP-LEARNING-PROJECT/src")

from model import FromScratchLM


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/data/processed/training_sequences/train.bin"
)

VALIDATION_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/data/processed/training_sequences/validation.bin"
)

CHECKPOINT_DIR = Path(
    "D:/DEEP-LEARNING-PROJECT/checkpoints"
)


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

WEIGHT_DECAY = 0.1

GRADIENT_CLIP = 1.0

MAX_STEPS = 10

VALIDATION_INTERVAL = 5

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

        self.sequence_count = len(self.data) // CONTEXT_LENGTH

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

print("=" * 60)
print("STEP 25 - TRAINING LOOP TEST")
print("=" * 60)

print()
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# DATA
# ============================================================

print()
print("Loading datasets...")

train_dataset = TokenDataset(TRAIN_FILE)
validation_dataset = TokenDataset(VALIDATION_FILE)

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

print(f"Training sequences:   {len(train_dataset):,}")
print(f"Validation sequences: {len(validation_dataset):,}")


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
    dropout=DROPOUT,
)

model = model.to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Parameters: {parameter_count:,}")


# ============================================================
# OPTIMIZER AND LOSS
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

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
# VALIDATION FUNCTION
# ============================================================

def evaluate():

    model.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():

        for input_ids, target_ids in validation_loader:

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                logits = model(input_ids)

                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    target_ids.reshape(-1)
                )

            total_loss += loss.item()
            batches += 1

            if batches >= 5:
                break

    model.train()

    return total_loss / batches


# ============================================================
# TRAINING LOOP
# ============================================================

print()
print("Starting training test...")
print(f"Maximum steps: {MAX_STEPS}")

model.train()

step = 0

while step < MAX_STEPS:

    for input_ids, target_ids in train_loader:

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
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
            GRADIENT_CLIP
        )

        scaler.step(optimizer)

        scaler.update()

        step += 1

        print(
            f"Step {step:03d} | "
            f"Training loss: {loss.item():.4f}"
        )

        if step % VALIDATION_INTERVAL == 0:

            validation_loss = evaluate()

            print(
                f"Step {step:03d} | "
                f"Validation loss: {validation_loss:.4f}"
            )

        if step >= MAX_STEPS:
            break


# ============================================================
# SAVE TEST CHECKPOINT
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

checkpoint_path = CHECKPOINT_DIR / "step25_test.pt"

torch.save(
    {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_loss": loss.item(),
    },
    checkpoint_path
)

print()
print("=" * 60)
print("TRAINING TEST COMPLETE")
print("=" * 60)

print(f"Steps completed: {step}")
print(f"Checkpoint: {checkpoint_path}")

if device.type == "cuda":

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print(f"Peak GPU memory: {peak_memory:.2f} MB")

print()
print("PASS: TRAINING LOOP TEST COMPLETED.")