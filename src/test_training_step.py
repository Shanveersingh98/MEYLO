from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "D:/DEEP-LEARNING-PROJECT/src")

from model import FromScratchLM


TRAIN_FILE = Path(
    "D:/DEEP-LEARNING-PROJECT/data/processed/training_sequences/train.bin"
)

CONTEXT_LENGTH = 1024
BATCH_SIZE = 1
PAD_ID = 0

VOCAB_SIZE = 8000
D_MODEL = 384
NUM_HEADS = 6
NUM_LAYERS = 6
D_FF = 1536
DROPOUT = 0.1

LEARNING_RATE = 3e-4


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


print("=" * 60)
print("STEP 24B - SINGLE TRAINING STEP TEST")
print("=" * 60)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()
print("Loading dataset...")

dataset = TokenDataset(TRAIN_FILE)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(f"Training sequences: {len(dataset):,}")
print(f"Batch size: {BATCH_SIZE}")

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

print("Model created.")

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(f"Parameters: {parameter_count:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)

print()
print("Getting one batch...")

input_ids, target_ids = next(iter(loader))

input_ids = input_ids.to(device)
target_ids = target_ids.to(device)

print(f"Input shape:  {tuple(input_ids.shape)}")
print(f"Target shape: {tuple(target_ids.shape)}")

print()
print("Running forward pass...")

model.train()

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

optimizer.zero_grad(set_to_none=True)

with torch.autocast(
    device_type=device.type,
    dtype=torch.float16,
    enabled=device.type == "cuda"
):

    logits = model(input_ids)

    loss = criterion(
        logits.reshape(-1, VOCAB_SIZE),
        target_ids.reshape(-1)
    )

print(f"Loss: {loss.item():.4f}")

print()
print("Running backward pass...")

loss.backward()

print("Backward pass completed.")

print()
print("Updating model weights...")

optimizer.step()

print("Optimizer step completed.")

if device.type == "cuda":

    peak_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print()
    print("GPU MEMORY")
    print("-" * 30)
    print(f"Peak allocated: {peak_memory:.2f} MB")

print()
print("=" * 60)
print("FINAL CHECK")
print("=" * 60)

if torch.isfinite(loss):
    print("PASS: Loss is finite.")
else:
    print("FAIL: Loss contains NaN or Inf.")

if parameter_count == 14_120_768:
    print("PASS: Parameter count is correct.")
else:
    print("FAIL: Parameter count is incorrect.")

print()
print("PASS: SINGLE TRAINING STEP COMPLETED.")