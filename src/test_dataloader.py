
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

TRAIN_FILE = Path('D:/DEEP-LEARNING-PROJECT/data/processed/training_sequences/train.bin')
VALIDATION_FILE = Path('D:/DEEP-LEARNING-PROJECT/data/processed/training_sequences/validation.bin')

CONTEXT_LENGTH = 1024
BATCH_SIZE = 2
PAD_ID = 0

class TokenDataset(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path

        self.data = np.memmap(
            file_path,
            dtype=np.uint16,
            mode='r'
        )

        if len(self.data) % CONTEXT_LENGTH != 0:
            raise ValueError(
                'Dataset size is not divisible by context length.'
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

        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        target_ids = torch.tensor(sequence[1:], dtype=torch.long)

        return input_ids, target_ids


print('=' * 60)
print('STEP 24A - DATALOADER TEST')
print('=' * 60)

print()
print('Loading training dataset...')

train_dataset = TokenDataset(TRAIN_FILE)

print(f'Training sequences: {len(train_dataset):,}')

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(f'Batch size: {BATCH_SIZE}')

print()
print('Getting one training batch...')

input_ids, target_ids = next(iter(train_loader))

print(f'Input shape:  {tuple(input_ids.shape)}')
print(f'Target shape: {tuple(target_ids.shape)}')

print()
print('Input dtype:', input_ids.dtype)
print('Target dtype:', target_ids.dtype)

print()
print('First input sequence:')
print(input_ids[0].tolist()[:30])

print()
print('First target sequence:')
print(target_ids[0].tolist()[:30])

print()
print('Checking padding...')

input_padding = int((input_ids == PAD_ID).sum())
target_padding = int((target_ids == PAD_ID).sum())

print(f'Padding in input:  {input_padding}')
print(f'Padding in target: {target_padding}')

print()
print('=' * 60)
print('FINAL CHECK')
print('=' * 60)

if input_ids.shape == (BATCH_SIZE, CONTEXT_LENGTH - 1):
    print('PASS: Input shape is correct.')
else:
    print('FAIL: Input shape is incorrect.')

if target_ids.shape == (BATCH_SIZE, CONTEXT_LENGTH - 1):
    print('PASS: Target shape is correct.')
else:
    print('FAIL: Target shape is incorrect.')

if input_ids.dtype == torch.long:
    print('PASS: Input dtype is correct.')
else:
    print('FAIL: Input dtype is incorrect.')

if target_ids.dtype == torch.long:
    print('PASS: Target dtype is correct.')
else:
    print('FAIL: Target dtype is incorrect.')

print()
print('PASS: STEP 24A DATALOADER TEST COMPLETE.')