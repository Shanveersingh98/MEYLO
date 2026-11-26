import sys
import torch


sys.path.insert(
    0,
    "D:/DEEP-LEARNING-PROJECT/src"
)

from model import FromScratchLM


# --------------------------------------------------
# Configuration
# --------------------------------------------------

VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024

BATCH_SIZE = 2

EXPECTED_PARAMETERS = 14_120_768


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("FROM-SCRATCH TRANSFORMER TEST")
print("=" * 60)

print()
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# --------------------------------------------------
# Create model
# --------------------------------------------------

print()
print("Creating model...")

model = FromScratchLM(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    d_model=384,
    num_heads=6,
    num_layers=6,
    d_ff=1536,
    dropout=0.1,
)

model = model.to(device)

print("Model created successfully.")


# --------------------------------------------------
# Count parameters
# --------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print()
print("=" * 60)
print("PARAMETER CHECK")
print("=" * 60)

print(
    f"Total parameters:      "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters:  "
    f"{trainable_parameters:,}"
)

print(
    f"Expected parameters:   "
    f"{EXPECTED_PARAMETERS:,}"
)

difference = (
    total_parameters - EXPECTED_PARAMETERS
)

print(
    f"Difference:            "
    f"{difference:,}"
)


# --------------------------------------------------
# Create random input
# --------------------------------------------------

print()
print("=" * 60)
print("FORWARD PASS TEST")
print("=" * 60)

input_ids = torch.randint(
    low=0,
    high=VOCAB_SIZE,
    size=(BATCH_SIZE, CONTEXT_LENGTH),
    device=device,
    dtype=torch.long,
)

print(
    f"Input shape:           "
    f"{tuple(input_ids.shape)}"
)

print(
    f"Input device:          "
    f"{input_ids.device}"
)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

model.eval()

with torch.no_grad():

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    logits = model(input_ids)


# --------------------------------------------------
# Output checks
# --------------------------------------------------

expected_shape = (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    VOCAB_SIZE
)

print()
print(
    f"Output shape:          "
    f"{tuple(logits.shape)}"
)

print(
    f"Expected shape:        "
    f"{expected_shape}"
)

print(
    f"Output dtype:          "
    f"{logits.dtype}"
)

print(
    f"Output device:         "
    f"{logits.device}"
)


# --------------------------------------------------
# Numerical checks
# --------------------------------------------------

has_nan = torch.isnan(logits).any().item()
has_inf = torch.isinf(logits).any().item()

print()
print("Numerical checks")
print("-" * 30)

print(f"Contains NaN:          {has_nan}")
print(f"Contains Inf:          {has_inf}")

# --------------------------------------------------
# GPU memory
# --------------------------------------------------

if device.type == "cuda":

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print()
    print("=" * 60)
    print("GPU MEMORY")
    print("=" * 60)

    print(
        f"Currently allocated:   "
        f"{allocated:.2f} MB"
    )

    print(
        f"Currently reserved:    "
        f"{reserved:.2f} MB"
    )

    print(
        f"Peak allocated:        "
        f"{peak:.2f} MB"
    )


# --------------------------------------------------
# Final verification
# --------------------------------------------------

print()
print("=" * 60)
print("FINAL TEST")
print("=" * 60)

parameter_ok = (
    total_parameters == EXPECTED_PARAMETERS
)

shape_ok = (
    tuple(logits.shape) == expected_shape
)

numeric_ok = (
    not has_nan and not has_inf
)

if parameter_ok:
    print("PASS: Parameter count correct.")
else:
    print("FAIL: Parameter count mismatch.")

if shape_ok:
    print("PASS: Output shape correct.")
else:
    print("FAIL: Output shape mismatch.")

if numeric_ok:
    print("PASS: No NaN or Inf detected.")
else:
    print("FAIL: NaN or Inf detected.")

if parameter_ok and shape_ok and numeric_ok:
    print()
    print("PASS: TRANSFORMER FORWARD PASS SUCCESSFUL.")
else:
    print()
    print("FAIL: Transformer test failed.")