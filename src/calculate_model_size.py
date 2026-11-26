VOCAB_SIZE = 8000
CONTEXT_LENGTH = 1024

NUM_LAYERS = 6
D_MODEL = 384
NUM_HEADS = 6
D_FF = 1536


def linear_params(input_size, output_size):
    return input_size * output_size + output_size


print("=" * 60)
print("TRANSFORMER MODEL SIZE CALCULATOR")
print("=" * 60)

print()
print("Architecture")
print("-" * 30)

print(f"Vocabulary size:     {VOCAB_SIZE:,}")
print(f"Context length:      {CONTEXT_LENGTH:,}")
print(f"Layers:              {NUM_LAYERS}")
print(f"Hidden size:         {D_MODEL}")
print(f"Attention heads:     {NUM_HEADS}")
print(f"Head dimension:      {D_MODEL // NUM_HEADS}")
print(f"FFN dimension:       {D_FF}")

print()
print("Checking attention dimensions...")

if D_MODEL % NUM_HEADS != 0:
    raise ValueError(
        "D_MODEL must be divisible by NUM_HEADS."
    )

print("PASS")

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

token_embedding = VOCAB_SIZE * D_MODEL
position_embedding = CONTEXT_LENGTH * D_MODEL

# --------------------------------------------------
# One Transformer block
# --------------------------------------------------

# Q, K, V and output projections
attention = (
    linear_params(D_MODEL, D_MODEL)
    + linear_params(D_MODEL, D_MODEL)
    + linear_params(D_MODEL, D_MODEL)
    + linear_params(D_MODEL, D_MODEL)
)

# Feed-forward network
ffn = (
    linear_params(D_MODEL, D_FF)
    + linear_params(D_FF, D_MODEL)
)

# Two LayerNorms
layer_norms = (
    2 * (2 * D_MODEL)
)

block_params = attention + ffn + layer_norms

all_blocks = NUM_LAYERS * block_params

# Final LayerNorm
final_norm = 2 * D_MODEL

# --------------------------------------------------
# Output head
# --------------------------------------------------

output_head = VOCAB_SIZE * D_MODEL + VOCAB_SIZE

# Untied embeddings
untied_total = (
    token_embedding
    + position_embedding
    + all_blocks
    + final_norm
    + output_head
)

# Tied embeddings
# Output weight reuses token embedding weights.
tied_total = (
    token_embedding
    + position_embedding
    + all_blocks
    + final_norm
    + VOCAB_SIZE
)

print()
print("=" * 60)
print("PARAMETER BREAKDOWN")
print("=" * 60)

print(f"Token embedding:     {token_embedding:,}")
print(f"Position embedding:  {position_embedding:,}")
print(f"One Transformer:     {block_params:,}")
print(f"All Transformer:     {all_blocks:,}")
print(f"Final LayerNorm:     {final_norm:,}")
print(f"Output head:         {output_head:,}")

print()
print("=" * 60)
print("TOTAL PARAMETERS")
print("=" * 60)

print(
    f"Untied embeddings:   "
    f"{untied_total:,} "
    f"({untied_total / 1_000_000:.2f}M)"
)

print(
    f"Tied embeddings:     "
    f"{tied_total:,} "
    f"({tied_total / 1_000_000:.2f}M)"
)

print()
print("=" * 60)
print("FP32 PARAMETER MEMORY")
print("=" * 60)

untied_fp32 = untied_total * 4
tied_fp32 = tied_total * 4

print(
    f"Untied: {untied_fp32 / (1024 ** 2):.2f} MB"
)

print(
    f"Tied:   {tied_fp32 / (1024 ** 2):.2f} MB"
)

print()
print("=" * 60)
print("FP16 PARAMETER MEMORY")
print("=" * 60)

untied_fp16 = untied_total * 2
tied_fp16 = tied_total * 2

print(
    f"Untied: {untied_fp16 / (1024 ** 2):.2f} MB"
)

print(
    f"Tied:   {tied_fp16 / (1024 ** 2):.2f} MB"
)

print()
print("=" * 60)
print("RECOMMENDATION")
print("=" * 60)

print(
    "For this first model, we will use tied embeddings."
)

print(
    "This reduces parameters while keeping the model architecture simple."
)