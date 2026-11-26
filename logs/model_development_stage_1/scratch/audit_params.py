"""
Architecture & Checkpoint Audit Script
Computes exact parameter counts and inspects all checkpoints.
"""
import sys
import os
import json
import hashlib

sys.path.insert(0, r"D:\DEEP-LEARNING-PROJECT\src")

import torch
from model import FromScratchLM

# ============================================================
# 1. Parameter Count Analysis
# ============================================================
print("=" * 70)
print("PARAMETER COUNT ANALYSIS")
print("=" * 70)

model = FromScratchLM(
    vocab_size=8000,
    context_length=1024,
    d_model=384,
    num_heads=6,
    num_layers=6,
    d_ff=1536,
    dropout=0.1,
)

total_params = 0
trainable_params = 0

# Detailed breakdown
print("\n--- Embedding Layers ---")
for name, param in model.named_parameters():
    if 'embedding' in name or 'output_bias' in name:
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
        print(f"  {name}: shape={list(param.shape)}, params={count:,}")

print("\n--- Transformer Blocks ---")
block_total = 0
for block_idx in range(6):
    block_params = 0
    print(f"\n  Block {block_idx}:")
    for name, param in model.named_parameters():
        if name.startswith(f"blocks.{block_idx}."):
            count = param.numel()
            block_params += count
            short_name = name.replace(f"blocks.{block_idx}.", "  ")
            print(f"    {short_name}: shape={list(param.shape)}, params={count:,}")
    print(f"  Block {block_idx} total: {block_params:,}")
    block_total += block_params

print(f"\n  All blocks total: {block_total:,}")

print("\n--- Final LayerNorm ---")
final_norm_params = 0
for name, param in model.named_parameters():
    if 'final_norm' in name:
        count = param.numel()
        final_norm_params += count
        print(f"  {name}: shape={list(param.shape)}, params={count:,}")

# Recount total properly
total_from_model = sum(p.numel() for p in model.parameters())
trainable_from_model = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "=" * 70)
print(f"Total parameters: {total_from_model:,}")
print(f"Trainable parameters: {trainable_from_model:,}")
print(f"Non-trainable parameters: {total_from_model - trainable_from_model:,}")

# Breakdown by component
emb_params = model.token_embedding.weight.numel() + model.position_embedding.weight.numel()
output_bias_params = model.output_bias.numel()
block_params_check = sum(p.numel() for block in model.blocks for p in block.parameters())
final_norm_total = sum(p.numel() for p in model.final_norm.parameters())

print("\n--- Summary Breakdown ---")
print(f"  Token Embedding:      {model.token_embedding.weight.numel():>12,}  (shape: {list(model.token_embedding.weight.shape)})")
print(f"  Position Embedding:   {model.position_embedding.weight.numel():>12,}  (shape: {list(model.position_embedding.weight.shape)})")
print(f"  Transformer Blocks:   {block_params_check:>12,}  (6 blocks)")
print(f"  Final LayerNorm:      {final_norm_total:>12,}  (shape: weight={list(model.final_norm.weight.shape)}, bias={list(model.final_norm.bias.shape)})")
print(f"  Output Bias:          {output_bias_params:>12,}  (shape: {list(model.output_bias.shape)})")
print(f"  ---")
print(f"  Sum:                  {emb_params + output_bias_params + block_params_check + final_norm_total:>12,}")
print(f"  model.parameters():   {total_from_model:>12,}")

# Note: weight tying means token_embedding.weight is reused in output projection
# so it is NOT double-counted in parameter count
print("\n--- Weight Tying Note ---")
print("  Output projection reuses token_embedding.weight via F.linear()")
print("  No separate output weight matrix exists — only output_bias is added")
print(f"  This saves {8000 * 384:,} parameters vs. a separate output projection layer")

# Per-block detailed breakdown
print("\n--- Per-Component in One Transformer Block ---")
block0 = model.blocks[0]
components = {
    "norm1 (pre-attention LN)": block0.norm1,
    "attention.qkv (combined Q/K/V)": block0.attention.qkv,
    "attention.out_proj": block0.attention.out_proj,
    "norm2 (pre-FFN LN)": block0.norm2,
    "feed_forward.network.0 (Linear d_model→d_ff)": block0.feed_forward.network[0],
    "feed_forward.network.2 (Linear d_ff→d_model)": block0.feed_forward.network[2],
}
for comp_name, comp in components.items():
    p = sum(p.numel() for p in comp.parameters())
    print(f"  {comp_name}: {p:,}")

# ============================================================
# 2. Checkpoint Inspection
# ============================================================
print("\n" + "=" * 70)
print("CHECKPOINT INSPECTION")
print("=" * 70)

checkpoint_paths = [
    r"D:\DEEP-LEARNING-PROJECT\checkpoints\ultimate_improvement\best.pt",
    r"D:\DEEP-LEARNING-PROJECT\checkpoints\experiment_2\best.pt",
    r"D:\DEEP-LEARNING-PROJECT\checkpoints\eos_v2_pilot\best.pt",
    r"D:\DEEP-LEARNING-PROJECT\checkpoints\best.pt",
]

for ckpt_path in checkpoint_paths:
    print(f"\n--- {os.path.relpath(ckpt_path, r'D:\DEEP-LEARNING-PROJECT')} ---")
    
    if not os.path.exists(ckpt_path):
        print(f"  FILE NOT FOUND!")
        continue
    
    file_size = os.path.getsize(ckpt_path)
    print(f"  File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    print(f"  Top-level keys: {list(ckpt.keys())}")
    
    # Config
    if 'config' in ckpt:
        config = ckpt['config']
        print(f"  Config: {json.dumps(config, indent=4)}")
    
    # Training info
    for key in ['step', 'optimizer_update', 'epoch', 'validation_loss', 'train_loss', 'best_val_loss', 'learning_rate']:
        if key in ckpt:
            print(f"  {key}: {ckpt[key]}")
    
    # Model state dict analysis
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
        print(f"  State dict keys count: {len(sd)}")
        total_sd_params = sum(v.numel() for v in sd.values())
        print(f"  State dict total params: {total_sd_params:,}")
        
        # Verify loading
        test_model = FromScratchLM(
            vocab_size=config.get('vocab_size', 8000),
            context_length=config.get('context_length', 1024),
            d_model=config.get('d_model', 384),
            num_heads=config.get('num_heads', 6),
            num_layers=config.get('num_layers', 6),
            d_ff=config.get('d_ff', 1536),
            dropout=config.get('dropout', 0.0),
        )
        
        # Check for missing/unexpected keys
        result = test_model.load_state_dict(sd, strict=False)
        if result.missing_keys:
            print(f"  ⚠ MISSING KEYS: {result.missing_keys}")
        else:
            print(f"  ✓ No missing keys")
        if result.unexpected_keys:
            print(f"  ⚠ UNEXPECTED KEYS: {result.unexpected_keys}")
        else:
            print(f"  ✓ No unexpected keys")
    
    # Optimizer state dict
    if 'optimizer_state_dict' in ckpt:
        opt_sd = ckpt['optimizer_state_dict']
        print(f"  Optimizer state dict present: Yes (param_groups: {len(opt_sd.get('param_groups', []))})")
    else:
        print(f"  Optimizer state dict present: No")
    
    del ckpt

# ============================================================
# 3. Tokenizer Round-Trip Test
# ============================================================
print("\n" + "=" * 70)
print("TOKENIZER ROUND-TRIP TEST")
print("=" * 70)

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(r"D:\DEEP-LEARNING-PROJECT\tokenizer\tokenizer.json")

print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Verify special token IDs
special_tokens = {
    "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
    "<|user|>": 4, "<|assistant|>": 5
}
print("\nSpecial Token Verification:")
all_ok = True
for token, expected_id in special_tokens.items():
    actual_id = tokenizer.token_to_id(token)
    status = "✓" if actual_id == expected_id else "✗"
    if actual_id != expected_id:
        all_ok = False
    print(f"  {status} {token}: expected={expected_id}, actual={actual_id}")

# Round-trip test
test_strings = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is fascinating.",
    "[BOS]\n<|user|>\nWhat is AI?\n<|assistant|>\nAI stands for Artificial Intelligence.\n[EOS]",
]

print("\nRound-Trip Encoding/Decoding:")
for s in test_strings:
    encoded = tokenizer.encode(s)
    decoded = tokenizer.decode(encoded.ids)
    match = "✓" if decoded.strip() == s.strip() or decoded == s else "≈"
    print(f"  {match} Original:  {repr(s[:60])}")
    print(f"    Encoded:  {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''} ({len(encoded.ids)} tokens)")
    print(f"    Decoded:  {repr(decoded[:60])}")

# Get merges count by reading JSON directly
import json
with open(r"D:\DEEP-LEARNING-PROJECT\tokenizer\tokenizer.json", 'r', encoding='utf-8') as f:
    tok_json = json.load(f)
vocab_count = len(tok_json['model']['vocab'])
merges_count = len(tok_json['model']['merges'])
added_tokens_count = len(tok_json['added_tokens'])
print(f"\nTokenizer JSON metadata:")
print(f"  Vocab entries: {vocab_count}")
print(f"  BPE merges: {merges_count}")
print(f"  Added tokens (special): {added_tokens_count}")
print(f"  Model type: {tok_json['model']['type']}")
print(f"  Pre-tokenizer: {tok_json['pre_tokenizer']['type']}")
print(f"  Decoder: {tok_json['decoder']['type']}")
print(f"  UNK token: {tok_json['model']['unk_token']}")
print(f"  Byte fallback: {tok_json['model']['byte_fallback']}")

print("\nDone.")
