# MEYLO Checkpoints

This directory houses trained checkpoints for the custom **FromScratchLM** language model.

## Active Checkpoint Location
```
checkpoints/ultimate_improvement/best.pt
```

## Model Architecture Specifications
- **Architecture**: FromScratchLM (Transformer Decoder)
- **Parameters**: 38,207,232 (38.2M)
- **Vocabulary Size**: 16,384 tokens (custom Byte-Pair Encoding)
- **Embedding Dimension ($d_{model}$)**: 512
- **Transformer Layers**: 8
- **Attention Heads**: 8 (64 dim/head)
- **Feed-Forward Dimension**: 2048
- **Context Length**: 512 tokens
- **Precision**: Float32 / BFloat16

> **Note**: PyTorch weights (`*.pt`) exceed GitHub's 100 MB file boundary (~161 MB each) and are excluded from Git commits. To restore trained weights, place your `best.pt` file into `checkpoints/ultimate_improvement/best.pt`.
