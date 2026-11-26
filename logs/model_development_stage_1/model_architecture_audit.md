# COMPLETE NEURAL MODEL ARCHITECTURE AUDIT
**Model Class**: `FromScratchLM` (`src/model.py`)  
**Active Checkpoint**: `checkpoints\ultimate_improvement\best.pt`  
**Evaluation Date**: 2026-11-13  
**Auditor**: Senior Machine Learning & PyTorch Architecture Specialist  

---

## 1. Architectural Blueprint & Parameter Count

The local language model `FromScratchLM` is a decoder-only, autoregressive transformer implementing Pre-Layer Normalization (Pre-LN), scaled dot-product attention with causal masking, and tied token embedding and output projection weights.

### Exact Parameter Count Verification

Every parameter tensor was extracted, inspected, and verified against PyTorch's computational graph:

| Component | Layer / Tensor Name | Tensor Shape | Parameter Count | Trainable? |
| :--- | :--- | :--- | :--- | :--- |
| **Token Embeddings** | `token_embedding.weight` | `[8000, 384]` | 3,072,000 | Yes |
| **Position Embeddings** | `position_embedding.weight` | `[1024, 384]` | 393,216 | Yes |
| **Output Bias** | `output_bias` | `[8000]` | 8,000 | Yes |
| **Final Normalization** | `final_norm.weight`<br>`final_norm.bias` | `[384]`<br>`[384]` | 384<br>384 | Yes<br>Yes |
| **Transformer Blocks (x6)** | 6 identical decoder blocks | (detailed below) | 10,646,784 | Yes |
| **TOTAL** | — | — | **14,120,768** | **14,120,768** |
| **Non-Trainable** | — | — | **0** | — |

> [!IMPORTANT]
> The historical parameter estimate of **14,120,768** is verified to the exact single parameter. All 14,120,768 parameters are active, non-frozen floating-point weights requiring gradient computation during training.

---

## 2. Transformer Block Dissection (Per Block)

The model stacks 6 identical `TransformerBlock` modules (`num_layers=6`). Each block consists of 1,774,464 parameters partitioned as follows:

```
Input x (batch, seq_len, 384)
  │
  ├──────────────────────────────────────────┐ (Residual Connection 1)
  ▼                                          │
norm1: LayerNorm(384) [768 params]           │
  ▼                                          │
attention: CausalSelfAttention [591,360]     │
  │  ├─ qkv: Linear(384 -> 1152) [443,520]   │
  │  ├─ F.scaled_dot_product_attention()     │
  │  ├─ out_proj: Linear(384 -> 384) [147,840]
  │  └─ resid_dropout: Dropout(0.1)          │
  ▼                                          │
  + ◄────────────────────────────────────────┘
  │
  ├──────────────────────────────────────────┐ (Residual Connection 2)
  ▼                                          │
norm2: LayerNorm(384) [768 params]           │
  ▼                                          │
feed_forward: FeedForward [1,181,568]        │
  │  ├─ Linear(384 -> 1536) [591,360]        │
  │  ├─ GELU() activation                    │
  │  ├─ Linear(1536 -> 384) [590,208]        │
  │  └─ Dropout(0.1)                         │
  ▼                                          │
  + ◄────────────────────────────────────────┘
  ▼
Output (batch, seq_len, 384)
```

### Detailed Tensor Breakdown per Block

1. **`norm1` (Pre-Attention LayerNorm)**:
   - `norm1.weight`: `[384]` (384 params)
   - `norm1.bias`: `[384]` (384 params)
   - Total: 768 params
2. **`attention.qkv` (Fused QKV Linear)**:
   - `weight`: `[1152, 384]` (442,368 params)
   - `bias`: `[1152]` (1,152 params)
   - Total: 443,520 params
3. **`attention.out_proj` (Attention Output Projection)**:
   - `weight`: `[384, 384]` (147,456 params)
   - `bias`: `[384]` (384 params)
   - Total: 147,840 params
4. **`norm2` (Pre-FFN LayerNorm)**:
   - `norm2.weight`: `[384]` (384 params)
   - `norm2.bias`: `[384]` (384 params)
   - Total: 768 params
5. **`feed_forward.network.0` (Up-Projection Linear)**:
   - `weight`: `[1536, 384]` (589,824 params)
   - `bias`: `[1536]` (1,536 params)
   - Total: 591,360 params
6. **`feed_forward.network.2` (Down-Projection Linear)**:
   - `weight`: `[384, 1536]` (589,824 params)
   - `bias`: `[384]` (384 params)
   - Total: 590,208 params

**Sum Per Block**: `768 + 443,520 + 147,840 + 768 + 591,360 + 590,208 = 1,774,464`.  
**Total for 6 Blocks**: `1,774,464 * 6 = 10,646,784`.

---

## 3. Structural & Algorithmic Audit Details

### 3.1 Embedding Configuration
- **Token Embedding**: Learned table of shape `[8000, 384]`.
- **Positional Encoding**: Learned absolute 1D positional embeddings of shape `[1024, 384]`.
- **Combination**: Element-wise addition `token_embeddings + position_embeddings`, followed by `self.dropout(x)`.
- **Evaluation**: The learned positional embeddings allow position-dependent features across the context length, but do not extrapolate beyond the trained 1024-token boundary.

### 3.2 Weight Tying (Embedding & Unembedding)
- Weight tying is implemented in `FromScratchLM.forward`:
  ```python
  logits = F.linear(
      x,
      self.token_embedding.weight,
      self.output_bias
  )
  ```
- **Technical Implication**: The output projection matrix directly reuses `token_embedding.weight`. No separate `nn.Linear(384, 8000)` layer exists. Instead, an independent learned 1D bias vector `self.output_bias` (`[8000]`) is added to the logits.
- **Parameter Savings**: Avoids adding `8000 * 384 = 3,072,000` additional parameters (~21.8% of the total model size), maintaining parameter efficiency within the 6 GB VRAM budget.

### 3.3 Attention Mechanism
- **Implementation**: Utilizes PyTorch's native `F.scaled_dot_product_attention` (SDPA).
- **Causal Masking**: Explicitly passed `is_causal=True`. This automatically invokes optimized CUDA FlashAttention or memory-efficient attention kernels where supported, eliminating manual creation of lower-triangular boolean mask tensors during inference.
- **Head Configuration**: `d_model=384`, `num_heads=6`, yielding `head_dim = 384 // 6 = 64`. This is mathematically standard and maps well onto NVIDIA GPU warp dimensions.

### 3.4 Feed-Forward Network
- **Expansion Ratio**: `d_model=384` expands to `d_ff=1536`, which is an exact $4\times$ standard expansion.
- **Activation Function**: `nn.GELU()`. Smooth, non-saturating non-linearity preventing dead neurons.
- **Linear Projections**: Both up-projection and down-projection include bias terms.

### 3.5 Normalization & Stability
- **Architecture**: Pre-LayerNorm (Pre-LN). The normalization is applied *prior* to entering the sub-layer (`norm1(x)` before attention, `norm2(x)` before FFN), and the residual connection bypasses the normalization.
- **Final Norm**: A dedicated `final_norm` (`nn.LayerNorm(384)`) is applied after the final transformer block, preceding the output projection.
- **Stability**: Pre-LN architectures provide superior gradient flow and enable training without complex warm-up schemes compared to legacy Post-LN transformers.

### 3.6 Context Window Enforcement
- In `FromScratchLM.forward(self, input_ids)`:
  ```python
  batch_size, sequence_length = input_ids.shape
  if sequence_length > self.context_length:
      raise ValueError(f"Sequence length {sequence_length} exceeds context length {self.context_length}.")
  ```
- Any sequence exceeding 1024 tokens triggers an explicit hard exception. In inference scripts (`generate.py` and `chat.py`), context truncation is enforced via slicing: `input_ids[:, -config["context_length"]:]`.

### 3.7 Parameter Initialization
- Linear layers: Weights sampled from $\mathcal{N}(0.0, 0.02^2)$; biases initialized to zero.
- Embedding layers: Sampled from $\mathcal{N}(0.0, 0.02^2)$.
- `output_bias`: Initialized to all zeros (`torch.zeros(vocab_size)`).

---

## 4. Active Checkpoint Verification (`ultimate_improvement\best.pt`)

The active production checkpoint was verified against the PyTorch model definition:

- **State Dict Size**: Exactly 77 keys.
- **Strict Loading Test**: `model.load_state_dict(checkpoint["model_state_dict"], strict=True)` passes with **zero missing keys** and **zero unexpected keys**.
- **Model Config Verification**:
  ```json
  {
      "vocab_size": 8000,
      "context_length": 1024,
      "d_model": 384,
      "num_heads": 6,
      "num_layers": 6,
      "d_ff": 1536,
      "dropout": 0.1,
      "peak_lr": 6e-05,
      "effective_batch_size": 16
  }
  ```
- **Recorded Training Progress**: Step / Update 1,500.
- **Optimizer State Present**: Yes, AdamW optimizer state dictionary included (param_groups: 1).

---

## 5. Architectural Findings & Assessment

1. **Clean Baseline Implementation**: The core `model.py` code is mathematically sound, adhering to modern transformer decoder practices (Pre-LN, SDPA with causal masking, GELU, weight tying).
2. **Missing RoPE / Modern Position Encodings**: The model relies on learned absolute positional embeddings (`nn.Embedding(1024, 384)`). Modern models (e.g., LLaMA, Mistral, Gemma) employ Rotary Position Embeddings (RoPE), which allow smoother length extrapolation and superior relative token distance awareness.
3. **Capacity Constraints**: With $d_{\text{model}}=384$ and 6 layers, the total non-embedding parameter capacity is only **10.65M parameters**. This restricts the amount of factual world knowledge the neural network can store in its weights, explaining why factual queries produce confabulations without external retrieval.
