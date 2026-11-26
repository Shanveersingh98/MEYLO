# Comprehensive Environment & Infrastructure Report

**Project Root:** `D:\DEEP-LEARNING-PROJECT`  
**Execution Host:** Windows 11 AMD64  
**Date:** 2026-10-27  

---

## 1. Verified Hardware Specifications

| Component | Specification | Operational Status |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop GPU | Active, CUDA-enabled |
| **CUDA Compute Capability** | 8.6 | Supported |
| **Dedicated VRAM** | 6.00 GB (6,144 MB) | Peak usage during training: **~1.85 GB** (30.8% of capacity) |
| **CPU Architecture** | x86_64 / AMD64 | 64-bit multi-core |
| **System Storage (D:)** | 146.48 GB Total, ~134.5 GB Free | Ample storage capacity |

---

## 2. Verified Software & Virtual Environment

| Software Layer | Version / Path | Verification Status |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Build 22631 | Verified |
| **Python Binary** | `D:\DEEP-LEARNING-PROJECT\.venv\Scripts\python.exe` | Python 3.12.10 (AMD64) |
| **PyTorch Version** | 2.14.0+cu130 | CUDA 13.0 Runtime active |
| **Tokenizers Library**| HuggingFace Tokenizers 0.21.0 | Byte-Pair Encoding verified |
| **NumPy Version** | 2.1.2 | Active |

---

## 3. Directory Layout & Storage Allocation

```
D:\DEEP-LEARNING-PROJECT\
├── .venv\                           # Isolated virtual environment (Python 3.12)
├── checkpoints\
│   ├── experiment_2\
│   │   ├── best.pt                  # PROTECTED REFERENCE CHECKPOINT (SHA-256: f717e5ef...)
│   │   └── latest.pt
│   ├── eos_v2_pilot\                # Protected historical checkpoints
│   └── ultimate_improvement\
│       ├── best.pt                  # NEW OPTIMAL CHECKPOINT (SHA-256: cecba494...)
│       └── latest.pt
├── data\
│   ├── experiment_2\                # Historical Experiment 2 datasets
│   └── ultimate_improvement\
│       ├── curriculum_raw\          # Raw JSONL curriculum files
│       ├── eval_suite\              # 12 Independent evaluation suites
│       └── training_sequences\      # Binary memmapped tokens and masks
├── logs\
│   ├── experiment_2\                # Historical Experiment 2 logs
│   ├── eos_v2_pilot\                # Historical Pilot logs
│   └── ultimate_improvement\        # All deliverables and logs for current mission
├── src\
│   ├── model.py                     # FromScratchLM PyTorch Transformer architecture
│   ├── generate.py                  # Single-prompt test inference script (Updated & Verified)
│   ├── chat.py                      # Interactive multi-turn CLI application (New)
│   ├── experiment_2\                # Historical experiment scripts
│   └── ultimate_improvement\        # Training, evaluation, and auditing scripts
└── tokenizer\
    └── tokenizer.json               # 8,000 vocab BPE tokenizer (SHA-256: 0f00fae1...)
```

---

## 4. Resource Utilization Profiling

1. **Training Resource Footprint:**
   - Effective Batch Size: 16 sequences (microbatch 1 $\times$ 16 gradient accumulation steps).
   - Peak GPU Memory Allocated: **1,852.4 MB** (~1.81 GB).
   - Speed: ~600 ms per optimizer update.
   - Total 1,500 Updates Wall-Clock Time: **15.15 minutes**.
2. **Inference Resource Footprint:**
   - GPU Memory Allocated: ~120 MB for weights + ~50 MB KV activation buffers.
   - Per-Token Latency: ~12 ms per token on RTX 3050 GPU.
   - CPU Inference: Verified fully functional with torch.float32 fallback.
