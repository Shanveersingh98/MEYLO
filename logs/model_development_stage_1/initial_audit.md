# INITIAL PROJECT & ENVIRONMENT AUDIT
**Project Directory**: `D:\DEEP-LEARNING-PROJECT`  
**Execution Timestamp**: 2026-11-13T12:58:23Z  
**Audit Stage**: Stage 1 — Model Audit, Independent Evaluation & Training Readiness  
**Author**: Senior Machine Learning & PyTorch Auditor  

---

## 1. Executive Summary

This initial audit establishes the forensic baseline for the from-scratch deep-learning chatbot project. The project hosts an autoregressive language model (`FromScratchLM`), a custom Byte-Pair Encoding (BPE) tokenizer, multiple training checkpoints across developmental phases, raw and tokenized datasets, and a hybrid inference application that integrates the local neural network with web search and deterministic tools.

In accordance with strict **Data-Preservation Rules**, zero existing model checkpoints, datasets, tokenizers, or source files have been deleted, overwritten, or modified. A full timestamped source backup was compiled and preserved prior to audit evaluations.

---

## 2. Hardware and Software Environment Verification

| Component | Target / Historical Specification | Verified Actual State | Status |
| :--- | :--- | :--- | :--- |
| **Operating System** | Windows 11 | Windows 11 (build 10.0.26100) | ✓ Verified |
| **Python Runtime** | Python 3.12 (`.venv\Scripts\python.exe`) | Python 3.12.5 (64-bit virtualenv) | ✓ Verified |
| **Deep Learning Framework** | PyTorch with CUDA | PyTorch 2.4.1+cu124 | ✓ Verified |
| **Compute Device** | NVIDIA RTX 3050 Laptop GPU (6 GB VRAM) | NVIDIA GeForce RTX 3050 Laptop GPU (6144 MB) | ✓ Verified |
| **System Memory** | ~8 GB System RAM | 7.82 GB Physical RAM available | ✓ Verified |
| **Git Repository** | Version control tracking | Not a Git repository (`.git` directory absent) | ⚠ Note |

> [!NOTE]
> Git is not initialized in this workspace (`fatal: not a git repository`). Consequently, version tracking and rollback rely entirely on snapshot backups. To enforce strict data preservation, a dedicated zip backup has been generated under `logs/model_development_stage_1/backups/`.

---

## 3. Timestamped Source Backup

- **Backup Archive**: `D:\DEEP-LEARNING-PROJECT\logs\model_development_stage_1\backups\source_backup_20261113_182820.zip`
- **Compressed Size**: 381,639 bytes
- **Contents**: Full recursive snapshot of `src/` (excluding `__pycache__`) and `tokenizer/tokenizer.json`.
- **Integrity**: Verified readable and non-destructive.

---

## 4. Key Source Code Inventory & SHA-256 Hashes

| Relative File Path | Size (Bytes) | SHA-256 Hash | Last Modified |
| :--- | :--- | :--- | :--- |
| `src\model.py` | 6,231 | `17f38407dad461d48260ff100843df02ca89fe73e81bab0d8a625126b5e817ae` | 2026-09-13 18:51 |
| `src\generate.py` | 6,395 | `3173c938b925336557dc1e0d01a7b0a587ad9f8becf9908b7fa85e8902cdcf36` | 2026-10-27 17:58 |
| `src\chat.py` | 22,225 | `ab76cab48457c21a6eb53bed7ad720bb018afadad64f98a316a8f3a315b0f506` | 2026-10-31 18:36 |
| `src\web_retrieval.py` | 83,214 | `df13070bbcb0b3541cea7550b97d2fb2dda589efce01a84f04df5194a0b8429a` | 2026-10-31 18:40 |
| `src\answer_synthesis.py` | 39,352 | `7eb82a30d01dc2acda668206e81546d5dedaffea45ac9d5b06cbc37d115bda62` | 2026-10-31 18:43 |
| `src\intent_router.py` | 6,212 | `71a6faf46bd54a4cfe529763e015eb283ba3b0a5c7e0cc05f5bc3db6c9b6c32b` | 2026-10-31 18:20 |
| `src\question_understanding.py` | 40,515 | `54213d00c7baf85b4960104339655c313e018ecf4a263665c0400c6047b19d04` | 2026-10-31 18:45 |
| `src\conversation_memory.py` | 12,577 | `5dc154a49c3f676d4b5b8f1c95ae35aedf54a96780d7833de31203a684080dbd` | 2026-10-31 18:18 |
| `src\source_selection.py` | 33,383 | `f5182634c2c62a80b01f63841422692c620887f6dc2fcf55c687d1ef0d384a2c` | 2026-10-31 18:35 |
| `src\train.py` | 18,730 | `ea0a263e326366ee008dcb4b48ee141144b5177d295af429c789d3fbf302c01e` | 2026-10-07 12:50 |
| `src\build_eos_v2_sequences.py` | 16,276 | `9f878263805bac3f9afb8c54b11f6b3db346f89d90d026fd8bd44f03844549fc` | 2026-10-13 21:39 |
| `src\ultimate_improvement\train_ultimate.py` | 21,339 | `464c3dbf2c79acd88c219eb39d953ccb12d94c671441f18bb5ac8231aac2c468` | 2026-10-27 12:16 |
| `src\ultimate_improvement\generate_ultimate_curriculum.py` | 26,720 | `86163f6d9f8f5bcd7230f0ee0dea822a65385a9e0c732062ff0642659b4a9815` | 2026-10-27 12:14 |
| `src\experiment_2\train_experiment_2.py` | 23,008 | `e7b45eb11c30ea1a4b3efa712fdef1771533f0075e42b8c98f1a4a0504c6ecf5` | 2026-10-19 22:45 |
| `tokenizer\tokenizer.json` | 540,619 | `0f00fae1c3852879cb0bb9a807b45aad93d7700ca4e3f903b1c607ac8c565b08` | 2026-09-09 18:23 |

---

## 5. Checkpoint Inventory & Integrity Analysis

All checkpoint `.pt` files were verified on disk. State dictionaries were inspected with PyTorch for structural integrity, parameter counts, and configuration fidelity.

| Checkpoint Identifier | File Size (Bytes) | SHA-256 Hash | Recorded Update / Step | Recorded Val Loss | Status & Keys |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`checkpoints\ultimate_improvement\best.pt`** *(Active)* | 169,528,553 | `cecba4946a8b5a0404e42e9e1790c0d780228da45e07b1910d190d89cbadf2fc` | Update 1,500 | 2.1963 | ✓ 77 keys, 0 missing, 0 unexpected |
| `checkpoints\ultimate_improvement\latest.pt` | 169,537,949 | `9eb9b09c6f95acb257f86af3dc070f73ad465785cfda0c071b2d4f1d1cfb9bd2` | Update 1,500 | 2.1963 | ✓ 77 keys, 0 missing, 0 unexpected |
| `checkpoints\experiment_2\best.pt` | 169,528,489 | `f717e5efff9dc6d58b0fc1d4872e74b0b03bd734029715a3ae14aa549b5b4ab1` | Update 1,750 | 3.6421 | ✓ 77 keys, 0 missing, 0 unexpected |
| `checkpoints\experiment_2\latest.pt` | 169,537,949 | `c6e57aaa73a2c558c91ff2be8cfeb3960097dba00071f475a31a2650e9c344f4` | Update 1,750 | 3.6421 | ✓ 77 keys, 0 missing, 0 unexpected |
| `checkpoints\eos_v2_pilot\best.pt` | 169,528,681 | `a55faea054b393c890793e128f70f1c4fde2254d21ade1ab8407e0a36bff491e` | Update 5,000 | 3.9620 | ✓ 77 keys, 0 missing, 0 unexpected |
| `checkpoints\best.pt` *(Original baseline)* | 169,529,193 | `8fc52d223f83255e12bee57838f820f279a8942546cbff63887d4e2e872db7f5` | Step 29,300 | 3.7745 | ✓ 77 keys, 0 missing, 0 unexpected |
| `checkpoints\latest.pt` | 169,538,717 | `528210f60c2120e6a2c9e53f100a282dcebb265be39958756f453a791a2269c0` | Step 30,000 | 3.7820 | ✓ 77 keys, 0 missing, 0 unexpected |
| `checkpoints\masked_best.pt` | 169,545,087 | `9dd2226561887013a2ddaa8b689f3e83db38fcb8b641d43ebf3880bebe39819b` | Step 10,000 | 4.1205 | ✓ 77 keys, 0 missing, 0 unexpected |
| `checkpoints\clean_50k_best.pt` | 169,546,157 | `1cd3306e4ca76d7efc3a913a8ff4b87c0c9e624b0d4176806f7feebdfb7ef995` | Step 50,000 | 3.8112 | ✓ 77 keys, 0 missing, 0 unexpected |

---

## 6. Tokenizer Verification

The project tokenizer file is located at `D:\DEEP-LEARNING-PROJECT\tokenizer\tokenizer.json`.

- **Model Type**: Byte-Pair Encoding (BPE)
- **Pre-tokenizer**: ByteLevel (add_prefix_space=False, trim_offsets=True, use_regex=True)
- **Decoder**: ByteLevel (add_prefix_space=True, trim_offsets=True, use_regex=True)
- **Vocabulary Size**: 8,000 tokens
- **BPE Merges**: 7,791 merge rules
- **Special Tokens & Exact IDs**:
  - `[PAD]` = `0` (Padding token)
  - `[UNK]` = `1` (Unknown token)
  - `[BOS]` = `2` (Beginning of sequence)
  - `[EOS]` = `3` (End of sequence)
  - `<|user|>` = `4` (User turn marker)
  - `<|assistant|>` = `5` (Assistant turn marker)
- **Round-Trip Integrity**: Tested on conversational and technical strings. Decoding perfectly reconstructs input strings without character loss or invalid unicode sequences.

---

## 7. Audit Readiness Checklist

- [x] Workspace environment verified (Windows 11, CUDA 12.4, RTX 3050 6GB).
- [x] No Git repository present; manual rollback protections established via timestamped zip backup.
- [x] All 9 historical checkpoints preserved and hashed.
- [x] Tokenizer verified with exact special token IDs.
- [x] Dedicated output directory `logs\model_development_stage_1\` created for all stage 1 artifacts.
- [x] Ready to proceed to Phase One (Architecture Audit) and Phase Two (Independent Neural Model Evaluation).
