"""
audit_pipeline_and_data.py
==========================
Comprehensive technical audit of training data pipelines, loss calculation,
validation integrity, duplicate leakage, and dataset distributions.

Computes:
1. Exact independent validation loss of checkpoints/ultimate_improvement/best.pt
   on both curriculum_heldout and oasst1_eos_v2 validation splits.
2. Sequence counts, mask percentages, active tokens per sequence.
3. Exact duplicate overlap between train and validation splits.
4. Dataset statistics across raw JSONL files and binary sequences.
5. Saves structured results to scratch/pipeline_audit_results.json.
"""

import sys
import os
import json
import time
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

PROJECT_ROOT = Path(r"D:\DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model import FromScratchLM

# Paths
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "ultimate_improvement" / "best.pt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"

# Datasets
ULTIMATE_DIR = PROJECT_ROOT / "data" / "ultimate_improvement" / "training_sequences"
ULTIMATE_TRAIN_TOK = ULTIMATE_DIR / "train_tokens.bin"
ULTIMATE_TRAIN_MSK = ULTIMATE_DIR / "train_mask.bin"
ULTIMATE_VAL_TOK = ULTIMATE_DIR / "heldout_tokens.bin"
ULTIMATE_VAL_MSK = ULTIMATE_DIR / "heldout_mask.bin"

OASST1_EOS_DIR = PROJECT_ROOT / "data" / "processed" / "training_sequences_eos_v2"
OASST1_TRAIN_TOK = OASST1_EOS_DIR / "train_tokens.bin"
OASST1_TRAIN_MSK = OASST1_EOS_DIR / "train_mask.bin"
OASST1_VAL_TOK = OASST1_EOS_DIR / "validation_tokens.bin"
OASST1_VAL_MSK = OASST1_EOS_DIR / "validation_mask.bin"

EXP2_DIR = PROJECT_ROOT / "data" / "experiment_2" / "training_sequences"
EXP2_TRAIN_TOK = EXP2_DIR / "train_tokens.bin"
EXP2_TRAIN_MSK = EXP2_DIR / "train_mask.bin"
EXP2_VAL_TOK = EXP2_DIR / "heldout_tokens.bin"
EXP2_VAL_MSK = EXP2_DIR / "heldout_mask.bin"

OUTPUT_RESULTS = PROJECT_ROOT / "logs" / "model_development_stage_1" / "scratch" / "pipeline_audit_results.json"

CONTEXT_LENGTH = 1024
VOCAB_SIZE = 8000

class BinarySeqDataset(Dataset):
    def __init__(self, token_path: Path, mask_path: Path):
        self.tokens = np.memmap(token_path, dtype=np.uint16, mode="r")
        self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
        assert len(self.tokens) == len(self.mask), f"Length mismatch: {len(self.tokens)} != {len(self.mask)}"
        assert len(self.tokens) % CONTEXT_LENGTH == 0, f"Length {len(self.tokens)} not divisible by {CONTEXT_LENGTH}"
        self.num_seqs = len(self.tokens) // CONTEXT_LENGTH

    def __len__(self):
        return self.num_seqs

    def __getitem__(self, idx):
        start = idx * CONTEXT_LENGTH
        end = start + CONTEXT_LENGTH
        seq = np.asarray(self.tokens[start:end], dtype=np.int64)
        m = np.asarray(self.mask[start:end], dtype=np.uint8)
        inp = torch.tensor(seq[:-1], dtype=torch.long)
        tgt = torch.tensor(seq[1:], dtype=torch.long)
        loss_m = torch.tensor(m[1:], dtype=torch.bool)
        return inp, tgt, loss_m

def evaluate_model_loss(model, dataset, device, batch_size=16):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction="none")
    total_loss = 0.0
    total_active_tokens = 0
    total_sequences = 0

    with torch.no_grad():
        for inp, tgt, msk in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            msk = msk.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(inp)
                tok_loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1)).reshape(tgt.shape)
                active = tok_loss[msk]
                total_loss += active.sum().item()
                total_active_tokens += active.numel()
                total_sequences += inp.size(0)

    mean_loss = total_loss / total_active_tokens if total_active_tokens > 0 else float("inf")
    ppl = math.exp(min(mean_loss, 20.0))
    return mean_loss, ppl, total_sequences, total_active_tokens

def check_duplicates_between_binaries(train_tok_path: Path, val_tok_path: Path):
    print(f"Checking duplicate sequences between:\n  Train: {train_tok_path}\n  Val:   {val_tok_path}")
    train_tokens = np.memmap(train_tok_path, dtype=np.uint16, mode="r")
    val_tokens = np.memmap(val_tok_path, dtype=np.uint16, mode="r")

    n_train = len(train_tokens) // CONTEXT_LENGTH
    n_val = len(val_tokens) // CONTEXT_LENGTH

    # Compute hashes of all train sequences
    train_hashes = set()
    train_hash_to_idx = {}
    for i in range(n_train):
        seq_bytes = train_tokens[i * CONTEXT_LENGTH : (i + 1) * CONTEXT_LENGTH].tobytes()
        h = hashlib.md5(seq_bytes).hexdigest()
        train_hashes.add(h)
        train_hash_to_idx[h] = i

    exact_duplicates = 0
    duplicate_indices = []
    for j in range(n_val):
        seq_bytes = val_tokens[j * CONTEXT_LENGTH : (j + 1) * CONTEXT_LENGTH].tobytes()
        h = hashlib.md5(seq_bytes).hexdigest()
        if h in train_hashes:
            exact_duplicates += 1
            duplicate_indices.append((j, train_hash_to_idx[h]))

    print(f"  Train sequences: {n_train:,}")
    print(f"  Val sequences:   {n_val:,}")
    print(f"  Exact duplicates: {exact_duplicates} ({exact_duplicates / n_val * 100:.2f}%)")
    return {
        "n_train": n_train,
        "n_val": n_val,
        "exact_duplicates": exact_duplicates,
        "duplicate_pct": round(exact_duplicates / n_val * 100, 2),
        "sample_duplicate_indices": duplicate_indices[:5]
    }

def analyze_mask_statistics(tok_path: Path, mask_path: Path, name: str):
    tokens = np.memmap(tok_path, dtype=np.uint16, mode="r")
    mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
    n_seq = len(tokens) // CONTEXT_LENGTH

    total_tokens = len(tokens)
    active_tokens = int(np.sum(mask))
    inactive_tokens = total_tokens - active_tokens
    active_pct = (active_tokens / total_tokens) * 100

    # Per-sequence stats
    seq_actives = []
    for i in range(min(n_seq, 500)):  # sample up to 500
        m_slice = mask[i * CONTEXT_LENGTH : (i + 1) * CONTEXT_LENGTH]
        seq_actives.append(int(np.sum(m_slice)))

    pad_count = int(np.sum(tokens == 0))
    eos_count = int(np.sum(tokens == 3))
    user_count = int(np.sum(tokens == 4))
    asst_count = int(np.sum(tokens == 5))

    return {
        "dataset_name": name,
        "num_sequences": n_seq,
        "total_tokens": total_tokens,
        "active_tokens": active_tokens,
        "inactive_tokens": inactive_tokens,
        "active_pct": round(active_pct, 2),
        "pad_token_count": pad_count,
        "pad_pct": round(pad_count / total_tokens * 100, 2),
        "eos_count": eos_count,
        "user_count": user_count,
        "asst_count": asst_count,
        "sample_mean_active_per_seq": round(float(np.mean(seq_actives)), 1),
        "sample_min_active": int(np.min(seq_actives)),
        "sample_max_active": int(np.max(seq_actives)),
    }

def analyze_jsonl_dataset(jsonl_path: Path):
    if not jsonl_path.exists():
        return {"exists": False}
    
    line_count = 0
    user_lens = []
    asst_lens = []
    has_empty_user = 0
    has_empty_asst = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            try:
                rec = json.loads(line)
                if "messages" in rec:
                    for m in rec["messages"]:
                        role = m.get("role", "")
                        content = m.get("content", m.get("text", ""))
                        if role == "user":
                            user_lens.append(len(content))
                            if not content.strip(): has_empty_user += 1
                        elif role == "assistant":
                            asst_lens.append(len(content))
                            if not content.strip(): has_empty_asst += 1
                elif "prompt" in rec and "response" in rec:
                    user_lens.append(len(rec["prompt"]))
                    asst_lens.append(len(rec["response"]))
                elif "question" in rec and "answer" in rec:
                    user_lens.append(len(rec["question"]))
                    asst_lens.append(len(rec["answer"]))
                elif "instruction" in rec and "response" in rec:
                    user_lens.append(len(rec["instruction"]))
                    asst_lens.append(len(rec["response"]))
            except Exception:
                pass
                
    return {
        "exists": True,
        "file_name": jsonl_path.name,
        "file_size_bytes": jsonl_path.stat().st_size,
        "record_count": line_count,
        "mean_user_chars": round(float(np.mean(user_lens)), 1) if user_lens else 0,
        "mean_asst_chars": round(float(np.mean(asst_lens)), 1) if asst_lens else 0,
        "empty_user_count": has_empty_user,
        "empty_asst_count": has_empty_asst
    }

def main():
    print("=" * 70)
    print("TRAINING PIPELINE & DATA AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load active model
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FromScratchLM(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 1. Independent validation loss calculation
    print("\n--- 1. Independent Validation Loss Recalculation ---")
    val_curriculum_ds = BinarySeqDataset(ULTIMATE_VAL_TOK, ULTIMATE_VAL_MSK)
    curr_loss, curr_ppl, curr_seqs, curr_tokens = evaluate_model_loss(model, val_curriculum_ds, device)
    print(f"  Ultimate Curriculum Val: Loss={curr_loss:.4f}, PPL={curr_ppl:.2f} ({curr_seqs} seqs, {curr_tokens} tokens)")

    val_oasst1_ds = BinarySeqDataset(OASST1_VAL_TOK, OASST1_VAL_MSK)
    oasst_loss, oasst_ppl, oasst_seqs, oasst_tokens = evaluate_model_loss(model, val_oasst1_ds, device)
    print(f"  OASST1 EOS v2 Val:       Loss={oasst_loss:.4f}, PPL={oasst_ppl:.2f} ({oasst_seqs} seqs, {oasst_tokens} tokens)")

    composite_loss = 0.5 * curr_loss + 0.5 * oasst_loss
    print(f"  Composite Val Loss:      {composite_loss:.4f}")
    print(f"  Reported in Checkpoint:  {ckpt.get('validation_loss', 'N/A')}")
    ckpt_val = ckpt.get("validation_loss", None)
    loss_match = abs(composite_loss - ckpt_val) < 0.05 if ckpt_val else False
    print(f"  Validation Loss Match:   {'✓ Verified' if loss_match else '⚠ Discrepancy'}")

    # 2. Check duplicate leakage between train and val
    print("\n--- 2. Duplicate Leakage Checks ---")
    dup_ultimate = check_duplicates_between_binaries(ULTIMATE_TRAIN_TOK, ULTIMATE_VAL_TOK)
    dup_oasst1 = check_duplicates_between_binaries(OASST1_TRAIN_TOK, OASST1_VAL_TOK)
    dup_exp2 = check_duplicates_between_binaries(EXP2_TRAIN_TOK, EXP2_VAL_TOK)

    # 3. Mask statistics
    print("\n--- 3. Loss Mask & Token Statistics ---")
    mask_stats = {
        "ultimate_train": analyze_mask_statistics(ULTIMATE_TRAIN_TOK, ULTIMATE_TRAIN_MSK, "ultimate_train"),
        "ultimate_heldout": analyze_mask_statistics(ULTIMATE_VAL_TOK, ULTIMATE_VAL_MSK, "ultimate_heldout"),
        "oasst1_train": analyze_mask_statistics(OASST1_TRAIN_TOK, OASST1_TRAIN_MSK, "oasst1_train"),
        "oasst1_val": analyze_mask_statistics(OASST1_VAL_TOK, OASST1_VAL_MSK, "oasst1_val")
    }
    for k, v in mask_stats.items():
        print(f"  {k}: {v['num_sequences']:,} seqs, {v['active_pct']}% active, {v['pad_pct']}% padding")

    # 4. JSONL Raw Data Catalog
    print("\n--- 4. Raw Dataset Files Catalog ---")
    jsonl_files = [
        PROJECT_ROOT / "data" / "ultimate_improvement" / "curriculum_raw" / "ultimate_train.jsonl",
        PROJECT_ROOT / "data" / "ultimate_improvement" / "curriculum_raw" / "ultimate_heldout.jsonl",
        PROJECT_ROOT / "data" / "experiment_2" / "curriculum_raw" / "category1_oasst1_cleaned.jsonl",
        PROJECT_ROOT / "data" / "experiment_2" / "curriculum_raw" / "category2_arithmetic_train.jsonl",
        PROJECT_ROOT / "data" / "experiment_2" / "curriculum_raw" / "category2_arithmetic_heldout.jsonl",
        PROJECT_ROOT / "data" / "experiment_2" / "curriculum_raw" / "category3_python_train.jsonl",
        PROJECT_ROOT / "data" / "experiment_2" / "curriculum_raw" / "category3_python_heldout.jsonl",
        PROJECT_ROOT / "data" / "experiment_2" / "curriculum_raw" / "category4_concepts_train.jsonl",
        PROJECT_ROOT / "data" / "experiment_2" / "curriculum_raw" / "category4_concepts_heldout.jsonl",
        PROJECT_ROOT / "data" / "processed" / "conversation_paths" / "train_paths.jsonl",
        PROJECT_ROOT / "data" / "processed" / "conversation_paths" / "validation_paths.jsonl",
    ]
    jsonl_stats = {}
    for jf in jsonl_files:
        st = analyze_jsonl_dataset(jf)
        jsonl_stats[jf.name] = st
        if st.get("exists"):
            print(f"  {jf.name:>35}: {st['record_count']:>7,} records ({st['file_size_bytes'] / (1024*1024):>5.2f} MB)")

    # Save all results
    all_results = {
        "independent_validation_loss": {
            "curriculum_heldout_loss": round(curr_loss, 4),
            "curriculum_heldout_ppl": round(curr_ppl, 2),
            "curriculum_heldout_seqs": curr_seqs,
            "oasst1_eos_v2_loss": round(oasst_loss, 4),
            "oasst1_eos_v2_ppl": round(oasst_ppl, 2),
            "oasst1_eos_v2_seqs": oasst_seqs,
            "composite_loss": round(composite_loss, 4),
            "checkpoint_reported_loss": ckpt.get("validation_loss"),
            "loss_match": loss_match
        },
        "duplicate_analysis": {
            "ultimate_improvement": dup_ultimate,
            "oasst1_eos_v2": dup_oasst1,
            "experiment_2": dup_exp2
        },
        "mask_statistics": mask_stats,
        "jsonl_statistics": jsonl_stats
    }

    OUTPUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {OUTPUT_RESULTS}")

if __name__ == "__main__":
    main()
