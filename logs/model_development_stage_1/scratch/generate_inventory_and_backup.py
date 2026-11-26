"""
generate_inventory_and_backup.py
================================
Computes SHA-256 hashes for all critical model, tokenizer, and checkpoint files.
Creates a timestamped backup zip/archive of source files before any changes.
Generates structured inventory data for initial_audit.md and artifact_manifest.json.
"""

import sys
import os
import json
import time
import hashlib
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(r"D:\DEEP-LEARNING-PROJECT")
STAGE1_DIR = PROJECT_ROOT / "logs" / "model_development_stage_1"
BACKUP_DIR = STAGE1_DIR / "backups"
SCRATCH_DIR = STAGE1_DIR / "scratch"

def compute_sha256(filepath: Path, max_bytes: int = None) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        if max_bytes:
            h.update(f.read(max_bytes))
        else:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()

def main():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Critical Source Files
    src_files = [
        PROJECT_ROOT / "src" / "model.py",
        PROJECT_ROOT / "src" / "generate.py",
        PROJECT_ROOT / "src" / "chat.py",
        PROJECT_ROOT / "src" / "web_retrieval.py",
        PROJECT_ROOT / "src" / "answer_synthesis.py",
        PROJECT_ROOT / "src" / "intent_router.py",
        PROJECT_ROOT / "src" / "question_understanding.py",
        PROJECT_ROOT / "src" / "conversation_memory.py",
        PROJECT_ROOT / "src" / "source_selection.py",
        PROJECT_ROOT / "src" / "train.py",
        PROJECT_ROOT / "src" / "build_eos_v2_sequences.py",
        PROJECT_ROOT / "src" / "ultimate_improvement" / "train_ultimate.py",
        PROJECT_ROOT / "src" / "ultimate_improvement" / "generate_ultimate_curriculum.py",
        PROJECT_ROOT / "src" / "ultimate_improvement" / "build_eval_suites.py",
        PROJECT_ROOT / "src" / "ultimate_improvement" / "evaluate_comprehensive.py",
        PROJECT_ROOT / "src" / "experiment_2" / "train_experiment_2.py",
        PROJECT_ROOT / "tokenizer" / "tokenizer.json",
    ]

    # 2. Checkpoints
    checkpoints = [
        PROJECT_ROOT / "checkpoints" / "ultimate_improvement" / "best.pt",
        PROJECT_ROOT / "checkpoints" / "ultimate_improvement" / "latest.pt",
        PROJECT_ROOT / "checkpoints" / "experiment_2" / "best.pt",
        PROJECT_ROOT / "checkpoints" / "experiment_2" / "latest.pt",
        PROJECT_ROOT / "checkpoints" / "eos_v2_pilot" / "best.pt",
        PROJECT_ROOT / "checkpoints" / "best.pt",
        PROJECT_ROOT / "checkpoints" / "latest.pt",
        PROJECT_ROOT / "checkpoints" / "masked_best.pt",
        PROJECT_ROOT / "checkpoints" / "clean_50k_best.pt",
    ]

    print("Computing SHA-256 for source files...")
    file_hashes = {}
    for p in src_files:
        if p.exists():
            h = compute_sha256(p)
            sz = p.stat().st_size
            rel = str(p.relative_to(PROJECT_ROOT))
            file_hashes[rel] = {
                "path": str(p),
                "size_bytes": sz,
                "sha256": h,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime))
            }
            print(f"  {rel}: {sz:,} bytes | {h[:16]}...")

    print("\nComputing SHA-256 for checkpoint best.pt files...")
    checkpoint_inventory = {}
    for cp in checkpoints:
        if cp.exists():
            sz = cp.stat().st_size
            rel = str(cp.relative_to(PROJECT_ROOT))
            # Full SHA-256
            h = compute_sha256(cp)
            checkpoint_inventory[rel] = {
                "path": str(cp),
                "size_bytes": sz,
                "sha256": h,
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cp.stat().st_mtime))
            }
            print(f"  {rel}: {sz:,} bytes | {h[:16]}...")

    # 3. Create timestamped zip backup of src/ and configs/
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_zip_path = BACKUP_DIR / f"source_backup_{ts}.zip"
    print(f"\nCreating timestamped source backup: {backup_zip_path}")
    with zipfile.ZipFile(backup_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(PROJECT_ROOT / "src"):
            if "__pycache__" in root:
                continue
            for f in files:
                full_p = Path(root) / f
                arcname = str(full_p.relative_to(PROJECT_ROOT))
                zf.write(full_p, arcname=arcname)
        # Also backup tokenizer
        tok_file = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
        if tok_file.exists():
            zf.write(tok_file, arcname="tokenizer/tokenizer.json")

    print(f"Backup created: {backup_zip_path.stat().st_size:,} bytes")

    output_manifest = {
        "timestamp": time.time(),
        "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backup_archive": str(backup_zip_path),
        "source_files": file_hashes,
        "checkpoints": checkpoint_inventory
    }

    out_file = SCRATCH_DIR / "file_inventory_and_hashes.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_manifest, f, indent=2)

    print(f"\nInventory saved to {out_file}")

if __name__ == "__main__":
    main()
