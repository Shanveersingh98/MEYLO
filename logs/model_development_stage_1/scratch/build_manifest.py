"""
build_manifest.py
=================
Computes SHA-256 hashes and file sizes for all generated artifacts in
D:\\DEEP-LEARNING-PROJECT\\logs\\model_development_stage_1\\
and writes artifact_manifest.json.
"""

import sys
import os
import json
import time
import hashlib
from pathlib import Path

STAGE1_DIR = Path(r"D:\DEEP-LEARNING-PROJECT\logs\model_development_stage_1")
MANIFEST_PATH = STAGE1_DIR / "artifact_manifest.json"

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    artifacts = {}
    
    # List of expected stage 1 files
    target_files = [
        "initial_audit.md",
        "model_architecture_audit.md",
        "independent_model_benchmark.json",
        "baseline_model_responses.json",
        "generation_comparison.json",
        "training_pipeline_audit.md",
        "validation_integrity_report.md",
        "training_data_quality_report.md",
        "model_limitations_report.md",
        "training_improvement_recommendation.md",
        "scripts/evaluate_local_model.py",
        "scripts/evaluate_generation_parameters.py",
        "scripts/audit_pipeline_and_data.py",
        "scratch/audit_params.py",
        "scratch/generate_inventory_and_backup.py",
        "scratch/file_inventory_and_hashes.json",
        "scratch/pipeline_audit_results.json",
        "backups/source_backup_20261113_182820.zip"
    ]

    for rel_path in target_files:
        full_p = STAGE1_DIR / rel_path
        if full_p.exists():
            sz = full_p.stat().st_size
            h = compute_sha256(full_p)
            artifacts[rel_path] = {
                "file_path": str(full_p),
                "size_bytes": sz,
                "sha256": h,
                "modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(full_p.stat().st_mtime))
            }
            print(f"  {rel_path}: {sz:,} bytes | {h[:16]}...")
            
    manifest = {
        "stage": "Stage 1: Complete Neural Model Audit, Independent Evaluation & Training Readiness",
        "timestamp": time.time(),
        "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_directory": str(STAGE1_DIR),
        "total_artifacts": len(artifacts),
        "artifacts": artifacts
    }
    
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nManifest written to {MANIFEST_PATH}")

if __name__ == "__main__":
    main()
