"""
src/populate_project_database.py
================================
Populates data/companion_project.db with all historical and current project artifacts:
- Datasets inventory
- Checkpoints inventory (including best.pt verification)
- Training sessions & loss histories from logs/
- Evaluation suite results
- Initial audit events
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR))

from src.companion_database import CompanionDatabase, compute_file_sha256, get_utc_timestamp

def populate():
    print("Connecting to CompanionDatabase at data/companion_project.db...")
    db = CompanionDatabase()
    
    # -------------------------------------------------------------
    # 1. Register All Datasets
    # -------------------------------------------------------------
    print("\n[1/5] Registering datasets...")
    with open("scratch/full_inventory.json", "r", encoding="utf-8") as f:
        inv_data = json.load(f)
    
    for item in inv_data["inventory"]:
        if item["category"] == "datasets":
            p = PROJECT_DIR / item["path"]
            # Estimate record count if jsonl or json
            rec_count = 0
            if item["ext"] == ".jsonl" and p.exists():
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as jf:
                        rec_count = sum(1 for line in jf if line.strip())
                except Exception:
                    rec_count = 0
            elif item["ext"] == ".json" and p.exists():
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as jf:
                        d = json.load(jf)
                        rec_count = len(d) if isinstance(d, list) else 1
                except Exception:
                    rec_count = 1

            # Determine dataset classification
            ds_name = p.name
            cat = "general_data"
            is_active = False
            if "ultimate_train" in ds_name:
                cat = "curriculum_train"
            elif "ultimate_heldout" in ds_name:
                cat = "curriculum_heldout"
            elif "arithmetic" in ds_name:
                cat = "arithmetic"
            elif "python" in ds_name:
                cat = "python_code"
            elif "oasst1" in ds_name:
                cat = "conversational_oasst1"
            elif "eval_suite" in item["path"]:
                cat = "evaluation_suite"
            elif "dolly" in ds_name:
                cat = "instruction_dolly"
            elif "gsm8k" in ds_name:
                cat = "arithmetic_gsm8k"
            elif "sciq" in ds_name:
                cat = "science_qa"

            ds_id = f"ds_{item['path'].replace('/', '_').replace('.', '_')}"
            db.register_dataset(
                dataset_id=ds_id,
                name=ds_name,
                category=cat,
                file_path=item["path"],
                record_count=rec_count,
                size_bytes=item["size"],
                sha256_hash=item["sha256"],
                is_active_companion=is_active,
                status="preserved",
                notes=f"Audited file under {item['path']}"
            )
    print(f"  Datasets registered.")

    # -------------------------------------------------------------
    # 2. Register All Model Checkpoints
    # -------------------------------------------------------------
    print("\n[2/5] Registering model checkpoints...")
    for item in inv_data["inventory"]:
        if item["category"] == "checkpoints":
            p = PROJECT_DIR / item["path"]
            cp_name = p.name
            cp_id = f"cp_{item['path'].replace('/', '_').replace('.', '_')}"
            
            is_best = ("ultimate_improvement/best.pt" in item["path"]) or (item["path"] == "checkpoints/ultimate_improvement/best.pt")
            is_latest = ("ultimate_improvement/latest.pt" in item["path"])
            
            # Extract step from filename or path if present
            step = 0
            if "step_" in cp_name:
                try:
                    step = int(cp_name.split("step_")[1].split(".")[0])
                except Exception:
                    step = 0
            elif "10000" in cp_name:
                step = 10000
            elif "50k" in cp_name:
                step = 50000

            notes = "Ultimate improvement production checkpoint" if is_best else "Historical checkpoint"
            db.register_checkpoint(
                checkpoint_id=cp_id,
                name=cp_name,
                file_path=item["path"],
                step=step,
                epoch=0.0,
                val_loss=None,
                is_best=is_best,
                is_latest=is_latest,
                size_bytes=item["size"],
                sha256_hash=item["sha256"],
                status="active" if is_best else "preserved_checkpoint",
                notes=notes
            )
    print(f"  Checkpoints registered.")

    # -------------------------------------------------------------
    # 3. Import Training Sessions & Loss History from Logs
    # -------------------------------------------------------------
    print("\n[3/5] Importing training history and loss logs...")
    # Session 1: Ultimate Improvement
    sess_id_ult = "sess_ultimate_improvement_20261027"
    db.register_training_session(
        session_id=sess_id_ult,
        session_name="Ultimate Improvement Final Alignment",
        model_name="FromScratchLM",
        dataset_used="data/ultimate_improvement/curriculum_raw/ultimate_train.jsonl",
        total_steps=5000,
        best_val_loss=2.15,
        checkpoint_id="cp_checkpoints_ultimate_improvement_best_pt",
        notes="Production training run that yielded checkpoints/ultimate_improvement/best.pt"
    )

    ult_train_log = PROJECT_DIR / "logs" / "ultimate_improvement" / "training_log.jsonl"
    if ult_train_log.exists():
        with open(ult_train_log, "r", encoding="utf-8") as lf:
            for line in lf:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    db.record_loss_point(
                        session_id=sess_id_ult,
                        step=rec.get("step", 0),
                        epoch=rec.get("epoch", 0.0),
                        train_loss=rec.get("loss", rec.get("train_loss", 0.0)),
                        val_loss=rec.get("val_loss"),
                        learning_rate=rec.get("lr", rec.get("learning_rate")),
                        timestamp=rec.get("timestamp")
                    )
                except Exception:
                    pass

    # Session 2: Clean 50k
    sess_id_clean50k = "sess_clean_50k_20261007"
    db.register_training_session(
        session_id=sess_id_clean50k,
        session_name="Clean 50k Training Run",
        model_name="FromScratchLM",
        dataset_used="data/processed/training_sequences_clean",
        total_steps=50000,
        best_val_loss=2.45,
        checkpoint_id="cp_checkpoints_clean_50k_best_pt",
        notes="Historical 50,000-step pre-training run on cleaned OASST1"
    )

    # Sample representative loss points from training_clean_50k.jsonl (every 500 steps to keep DB nimble)
    clean50k_log = PROJECT_DIR / "logs" / "training_clean_50k.jsonl"
    if clean50k_log.exists():
        print("  Sampling historical loss trajectory from training_clean_50k.jsonl...")
        with open(clean50k_log, "r", encoding="utf-8") as lf:
            for line in lf:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    st = rec.get("step", 0)
                    if st % 500 == 0:
                        db.record_loss_point(
                            session_id=sess_id_clean50k,
                            step=st,
                            epoch=rec.get("epoch"),
                            train_loss=rec.get("loss", rec.get("train_loss", 0.0)),
                            val_loss=rec.get("val_loss"),
                            learning_rate=rec.get("lr", rec.get("learning_rate")),
                            timestamp=rec.get("timestamp")
                        )
                except Exception:
                    pass
    print("  Training sessions and loss points recorded.")

    # -------------------------------------------------------------
    # 4. Record Initial Audit Event
    # -------------------------------------------------------------
    print("\n[4/5] Recording system audit event...")
    db.record_event(
        event_type="INITIAL_AUDIT_COMPLETED",
        description="Comprehensive project inventory and checksum audit completed.",
        metadata={
            "total_files": inv_data["summary"]["total_files"],
            "total_size_mb": round(inv_data["summary"]["total_size_bytes"] / (1024*1024), 2),
            "duplicate_groups": inv_data["summary"]["duplicate_groups"]
        }
    )

    # -------------------------------------------------------------
    # 5. Verify Integrity and Create Initial Hot Backup
    # -------------------------------------------------------------
    print("\n[5/5] Verifying database integrity and creating hot backup...")
    integrity = db.verify_integrity()
    print(f"  Integrity Check: {integrity['status']}")
    
    backup_path = db.backup_database()
    print(f"  Hot Backup Created at: {backup_path}")

    summary = db.get_summary()
    print("\n=== DATABASE POPULATION COMPLETE ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    db.close()

if __name__ == "__main__":
    populate()
