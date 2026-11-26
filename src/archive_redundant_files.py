"""
src/archive_redundant_files.py
==============================
Safely archives redundant in-place backup files (*bak*) from src/
into backups/archive/src_legacy_backups/.
Zero data loss: All files are preserved with their original SHA256 checksums verified.
All operations are logged to data/companion_project.db and data/cleanup_manifest.json.
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR))

from src.companion_database import CompanionDatabase, compute_file_sha256, get_utc_timestamp

SRC_DIR = PROJECT_DIR / "src"
ARCHIVE_DIR = PROJECT_DIR / "backups" / "archive" / "src_legacy_backups"
MANIFEST_FILE = PROJECT_DIR / "data" / "cleanup_manifest.json"

def archive_files():
    print("=== SAFE FILE ARCHIVING & AUDIT ===")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    db = CompanionDatabase()
    
    # Load existing manifest if present
    existing_ops = []
    if MANIFEST_FILE.exists():
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as mf:
                existing_ops = json.load(mf).get("operations", [])
        except Exception:
            existing_ops = []
    
    # Identify all files with 'bak' in name in src/
    bak_files = [f for f in SRC_DIR.iterdir() if f.is_file() and "bak" in f.name]
    print(f"Identified {len(bak_files)} in-place backup files in {SRC_DIR.name}/:")
    for f in bak_files:
        print(f"  - {f.name} ({f.stat().st_size:,} bytes)")
    
    new_ops = []
    
    for f in bak_files:
        orig_path = str(f.relative_to(PROJECT_DIR)).replace("\\", "/")
        dest_file = ARCHIVE_DIR / f.name
        dest_path = str(dest_file.relative_to(PROJECT_DIR)).replace("\\", "/")
        f_size = f.stat().st_size
        f_hash = compute_file_sha256(f)
        
        # Copy to archive first to verify integrity
        shutil.copy2(f, dest_file)
        dest_hash = compute_file_sha256(dest_file)
        
        assert f_hash == dest_hash, f"Hash mismatch during archiving of {f.name}!"
        
        # Now remove original from src/
        f.unlink()
        
        op_id = f"op_arch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        reason = "Quarantined loose backup file from active src/ directory into organized backups archive."
        
        db.record_cleanup_op(
            operation_id=op_id,
            operation_type="archive_move",
            original_path=orig_path,
            target_path=dest_path,
            file_size=f_size,
            sha256_hash=f_hash,
            reason=reason,
            status="verified"
        )
        
        new_ops.append({
            "operation_id": op_id,
            "original_path": orig_path,
            "target_path": dest_path,
            "file_size": f_size,
            "sha256": f_hash,
            "reason": reason,
            "timestamp": get_utc_timestamp(),
            "status": "archived_verified"
        })
        print(f"  [ARCHIVED] {f.name} -> {dest_path}")
        
    all_ops = existing_ops + new_ops
    manifest = {
        "manifest_version": "1.0",
        "timestamp": get_utc_timestamp(),
        "total_files_archived": len(all_ops),
        "total_bytes_archived": sum(op["file_size"] for op in all_ops),
        "archive_destination": str(ARCHIVE_DIR.relative_to(PROJECT_DIR)).replace("\\", "/"),
        "operations": all_ops
    }
    
    with open(MANIFEST_FILE, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)
        
    print(f"\nCleanup manifest updated at {MANIFEST_FILE}")
    db.record_event(
        event_type="FILE_ARCHIVING_COMPLETED",
        description=f"Archived {len(new_ops)} loose backup files from src/ into backups/archive/src_legacy_backups/",
        metadata={"total_files": len(all_ops), "total_bytes": manifest["total_bytes_archived"]}
    )
    
    db.backup_database()
    summary = db.get_summary()
    print(f"Database recorded {summary['cleanup_operations_recorded']} cleanup operations.")
    db.close()

if __name__ == "__main__":
    archive_files()
