import hashlib
import json
import zipfile
from pathlib import Path

FILES_TO_CLEAN = [
    "checkpoints/clean_50k_step_50000.pt",
    "checkpoints/clean_50k_latest.pt",
    "checkpoints/clean_50k_step_5000.pt",
    "checkpoints/clean_50k_step_10000.pt",
    "checkpoints/clean_50k_step_20000.pt",
    "checkpoints/clean_50k_step_30000.pt",
    "checkpoints/clean_50k_step_40000.pt",
    "checkpoints/eos_v2_pilot/step_5000.pt",
    "checkpoints/eos_v2_pilot/latest.pt",
    "checkpoints/eos_v2_pilot/step_2500.pt",
    "checkpoints/phase3_latest.pt",
    "checkpoints/masked_latest.pt",
    "checkpoints/diagnostic_overfit_latest.pt",
    "checkpoints/clean_dataset_test_latest.pt",
    "checkpoints/step25_test.pt",
]

ACTIVE_CHECKPOINT = "checkpoints/ultimate_improvement/best.pt"
EXPECTED_ACTIVE_SHA256 = "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC"

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest().upper()

def create_recovery_archive():
    backup_dir = Path("backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    archive_path = backup_dir / "checkpoints_recovery_archive.zip"
    manifest_path = backup_dir / "recovery_manifest.json"
    
    print("=" * 70)
    print("STEP 1: Verify all 15 files exist and record hashes")
    print("=" * 70)
    
    manifest = {
        "created_at": "2026-11-22T20:05:00+05:30",
        "archive_file": str(archive_path.name),
        "total_files": len(FILES_TO_CLEAN),
        "files": []
    }
    
    total_bytes = 0
    for rel_str in FILES_TO_CLEAN:
        p = Path(rel_str)
        if not p.exists():
            raise FileNotFoundError(f"File to clean does not exist: {p}")
        size = p.stat().st_size
        total_bytes += size
        sha = compute_sha256(p)
        manifest["files"].append({
            "path": rel_str,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2),
            "sha256": sha
        })
        print(f"  Recorded: {rel_str:<42} ({size/(1024*1024):.2f} MB, SHA: {sha[:16]}...)")
        
    manifest["total_size_bytes"] = total_bytes
    manifest["total_size_mb"] = round(total_bytes / (1024 * 1024), 2)
    manifest["total_size_gb"] = round(total_bytes / (1024 * 1024 * 1024), 3)
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path} (Total: {manifest['total_size_mb']} MB)")
    
    print("\n" + "=" * 70)
    print("STEP 2: Create verified recovery archive")
    print("=" * 70)
    
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for item in manifest["files"]:
            p = Path(item["path"])
            arcname = item["path"]
            print(f"  Archiving: {item['path']} -> {arcname}...")
            zf.write(p, arcname=arcname)
            
    archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
    print(f"Archive created: {archive_path} ({archive_size_mb:.2f} MB)")
    
    print("\n" + "=" * 70)
    print("STEP 3: Verify archive integrity and SHA-256 match")
    print("=" * 70)
    
    with zipfile.ZipFile(archive_path, "r") as zf:
        test_res = zf.testzip()
        if test_res is not None:
            raise RuntimeError(f"Corrupted file in zip archive: {test_res}")
            
        for item in manifest["files"]:
            with zf.open(item["path"]) as zf_entry:
                h = hashlib.sha256()
                while chunk := zf_entry.read(1024 * 1024 * 4):
                    h.update(chunk)
                extracted_sha = h.hexdigest().upper()
                if extracted_sha != item["sha256"]:
                    raise ValueError(f"Hash mismatch in archive for {item['path']}! {extracted_sha} != {item['sha256']}")
                print(f"  Verified in Archive: {item['path']} (SHA-256 Matches 100%)")
                
    print("\nRECOVERY ARCHIVE VERIFIED 100% HEALTHY AND INTELLECTUALLY COMPLETE.")
    return manifest

if __name__ == "__main__":
    create_recovery_archive()
