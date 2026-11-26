import hashlib
import json
import os
import zipfile
from pathlib import Path

ACTIVE_CHECKPOINT = "checkpoints/ultimate_improvement/best.pt"
EXPECTED_ACTIVE_SHA256 = "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC"
ARCHIVE_PATH = Path("backups/checkpoints_recovery_archive.zip")
MANIFEST_PATH = Path("backups/recovery_manifest.json")

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024 * 4):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    print("=" * 70)
    print("STEP 1: PRE-CLEANUP SAFETY ASSERTIONS")
    print("=" * 70)

    # 1. Assert recovery archive exists and is non-empty
    if not ARCHIVE_PATH.exists() or ARCHIVE_PATH.stat().st_size < 1000000000:
        raise RuntimeError(f"Recovery archive missing or suspiciously small: {ARCHIVE_PATH}")
    print(f"Archive verified present: {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Read manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Manifest verified present: {len(manifest['files'])} files logged.")

    # 3. Assert active model exists and matches expected SHA-256
    active_path = Path(ACTIVE_CHECKPOINT)
    if not active_path.exists():
        raise FileNotFoundError(f"Active model missing: {active_path}")
    active_sha = compute_sha256(active_path)
    if active_sha != EXPECTED_ACTIVE_SHA256:
        raise ValueError(f"Active model hash mismatch before cleanup! Got {active_sha}")
    print(f"Active model integrity verified: {active_sha}")

    # 4. Check initial checkpoints directory stats
    all_initial_pt = list(Path("checkpoints").rglob("*.pt"))
    initial_bytes = sum(f.stat().st_size for f in all_initial_pt)
    print(f"Total .pt files before cleanup: {len(all_initial_pt)} ({initial_bytes / (1024*1024):.2f} MB)")

    print("\n" + "=" * 70)
    print("STEP 2: SAFELY REMOVE 15 VERIFIED REDUNDANT FILES")
    print("=" * 70)

    deleted_bytes = 0
    deleted_count = 0
    for item in manifest["files"]:
        p = Path(item["path"])
        if p.exists():
            size = p.stat().st_size
            # Sanity check: must match manifest SHA256 before unlinking
            current_sha = compute_sha256(p)
            if current_sha != item["sha256"]:
                raise ValueError(f"Hash mismatch on {p} before deletion! Aborting.")
            p.unlink()
            deleted_bytes += size
            deleted_count += 1
            print(f"  Safely deleted: {item['path']} ({size / (1024*1024):.2f} MB)")
        else:
            print(f"  Already absent: {item['path']}")

    print(f"\nSuccessfully deleted {deleted_count} files, recovering {deleted_bytes / (1024*1024):.2f} MB ({deleted_bytes / (1024*1024*1024):.3f} GB).")

    print("\n" + "=" * 70)
    print("STEP 3: POST-CLEANUP VERIFICATION")
    print("=" * 70)

    # 1. Verify active model still exists and has exact same SHA-256
    post_active_sha = compute_sha256(active_path)
    if post_active_sha != EXPECTED_ACTIVE_SHA256:
        raise RuntimeError(f"FATAL: Active model modified during cleanup! Got {post_active_sha}")
    print(f"Active model hash POST-CLEANUP: {post_active_sha} (100% MATCH)")

    # 2. Count remaining .pt files
    remaining_pt = list(Path("checkpoints").rglob("*.pt"))
    remaining_bytes = sum(f.stat().st_size for f in remaining_pt)
    print(f"Total .pt files remaining: {len(remaining_pt)}")
    for f in sorted(remaining_pt):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  Preserved: {f.as_posix():<45} ({size_mb:.2f} MB)")
    print(f"Total remaining checkpoint storage: {remaining_bytes / (1024*1024):.2f} MB ({remaining_bytes / (1024*1024*1024):.3f} GB)")

    summary = {
        "deleted_count": deleted_count,
        "deleted_mb": round(deleted_bytes / (1024*1024), 2),
        "deleted_gb": round(deleted_bytes / (1024*1024*1024), 3),
        "remaining_count": len(remaining_pt),
        "remaining_mb": round(remaining_bytes / (1024*1024), 2),
        "remaining_gb": round(remaining_bytes / (1024*1024*1024), 3),
        "active_model_sha256": post_active_sha,
        "archive_path": str(ARCHIVE_PATH),
        "archive_size_mb": round(ARCHIVE_PATH.stat().st_size / (1024*1024), 2)
    }

    with open("data/post_cleanup_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPOST-CLEANUP VERIFICATION COMPLETE: ALL CHECKS PASSED.")

if __name__ == "__main__":
    main()
