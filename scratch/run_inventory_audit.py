import os
import hashlib
import json
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
EXCLUDE_DIRS = {".venv", ".cache", "Python312", ".git"}

def sha256_file(filepath):
    h = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"

inventory = []
hashes = defaultdict(list)

for root, dirs, files in os.walk(PROJECT_DIR):
    # filter out excluded dirs
    rel_root = os.path.relpath(root, PROJECT_DIR)
    parts = Path(rel_root).parts
    if parts and parts[0] in EXCLUDE_DIRS:
        continue
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    
    for f in files:
        full_path = Path(root) / f
        rel_path = full_path.relative_to(PROJECT_DIR).as_posix()
        try:
            size = full_path.stat().st_size
            mtime = full_path.stat().st_mtime
        except Exception:
            size = 0
            mtime = 0
        
        # Don't hash giant files > 500MB if any, but our checkpoints are ~170MB, safe to hash
        file_hash = sha256_file(full_path)
        hashes[file_hash].append(rel_path)
        
        # Determine category
        cat = "other"
        ext = full_path.suffix.lower()
        if "checkpoint" in rel_path or ext == ".pt":
            cat = "checkpoints"
        elif "data" in rel_path.split("/")[0] if "/" in rel_path else False:
            cat = "datasets"
        elif rel_path.startswith("src/"):
            cat = "source_code"
        elif rel_path.startswith("tests/"):
            cat = "tests"
        elif rel_path.startswith("logs/"):
            cat = "logs"
        elif rel_path.startswith("tokenizer/"):
            cat = "tokenizer"
        elif rel_path.startswith("backups/"):
            cat = "backups"
        elif rel_path.startswith("scratch/"):
            cat = "scratch"
        elif ext in [".bat", ".sh", ".cmd"]:
            cat = "launch_scripts"
        elif ext in [".md", ".txt"]:
            cat = "documentation"
        
        inventory.append({
            "path": rel_path,
            "category": cat,
            "size": size,
            "mtime": mtime,
            "sha256": file_hash,
            "ext": ext
        })

duplicates = {h: p_list for h, p_list in hashes.items() if len(p_list) > 1 and not h.startswith("ERROR")}

summary = {
    "total_files": len(inventory),
    "total_size_bytes": sum(item["size"] for item in inventory),
    "duplicate_groups": len(duplicates),
    "category_counts": defaultdict(int),
    "category_sizes": defaultdict(int)
}

for item in inventory:
    summary["category_counts"][item["category"]] += 1
    summary["category_sizes"][item["category"]] += item["size"]

print(f"Total files audited: {summary['total_files']}")
print(f"Total size: {summary['total_size_bytes'] / (1024*1024):.2f} MB")
print(f"Duplicate hash groups: {len(duplicates)}")
print("\nBy category:")
for cat, cnt in summary["category_counts"].items():
    sz = summary["category_sizes"][cat] / (1024*1024)
    print(f"  {cat:15}: {cnt:4} files, {sz:9.2f} MB")

print("\n--- Duplicate Hash Groups Preview ---")
for h, paths in list(duplicates.items())[:10]:
    print(f"Hash {h[:12]}... ({len(paths)} copies):")
    for p in paths:
        print(f"    {p}")

with open("scratch/full_inventory.json", "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "duplicates": duplicates, "inventory": inventory}, f, indent=2)

print("\nSaved inventory to scratch/full_inventory.json")
