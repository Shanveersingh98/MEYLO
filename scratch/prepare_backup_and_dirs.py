"""
scratch/prepare_backup_and_dirs.py
Creates backups of critical files into backups/two_model_implementation_20261120/
and creates runtime/llama.cpp and models/external/qwen2.5-1.5b-instruct/
"""
import os
import shutil
import hashlib
import sqlite3

PROJECT_DIR = "D:/DEEP-LEARNING-PROJECT"
BACKUP_DIR = os.path.join(PROJECT_DIR, "backups", "two_model_implementation_20261120")
os.makedirs(BACKUP_DIR, exist_ok=True)

files_to_backup = [
    "src/chat.py",
    "src/companion_rules.py",
    "src/question_understanding.py",
    "src/web_retrieval.py",
    "src/source_selection.py",
    "src/answer_synthesis.py",
    "src/conversation_memory.py",
    "RUN_CHATBOT.bat",
    "start_with_web.bat",
    "start_without_web.bat",
]

print("=== BACKING UP CRITICAL FILES ===")
for rel_path in files_to_backup:
    src_path = os.path.join(PROJECT_DIR, rel_path)
    dst_path = os.path.join(BACKUP_DIR, os.path.basename(rel_path))
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Backed up: {rel_path} -> {dst_path}")
    else:
        print(f"Warning: File not found to backup: {rel_path}")

# Create destination directories
models_dir = os.path.join(PROJECT_DIR, "models", "external", "qwen2.5-1.5b-instruct")
runtime_dir = os.path.join(PROJECT_DIR, "runtime", "llama.cpp")
cache_dir = os.path.join(PROJECT_DIR, ".cache")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(runtime_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

print("\nCreated directories:")
print(" - Models :", models_dir)
print(" - Runtime:", runtime_dir)
print(" - Cache  :", cache_dir)

# Verify Checkpoint
ckpt_path = os.path.join(PROJECT_DIR, "checkpoints/ultimate_improvement/best.pt")
h = hashlib.sha256()
with open(ckpt_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
sha = h.hexdigest().upper()
expected = "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC"
assert sha == expected, f"Checkpoint SHA-256 mismatch! Got {sha}"
print(f"\nCheckpoint SHA-256 Verified: {sha}")

# Verify DB
db_path = os.path.join(PROJECT_DIR, "data/companion_project.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("PRAGMA integrity_check;")
res = c.fetchall()
conn.close()
assert res == [('ok',)], f"DB integrity failed: {res}"
print(f"Database Integrity Verified: {res}")
print("\nPhase 1 Backup and Directory Preparation Complete.")
