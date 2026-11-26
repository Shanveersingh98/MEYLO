"""
scratch/verify_checkpoint_and_db.py
"""
import hashlib
import sqlite3
from pathlib import Path

# 1. SHA-256 Checkpoint Verification
ckpt_path = Path("checkpoints/ultimate_improvement/best.pt")
h = hashlib.sha256()
with open(ckpt_path, "rb") as f:
    while chunk := f.read(1024 * 1024):
        h.update(chunk)
sha = h.hexdigest().upper()
expected_sha = "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC"
print("Checkpoint Path :", ckpt_path)
print("Current SHA-256 :", sha)
print("Expected SHA-256:", expected_sha)
print("SHA-256 Match   :", sha == expected_sha)

# 2. Database Integrity & Table Counts
db_path = Path("data/companion_project.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("PRAGMA integrity_check;")
status = c.fetchall()
print("\nDatabase Integrity Check:", status)

c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in c.fetchall()]
counts = {}
for t in tables:
    c.execute(f"SELECT count(*) FROM {t};")
    counts[t] = c.fetchone()[0]
conn.close()

expected_counts = {
    'datasets': 84,
    'sqlite_sequence': 9,
    'dataset_processing_history': 1,
    'data_exclusions': 3,
    'checkpoints': 28,
    'training_sessions': 2,
    'training_loss_history': 131,
    'evaluation_results': 5,
    'cleanup_operations': 14,
    'application_events': 10
}
print("\nCurrent Table Counts:", counts)
print("Counts Match Baseline:", counts == expected_counts)
