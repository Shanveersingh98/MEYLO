"""
scratch/audit_project_integrity.py
Independent Audit - Requirement 5: Verify Project Integrity
"""
import sys
import os
import hashlib
import sqlite3
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

results = {}

print("==================================================")
print("1. CHECKPOINT SHA-256 INTEGRITY")
print("==================================================")
ckpt_path = PROJECT_DIR / "checkpoints/ultimate_improvement/best.pt"
assert ckpt_path.exists(), "best.pt must exist"
h = hashlib.sha256()
with open(ckpt_path, "rb") as f:
    while chunk := f.read(65536):
        h.update(chunk)
ckpt_sha = h.hexdigest().upper()
expected_ckpt_sha = "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC"
ckpt_match = (ckpt_sha == expected_ckpt_sha)
print(f"File: {ckpt_path.name}")
print(f"Size: {ckpt_path.stat().st_size:,} bytes")
print(f"SHA-256: {ckpt_sha}")
print(f"Matches Expected ({expected_ckpt_sha}): {ckpt_match}")
assert ckpt_match, "Checkpoint SHA-256 mismatch!"
results["checkpoint"] = {"path": str(ckpt_path), "size": ckpt_path.stat().st_size, "sha256": ckpt_sha, "verified": True}

print("\n==================================================")
print("2. QWEN MODEL FILE INTEGRITY")
print("==================================================")
qwen_path = PROJECT_DIR / "models/external/qwen2.5-1.5b-instruct/model.gguf"
assert qwen_path.exists(), "model.gguf must exist"
h_qwen = hashlib.sha256()
with open(qwen_path, "rb") as f:
    while chunk := f.read(65536):
        h_qwen.update(chunk)
qwen_sha = h_qwen.hexdigest().upper()
expected_qwen_sha = "6A1A2EB6D15622BF3C96857206351BA97E1AF16C30D7A74EE38970E434E9407E"
qwen_match = (qwen_sha == expected_qwen_sha)
print(f"File: {qwen_path.name}")
print(f"Size: {qwen_path.stat().st_size:,} bytes")
print(f"SHA-256: {qwen_sha}")
print(f"Matches Expected: {qwen_match}")
assert qwen_match, "Qwen GGUF SHA-256 mismatch!"
results["qwen_model"] = {"path": str(qwen_path), "size": qwen_path.stat().st_size, "sha256": qwen_sha, "verified": True}

print("\n==================================================")
print("3. SQLITE DATABASE INTEGRITY")
print("==================================================")
db_path = PROJECT_DIR / "data/companion_project.db"
assert db_path.exists(), "companion_project.db must exist"
conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
cur.execute("PRAGMA integrity_check;")
db_status = cur.fetchone()[0]
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [t[0] for t in cur.fetchall()]
table_counts = {}
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM {t};")
    table_counts[t] = cur.fetchone()[0]
conn.close()
print(f"Database: {db_path.name}")
print(f"Integrity Check: {db_status}")
print(f"Table Counts ({len(tables)} tables): {table_counts}")
assert db_status.lower() == "ok", "Database integrity check failed!"
assert len(tables) >= 5, "Database must have tables"
results["database"] = {"status": db_status, "tables": table_counts, "verified": True}

print("\n==================================================")
print("4. DATASET PRESERVATION")
print("==================================================")
data_dir = PROJECT_DIR / "data"
data_files = [f for f in data_dir.glob("**/*") if f.is_file()]
print(f"Total files in data/: {len(data_files)}")
assert len(data_files) > 10, "Data directory must contain preserved datasets"
results["datasets"] = {"file_count": len(data_files), "verified": True}

print("\n==================================================")
print("5. TOKENIZER & CONFIGS AVAILABILITY")
print("==================================================")
tok_path = PROJECT_DIR / "tokenizer/tokenizer.json"
assert tok_path.exists(), "tokenizer.json must exist"

# Check self-contained config in checkpoint
import torch
ckpt_meta = torch.load(str(ckpt_path), map_location="cpu")
custom_cfg = ckpt_meta.get("config", {})
assert "vocab_size" in custom_cfg and "d_model" in custom_cfg, "best.pt must contain valid model config"

# Check external model config
qwen_cfg_path = PROJECT_DIR / "models/external/qwen2.5-1.5b-instruct/model-config.json"
assert qwen_cfg_path.exists(), "Qwen model-config.json must exist"

print(f"Tokenizer: {tok_path.name} (exists={tok_path.exists()})")
print(f"Custom Model Config in best.pt: {custom_cfg}")
print(f"External Model Config in {qwen_cfg_path.name}: exists={qwen_cfg_path.exists()}")
results["tokenizers_and_configs"] = {
    "tokenizer": str(tok_path),
    "custom_config": custom_cfg,
    "qwen_config": str(qwen_cfg_path),
    "verified": True
}

print("\n==================================================")
print("6. LAUNCH SCRIPTS VERIFICATION")
print("==================================================")
launch_scripts = [
    "RUN_CHATBOT.bat",
    "start_qwen.bat",
    "start_with_web.bat",
    "start_without_web.bat",
    "help.bat",
    "commands.bat"
]
for s in launch_scripts:
    p = PROJECT_DIR / s
    assert p.exists(), f"Script {s} must exist"
    assert p.stat().st_size > 0, f"Script {s} must not be empty"
    print(f"  {s}: OK ({p.stat().st_size} bytes)")
results["launch_scripts"] = {"verified": True, "scripts": launch_scripts}

print("\n==================================================")
print("7. CORE APPLICATION IMPORTS & PATHS")
print("==================================================")
modules = [
    "chat",
    "model_manager",
    "web_retrieval",
    "source_selection",
    "question_understanding",
    "intent_router"
]
for m in modules:
    mod = __import__(m)
    print(f"  Module '{m}': Imported successfully from {getattr(mod, '__file__', 'unknown')}")
results["imports"] = {"verified": True, "modules": modules}

print("\nAUDIT 5 ALL CHECKS PASSED: Project integrity 100% verified!")
