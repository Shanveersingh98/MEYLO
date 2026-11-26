"""
scratch/download_qwen_model.py
Download Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M) into models/external/qwen2.5-1.5b-instruct/model.gguf
and verify SHA-256 hash against official Hugging Face release.
"""
import os
import sys
import json
import hashlib
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
EXPECTED_SHA256 = "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e"
EXPECTED_SIZE = 1117320736

TARGET_DIR = "D:/DEEP-LEARNING-PROJECT/models/external/qwen2.5-1.5b-instruct"
TARGET_FILE = os.path.join(TARGET_DIR, "model.gguf")
CONFIG_FILE = os.path.join(TARGET_DIR, "model-config.json")
INFO_FILE = os.path.join(TARGET_DIR, "download-info.txt")

os.makedirs(TARGET_DIR, exist_ok=True)
CACHE_DIR = "D:/DEEP-LEARNING-PROJECT/.cache/huggingface"
os.makedirs(CACHE_DIR, exist_ok=True)

print(f"=== DOWNLOADING {FILENAME} FROM {REPO_ID} ===")
print("Target Destination:", TARGET_FILE)
print("Cache Directory    :", CACHE_DIR)

downloaded_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILENAME,
    local_dir=TARGET_DIR,
    cache_dir=CACHE_DIR
)

# Rename to model.gguf if downloaded as original filename
if os.path.exists(downloaded_path) and downloaded_path != TARGET_FILE:
    if os.path.exists(TARGET_FILE):
        os.remove(TARGET_FILE)
    os.rename(downloaded_path, TARGET_FILE)

file_size = os.path.getsize(TARGET_FILE)
print(f"Downloaded file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

# Verify SHA-256
print("Computing SHA-256 hash...")
h = hashlib.sha256()
with open(TARGET_FILE, "rb") as f:
    while chunk := f.read(4 * 1024 * 1024):
        h.update(chunk)
computed_sha256 = h.hexdigest().lower()

print("Computed SHA-256 :", computed_sha256)
print("Expected SHA-256 :", EXPECTED_SHA256)
assert computed_sha256 == EXPECTED_SHA256, "SHA-256 mismatch!"
print("SHA-256 VERIFICATION: SUCCESS!")

# Write model-config.json
config_data = {
    "model_name": "Qwen2.5-1.5B-Instruct",
    "format": "GGUF",
    "quantization": "Q4_K_M",
    "parameters": "1.54B",
    "context_length": 4096,
    "repo_id": REPO_ID,
    "filename": FILENAME,
    "sha256": computed_sha256,
    "size_bytes": file_size,
    "license": "Apache-2.0",
    "intended_use": "Direct instruction-following local chat inference"
}
with open(CONFIG_FILE, "w", encoding="utf-8") as f:
    json.dump(config_data, f, indent=4)
print("Wrote:", CONFIG_FILE)

# Write download-info.txt
info_text = f"""Model: Qwen2.5-1.5B-Instruct
Source: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF
File: {FILENAME}
Quantization: Q4_K_M (4-bit medium k-quant)
File Size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)
SHA-256 Checksum: {computed_sha256}
License: Apache-2.0
Inference Runtime: llama.cpp (portable x86_64 build 12567 in runtime/llama.cpp)
Downloaded To: D:\\DEEP-LEARNING-PROJECT\\models\\external\\qwen2.5-1.5b-instruct\\model.gguf
"""
with open(INFO_FILE, "w", encoding="utf-8") as f:
    f.write(info_text)
print("Wrote:", INFO_FILE)
print("\nPhase 2 Model Download & Verification Complete.")
