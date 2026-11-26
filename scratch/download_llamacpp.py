"""
scratch/download_llamacpp.py
Download portable llama.cpp Windows binaries and extract to runtime/llama.cpp
"""
import os
import urllib.request
import zipfile
import shutil

RUNTIME_DIR = "D:/DEEP-LEARNING-PROJECT/runtime/llama.cpp"
ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b12567/llama-b12567-bin-win-cpu-x64.zip"
ZIP_PATH = os.path.join(RUNTIME_DIR, "llama_bin.zip")

print(f"Downloading llama.cpp from {ZIP_URL}...")
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
req = urllib.request.Request(ZIP_URL, headers=headers)
with urllib.request.urlopen(req, timeout=60) as resp, open(ZIP_PATH, "wb") as out:
    shutil.copyfileobj(resp, out)

print(f"Downloaded zip size: {os.path.getsize(ZIP_PATH) / (1024*1024):.2f} MB")

print("Extracting files...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(RUNTIME_DIR)

os.remove(ZIP_PATH)
print("Extracted files in runtime/llama.cpp:")
for f in os.listdir(RUNTIME_DIR):
    fpath = os.path.join(RUNTIME_DIR, f)
    if os.path.isfile(fpath):
        print(f"  - {f} ({os.path.getsize(fpath) / (1024*1024):.2f} MB)")
