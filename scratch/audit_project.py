"""
scratch/audit_project.py
Phase 1: Project & Hardware Audit Script (Standard Library + PyTorch)
"""
import sys
import os
import hashlib
import sqlite3
import shutil
import ctypes
import torch

print("=== SYSTEM & HARDWARE AUDIT ===")
print("Python Executable :", sys.executable)
print("Python Version    :", sys.version.replace('\n', ' '))
print("Current Working Dir:", os.getcwd())

# System Memory (RAM) via Windows API
class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]

mem = MEMORYSTATUSEX()
mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
total_ram = mem.ullTotalPhys / (1024**3)
avail_ram = mem.ullAvailPhys / (1024**3)
used_ram = total_ram - avail_ram
print(f"System RAM Total  : {total_ram:.2f} GB")
print(f"System RAM Avail  : {avail_ram:.2f} GB")
print(f"System RAM Used   : {used_ram:.2f} GB ({mem.dwMemoryLoad}%)")

# Disk Space
total, used, free = shutil.disk_usage("D:/DEEP-LEARNING-PROJECT")
print(f"Disk Space on D:  : Free={free / (1024**3):.2f} GB, Total={total / (1024**3):.2f} GB")

# CUDA & GPU
print("CUDA Available    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name   :", torch.cuda.get_device_name(0))
    print(f"GPU VRAM Total    : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"GPU VRAM Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.2f} MB")
    print(f"GPU VRAM Reserved : {torch.cuda.memory_reserved(0) / (1024**2):.2f} MB")

# Checkpoint SHA-256
ckpt_path = "checkpoints/ultimate_improvement/best.pt"
if os.path.exists(ckpt_path):
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    sha = h.hexdigest().upper()
    expected = "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC"
    print("Checkpoint Path   :", ckpt_path)
    print("Checkpoint SHA-256:", sha)
    print("Matches Expected  :", sha == expected)
else:
    print("CHECKPOINT NOT FOUND!")

# DB Integrity
db_path = "data/companion_project.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("PRAGMA integrity_check;")
    print("DB Integrity      :", c.fetchall())
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in c.fetchall()]
    counts = {}
    for t in tables:
        c.execute(f"SELECT count(*) FROM {t};")
        counts[t] = c.fetchone()[0]
    conn.close()
    print("DB Table Counts   :", counts)
else:
    print("DB NOT FOUND!")
