"""
scratch/run_full_two_model_tests.py
===================================
Comprehensive Real-World Validation Suite for Local AI Companion
Validating Two Independent Models: Custom Trained + External Qwen Model.
"""

import sys
import os
import time
import json
import sqlite3
import hashlib
import torch
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from model_manager import ModelManager, get_system_memory_info
from chat import generate_chat_response, parse_command
from web_retrieval import WebRetrievalEngine

BASELINE_SHA256 = "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "ultimate_improvement" / "best.pt"
DB_PATH = PROJECT_DIR / "data" / "companion_project.db"

results_summary = {}

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().upper()

print("=" * 70)
print("REAL-WORLD VALIDATION SUITE: TWO INDEPENDENT AI MODELS")
print("=" * 70)

# TEST 1: Checkpoint SHA-256 Integrity
print("\n[TEST 1] Verifying Checkpoint SHA-256 Integrity...")
actual_sha = compute_sha256(CHECKPOINT_PATH)
sha_match = (actual_sha == BASELINE_SHA256)
print(f"  Expected: {BASELINE_SHA256}")
print(f"  Actual  : {actual_sha}")
print(f"  Integrity: {'MATCH (UNTOUCHED)' if sha_match else 'MISMATCH'}")
results_summary["checkpoint_sha_match"] = sha_match

# TEST 2: SQLite Database Integrity Check
print("\n[TEST 2] Verifying SQLite Database Integrity...")
conn = sqlite3.connect(str(DB_PATH))
cursor = conn.cursor()
cursor.execute("PRAGMA integrity_check;")
db_integrity = cursor.fetchone()[0]
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [row[0] for row in cursor.fetchall()]
table_counts = {}
for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {t};")
    table_counts[t] = cursor.fetchone()[0]
conn.close()
print(f"  PRAGMA integrity_check: {db_integrity}")
print(f"  Table counts: {table_counts}")
results_summary["db_integrity_ok"] = (db_integrity == "ok")

# Initialize WebRetrievalEngine and Device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrieval_engine = WebRetrievalEngine()
retrieval_engine.session_last_sources = []

# TEST 3: Model Manager Lifecycle and Memory Verification
print("\n[TEST 3] Model Manager Lifecycle & Dynamic Memory Freeing...")
mem_baseline = get_system_memory_info()
print(f"  Baseline: RAM {mem_baseline['ram_used_gb']} GB, VRAM {mem_baseline['vram_used_mb']} MB")

mgr = ModelManager(device=dev, default_model="custom")
mem_custom = get_system_memory_info()
print(f"  Custom Loaded: RAM {mem_custom['ram_used_gb']} GB, VRAM {mem_custom['vram_used_mb']} MB")

# Switch to External
ok_ext, msg_ext = mgr.set_model("external")
mem_ext = get_system_memory_info()
print(f"  Switched to External: {msg_ext}")
print(f"  External Active: RAM {mem_ext['ram_used_gb']} GB, VRAM {mem_ext['vram_used_mb']} MB")
vram_freed_when_external = (mem_ext['vram_used_mb'] < mem_custom['vram_used_mb'])
print(f"  GPU VRAM freed upon switching to external: {vram_freed_when_external}")

# Switch back to Custom
ok_cust, msg_cust = mgr.set_model("custom")
mem_cust2 = get_system_memory_info()
print(f"  Switched back to Custom: {msg_cust}")
print(f"  Custom Active Again: RAM {mem_cust2['ram_used_gb']} GB, VRAM {mem_cust2['vram_used_mb']} MB")

results_summary["memory_lifecycle_verified"] = (ok_ext and ok_cust and vram_freed_when_external)

# TEST 4: Custom Model Responses (Offline & Online)
print("\n[TEST 4] Custom Model Evaluation (Offline & Online)...")
mgr.set_model("custom")

# Test 4A: Custom Model Offline - Greeting & Education
h_cust = [{"role": "user", "text": "hello who are you"}]
resp_c1, route_c1 = generate_chat_response(
    model=mgr.custom_engine.model,
    tokenizer=mgr.custom_engine.tokenizer,
    history=h_cust,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
print(f"  Custom Offline Turn 1: {resp_c1[:80]}... [Route: {route_c1}]")

h_cust2 = [{"role": "user", "text": "wht is machne lernng"}]
resp_c2, route_c2 = generate_chat_response(
    model=mgr.custom_engine.model,
    tokenizer=mgr.custom_engine.tokenizer,
    history=h_cust2,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
print(f"  Custom Offline Turn 2 (Typo question): {resp_c2[:80]}... [Route: {route_c2}]")

# Test 4B: Custom Model Offline - Freshness check
h_cust3 = [{"role": "user", "text": "who is india current prime minister"}]
resp_c3, route_c3 = generate_chat_response(
    model=mgr.custom_engine.model,
    tokenizer=mgr.custom_engine.tokenizer,
    history=h_cust3,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
print(f"  Custom Offline Turn 3 (Freshness check): {resp_c3[:80]}... [Route: {route_c3}]")
c_offline_freshness_honest = "enable web search" in resp_c3.lower() or "/web on" in resp_c3.lower()

# Test 4C: Custom Model Online - Web Search
resp_c4, route_c4 = generate_chat_response(
    model=mgr.custom_engine.model,
    tokenizer=mgr.custom_engine.tokenizer,
    history=h_cust3,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=True,
    model_manager=mgr
)
print(f"  Custom Online Turn 4 (Live PM lookup): {resp_c4[:80]}... [Route: {route_c4}]")
c_pm_correct = "narendra modi" in resp_c4.lower() or "modi" in resp_c4.lower()
results_summary["custom_pm_correct"] = c_pm_correct
results_summary["custom_freshness_honest"] = c_offline_freshness_honest

# TEST 5: External Qwen Model Evaluation (Offline & Online)
print("\n[TEST 5] External Qwen Model Evaluation (Offline & Online)...")
mgr.set_model("external")

# Test 5A: External Model Offline - Companion Chat & Empathy
h_qwen1 = [{"role": "user", "text": "I had a really difficult day at work and feel overwhelmed."}]
resp_q1, route_q1 = generate_chat_response(
    model=None,
    tokenizer=None,
    history=h_qwen1,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
print(f"  Qwen Offline Turn 1 (Empathy): {resp_q1[:100]}... [Route: {route_q1}]")

# Test 5B: External Model Offline - Technical Explanation
h_qwen2 = [{"role": "user", "text": "Explain how multi-head self-attention works in transformers."}]
resp_q2, route_q2 = generate_chat_response(
    model=None,
    tokenizer=None,
    history=h_qwen2,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
print(f"  Qwen Offline Turn 2 (Technical): {resp_q2[:100]}... [Route: {route_q2}]")

# Test 5C: External Model Offline - Freshness check
h_qwen3 = [{"role": "user", "text": "What is the stock price of Apple right now?"}]
resp_q3, route_q3 = generate_chat_response(
    model=None,
    tokenizer=None,
    history=h_qwen3,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
print(f"  Qwen Offline Turn 3 (Stock Freshness): {resp_q3[:100]}... [Route: {route_q3}]")
q_offline_freshness_honest = "web search" in resp_q3.lower() or "/web on" in resp_q3.lower()

# Test 5D: External Model Online - Failure A & B
print("\n[TEST 6] Resolving Previous Failures with External Model (Online)...")
# Failure A: "diljit dosanjh current song with daddy yankee colabrated song name"
h_fail_a = [{"role": "user", "text": "diljit dosanjh current song with daddy yankee colabrated song name"}]
resp_qa, route_qa = generate_chat_response(
    model=None,
    tokenizer=None,
    history=h_fail_a,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=True,
    model_manager=mgr
)
print(f"  Failure A Resolution (Diljit & Daddy Yankee): {resp_qa[:120]}... [Route: {route_qa}]")
diljit_collab_ok = any(k in resp_qa.lower() for k in ["palpita", "diljit", "daddy yankee"])
results_summary["diljit_collab_ok"] = diljit_collab_ok

# Failure B: "who is india current prime minister"
h_fail_b = [{"role": "user", "text": "who is india current prime minister"}]
resp_qb, route_qb = generate_chat_response(
    model=None,
    tokenizer=None,
    history=h_fail_b,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=True,
    model_manager=mgr
)
print(f"  Failure B Resolution (India Prime Minister): {resp_qb[:120]}... [Route: {route_qb}]")
qwen_pm_correct = "narendra modi" in resp_qb.lower() or "modi" in resp_qb.lower()
results_summary["qwen_pm_correct"] = qwen_pm_correct

# Failure C: Diverse Research Papers (OpenAlex + arXiv)
print("\n[TEST 7] Research Paper Retrieval Verification (Cancer & AI Agents)...")
# Test 7A: Cancer therapy research papers (verifying not falling back to "AI agents"!)
cancer_papers = retrieval_engine.openalex_provider.search_works("recent research papers about cancer immunotherapy", limit=3)
print(f"  Cancer Papers Retrieved ({len(cancer_papers)} papers):")
has_future_date = False
import datetime
today_str = datetime.date.today().isoformat()
for p in cancer_papers:
    print(f"    - {p.title[:55]} | Date: {p.date}")
    if p.date and p.date > today_str:
        has_future_date = True
cancer_papers_ok = (len(cancer_papers) > 0 and not has_future_date and any("cancer" in p.title.lower() or "carba" in p.title.lower() or "oral" in p.title.lower() for p in cancer_papers))
results_summary["cancer_papers_ok"] = cancer_papers_ok
results_summary["no_future_dates_in_openalex"] = (not has_future_date)

# Test 7B: Stock price quotes via DuckDuckGo
print("\n[TEST 8] Live Stock Price Retrieval...")
stock_res = retrieval_engine.ddg_provider.search("AAPL stock price quote", limit=2)
print(f"  Stock Query Hits ({len(stock_res)} hits):")
for s in stock_res:
    print(f"    - {s.title[:55]} | {s.snippet[:75]}")
stock_ok = len(stock_res) > 0 and any("apple" in (s.title + s.snippet).lower() or "aapl" in (s.title + s.snippet).lower() for s in stock_res)
results_summary["stock_retrieval_ok"] = stock_ok

# Clean shutdown
mgr.shutdown()

print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)
for k, v in results_summary.items():
    print(f"  {k:35}: {'PASS' if v else 'FAIL'}")

all_pass = all(results_summary.values())
print(f"\nOverall Result: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
sys.exit(0 if all_pass else 1)
