"""
scratch/run_full_system_verification.py
Comprehensive regression and acceptance test suite for Local AI Companion.
Validates:
1. Checkpoint integrity (SHA-256).
2. SQLite DB integrity & table counts.
3. Custom Model offline and online QA.
4. External Qwen Model offline and online QA (including Punjab CM).
5. Mathematical capabilities without function tools.
6. Seamless model switching (/model custom <-> /model external).
7. Crash simulation & error handling resilience.
8. Topic disambiguation (Punjab India vs Pakistan, cancer, stocks).
"""
import sys
import os
import time
import json
import hashlib
import sqlite3
import torch
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from model_manager import ModelManager
from chat import generate_chat_response
from web_retrieval import WebRetrievalEngine

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "checks": {},
    "passed": 0,
    "failed": 0
}

def record_check(name: str, passed: bool, details: str = ""):
    results["checks"][name] = {"passed": passed, "details": details}
    if passed:
        results["passed"] += 1
        print(f"[PASS] {name}: {details}")
    else:
        results["failed"] += 1
        print(f"[FAIL] {name}: {details}")

print("==================================================")
print("1. CHECKPOINT SHA-256 INTEGRITY")
print("==================================================")
ckpt_path = PROJECT_DIR / "checkpoints/ultimate_improvement/best.pt"
if ckpt_path.exists():
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest().upper()
    expected_sha = "CECBA4946A8B5A0404E42E9E1790C0D780228DA45E07B1910D190D89CBADF2FC"
    match = (actual_sha == expected_sha)
    record_check("checkpoint_sha256", match, f"SHA256={actual_sha} (Matches expected: {match})")
else:
    record_check("checkpoint_sha256", False, "Checkpoint best.pt not found")

print("\n==================================================")
print("2. SQLITE DATABASE INTEGRITY")
print("==================================================")
db_path = PROJECT_DIR / "data/companion_project.db"
if db_path.exists():
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check;")
        status = cur.fetchone()[0]
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cur.fetchall()]
        table_counts = {}
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t};")
            table_counts[t] = cur.fetchone()[0]
        conn.close()
        db_ok = (status.lower() == "ok")
        record_check("database_integrity", db_ok, f"Integrity={status}, Tables={table_counts}")
    except Exception as e:
        record_check("database_integrity", False, f"DB error: {str(e)}")
else:
    record_check("database_integrity", False, "DB file not found")

print("\n==================================================")
print("3. MODEL MANAGER INITIALIZATION & SPECS")
print("==================================================")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrieval_engine = WebRetrievalEngine()
mgr = ModelManager(device=dev, default_model="external")
status = mgr.get_status()
record_check("model_manager_init", True, f"Active={mgr.active_model_name}, Device={mgr.device}")

print("\n==================================================")
print("4. EXTERNAL QWEN MODEL — OFFLINE & MATH CAPABILITIES")
print("==================================================")
mgr.set_model("external")
# Math calculus
h1 = [{"role": "user", "text": "Differentiation of sin(x)/cos(x)"}]
r1, route1 = generate_chat_response(None, None, h1, dev, retrieval_engine, False, model_manager=mgr)
math_ok = "sec" in r1.lower() or "tan" in r1.lower()
record_check("qwen_offline_calculus", math_ok and "[Tool:" not in r1, f"Route={route1}, Ans: {r1[:120]}...")

# Math arithmetic
h2 = [{"role": "user", "text": "What is 45 * 12?"}]
r2, route2 = generate_chat_response(None, None, h2, dev, retrieval_engine, False, model_manager=mgr)
arith_ok = "540" in r2
record_check("qwen_offline_arithmetic", arith_ok and "[Tool:" not in r2, f"Route={route2}, Ans: {r2.strip()}")

print("\n==================================================")
print("5. EXTERNAL QWEN MODEL — LIVE WEB RETRIEVAL (PUNJAB CM)")
print("==================================================")
h3 = [{"role": "user", "text": "name of current cm of punjab india"}]
r3, route3 = generate_chat_response(None, None, h3, dev, retrieval_engine, True, model_manager=mgr)
bhagwant_ok = "bhagwant mann" in r3.lower() or "bhagwant" in r3.lower()
no_pakistan = "maryam" not in r3.lower() and "pakistan" not in r3.lower()
record_check("qwen_web_punjab_cm", bhagwant_ok and no_pakistan, f"Route={route3}, Ans: {r3.strip()}")

print("\n==================================================")
print("6. TOPIC DISAMBIGUATION — CANCER & STOCKS")
print("==================================================")
# Cancer query: must NOT return AI agent papers
bundle_cancer = retrieval_engine.search_and_retrieve("latest cancer research treatments", max_results=3, fetch_pages=False)
cancer_titles = " ".join([it.title.lower() for it in bundle_cancer.items])
cancer_clean = "agent" not in cancer_titles and "llm" not in cancer_titles
record_check("cancer_disambiguation", cancer_clean, f"Titles: {[it.title for it in bundle_cancer.items]}")

# Stock query
bundle_stock = retrieval_engine.search_and_retrieve("AAPL stock price today", max_results=3, fetch_pages=False)
stock_titles = " ".join([it.title.lower() for it in bundle_stock.items])
stock_clean = any(k in stock_titles for k in ["aapl", "apple", "stock", "nasdaq", "quote"])
record_check("stock_retrieval", stock_clean, f"Titles: {[it.title for it in bundle_stock.items]}")

print("\n==================================================")
print("7. MODEL SWITCHING & CUSTOM MODEL EVALUATION")
print("==================================================")
mgr.set_model("custom")
custom_model = mgr.custom_engine.model
custom_tok = mgr.custom_engine.tokenizer
record_check("model_switch_custom", mgr.active_model_name == "custom", f"Active={mgr.active_model_name}")

# Custom model conversational
h_cust = [{"role": "user", "text": "I feel really overwhelmed with my studies today"}]
r_cust, route_cust = generate_chat_response(custom_model, custom_tok, h_cust, dev, retrieval_engine, False, model_manager=mgr)
record_check("custom_model_conversational", len(r_cust) > 10, f"Route={route_cust}, Ans: {r_cust[:120]}...")

# Switch back to external
mgr.set_model("external")
record_check("model_switch_external", mgr.active_model_name == "external", f"Active={mgr.active_model_name}")

print("\n==================================================")
print("8. CRASH SIMULATION RESILIENCE")
print("==================================================")
# Test invalid URL/port handling: chat must not crash
class MockBadEngine:
    def generate(self, *args, **kwargs):
        import urllib.error
        raise urllib.error.HTTPError("http://localhost:8080/v1/chat/completions", 400, "Bad Request", {}, None)

old_ext = mgr.external_engine
mgr.external_engine = MockBadEngine()
h_err = [{"role": "user", "text": "trigger crash test"}]
try:
    r_err, route_err = generate_chat_response(None, None, h_err, dev, retrieval_engine, False, model_manager=mgr)
    record_check("crash_resilience", True, f"Recovered gracefully: Route={route_err}, Reply={r_err[:80]}...")
except Exception as e:
    record_check("crash_resilience", False, f"Chat crashed on exception: {e}")
finally:
    mgr.external_engine = old_ext

mgr.shutdown()

print("\n==================================================")
print(f"VERIFICATION SUMMARY: {results['passed']} PASSED, {results['failed']} FAILED")
print("==================================================")
with open(PROJECT_DIR / "scratch/regression_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

if results["failed"] == 0:
    print("\nALL VERIFICATION CHECKS PASSED PERFECTLY!")
    sys.exit(0)
else:
    print(f"\n{results['failed']} CHECKS FAILED.")
    sys.exit(1)
