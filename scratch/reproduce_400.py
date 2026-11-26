"""
scratch/reproduce_400.py
Reproduce HTTP 400 error on: name of current cm of punjab india
"""
import urllib.request
import urllib.error
import json
import traceback
import sys
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from web_retrieval import WebRetrievalEngine
from model_manager import ModelManager

retrieval = WebRetrievalEngine()
query = "name of current cm of punjab india"
print(f"Searching for: '{query}'")
bundle = retrieval.search_and_retrieve(query, max_results=3, fetch_pages=True)
print(f"Results found: {len(bundle.items)}")
for idx, it in enumerate(bundle.items, 1):
    text_len = len(it.full_text or it.snippet or "")
    print(f"  Source {idx}: {it.title} | {it.url} | text length: {text_len}")

evidence_lines = []
for idx, it in enumerate(bundle.items, 1):
    date_str = f" [Date: {it.date}]" if getattr(it, "date", None) else ""
    pub_str = f" [Publisher: {it.publisher}]" if getattr(it, "publisher", None) else ""
    evidence_lines.append(f"Source [{idx}]: {it.title}{date_str}{pub_str}\nURL: {it.url}\nInformation: {it.full_text or it.snippet}")
evidence_str = "\n\n".join(evidence_lines)
print(f"Total evidence_str length (chars): {len(evidence_str)}")

mgr = ModelManager(default_model="external", device="cpu")
print("Calling external_engine.generate...")
try:
    reply, lat = mgr.external_engine.generate(
        prompt=query,
        history=[],
        context_evidence=evidence_str,
        system_instruction="You are a helpful, intelligent local AI Companion with live web access. Answer the user's question accurately, concisely, and naturally based on the verified web evidence provided below. Ground your facts strictly in the retrieved evidence."
    )
    print(f"Generated reply ({lat:.1f}ms): {reply[:300]}")
except urllib.error.HTTPError as e:
    print(f"HTTP ERROR CAUGHT: {e.code} {e.reason}")
    body = e.read().decode("utf-8")
    print(f"HTTP Error Body: {body}")
except Exception as e:
    print(f"Exception: {e}")
    traceback.print_exc()
