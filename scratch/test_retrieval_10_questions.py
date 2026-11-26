import sys
import json
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from web_retrieval import WebRetrievalEngine

engine = WebRetrievalEngine()

test_queries = [
    "What is deep learning?",
    "What is ChatGPT?",
    "What is machine learning?",
    "Explain the difference between AI, machine learning, and deep learning.",
    "Give three real-world applications of deep learning.",
    "Explain supervised learning in simple words.",
    "What is a neural network?",
    "Summarize an accessible public educational source about deep learning: https://en.wikipedia.org/wiki/Deep_learning",
    "What are the latest release updates for Python 3.12?"
]

for q in test_queries:
    clean_q = engine.extract_search_query(q)
    should, reason = engine.should_search_web(q, web_mode_enabled=True)
    bundle = engine.search_and_retrieve(q, max_results=3, fetch_pages=True)
    print("=" * 70)
    print(f"QUERY: {q}")
    print(f"CLEAN_Q: {clean_q} | SHOULD_SEARCH: {should} ({reason})")
    print(f"HAS_RESULTS: {bundle.has_results} | COUNT: {len(bundle.items)} | ERROR: {bundle.error_message}")
    for item in bundle.items:
        print(f"  [{item.rank}] {item.title} -> {item.url}")
        print(f"      Snippet: {item.snippet[:120]}...")
        if item.full_text:
            print(f"      Full text (first 150 chars): {item.full_text[:150]}...")
