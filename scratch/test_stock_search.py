"""
scratch/test_stock_search.py
Test DuckDuckGo retrieval for live stock quotes and market queries.
"""
import sys
sys.path.insert(0, "src")
from web_retrieval import DuckDuckGoProvider

ddg = DuckDuckGoProvider()
for q in ["AAPL stock price today", "NVIDIA stock price quote"]:
    hits = ddg.search(q, limit=2)
    print(f"\nQuery: '{q}' -> Hits: {len(hits)}")
    for h in hits:
        print("Title:", h.title)
        print("URL  :", h.url)
        print("Snip :", h.snippet[:120])
