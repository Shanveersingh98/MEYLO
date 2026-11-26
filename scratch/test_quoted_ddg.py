import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.web_retrieval import DuckDuckGoProvider

ddg = DuckDuckGoProvider()
for q in ['"Shanveer Singh Khosa" LinkedIn', 'Shanveer Singh Khosa LinkedIn', 'site:linkedin.com/in/ "Shanveer Singh Khosa"']:
    res = ddg.search(q, limit=3)
    print(f"Query: {q} -> Found {len(res)} results")
    for r in res:
        print(f"   [{r.title}] ({r.url})")
