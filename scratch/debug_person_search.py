import sys, os
sys.path.insert(0, os.path.abspath("."))
from src.intent_router import IntentRouter
from src.web_retrieval import WebRetrievalEngine, DuckDuckGoProvider

q = "tell me who is the shanveer singh khosa on the linkedin"
router = IntentRouter()
dec = router.classify(q)
print("Intent:", dec.intent)
print("Entities:", dec.extracted_entities)
print("Search queries:", dec.search_queries)

engine = WebRetrievalEngine()
ddg = engine.providers[0]
print("Provider:", ddg)

for sq in dec.search_queries:
    res = ddg.search(sq, limit=5)
    print(f"Query '{sq}' -> {len(res)} results")
    for r in res:
        print("   Title:", r.title)
        print("   URL:", r.url)
