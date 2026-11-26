import sys
import os
sys.path.insert(0, os.path.abspath("."))
import re
import urllib.parse
from src.intent_router import IntentRouter, INTENT_PERSON_PROFILE, INTENT_HEALTH_SAFETY, SUB_HEALTH_URGENT, SUB_HEALTH_GENERAL
from src.web_retrieval import WebRetrievalEngine, SearchResult, EvidenceItem, EvidenceBundle, DuckDuckGoProvider

def test_person_synthesis():
    query = "tell me who is the shanveer singh khosa on the linkedin"
    router = IntentRouter()
    decision = router.classify(query)
    print("Decision:", decision.intent, decision.extracted_entities)
    
    # Test DDG search with person name
    ddg = DuckDuckGoProvider()
    results = ddg.search("Shanveer Singh Khosa LinkedIn", limit=5)
    print(f"Retrieved {len(results)} results:")
    for r in results:
        print(f"  - [{r.title}] ({r.url})")
        print(f"    Snippet: {r.snippet[:150]}")

def test_health_query():
    query = "i have pain in my heart"
    router = IntentRouter()
    decision = router.classify(query)
    print("Health Decision:", decision.intent, decision.sub_intent, "Urgent:", decision.sub_intent == SUB_HEALTH_URGENT)
    print("Immediate guidance:\n", decision.immediate_safety_response)

if __name__ == "__main__":
    test_person_synthesis()
    print("-" * 50)
    test_health_query()
