import sys, os
sys.path.insert(0, os.path.abspath("."))
import re
from src.intent_router import IntentRouter, INTENT_PERSON_PROFILE, INTENT_HEALTH_SAFETY, SUB_HEALTH_URGENT
from src.web_retrieval import WebRetrievalEngine, DuckDuckGoProvider, EvidenceBundle, EvidenceItem

def test_full_flow():
    router = IntentRouter()
    ddg = DuckDuckGoProvider()
    
    # Query A: Shanveer Singh Khosa
    q_a = "tell me who is the shanveer singh khosa on the linkedin"
    d_a = router.classify(q_a)
    print("Query A Intent:", d_a.intent, d_a.extracted_entities)
    
    # Simulate person search
    results_a = ddg.search("Shanveer Singh Khosa LinkedIn", limit=3)
    items_a = []
    for i, r in enumerate(results_a, 1):
        items_a.append(EvidenceItem(
            title=r.title,
            url=r.url,
            snippet=r.snippet,
            full_text=r.snippet,
            rank=i
        ))
    bundle_a = EvidenceBundle(query=q_a, items=items_a, has_results=bool(items_a))
    
    # Test person synthesis logic
    clean_name = d_a.extracted_entities.get("person_name", "").title()
    print("--- SYNTHESIZED PERSON ANSWER ---")
    print("Name:", clean_name)
    print("Primary URL:", bundle_a.items[0].url if bundle_a.items else "None")
    for it in bundle_a.items:
        print("Snippet:", it.snippet)
        
    print("=" * 60)
    
    # Query B: "i have pain in my heart"
    q_b = "i have pain in my heart"
    d_b = router.classify(q_b)
    print("Query B Intent:", d_b.intent, d_b.sub_intent)
    print("Immediate notice:\n", d_b.immediate_safety_response)

if __name__ == "__main__":
    test_full_flow()
