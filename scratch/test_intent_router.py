import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.intent_router import IntentRouter

router = IntentRouter()

test_queries = [
    ("tell me who is the shanveer singh khosa on the linkedin", "person_profile"),
    ("Find the LinkedIn profile of Shanveer Singh Khosa", "person_profile"),
    ("Who is the CEO of OpenAI?", "person_profile"),
    ("i have pain in my heart", "health_safety"),
    ("My chest hurts", "health_safety"),
    ("What are the common causes of chest pain?", "health_safety"),
    ("What is supervised learning?", "educational_concept"),
    ("What is deep learning?", "educational_concept"),
    ("Explain the difference between supervised and self-supervised learning", "comparison"),
    ("Hello! How are you?", "general_conversation"),
    ("What is 45 * 12?", "deterministic_tool"),
    ("What does torchvision do?", "technical_programming"),
    ("What is the latest release of PyTorch?", "current_events"),
    ("What did the second source say?", "followup_reference"),
    ("thing", "ambiguous_unspecified")
]

print("=" * 60)
print("TESTING INTENT ROUTER CLASSIFICATIONS")
print("=" * 60)
all_pass = True
for q, expected in test_queries:
    dec = router.classify(q, has_cached_web=True)
    passed = dec.intent == expected
    if not passed:
        all_pass = False
    print(f"[{'PASS' if passed else 'FAIL'}] '{q}'\n  -> Intent: {dec.intent} (Sub: {dec.sub_intent}) | Provider: {dec.preferred_provider} | Safety: {bool(dec.immediate_safety_response)}")
    if dec.search_queries:
        print(f"     Search queries: {dec.search_queries}")

print("=" * 60)
print(f"OVERALL INTENT TEST RESULT: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
