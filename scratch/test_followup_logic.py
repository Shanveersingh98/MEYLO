import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
from web_retrieval import WebRetrievalEngine, EvidenceBundle, EvidenceItem

engine = WebRetrievalEngine()
# Seed with supervised learning evidence
bundle = engine.search_and_retrieve("What is supervised learning?")
print("Seeded bundle:", bundle.items[0].title)

for follow_up in [
    "Explain that in simpler words.",
    "Give me a shorter summary.",
    "What did the second source say?",
    "What are the main points from the first source?"
]:
    print("\n--- TEST FOLLOW-UP:", follow_up)
    res = engine.handle_followup_reference(follow_up)
    print("RESULT:\n", res)
