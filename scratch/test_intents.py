import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
from intent_router import IntentRouter

router = IntentRouter()

test_cases = [
    # 1. General conversation
    ("hello there!", "general_conversation"),
    ("how are you doing today?", "general_conversation"),
    ("who are you?", "general_conversation"),
    ("thank you so much", "general_conversation"),

    # 2. Educational & factual
    ("what is deep learning?", "educational_concept"),
    ("explain supervised learning in simple words", "educational_concept"),
    ("what is self-supervised learning?", "educational_concept"),
    ("what is reinforcement learning?", "educational_concept"),

    # 3. People, public profiles, companies
    ("tell me who is the shanveer singh khosa on the linkedin", "person_profile"),
    ("Who is Shanveer Singh Khosa on LinkedIn?", "person_profile"),
    ("Find the LinkedIn profile of Satya Nadella", "person_profile"),
    ("who is the CEO of Microsoft", "person_profile"),
    ("who founded Apple", "person_profile"),
    ("What professional information is publicly available about Geoffrey Hinton?", "person_profile"),

    # 4. Current events
    ("what is the latest release of PyTorch?", "current_events"),
    ("what are the new features in Python 3.12?", "current_events"),

    # 5. Technical & programming
    ("find the official documentation for torch.nn.Linear", "technical_programming"),
    ("what does torchvision.transforms do?", "technical_programming"),

    # 6. Health & safety
    ("i have pain in my heart", "health_safety"),
    ("my chest hurts and i have trouble breathing", "health_safety"),
    ("what are the common causes of chest pain?", "health_safety"),
    ("what are the emergency warning signs of a heart attack?", "health_safety"),

    # 7. Product & service
    ("what is the price of an RTX 4090 GPU?", "product_service"),
    ("review of MacBook Pro M3 for deep learning", "product_service"),

    # 8. Calculations & tool operations
    ("what is 45 * 12 + 100", "deterministic_tool"),
    ("calculate 1024 / 8", "deterministic_tool"),

    # 9. Document summary
    ("summarize https://en.wikipedia.org/wiki/Artificial_intelligence", "document_summary"),

    # 10. Comparisons
    ("explain the difference between AI, machine learning, and deep learning", "comparison"),
    ("compare supervised learning and self-supervised learning", "comparison"),

    # 11. Follow-up
    ("what did the second source say?", "followup_reference"),
    ("explain that in simpler words", "followup_reference"),

    # 12. Ambiguous
    ("stuff", "ambiguous_unspecified"),
]

passed = 0
for prompt, expected in test_cases:
    decision = router.classify(prompt, has_cached_web=True)
    ok = (decision.intent == expected)
    if ok:
        passed += 1
        print(f"[PASS] '{prompt}' -> {decision.intent} ({decision.sub_intent or ''})")
    else:
        print(f"[FAIL] '{prompt}' -> GOT {decision.intent}, EXPECTED {expected}")

print(f"\nTotal: {passed}/{len(test_cases)} passed.")
