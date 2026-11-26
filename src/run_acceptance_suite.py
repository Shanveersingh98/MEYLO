"""
src/run_acceptance_suite.py
===========================
Automated Acceptance Test Suite for From-Scratch Chatbot Universal QA & Retrieval.
Executes against actual runtime application components:
- IntentRouter
- WebRetrievalEngine & Search Providers
- RobustContentExtractor & SSRF Security
- Chat generation & follow-up dialogues
Outputs all required JSON reports to logs/universal_question_answering/
"""

import sys
import os
import time
import json
import re
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

import torch
from intent_router import (
    IntentRouter,
    INTENT_CONVERSATION, INTENT_EDUCATIONAL, INTENT_PERSON_PROFILE,
    INTENT_CURRENT_EVENTS, INTENT_TECHNICAL, INTENT_HEALTH_SAFETY,
    INTENT_PRODUCT_SERVICE, INTENT_TOOL, INTENT_DOCUMENT_SUMMARY,
    INTENT_COMPARISON, INTENT_FOLLOWUP, INTENT_AMBIGUOUS,
    SUB_HEALTH_URGENT, SUB_HEALTH_GENERAL
)
from web_retrieval import (
    WebRetrievalEngine, RobustContentExtractor, extract_text_from_html,
    fetch_page_content, is_safe_url, clean_mathematical_notation,
    EvidenceBundle, EvidenceItem
)
from chat import load_chatbot, generate_chat_response

REPORTS_DIR = PROJECT_DIR / "logs" / "universal_question_answering"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("UNIVERSAL QUESTION ANSWERING & INTELLIGENT RETRIEVAL ACCEPTANCE SUITE")
print("=" * 70)

# Load chatbot
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading chatbot on device: {device}...")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
router = retrieval_engine.intent_router
print("Chatbot and retrieval engine ready.\n")


# =====================================================================
# SUITE 1: INTENT ROUTING RESULTS
# =====================================================================
print("--- [Suite 1/6] Running Universal Intent Routing Suite ---")
intent_test_cases = [
    # 1. General conversation
    ("hello there!", INTENT_CONVERSATION, None, False),
    ("how are you doing today?", INTENT_CONVERSATION, None, False),
    ("who are you?", INTENT_CONVERSATION, None, False),
    ("thank you so much", INTENT_CONVERSATION, None, False),
    ("good morning friend", INTENT_CONVERSATION, None, False),

    # 2. Educational & factual
    ("what is deep learning?", INTENT_EDUCATIONAL, None, True),
    ("explain supervised learning in simple words", INTENT_EDUCATIONAL, None, True),
    ("what is self-supervised learning?", INTENT_EDUCATIONAL, None, True),
    ("what is reinforcement learning?", INTENT_EDUCATIONAL, None, True),

    # 3. People, public profiles, companies
    ("tell me who is the shanveer singh khosa on the linkedin", INTENT_PERSON_PROFILE, None, True),
    ("Who is Shanveer Singh Khosa on LinkedIn?", INTENT_PERSON_PROFILE, None, True),
    ("Find the LinkedIn profile of Satya Nadella", INTENT_PERSON_PROFILE, None, True),
    ("who is the CEO of Microsoft", INTENT_PERSON_PROFILE, None, True),
    ("who founded Apple", INTENT_PERSON_PROFILE, None, True),
    ("What professional information is publicly available about Geoffrey Hinton?", INTENT_PERSON_PROFILE, None, True),

    # 4. Current events
    ("what is the latest release of PyTorch?", INTENT_CURRENT_EVENTS, None, True),
    ("what are the new features in Python 3.12?", INTENT_CURRENT_EVENTS, None, True),

    # 5. Technical & programming
    ("find the official documentation for torch.nn.Linear", INTENT_TECHNICAL, None, True),
    ("what does torchvision.transforms do?", INTENT_TECHNICAL, None, True),

    # 6. Health & safety
    ("i have pain in my heart", INTENT_HEALTH_SAFETY, SUB_HEALTH_URGENT, True),
    ("my chest hurts and i have trouble breathing", INTENT_HEALTH_SAFETY, SUB_HEALTH_URGENT, True),
    ("what are the common causes of chest pain?", INTENT_HEALTH_SAFETY, SUB_HEALTH_GENERAL, True),
    ("what are the emergency warning signs of a heart attack?", INTENT_HEALTH_SAFETY, SUB_HEALTH_GENERAL, True),

    # 7. Product & service
    ("what is the price of an RTX 4090 GPU?", INTENT_PRODUCT_SERVICE, None, True),
    ("review of MacBook Pro M3 for deep learning", INTENT_PRODUCT_SERVICE, None, True),

    # 8. Calculations & tools
    ("what is 45 * 12 + 100", INTENT_TOOL, None, False),
    ("calculate 1024 / 8", INTENT_TOOL, None, False),

    # 9. Document summary
    ("summarize https://en.wikipedia.org/wiki/Artificial_intelligence", INTENT_DOCUMENT_SUMMARY, None, True),

    # 10. Comparisons
    ("explain the difference between AI, machine learning, and deep learning", INTENT_COMPARISON, None, True),
    ("compare supervised learning and self-supervised learning", INTENT_COMPARISON, None, True),

    # 11. Follow-up
    ("what did the second source say?", INTENT_FOLLOWUP, None, False),
    ("explain that in simpler words", INTENT_FOLLOWUP, None, False),

    # 12. Ambiguous
    ("stuff", INTENT_AMBIGUOUS, None, False),
]

intent_results = []
for prompt, exp_intent, exp_sub, exp_web in intent_test_cases:
    t0 = time.time()
    decision = router.classify(prompt, has_cached_web=True)
    lat = round((time.time() - t0) * 1000, 2)

    passed_intent = (decision.intent == exp_intent)
    passed_sub = (decision.sub_intent == exp_sub) if exp_sub else True
    passed_web = (decision.needs_web_search == exp_web)
    passed = passed_intent and passed_sub and passed_web

    record = {
        "prompt": prompt,
        "expected_intent": exp_intent,
        "actual_intent": decision.intent,
        "expected_sub_intent": exp_sub,
        "actual_sub_intent": decision.sub_intent,
        "expected_needs_web": exp_web,
        "actual_needs_web": decision.needs_web_search,
        "preferred_provider": decision.preferred_provider,
        "confidence": decision.confidence,
        "latency_ms": lat,
        "passed": passed
    }
    intent_results.append(record)
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] '{prompt}' -> {decision.intent} ({decision.sub_intent or 'none'}) [{lat}ms]")

with open(REPORTS_DIR / "intent_routing_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(intent_results),
        "passed": sum(1 for r in intent_results if r["passed"]),
        "failed": sum(1 for r in intent_results if not r["passed"]),
        "results": intent_results
    }, f, indent=2)


# =====================================================================
# SUITE 2: PUBLIC PROFILE SEARCH RESULTS
# =====================================================================
print("\n--- [Suite 2/6] Running Public Profile Search Suite ---")
profile_cases = [
    ("tell me who is the shanveer singh khosa on the linkedin", "Shanveer Singh Khosa", False),
    ("Who is Shanveer Singh Khosa on LinkedIn?", "Shanveer Singh Khosa", False),
    ("Find the LinkedIn profile of Satya Nadella", "Satya Nadella", True),
    ("Who is Geoffrey Hinton?", "Geoffrey Hinton", True),
    ("What professional information is publicly available about Geoffrey Hinton?", "Geoffrey Hinton", True),
    ("Who is the CEO of Microsoft?", "Satya Nadella", True),
    ("Who founded Apple?", "Apple Inc.", True)
]

profile_results = []
for query, target_entity, expects_verified in profile_cases:
    t0 = time.time()
    history = [{"role": "user", "text": query}]
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=True
    )
    lat = round((time.time() - t0) * 1000, 2)

    has_educational_error = "Could not locate verified educational sources" in reply
    has_profile_notice = "[Profile Notice:" in reply or "Public Professional Profile" in reply
    identity_ambiguity_noted = "Identity Ambiguity Notice" in reply
    privacy_noted = "Data Privacy & Verification Note" in reply
    no_hallucinated_tokens = not any(w in reply.lower() for w in ["mach is a", "burgpt", "dfecdm", "rusearching"])

    if not expects_verified:
        passed = (not has_educational_error) and has_profile_notice and ("Shanveer Singh Khosa" in reply)
    else:
        passed = (not has_educational_error) and ("Public Professional Profile" in reply) and (target_entity.lower() in reply.lower())

    record = {
        "query": query,
        "target_entity": target_entity,
        "route": route,
        "response_preview": reply[:250].replace("\n", " "),
        "has_educational_error": has_educational_error,
        "has_profile_notice": has_profile_notice,
        "identity_ambiguity_noted": identity_ambiguity_noted,
        "privacy_noted": privacy_noted,
        "latency_ms": lat,
        "passed": passed
    }
    profile_results.append(record)
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] '{query}' -> route: {route} [{lat}ms]")

with open(REPORTS_DIR / "profile_search_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(profile_results),
        "passed": sum(1 for r in profile_results if r["passed"]),
        "failed": sum(1 for r in profile_results if not r["passed"]),
        "results": profile_results
    }, f, indent=2)


# =====================================================================
# SUITE 3: HEALTH AND SAFETY RESPONSE TESTS
# =====================================================================
print("\n--- [Suite 3/6] Running Health and Safety Triage Suite ---")
health_cases = [
    ("i have pain in my heart", True),
    ("my chest hurts and i have trouble breathing", True),
    ("what are the common causes of chest pain?", False),
    ("what are the emergency warning signs of a heart attack?", False),
    ("what is hypertension?", False)
]

health_results = []
for query, is_urgent in health_cases:
    t0 = time.time()
    history = [{"role": "user", "text": query}]
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=True
    )
    lat = round((time.time() - t0) * 1000, 2)

    has_emergency_notice = "[!] EMERGENCY MEDICAL NOTICE [!]" in reply
    has_emergency_numbers = "911" in reply or "999" in reply or "112" in reply
    has_disclaimer = "Disclaimer:" in reply
    has_red_flags = "Red Flag" in reply or "warning signs" in reply.lower()
    has_raw_scraped_boilerplate = any(bp in reply.lower() for bp in ["skip to main", "cookie policy", "terms of use", "categories"])

    if is_urgent:
        passed = has_emergency_notice and has_emergency_numbers and has_disclaimer and has_red_flags and not has_raw_scraped_boilerplate
    else:
        passed = has_disclaimer and has_red_flags and not has_raw_scraped_boilerplate

    record = {
        "query": query,
        "is_urgent": is_urgent,
        "route": route,
        "has_emergency_notice": has_emergency_notice,
        "has_emergency_numbers": has_emergency_numbers,
        "has_disclaimer": has_disclaimer,
        "has_red_flags": has_red_flags,
        "has_raw_scraped_boilerplate": has_raw_scraped_boilerplate,
        "response_preview": reply[:250].replace("\n", " "),
        "latency_ms": lat,
        "passed": passed
    }
    health_results.append(record)
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] '{query}' (urgent={is_urgent}) -> route: {route} [{lat}ms]")

with open(REPORTS_DIR / "health_response_tests.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(health_results),
        "passed": sum(1 for r in health_results if r["passed"]),
        "failed": sum(1 for r in health_results if not r["passed"]),
        "results": health_results
    }, f, indent=2)


# =====================================================================
# SUITE 4: CONTENT EXTRACTION TESTS
# =====================================================================
print("\n--- [Suite 4/6] Running DOM-Aware Content Extraction Suite ---")
extraction_cases = [
    {
        "name": "Nested Div & Breadcrumb Bug Reproduction",
        "html": """
        <div id="content">
          <div class="breadcrumbs">
            <a href="/health">Health</a> &gt; <a href="/cardio">Cardiology</a> &gt; Categories
          </div>
          <div class="article-body">
            <h1>Emergency Evaluation of Acute Chest Symptoms</h1>
            <p>Acute chest discomfort requires immediate systematic clinical evaluation to rule out acute coronary syndrome.</p>
            <p>Potential etiologies include cardiac ischemia, aortic dissection, and pulmonary embolism.</p>
          </div>
        </div>
        """,
        "must_contain": ["Emergency Evaluation", "Acute chest discomfort", "Potential etiologies"],
        "must_not_contain": ["Categories", "Cardiology", "breadcrumbs"]
    },
    {
        "name": "Advertising, Sidebar & Newsletter Stripping",
        "html": """
        <html><body>
          <div class="sidebar widget">
            <h3>Trending Stories</h3>
            <p>Top 10 diet tips you will not believe!</p>
          </div>
          <article>
            <h1>Deep Neural Network Architectures</h1>
            <p>A deep neural network consists of multiple sequential hidden layers that perform hierarchical representation learning.</p>
          </article>
          <div class="advertisement promo newsletter">
            <p>Subscribe to our daily newsletter for special discounts and offers!</p>
          </div>
        </body></html>
        """,
        "must_contain": ["Deep Neural Network Architectures", "hierarchical representation learning"],
        "must_not_contain": ["Trending Stories", "diet tips", "newsletter", "discounts and offers"]
    },
    {
        "name": "Mathematical Notation Cleanup",
        "html": """
        <article>
          <p>The conditional probability density is given by {\\displaystyle p(y\\mid x)} = \\frac{1}{\\sqrt{2\\pi}}.</p>
        </article>
        """,
        "must_contain": ["p(y | x)", "(1 / sqrt(2pi))"],
        "must_not_contain": ["\\displaystyle", "\\mid", "\\frac"]
    },
    {
        "name": "HTML Entities & Character Escapes",
        "html": """
        <article>
          <p>Tom &amp; Jerry &quot;enjoyed&quot; learning about Artificial Intelligence &mdash; it was &gt; 99% accurate.</p>
        </article>
        """,
        "must_contain": ["Tom & Jerry", '"enjoyed"', "> 99% accurate"],
        "must_not_contain": ["&amp;", "&quot;", "&gt;"]
    },
    {
        "name": "Client-Hydrated Metadata Harvesting (LinkedIn Pattern)",
        "html": """
        <html>
          <head>
            <title>Satya Nadella - Chairman and CEO at Microsoft | LinkedIn</title>
            <meta name="description" content="Chairman and CEO at Microsoft. As chairman and CEO of Microsoft, I define my mission as empowering every person. Experience: Microsoft. Location: Redmond.">
            <script type="application/ld+json">
              {"@context":"http://schema.org","@type":"Person","name":"Satya Nadella","jobTitle":"CEO"}
            </script>
          </head>
          <body>
            <div id="app-root"><!-- empty client hydration target --></div>
          </body>
        </html>
        """,
        "must_contain": ["Satya Nadella", "Chairman and CEO at Microsoft", "Experience: Microsoft"],
        "must_not_contain": ["app-root", "empty client hydration"]
    }
]

extraction_results = []
for case in extraction_cases:
    t0 = time.time()
    extracted = extract_text_from_html(case["html"])
    lat = round((time.time() - t0) * 1000, 2)

    has_required = all(req in extracted for req in case["must_contain"])
    lacks_prohibited = not any(proh.lower() in extracted.lower() for proh in case["must_not_contain"])
    passed = has_required and lacks_prohibited

    record = {
        "test_name": case["name"],
        "raw_length": len(case["html"]),
        "extracted_length": len(extracted),
        "extracted_sample": extracted[:200].replace("\n", " "),
        "has_required": has_required,
        "lacks_prohibited": lacks_prohibited,
        "latency_ms": lat,
        "passed": passed
    }
    extraction_results.append(record)
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] {case['name']} [{lat}ms]")

with open(REPORTS_DIR / "content_extraction_tests.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(extraction_results),
        "passed": sum(1 for r in extraction_results if r["passed"]),
        "failed": sum(1 for r in extraction_results if not r["passed"]),
        "results": extraction_results
    }, f, indent=2)


# =====================================================================
# SUITE 5: WEB RETRIEVAL PROVIDER & SECURITY TESTS
# =====================================================================
print("\n--- [Suite 5/6] Running Web Retrieval Providers & SSRF Security Suite ---")
retrieval_cases = [
    ("Wikipedia Exact Concept: Deep learning", lambda: retrieval_engine.wikipedia_provider.fetch_exact_concept("Deep learning")),
    ("Wikidata Entity Search: PyTorch", lambda: retrieval_engine.wikidata_provider.search("PyTorch", limit=2)),
    ("PubMed Medical Search: chest pain causes", lambda: retrieval_engine.pubmed_provider.search("chest pain causes", limit=2)),
    ("OpenAlex Researcher Search: Geoffrey Hinton", lambda: retrieval_engine.openalex_provider.search("Geoffrey Hinton", limit=2)),
    ("GitHub User Search: torvalds", lambda: retrieval_engine.github_provider.search("torvalds", limit=2)),
    ("SSRF Check Loopback: http://127.0.0.1:8080", lambda: is_safe_url("http://127.0.0.1:8080")[0]),
    ("SSRF Check Localhost: http://localhost/admin", lambda: is_safe_url("http://localhost/admin")[0]),
    ("SSRF Check AWS Metadata: http://169.254.169.254/latest", lambda: is_safe_url("http://169.254.169.254/latest")[0]),
    ("SSRF Check Private Subnet: http://192.168.1.1", lambda: is_safe_url("http://192.168.1.1")[0]),
    ("SSRF Check Disallowed Scheme: file:///etc/passwd", lambda: is_safe_url("file:///etc/passwd")[0]),
    ("SSRF Check Valid Public HTTPS: https://en.wikipedia.org", lambda: is_safe_url("https://en.wikipedia.org")[0]),
]

retrieval_results = []
for name, fn in retrieval_cases:
    t0 = time.time()
    try:
        res = fn()
        lat = round((time.time() - t0) * 1000, 2)
        if "SSRF Check" in name:
            if "Valid Public" in name:
                passed = (res is True)
            else:
                passed = (res is False)
        else:
            passed = bool(res)
        err = None
    except Exception as e:
        lat = round((time.time() - t0) * 1000, 2)
        passed = False
        err = str(e)
        res = None

    record = {
        "name": name,
        "latency_ms": lat,
        "passed": passed,
        "error": err
    }
    retrieval_results.append(record)
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] {name} [{lat}ms]")

with open(REPORTS_DIR / "web_retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(retrieval_results),
        "passed": sum(1 for r in retrieval_results if r["passed"]),
        "failed": sum(1 for r in retrieval_results if not r["passed"]),
        "results": retrieval_results
    }, f, indent=2)


# =====================================================================
# SUITE 6: CONVERSATION REGRESSION RESULTS
# =====================================================================
print("\n--- [Suite 6/6] Running Conversation Regression Suite ---")
dialogue_history = []
dialogue_steps = [
    # Turn 1: Educational question
    ("What is supervised learning?", "web_grounded_evidence"),
    # Turn 2: Follow-up simpler words
    ("Explain that in simpler words.", "web_cache_followup"),
    # Turn 3: Follow-up specific source
    ("What did the second source say?", "web_cache_followup"),
    # Turn 4: Follow-up shorter summary
    ("Give me a shorter summary.", "web_cache_followup"),
    # Turn 5: Math calculator
    ("45 * 12", "calculator"),
    # Turn 6: General conversation
    ("thank you!", "general_conversation"),
    # Turn 7: Ambiguous clarification
    ("stuff", "clarification")
]

conv_results = []
for q, exp_route in dialogue_steps:
    t0 = time.time()
    dialogue_history.append({"role": "user", "text": q})
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=dialogue_history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=True
    )
    dialogue_history.append({"role": "assistant", "text": reply})
    lat = round((time.time() - t0) * 1000, 2)

    passed = (route == exp_route) or (exp_route == "general_conversation" and route in ["conversation", "neural_model"])
    record = {
        "turn": len(conv_results) + 1,
        "input": q,
        "expected_route": exp_route,
        "actual_route": route,
        "response_preview": reply[:200].replace("\n", " "),
        "latency_ms": lat,
        "passed": passed
    }
    conv_results.append(record)
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] Turn {len(conv_results)}: '{q}' -> {route} [{lat}ms]")

# Offline mode test (/web off)
offline_q = "What is reinforcement learning?"
history_offline = [{"role": "user", "text": offline_q}]
t0 = time.time()
reply_offline, route_offline = generate_chat_response(
    model=model,
    tokenizer=tokenizer,
    history=history_offline,
    device=device,
    retrieval_engine=retrieval_engine,
    tool_assisted=True,
    web_search_enabled=False
)
lat_off = round((time.time() - t0) * 1000, 2)
passed_off = "offline" in reply_offline.lower() or "offline_notice" in route_offline
conv_results.append({
    "turn": len(conv_results) + 1,
    "input": f"/web off -> {offline_q}",
    "expected_route": "offline_notice",
    "actual_route": route_offline,
    "response_preview": reply_offline[:200].replace("\n", " "),
    "latency_ms": lat_off,
    "passed": passed_off
})
print(f"  [{'PASS' if passed_off else 'FAIL'}] Offline Mode (/web off) -> {route_offline} [{lat_off}ms]")

with open(REPORTS_DIR / "conversation_regression_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(conv_results),
        "passed": sum(1 for r in conv_results if r["passed"]),
        "failed": sum(1 for r in conv_results if not r["passed"]),
        "results": conv_results
    }, f, indent=2)


# =====================================================================
# FINAL ACCEPTANCE SUMMARY
# =====================================================================
all_suites = [
    ("Intent Routing", intent_results),
    ("Public Profile Search", profile_results),
    ("Health & Safety Triage", health_results),
    ("Content Extraction", extraction_results),
    ("Web Retrieval & SSRF Security", retrieval_results),
    ("Conversation & Controls", conv_results)
]

total_tests = sum(len(res) for _, res in all_suites)
total_passed = sum(sum(1 for r in res if r["passed"]) for _, res in all_suites)
total_failed = total_tests - total_passed
pass_rate = round((total_passed / total_tests) * 100, 2)

print("\n" + "=" * 70)
print(f"OVERALL ACCEPTANCE SUITE RESULTS: {total_passed}/{total_tests} PASSED ({pass_rate}%)")
print("=" * 70)
for name, res in all_suites:
    p = sum(1 for r in res if r["passed"])
    t = len(res)
    print(f"  * {name:<32}: {p:>2}/{t:<2} passed ({round(p/t*100, 1)}%)")

final_summary = {
    "timestamp": time.time(),
    "model_checkpoint": "checkpoints/ultimate_improvement/best.pt",
    "total_tests": total_tests,
    "total_passed": total_passed,
    "total_failed": total_failed,
    "pass_rate_percent": pass_rate,
    "suites": {
        name: {
            "total": len(res),
            "passed": sum(1 for r in res if r["passed"]),
            "failed": sum(1 for r in res if not r["passed"]),
            "pass_rate": round((sum(1 for r in res if r["passed"]) / len(res)) * 100, 1)
        }
        for name, res in all_suites
    }
}

with open(REPORTS_DIR / "final_acceptance_results.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=2)

print(f"\nSaved all structured reports to: {REPORTS_DIR}")
