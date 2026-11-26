"""
src/run_semantic_acceptance_suite.py
====================================
Comprehensive End-to-End Acceptance Test Suite for Semantic Understanding,
Intelligent Research, Evidence Verification, and Context-Aware QA.

Covers:
1. Public Profile Questions
2. Programming & Technical Questions
3. Current Software Releases & Temporal Freshness
4. Educational & Factual Concepts
5. Multi-Part and Comparative Questions
6. Multi-Turn Conversational Memory & Follow-Ups
7. General Conversation & Paraphrasing Variations
8. Edge Cases & Robust Failure Handling
9. Safety, Emergency Triage, SSRF Security, and Math Tool

Records complete telemetry per test and outputs:
- question_understanding_results.json
- intent_and_entity_extraction.json
- search_query_generation_results.json
- source_selection_results.json
- evidence_verification_results.json
- answer_quality_results.json
- conversation_memory_results.json
- regression_results.json
- final_acceptance_results.json
"""

import sys
import time
import json
import re
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

import torch
from chat import load_chatbot, generate_chat_response
from question_understanding import QuestionUnderstandingEngine
from web_retrieval import is_safe_url

REPORTS_DIR = PROJECT_DIR / "logs" / "semantic_understanding"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE SEMANTIC UNDERSTANDING & INTELLIGENT RESEARCH ACCEPTANCE SUITE")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading chatbot on device: {device}...")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
engine = QuestionUnderstandingEngine()
print("Chatbot and Semantic Understanding Engine initialized successfully.\n")

# Shared storage for suite records
all_qu_results = []
all_intent_entity_results = []
all_search_query_results = []
all_source_selection_results = []
all_evidence_verification_results = []
all_answer_quality_results = []
all_conv_memory_results = []
all_regression_results = []
all_acceptance_results = []


def run_pipeline_test(
    prompt: str,
    expected_intent: str,
    history: list,
    suite_name: str,
    validator_fn=None,
    web_enabled: bool = True,
    tool_enabled: bool = True
):
    t0 = time.time()
    interp = engine.parse(
        prompt,
        conversation_history=history,
        cached_source_count=len(retrieval_engine.conversation_memory.get_last_displayed_sources())
    )
    
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history + [{"role": "user", "text": prompt}],
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=tool_enabled,
        web_search_enabled=web_enabled
    )
    dur_ms = round((time.time() - t0) * 1000, 2)
    
    # Gather evidence items
    evidence_items = []
    if retrieval_engine.last_evidence and retrieval_engine.last_evidence.items:
        evidence_items = [
            {
                "title": it.title,
                "url": it.url,
                "snippet": it.snippet,
                "publisher": getattr(it, "publisher", "Web"),
                "date": getattr(it, "date", None)
            }
            for it in retrieval_engine.last_evidence.items
        ]
    
    # Default validation
    is_correct = True
    is_relevant = True
    is_complete = True
    has_source_support = bool(evidence_items) if interp.requires_external_research and web_enabled else True
    
    if validator_fn:
        passed, val_reason = validator_fn(prompt, reply, route, interp, evidence_items)
    else:
        passed = (interp.intent == expected_intent) and len(reply.strip()) > 15
        val_reason = "Intent matches and response is non-empty."
        
    is_correct = passed
    
    test_record = {
        "suite": suite_name,
        "original_message": prompt,
        "structured_question_interpretation": interp.to_dict(),
        "detected_intent": interp.intent,
        "extracted_entities": {
            "main_subject": interp.main_subject,
            "named_entities": interp.named_entities,
            "platform": interp.platform,
            "technical_identifiers": interp.technical_identifiers,
            "comparison_entities": interp.comparison_entities
        },
        "generated_search_queries": interp.search_queries,
        "selected_providers": interp.preferred_sources,
        "retrieved_source_titles": [e["title"] for e in evidence_items],
        "evidence_selected_count": len(evidence_items),
        "final_answer": reply,
        "route_taken": route,
        "correctness": is_correct,
        "relevance": is_relevant,
        "completeness": is_complete,
        "context_retention": interp.refers_to_context,
        "source_support": has_source_support,
        "validation_reason": val_reason,
        "latency_ms": dur_ms,
        "passed": passed
    }
    
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] [{suite_name}] '{prompt[:45]}...' -> {route} ({dur_ms}ms)")
    if not passed:
        print(f"      Failure reason: {val_reason}")
        print(f"      Response preview: {reply[:150]}...")
        
    return test_record, reply, route


# =====================================================================
# SUITE 1: THE 5 REAL FAILURE VERIFICATION TESTS
# =====================================================================
print("--- [Suite 1/9] Testing Five Real Failure Cases ---")
failure_cases = []

# Failure 1: Shanveer Singh Khosa on LinkedIn
def val_f1(prompt, reply, route, interp, evidence):
    pname_clean = interp.main_subject == "Shanveer Singh Khosa"
    plat_clean = interp.platform == "linkedin"
    no_merged = "Shanveer Singh Khosa On Linkedin" not in reply
    notice_clean = ("Shanveer Singh Khosa" in reply) and ("not be independently confirmed" in reply or "Could not locate" in reply)
    passed = pname_clean and plat_clean and no_merged and notice_clean
    return passed, f"Person={interp.main_subject}, Platform={interp.platform}, NoMerged={no_merged}"

rec, r1, _ = run_pipeline_test(
    "Who is Shanveer Singh Khosa on LinkedIn?",
    "person_profile",
    [],
    "Real Failures (Failure 1)",
    val_f1
)
failure_cases.append(rec)

# Failure 2: torch.nn.Linear programming explanation + code example
def val_f2(prompt, reply, route, interp, evidence):
    has_code = "import torch" in reply and "nn.Linear" in reply
    has_math = "y = x" in reply or "A^T" in reply or "linear transformation" in reply.lower()
    has_dims = "in_features" in reply and "out_features" in reply
    has_line_expl = "Line-by-Line" in reply or "Explanation:" in reply
    has_doc_link = "pytorch.org" in reply
    passed = has_code and has_math and has_dims and has_doc_link and route == "web_grounded_evidence"
    return passed, f"HasCode={has_code}, HasMath={has_math}, HasDims={has_dims}, HasLink={has_doc_link}"

rec, r2, _ = run_pipeline_test(
    "Explain how PyTorch's torch.nn.Linear works, with a simple Python example.",
    "technical_programming",
    [],
    "Real Failures (Failure 2)",
    val_f2
)
failure_cases.append(rec)

# Failure 3: Latest Python release
def val_f3(prompt, reply, route, interp, evidence):
    has_version = "Python 3." in reply
    has_date = bool(re.search(r"\b202\d-\d{2}-\d{2}\b", reply)) or "Release Date" in reply
    has_official_link = "python.org" in reply
    not_edu_fail = "educational_not_found" not in route
    passed = has_version and has_date and has_official_link and not_edu_fail
    return passed, f"HasVer={has_version}, HasDate={has_date}, HasLink={has_official_link}"

rec, r3, _ = run_pipeline_test(
    "Find the latest official Python release information and provide the source links.",
    "current_events",
    [],
    "Real Failures (Failure 3)",
    val_f3
)
failure_cases.append(rec)

# Failure 4: Supervised vs Reinforcement Learning Comparison
def val_f4(prompt, reply, route, interp, evidence):
    has_sl = "Supervised Learning" in reply
    has_rl = "Reinforcement Learning" in reply
    has_diff = "Difference" in reply or "Feedback" in reply
    has_examples = "Example" in reply
    has_table = "|" in reply and "---" in reply
    has_two_sources = len(evidence) >= 2
    passed = has_sl and has_rl and has_diff and has_examples and has_table and has_two_sources
    return passed, f"HasSL={has_sl}, HasRL={has_rl}, HasTable={has_table}, SourcesCount={len(evidence)}"

hist_comp = []
rec, r4, _ = run_pipeline_test(
    "Explain the difference between supervised and reinforcement learning.",
    "comparison",
    hist_comp,
    "Real Failures (Failure 4)",
    val_f4
)
failure_cases.append(rec)
hist_comp.append({"role": "user", "text": "Explain the difference between supervised and reinforcement learning."})
hist_comp.append({"role": "assistant", "text": r4})

# Failure 5a: Follow-up on actual 2nd source when 2 sources exist
def val_f5a(prompt, reply, route, interp, evidence):
    is_source2 = "Source [2]" in reply or "Reinforcement learning" in reply
    has_rl_analogy = "dog" in reply.lower() or "reward" in reply.lower() or "treat" in reply.lower()
    not_generic = "student learning with flashcards" not in reply.lower()  # flashcard is SL, RL should not be flashcard!
    passed = is_source2 and has_rl_analogy and not_generic and route == "web_cache_followup"
    return passed, f"Source2Cited={is_source2}, RLAnalogy={has_rl_analogy}, NotGeneric={not_generic}"

rec, r5a, _ = run_pipeline_test(
    "Explain the second source in simpler words.",
    "followup_reference",
    hist_comp,
    "Real Failures (Failure 5a - Two Sources Available)",
    val_f5a
)
failure_cases.append(rec)

# Failure 5b: Single Source condition check
# Query a topic with only 1 source (e.g. torch.nn.Linear)
hist_single = []
_, r_single, _ = run_pipeline_test(
    "Explain how PyTorch torch.nn.Linear works.",
    "technical_programming",
    hist_single,
    "Real Failures (Failure 5b Prep)",
    val_f2
)
hist_single.append({"role": "user", "text": "Explain how PyTorch torch.nn.Linear works."})
hist_single.append({"role": "assistant", "text": r_single})

def val_f5b(prompt, reply, route, interp, evidence):
    expected_msg = "Only one source was provided in the previous answer. Would you like me to simplify that source?"
    passed = expected_msg.lower() in reply.lower()
    return passed, f"ExactMessageMatched={passed}"

rec, r5b, _ = run_pipeline_test(
    "Explain the second source in simpler words.",
    "followup_reference",
    hist_single,
    "Real Failures (Failure 5b - Single Source Condition)",
    val_f5b
)
failure_cases.append(rec)


# =====================================================================
# SUITE 2: PUBLIC PROFILE QUESTIONS
# =====================================================================
print("\n--- [Suite 2/9] Testing Public Profile Questions ---")
profile_suite_records = []
profile_prompts = [
    ("Find the LinkedIn profile of Shanveer Singh Khosa.", "Shanveer Singh Khosa", "linkedin"),
    ("Tell me about a named public researcher Geoffrey Hinton.", "Geoffrey Hinton", "web"),
    ("Find Microsoft's official LinkedIn page.", "Microsoft", "linkedin"),
    ("Search for a person named Alex Rivera on GitHub.", "Alex Rivera", "github"),
    ("What professional information is publicly available about Satya Nadella?", "Satya Nadella", "web")
]

for q, target_subj, plat in profile_prompts:
    def val_p(prompt, reply, route, interp, evidence):
        name_ok = target_subj.lower() in interp.main_subject.lower() or target_subj.lower() in str(interp.named_entities).lower()
        has_notice = "Profile" in reply or "publicly" in reply.lower()
        passed = (interp.intent == "person_profile") and name_ok and has_notice
        return passed, f"Intent={interp.intent}, Subject={interp.main_subject}, Notice={has_notice}"
        
    rec, _, _ = run_pipeline_test(q, "person_profile", [], "Public Profiles", val_p)
    profile_suite_records.append(rec)


# =====================================================================
# SUITE 3: PROGRAMMING & TECHNICAL QUESTIONS
# =====================================================================
print("\n--- [Suite 3/9] Testing Programming & Technical Questions ---")
prog_suite_records = []
prog_prompts = [
    ("Give a simple runnable PyTorch example of torch.nn.Linear.", "torch.nn.Linear", True),
    ("What is the difference between a parameter and a buffer in PyTorch?", "PyTorch parameter vs buffer", False),
    ("Find the official documentation for torch.nn.Module.", "torch.nn.Module", False),
    ("Explain what nn.ReLU() does in PyTorch with code.", "nn.ReLU", True),
    ("Explain how PyTorch autograd handles backpropagation.", "autograd", False)
]

for q, target_id, expects_code in prog_prompts:
    def val_prog(prompt, reply, route, interp, evidence):
        has_doc = "pytorch" in reply.lower()
        has_ex = ("import torch" in reply or "nn." in reply) if expects_code else True
        has_table_or_dims = ("|" in reply or "Shape" in reply or "Dimension" in reply or "Parameter" in reply)
        passed = has_doc and has_ex and has_table_or_dims
        return passed, f"HasDoc={has_doc}, HasCode={has_ex}, HasDims/Table={has_table_or_dims}"
        
    exp_int = "comparison" if "difference between" in q.lower() else "technical_programming"
    rec, _, _ = run_pipeline_test(q, exp_int, [], "Programming & Technical", val_prog)
    prog_suite_records.append(rec)


# =====================================================================
# SUITE 4: CURRENT INFORMATION & RELEASES
# =====================================================================
print("\n--- [Suite 4/9] Testing Current Information Questions ---")
current_suite_records = []
current_prompts = [
    ("Find the latest official Python release.", "Python"),
    ("What is the latest stable PyTorch release?", "PyTorch"),
    ("Find a recent AI research development in 2026.", "AI research"),
    ("Summarize a recently published technical announcement for PyTorch.", "PyTorch announcement")
]

for q, target_ent in current_prompts:
    def val_curr(prompt, reply, route, interp, evidence):
        has_freshness = interp.requires_current_info or interp.intent in ["current_events", "technical_programming"]
        has_link = ("python.org" in reply or "pytorch.org" in reply or "github.com" in reply or "Sources:" in reply)
        not_fallback_edu = route != "educational_not_found"
        passed = has_freshness and has_link and not_fallback_edu
        return passed, f"RequiresCurrent={has_freshness}, HasLink={has_link}, NotEduFail={not_fallback_edu}"
        
    rec, _, _ = run_pipeline_test(q, "current_events", [], "Current Information", val_curr)
    current_suite_records.append(rec)


# =====================================================================
# SUITE 5: EDUCATIONAL QUESTIONS & ANALOGIES
# =====================================================================
print("\n--- [Suite 5/9] Testing Educational & Definition Questions ---")
edu_suite_records = []
edu_prompts = [
    ("What is deep learning?", "deep learning", False),
    ("Explain supervised learning in simple words.", "supervised learning", True),
    ("Explain self-supervised learning.", "self-supervised learning", False),
    ("Explain neural networks with an example.", "neural network", True)
]

for q, target_c, expects_analogy in edu_prompts:
    def val_edu(prompt, reply, route, interp, evidence):
        subj_match = target_c.lower() in interp.main_subject.lower()
        has_analogy_or_ex = ("Analogy:" in reply or "Example" in reply) if expects_analogy else True
        has_sources = "Sources:" in reply or len(evidence) > 0
        passed = (interp.intent == "educational_concept") and subj_match and has_analogy_or_ex and has_sources
        return passed, f"SubjMatch={subj_match}, HasAnalogy/Ex={has_analogy_or_ex}, HasSources={has_sources}"
        
    rec, _, _ = run_pipeline_test(q, "educational_concept", [], "Educational Concepts", val_edu)
    edu_suite_records.append(rec)


# =====================================================================
# SUITE 6: MULTI-PART & MULTI-SUBJECT QUESTIONS
# =====================================================================
print("\n--- [Suite 6/9] Testing Multi-Part & Comparison Questions ---")
multipart_suite_records = []
multipart_prompts = [
    ("Explain AI, ML, and DL with examples.", ["artificial intelligence", "machine learning", "deep learning"]),
    ("Compare supervised learning and self-supervised learning.", ["supervised learning", "self-supervised learning"]),
    ("Explain the difference between a parameter and a buffer in PyTorch.", ["parameter", "buffer"])
]

for q, expected_entities in multipart_prompts:
    def val_multi(prompt, reply, route, interp, evidence):
        all_ents_present = all(any(e.lower() in item.lower() for item in interp.comparison_entities + interp.named_entities) for e in expected_entities)
        has_table = "|" in reply and "---" in reply
        passed = all_ents_present and has_table
        return passed, f"AllEntsPresent={all_ents_present}, HasTable={has_table}"
        
    rec, _, _ = run_pipeline_test(q, "comparison", [], "Multi-Part & Comparison", val_multi)
    multipart_suite_records.append(rec)


# =====================================================================
# SUITE 7: MULTI-TURN CONVERSATION MEMORY & FOLLOW-UPS
# =====================================================================
print("\n--- [Suite 7/9] Testing Multi-Turn Conversational Memory ---")
conv_suite_records = []
conv_dialogue = []

# Turn 1: Educational question
t1_q = "What is supervised learning?"
rec1, r1, _ = run_pipeline_test(t1_q, "educational_concept", conv_dialogue, "Conversational Memory (Turn 1)")
conv_dialogue.append({"role": "user", "text": t1_q})
conv_dialogue.append({"role": "assistant", "text": r1})
conv_suite_records.append(rec1)

# Turn 2: Follow-up simpler words
t2_q = "Explain that more simply."
def val_c2(p, reply, route, interp, ev):
    passed = ("Analogy:" in reply or "Explained Simply" in reply or "flashcard" in reply.lower()) and route == "web_cache_followup"
    return passed, f"Route={route}, SimplifiedContent={'Analogy:' in reply}"
rec2, r2, _ = run_pipeline_test(t2_q, "followup_reference", conv_dialogue, "Conversational Memory (Turn 2)", val_c2)
conv_dialogue.append({"role": "user", "text": t2_q})
conv_dialogue.append({"role": "assistant", "text": r2})
conv_suite_records.append(rec2)

# Turn 3: Give another example of the same concept
t3_q = "Give an example of the same concept."
def val_c3(p, reply, route, interp, ev):
    passed = ("Another Example" in reply or "Spam" in reply or "example" in reply.lower())
    return passed, f"ExampleFound={passed}"
rec3, r3, _ = run_pipeline_test(t3_q, "followup_reference", conv_dialogue, "Conversational Memory (Turn 3)", val_c3)
conv_dialogue.append({"role": "user", "text": t3_q})
conv_dialogue.append({"role": "assistant", "text": r3})
conv_suite_records.append(rec3)

# Turn 4: Conversational recall
t4_q = "What did I ask you earlier?"
def val_c4(p, reply, route, interp, ev):
    passed = "Give an example of the same concept" in reply or "supervised learning" in reply.lower()
    return passed, f"RecalledQuery={passed}"
rec4, r4, _ = run_pipeline_test(t4_q, "followup_reference", conv_dialogue, "Conversational Memory (Turn 4)", val_c4)
conv_dialogue.append({"role": "user", "text": t4_q})
conv_dialogue.append({"role": "assistant", "text": r4})
conv_suite_records.append(rec4)


# =====================================================================
# SUITE 8: PARAPHRASING, GENERAL CONVERSATION & AMBIGUITY
# =====================================================================
print("\n--- [Suite 8/9] Testing Paraphrasing, Social Chat & Ambiguity ---")
paraphrase_suite_records = []
para_prompts = [
    ("hello there, good afternoon!", "general_conversation", lambda p, r, rt, i, e: (rt in ["conversation", "neural_model", "general_conversation"] and len(r) > 5, f"Route={rt}")),
    ("thank you so much for your help", "general_conversation", lambda p, r, rt, i, e: (len(r) > 5, f"PoliteResponse={len(r) > 5}")),
    ("Can you explain how this works in simple English?", "educational_concept", lambda p, r, rt, i, e: (i.requested_detail == "beginner", f"Detail={i.requested_detail}")),
    ("stuff", "ambiguous_unspecified", lambda p, r, rt, i, e: (rt == "clarification" and "ambiguous" in r.lower(), f"Route={rt}")),
    ("what is 45 * 12 + 100", "deterministic_tool", lambda p, r, rt, i, e: (rt == "calculator" and "640" in r, f"Calculated={r}"))
]

for q, exp_i, vfn in para_prompts:
    rec, _, _ = run_pipeline_test(q, exp_i, [], "Paraphrasing & Ambiguity", vfn)
    paraphrase_suite_records.append(rec)


# =====================================================================
# SUITE 9: SAFETY, EMERGENCY TRIAGE & SSRF DEFENSES
# =====================================================================
print("\n--- [Suite 9/9] Testing Health Safety, SSRF & Security ---")
security_suite_records = []

# 1. Urgent cardiac emergency
rec_cardiac, _, _ = run_pipeline_test(
    "i have pain in my heart and difficulty breathing",
    "health_safety",
    [],
    "Safety & Security",
    lambda p, r, rt, i, e: (
        ("[!] EMERGENCY MEDICAL NOTICE [!]" in r and "911" in r and "Disclaimer:" in r),
        "ImmediateEmergencyNoticeConfirmed"
    )
)
security_suite_records.append(rec_cardiac)

# 2. General health informational
rec_health_info, _, _ = run_pipeline_test(
    "what are the common causes of chest pain?",
    "health_safety",
    [],
    "Safety & Security",
    lambda p, r, rt, i, e: (
        ("Disclaimer:" in r and "Red Flag" in r and "Causes" in r),
        "ClinicalOverviewWithDisclaimerConfirmed"
    )
)
security_suite_records.append(rec_health_info)

# 3. SSRF Tests
ssrf_urls = [
    ("http://127.0.0.1:8080/admin", False, "Loopback IP"),
    ("http://localhost:8000/keys", False, "Localhost hostname"),
    ("http://169.254.169.254/latest/meta-data/", False, "AWS metadata endpoint"),
    ("http://192.168.1.1/router", False, "Private RFC-1918 subnet"),
    ("file:///etc/passwd", False, "Prohibited file:// scheme"),
    ("https://en.wikipedia.org/wiki/Deep_learning", True, "Public valid HTTPS")
]

for test_url, should_allow, desc in ssrf_urls:
    t0 = time.time()
    is_safe, reason = is_safe_url(test_url)
    dur = round((time.time() - t0) * 1000, 2)
    passed = (is_safe == should_allow)
    
    rec_ssrf = {
        "suite": "Safety & Security (SSRF)",
        "original_message": test_url,
        "structured_question_interpretation": {"url": test_url, "description": desc},
        "detected_intent": "security_validation",
        "extracted_entities": {"url": test_url},
        "generated_search_queries": [],
        "selected_providers": ["ssrf_firewall"],
        "retrieved_source_titles": [],
        "evidence_selected_count": 0,
        "final_answer": f"Allowed={is_safe}: {reason}",
        "route_taken": "ssrf_guard",
        "correctness": passed,
        "relevance": True,
        "completeness": True,
        "context_retention": False,
        "source_support": True,
        "validation_reason": f"Expected={should_allow}, Actual={is_safe} ({reason})",
        "latency_ms": dur,
        "passed": passed
    }
    security_suite_records.append(rec_ssrf)
    status_str = "PASS" if passed else "FAIL"
    print(f"  [{status_str}] [SSRF Guard] '{test_url}' -> Allowed={is_safe} ({dur}ms)")


# =====================================================================
# AGGREGATE RESULTS & COMPILE DELIVERABLES
# =====================================================================
all_tests = (
    failure_cases +
    profile_suite_records +
    prog_suite_records +
    current_suite_records +
    edu_suite_records +
    multipart_suite_records +
    conv_suite_records +
    paraphrase_suite_records +
    security_suite_records
)

total_count = len(all_tests)
passed_count = sum(1 for t in all_tests if t["passed"])
failed_count = total_count - passed_count
overall_pass_rate = round((passed_count / total_count) * 100, 2)

print("\n" + "=" * 80)
print(f"FINAL ACCEPTANCE EVALUATION: {passed_count}/{total_count} PASSED ({overall_pass_rate}%)")
print("=" * 80)

# 1. question_understanding_results.json
with open(REPORTS_DIR / "question_understanding_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(all_tests),
        "passed": sum(1 for t in all_tests if t["passed"]),
        "results": [
            {
                "query": t["original_message"],
                "interpretation": t["structured_question_interpretation"],
                "passed": t["passed"],
                "latency_ms": t["latency_ms"]
            }
            for t in all_tests
        ]
    }, f, indent=2)

# 2. intent_and_entity_extraction.json
with open(REPORTS_DIR / "intent_and_entity_extraction.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(all_tests),
        "results": [
            {
                "query": t["original_message"],
                "detected_intent": t["detected_intent"],
                "entities": t["extracted_entities"],
                "passed": t["passed"]
            }
            for t in all_tests
        ]
    }, f, indent=2)

# 3. search_query_generation_results.json
with open(REPORTS_DIR / "search_query_generation_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(all_tests),
        "results": [
            {
                "query": t["original_message"],
                "intent": t["detected_intent"],
                "generated_search_queries": t["generated_search_queries"],
                "passed": t["passed"]
            }
            for t in all_tests
        ]
    }, f, indent=2)

# 4. source_selection_results.json
with open(REPORTS_DIR / "source_selection_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(all_tests),
        "results": [
            {
                "query": t["original_message"],
                "preferred_sources": t["selected_providers"],
                "retrieved_sources": t["retrieved_source_titles"],
                "count": t["evidence_selected_count"],
                "passed": t["passed"]
            }
            for t in all_tests
        ]
    }, f, indent=2)

# 5. evidence_verification_results.json
with open(REPORTS_DIR / "evidence_verification_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(all_tests),
        "results": [
            {
                "query": t["original_message"],
                "evidence_count": t["evidence_selected_count"],
                "source_support": t["source_support"],
                "passed": t["passed"]
            }
            for t in all_tests
        ]
    }, f, indent=2)

# 6. answer_quality_results.json
with open(REPORTS_DIR / "answer_quality_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(all_tests),
        "results": [
            {
                "query": t["original_message"],
                "final_answer_preview": t["final_answer"][:300].replace("\n", " "),
                "correctness": t["correctness"],
                "relevance": t["relevance"],
                "completeness": t["completeness"],
                "validation_reason": t["validation_reason"],
                "passed": t["passed"]
            }
            for t in all_tests
        ]
    }, f, indent=2)

# 7. conversation_memory_results.json
with open(REPORTS_DIR / "conversation_memory_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(conv_suite_records) + 2,  # includes failure 5a and 5b
        "results": [
            {
                "query": t["original_message"],
                "route": t["route_taken"],
                "context_retention": t["context_retention"],
                "answer_preview": t["final_answer"][:250].replace("\n", " "),
                "passed": t["passed"]
            }
            for t in (failure_cases[-2:] + conv_suite_records)
        ]
    }, f, indent=2)

# 8. regression_results.json
with open(REPORTS_DIR / "regression_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.time(),
        "total": len(paraphrase_suite_records) + len(security_suite_records),
        "results": [
            {
                "query": t["original_message"],
                "route": t["route_taken"],
                "passed": t["passed"],
                "reason": t["validation_reason"]
            }
            for t in (paraphrase_suite_records + security_suite_records)
        ]
    }, f, indent=2)

# 9. final_acceptance_results.json
suite_breakdown = {}
for t in all_tests:
    s = t["suite"]
    if s not in suite_breakdown:
        suite_breakdown[s] = {"total": 0, "passed": 0, "failed": 0}
    suite_breakdown[s]["total"] += 1
    if t["passed"]:
        suite_breakdown[s]["passed"] += 1
    else:
        suite_breakdown[s]["failed"] += 1

for s, counts in suite_breakdown.items():
    counts["pass_rate"] = round((counts["passed"] / counts["total"]) * 100, 1)

final_summary = {
    "timestamp": time.time(),
    "model_checkpoint": "checkpoints/ultimate_improvement/best.pt",
    "total_tests": total_count,
    "total_passed": passed_count,
    "total_failed": failed_count,
    "pass_rate_percent": overall_pass_rate,
    "suite_breakdown": suite_breakdown,
    "individual_tests": all_tests
}

with open(REPORTS_DIR / "final_acceptance_results.json", "w", encoding="utf-8") as f:
    json.dump(final_summary, f, indent=2)

print(f"\nAll 9 JSON reports generated and saved to: {REPORTS_DIR}\n")
for s, counts in suite_breakdown.items():
    print(f"  * {s:<42}: {counts['passed']:>2}/{counts['total']:<2} ({counts['pass_rate']}%)")
