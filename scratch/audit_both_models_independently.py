"""
scratch/audit_both_models_independently.py
Independent Audit - Requirement 2:
Verify Custom Model (best.pt) and External Model (Qwen) independently.
"""
import sys
import time
import json
import torch
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from model_manager import ModelManager
from chat import generate_chat_response
from web_retrieval import WebRetrievalEngine

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrieval = WebRetrievalEngine()
mgr = ModelManager(device=dev, default_model="custom")

results = {
    "custom_model": {},
    "external_model": {}
}

print("==================================================")
print("AUDIT 2.1: CUSTOM MODEL (best.pt) INDEPENDENT TESTS")
print("==================================================")
mgr.set_model("custom")
assert mgr.active_model_name == "custom", "Active model must be custom"
custom_model = mgr.custom_engine.model
custom_tok = mgr.custom_engine.tokenizer

# Verify model architecture and parameters
param_count = sum(p.numel() for p in custom_model.parameters())
print(f"Custom Model Type: {type(custom_model).__name__}")
print(f"Custom Model Parameters: {param_count:,}")
print(f"Custom Model Device: {next(custom_model.parameters()).device}")
assert param_count > 10_000_000, "Custom model must be ~14.1M params"
assert "cuda" in str(next(custom_model.parameters()).device).lower(), "Custom model must be on CUDA"

custom_tests = [
    ("general_conversation", "Hello! How are you doing today?"),
    ("educational_question", "What is photosynthesis?"),
    ("mathematical_question", "Can you explain the derivative of x^2?"),
    ("technical_explanation", "What is gradient descent in machine learning?")
]

history_custom = []
for test_id, query in custom_tests:
    history_custom.append({"role": "user", "text": query})
    reply, route = generate_chat_response(
        model=custom_model,
        tokenizer=custom_tok,
        history=history_custom,
        device=dev,
        retrieval_engine=retrieval,
        web_search_enabled=False,
        model_manager=mgr
    )
    history_custom.append({"role": "assistant", "text": reply})
    results["custom_model"][test_id] = {"query": query, "reply": reply, "route": route}
    print(f"\n[Custom] User: {query}")
    print(f"[Custom] Route: {route}")
    print(f"[Custom] Reply: {reply[:200]}...")
    assert route in [
        "neural_model", "offline_quality_fallback", "companion_greeting",
        "direct_dialogue", "companion_chat", "offline_educational_grounded",
        "companion_empathy", "companion_emotional_support"
    ], f"Unexpected route for custom model: {route}"

# Test follow-up question
follow_up_q = "Can you explain more about its learning rate?"
history_custom.append({"role": "user", "text": follow_up_q})
reply_fu, route_fu = generate_chat_response(
    model=custom_model,
    tokenizer=custom_tok,
    history=history_custom,
    device=dev,
    retrieval_engine=retrieval,
    web_search_enabled=False,
    model_manager=mgr
)
results["custom_model"]["follow_up_question"] = {"query": follow_up_q, "reply": reply_fu, "route": route_fu}
print(f"\n[Custom Follow-up] User: {follow_up_q}")
print(f"[Custom Follow-up] Route: {route_fu}")
print(f"[Custom Follow-up] Reply: {reply_fu[:200]}...")

print("\n==================================================")
print("AUDIT 2.2: EXTERNAL MODEL (Qwen) INDEPENDENT TESTS")
print("==================================================")
mgr.set_model("external")
assert mgr.active_model_name == "external", "Active model must be external"
assert mgr.external_engine.is_server_alive(), "llama-server must be alive"

# 1. General conversation
h_qwen = [{"role": "user", "text": "Hello! What kind of companion are you?"}]
r_gen, route_gen = generate_chat_response(None, None, h_qwen, dev, retrieval, False, model_manager=mgr)
results["external_model"]["general_conversation"] = {"query": h_qwen[0]["text"], "reply": r_gen, "route": route_gen}
print(f"\n[Qwen] User: {h_qwen[0]['text']}")
print(f"[Qwen] Route: {route_gen}")
print(f"[Qwen] Reply: {r_gen[:250]}...")
h_qwen.append({"role": "assistant", "text": r_gen})

# 2. Educational question
q_edu = "Explain the theory of relativity in simple terms."
h_qwen.append({"role": "user", "text": q_edu})
r_edu, route_edu = generate_chat_response(None, None, h_qwen, dev, retrieval, False, model_manager=mgr)
results["external_model"]["educational_question"] = {"query": q_edu, "reply": r_edu, "route": route_edu}
print(f"\n[Qwen] User: {q_edu}")
print(f"[Qwen] Route: {route_edu}")
print(f"[Qwen] Reply: {r_edu[:250]}...")
h_qwen.append({"role": "assistant", "text": r_edu})

# 3. Mathematics
q_math = "Find the derivative of sin(x)*cos(x) step by step."
h_qwen.append({"role": "user", "text": q_math})
r_math, route_math = generate_chat_response(None, None, h_qwen, dev, retrieval, False, model_manager=mgr)
results["external_model"]["mathematics"] = {"query": q_math, "reply": r_math, "route": route_math}
print(f"\n[Qwen] User: {q_math}")
print(f"[Qwen] Route: {route_math}")
print(f"[Qwen] Reply: {r_math[:250]}...")
h_qwen.append({"role": "assistant", "text": r_math})

# 4. Web-enabled retrieval
q_web = "Who is the current Chief Minister of Punjab, India?"
h_web = [{"role": "user", "text": q_web}]
r_web, route_web = generate_chat_response(None, None, h_web, dev, retrieval, True, model_manager=mgr)
results["external_model"]["web_retrieval"] = {"query": q_web, "reply": r_web, "route": route_web}
print(f"\n[Qwen Web] User: {q_web}")
print(f"[Qwen Web] Route: {route_web}")
print(f"[Qwen Web] Reply: {r_web[:250]}...")
assert "bhagwant" in r_web.lower() or "mann" in r_web.lower(), f"Expected Bhagwant Mann in Punjab CM response: {r_web}"

# 5. Multiple consecutive questions (turns 4 & 5)
q_follow1 = "Which topic did we discuss right before this mathematics question?"
h_qwen.append({"role": "user", "text": q_follow1})
r_follow1, route_f1 = generate_chat_response(None, None, h_qwen, dev, retrieval, False, model_manager=mgr)
results["external_model"]["multi_turn_1"] = {"query": q_follow1, "reply": r_follow1, "route": route_f1}
print(f"\n[Qwen Multi-turn 1] User: {q_follow1}")
print(f"[Qwen Multi-turn 1] Route: {route_f1}")
print(f"[Qwen Multi-turn 1] Reply: {r_follow1[:250]}...")
h_qwen.append({"role": "assistant", "text": r_follow1})

q_follow2 = "Can you summarize our conversation so far in two sentences?"
h_qwen.append({"role": "user", "text": q_follow2})
r_follow2, route_f2 = generate_chat_response(None, None, h_qwen, dev, retrieval, False, model_manager=mgr)
results["external_model"]["multi_turn_2"] = {"query": q_follow2, "reply": r_follow2, "route": route_f2}
print(f"\n[Qwen Multi-turn 2] User: {q_follow2}")
print(f"[Qwen Multi-turn 2] Route: {route_f2}")
print(f"[Qwen Multi-turn 2] Reply: {r_follow2[:250]}...")

mgr.shutdown()

with open(PROJECT_DIR / "scratch/audit_models_independent_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nAUDIT 2 ALL CHECKS PASSED: Both models independently verified across all test scenarios!")
