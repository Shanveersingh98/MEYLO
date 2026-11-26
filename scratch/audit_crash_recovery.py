"""
scratch/audit_crash_recovery.py
Independent Audit - Requirement 1: Verify Crash Recovery
1. Reproduce raw HTTP 400 context-overflow scenario directly against llama-server.
2. Capture the actual server error response body and JSON structure.
3. Verify whether compact-context retry in ExternalModelEngine works.
4. Verify chatbot remains interactive on subsequent turns after error.
5. Verify no automatic model switching or hidden fallback occurs.
"""
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import torch

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model_manager import ModelManager
from chat import generate_chat_response
from web_retrieval import WebRetrievalEngine

print("==================================================")
print("AUDIT 1.1: STARTING SERVER & CAPTURING RAW HTTP 400")
print("==================================================")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mgr = ModelManager(device=dev, default_model="external")
mgr.set_model("external")

# Make sure server is up
is_alive = mgr.external_engine.is_server_alive()
print(f"llama-server alive on port {mgr.external_engine.port}: {is_alive}")
assert is_alive, "llama-server must be alive"

# Construct a massive prompt that guaranteed exceeds 4096 tokens (e.g. 5,000 tokens ~ 25,000 chars)
massive_text = "The quick brown fox jumps over the lazy dog. " * 800
print(f"Massive text length (chars): {len(massive_text)}")

raw_payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Summarize this text: {massive_text}"}
    ],
    "max_tokens": 512
}

raw_req = urllib.request.Request(
    f"http://127.0.0.1:{mgr.external_engine.port}/v1/chat/completions",
    data=json.dumps(raw_payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

captured_code = None
captured_reason = None
captured_body = None

try:
    with urllib.request.urlopen(raw_req, timeout=30) as resp:
        print("UNEXPECTED: Request succeeded instead of 400!")
except urllib.error.HTTPError as e:
    captured_code = e.code
    captured_reason = e.reason
    captured_body = e.read().decode("utf-8")
    print(f"\n[SUCCESSFULLY REPRODUCED HTTP {captured_code}]")
    print(f"HTTP Status: {captured_code} {captured_reason}")
    print(f"Raw Response Body:\n{captured_body}\n")

assert captured_code == 400, f"Expected HTTP 400, got {captured_code}"
assert "exceed" in captured_body.lower() or "context" in captured_body.lower(), "Expected context exceed error message"

print("==================================================")
print("AUDIT 1.2: VERIFY COMPACT RETRY IN GENERATE()")
print("==================================================")
# Test ExternalModelEngine.generate with context_evidence that would overflow if not bounded/retried
# We pass an oversized context_evidence, but with a real prompt
test_prompt = "What is the capital of France?"
# We pass oversized context
oversized_evidence = "Historical context: France is a country in Europe. " * 400

reply, latency = mgr.external_engine.generate(
    prompt=test_prompt,
    history=[],
    context_evidence=oversized_evidence,
    system_instruction="You are a helpful assistant. Answer based on the evidence."
)

print(f"Generate Reply ({latency:.1f}ms): {reply}")
has_paris = "paris" in reply.lower()
print(f"Contains correct answer 'Paris': {has_paris}")
assert has_paris, "Compact retry should have succeeded and returned Paris"

print("==================================================")
print("AUDIT 1.3: VERIFY CHATBOT INTERACTIVITY & NO FALLBACK")
print("==================================================")
retrieval_engine = WebRetrievalEngine()

# First turn: User asks a question
h1 = [{"role": "user", "text": "Hello, my name is Alex."}]
r1, route1 = generate_chat_response(None, None, h1, dev, retrieval_engine, False, model_manager=mgr)
print(f"Turn 1 (Hello): Route={route1}, ActiveModel={mgr.active_model_name}")
print(f"Bot 1: {r1[:120]}...\n")
assert mgr.active_model_name == "external", "Active model must remain external"

# Second turn: trigger simulated server error
print("Simulating server failure during generation...")
class FailOnceEngine:
    def __init__(self, real_engine):
        self.real = real_engine
        self.failed = False
    def generate(self, *args, **kwargs):
        if not self.failed:
            self.failed = True
            raise urllib.error.HTTPError("http://127.0.0.1:8080/v1/chat/completions", 400, "Bad Request: context limit exceeded", {}, None)
        return self.real.generate(*args, **kwargs)

real_engine = mgr.external_engine
mgr.external_engine = FailOnceEngine(real_engine)

h2 = [
    {"role": "user", "text": "Hello, my name is Alex."},
    {"role": "assistant", "text": r1},
    {"role": "user", "text": "Tell me a very long story with huge context."}
]
r2, route2 = generate_chat_response(None, None, h2, dev, retrieval_engine, False, model_manager=mgr)
print(f"Turn 2 (Simulated 400): Route={route2}, ActiveModel={mgr.active_model_name}")
print(f"Bot 2: {r2}\n")
assert "error" in r2.lower() or "apologize" in r2.lower(), "Should gracefully inform user of error"
assert mgr.active_model_name == "external", "Active model MUST NOT automatically switch to custom!"
assert route2.startswith("qwen_"), f"Route must be external Qwen, got: {route2}"

# Third turn: Subsequent query MUST continue working smoothly
h3 = h2 + [
    {"role": "assistant", "text": r2},
    {"role": "user", "text": "What was my name that I told you earlier?"}
]
r3, route3 = generate_chat_response(None, None, h3, dev, retrieval_engine, False, model_manager=mgr)
print(f"Turn 3 (Subsequent turn): Route={route3}, ActiveModel={mgr.active_model_name}")
print(f"Bot 3: {r3}\n")
assert mgr.active_model_name == "external", "Active model must remain external"
assert "alex" in r3.lower(), "Qwen should remember context and name Alex"

mgr.external_engine = real_engine
mgr.shutdown()
print("AUDIT 1 ALL CHECKS PASSED: Crash recovery, raw 400 capture, compact retry, session interactivity, and zero fallback verified!")
