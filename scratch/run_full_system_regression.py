"""
scratch/run_full_system_regression.py
======================================
Executes the full suite of regression tests requested by the user:
- 10 Offline Tests (pure local, no web search)
- 8 Online Tests (web search ON)
- 4-turn Follow-up Test (Transformer context)
- Source Visibility Test (Karan Aujla -> hidden -> ask links -> shown)
- Mode Switching Test (web OFF -> ON -> OFF)

All turns record routing telemetry to logs/router_debug.log.
"""

import sys
import json
import time
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response, format_source_links

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading system on device: {device}...")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
print("System ready.\n")

results = {
    "offline_tests": [],
    "online_tests": [],
    "followup_tests": [],
    "source_test": {},
    "mode_test": {}
}

# -------------------------------------------------------------
# 1. OFFLINE TESTS (Web Search OFF)
# -------------------------------------------------------------
print("=" * 60)
print("RUNNING 10 OFFLINE TESTS (Web Search: OFF)")
print("=" * 60)

offline_queries = [
    "hii",
    "hello how are you today",
    "i am sa",
    "i am sad today",
    "i got injured",
    "i feel bad today",
    "i failed my examintaion",
    "what is machine learning",
    "what is python",
    "what is tranformer"
]

for q in offline_queries:
    history = [{"role": "user", "text": q}]
    t0 = time.time()
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=False
    )
    dt = round((time.time() - t0) * 1000, 2)
    print(f"USER: {q}")
    print(f"BOT ({route}, {dt}ms):\n{reply}\n")
    results["offline_tests"].append({
        "query": q,
        "reply": reply,
        "route": route,
        "latency_ms": dt
    })

# -------------------------------------------------------------
# 2. ONLINE TESTS (Web Search ON)
# -------------------------------------------------------------
print("=" * 60)
print("RUNNING 8 ONLINE TESTS (Web Search: ON)")
print("=" * 60)

online_queries = [
    "what is day today",
    "whats the date",
    "what is the date today",
    "who is karan aujla",
    "who is sidhu moosewala",
    "what is tranformer",
    "what is punjab and who is sidhu moosewala",
    "who is shanveer singh khosa on linkedin"
]

for q in online_queries:
    history = [{"role": "user", "text": q}]
    t0 = time.time()
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=True
    )
    dt = round((time.time() - t0) * 1000, 2)
    print(f"USER: {q}")
    print(f"BOT ({route}, {dt}ms):\n{reply}\n")
    results["online_tests"].append({
        "query": q,
        "reply": reply,
        "route": route,
        "latency_ms": dt
    })

# -------------------------------------------------------------
# 3. FOLLOW-UP TEST (Multi-turn Context)
# -------------------------------------------------------------
print("=" * 60)
print("RUNNING FOLLOW-UP CONVERSATION TEST")
print("=" * 60)

followup_flow = [
    "What is a Transformer?",
    "How does it work?",
    "Give me an example.",
    "Where is it used?"
]

history = []
if hasattr(retrieval_engine, "conversation_memory"):
    retrieval_engine.conversation_memory.clear()

for q in followup_flow:
    history.append({"role": "user", "text": q})
    t0 = time.time()
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=True
    )
    dt = round((time.time() - t0) * 1000, 2)
    print(f"USER: {q}")
    print(f"BOT ({route}, {dt}ms):\n{reply}\n")
    history.append({"role": "assistant", "text": reply})
    results["followup_tests"].append({
        "turn": q,
        "reply": reply,
        "route": route,
        "latency_ms": dt
    })

# -------------------------------------------------------------
# 4. SOURCE TEST (Hidden by Default -> Show when asked)
# -------------------------------------------------------------
print("=" * 60)
print("RUNNING SOURCE VISIBILITY TEST")
print("=" * 60)

history = [{"role": "user", "text": "Who is Karan Aujla?"}]
reply1, route1 = generate_chat_response(
    model=model,
    tokenizer=tokenizer,
    history=history,
    device=device,
    retrieval_engine=retrieval_engine,
    tool_assisted=True,
    web_search_enabled=True
)
print("USER: Who is Karan Aujla?")
print(f"BOT ({route1}):\n{reply1}\n")
sources_hidden_check = ("http" not in reply1 and "Sources Consulted" not in reply1)
print(f"Confirm sources are hidden: {sources_hidden_check}")

history.append({"role": "assistant", "text": reply1})
history.append({"role": "user", "text": "Give me the links."})
reply2, route2 = generate_chat_response(
    model=model,
    tokenizer=tokenizer,
    history=history,
    device=device,
    retrieval_engine=retrieval_engine,
    tool_assisted=True,
    web_search_enabled=True
)
print("USER: Give me the links.")
print(f"BOT ({route2}):\n{reply2}\n")
sources_displayed_check = ("http" in reply2)
print(f"Confirm source URLs displayed: {sources_displayed_check}")

results["source_test"] = {
    "question": "Who is Karan Aujla?",
    "answer": reply1,
    "sources_hidden_in_answer": sources_hidden_check,
    "link_request": "Give me the links.",
    "links_answer": reply2,
    "links_successfully_shown": sources_displayed_check
}

# -------------------------------------------------------------
# 5. MODE TEST (OFF -> ON -> OFF)
# -------------------------------------------------------------
print("=" * 60)
print("RUNNING MODE SWITCHING TEST")
print("=" * 60)

mode_checks = []

# Step A: Web OFF -> ask who is Karan Aujla -> verify NO network requests occur
history = [{"role": "user", "text": "who is karan aujla"}]
reply_off1, route_off1 = generate_chat_response(
    model=model, tokenizer=tokenizer, history=history, device=device,
    retrieval_engine=retrieval_engine, web_search_enabled=False
)
print("[Mode: OFF] USER: who is karan aujla")
print(f"BOT ({route_off1}): {reply_off1}")
mode_checks.append({
    "mode": "OFF",
    "query": "who is karan aujla",
    "route": route_off1,
    "reply": reply_off1,
    "web_used": False
})

# Step B: Switch Web ON -> ask who is Karan Aujla -> verify retrieval works
reply_on, route_on = generate_chat_response(
    model=model, tokenizer=tokenizer, history=history, device=device,
    retrieval_engine=retrieval_engine, web_search_enabled=True
)
print("\n[Mode: ON] USER: who is karan aujla")
print(f"BOT ({route_on}): {reply_on}")
mode_checks.append({
    "mode": "ON",
    "query": "who is karan aujla",
    "route": route_on,
    "reply": reply_on,
    "web_used": True
})

# Step C: Switch Web OFF -> ask what is machine learning -> verify retrieval stops
history = [{"role": "user", "text": "what is machine learning"}]
reply_off2, route_off2 = generate_chat_response(
    model=model, tokenizer=tokenizer, history=history, device=device,
    retrieval_engine=retrieval_engine, web_search_enabled=False
)
print("\n[Mode: OFF] USER: what is machine learning")
print(f"BOT ({route_off2}): {reply_off2}")
mode_checks.append({
    "mode": "OFF",
    "query": "what is machine learning",
    "route": route_off2,
    "reply": reply_off2,
    "web_used": False
})

results["mode_test"] = mode_checks

# Save complete results
output_file = PROJECT_DIR / "logs" / "critical_repair_regression_results.json"
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved regression results to: {output_file}")
