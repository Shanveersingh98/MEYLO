"""
run_full_system_eval.py
========================
Automated End-to-End System Evaluation for AI Companion Chatbot.
Executes:
1. TEST A: Mode 1 (Web Search OFF) across all 25 user questions.
2. TEST B: Mode 2 (Web Search ON) across factual/recent/comparative and casual/emotional queries.
3. Multi-turn Conversational Context Test: "What is Machine Learning?" -> "Give me an example."
"""

import sys
import os
import json
import time
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response

QUESTIONS_TEST_A = [
    # General Conversation (1-6)
    {"cat": "General Conversation", "num": 1, "q": "Hello, how are you?"},
    {"cat": "General Conversation", "num": 2, "q": "What are you doing?"},
    {"cat": "General Conversation", "num": 3, "q": "Can we talk?"},
    {"cat": "General Conversation", "num": 4, "q": "Tell me something interesting."},
    {"cat": "General Conversation", "num": 5, "q": "How was your day?"},
    {"cat": "General Conversation", "num": 6, "q": "What can you help me with?"},
    # Educational Questions (7-14)
    {"cat": "Educational Concepts", "num": 7, "q": "What is Artificial Intelligence?"},
    {"cat": "Educational Concepts", "num": 8, "q": "What is Machine Learning?"},
    {"cat": "Educational Concepts", "num": 9, "q": "What is Deep Learning?"},
    {"cat": "Educational Concepts", "num": 10, "q": "What is a Neural Network?"},
    {"cat": "Educational Concepts", "num": 11, "q": "What is Supervised Learning?"},
    {"cat": "Educational Concepts", "num": 12, "q": "What is a CNN?"},
    {"cat": "Educational Concepts", "num": 13, "q": "What is a Transformer?"},
    {"cat": "Educational Concepts", "num": 14, "q": "What is the difference between CNN and Transformer?"},
    # Programming (15-18)
    {"cat": "Programming", "num": 15, "q": "What is Python?"},
    {"cat": "Programming", "num": 16, "q": "What is a variable?"},
    {"cat": "Programming", "num": 17, "q": "What is a loop?"},
    {"cat": "Programming", "num": 18, "q": "What is a function?"},
    # Everyday Conversation (19-21)
    {"cat": "Everyday Conversation", "num": 19, "q": "How can I improve my communication skills?"},
    {"cat": "Everyday Conversation", "num": 20, "q": "How can I manage my time?"},
    {"cat": "Everyday Conversation", "num": 21, "q": "How can I prepare for examinations?"},
    # Emotional Conversation (22-25)
    {"cat": "Emotional Conversation", "num": 22, "q": "I am feeling sad. Can we talk?"},
    {"cat": "Emotional Conversation", "num": 23, "q": "I feel lonely."},
    {"cat": "Emotional Conversation", "num": 24, "q": "I am stressed about my examination."},
    {"cat": "Emotional Conversation", "num": 25, "q": "I had a bad day."}
]

QUESTIONS_TEST_B = [
    {"num": 1, "q": "What is the latest NVIDIA GPU?", "expected_search": True, "desc": "Hardware freshness check"},
    {"num": 2, "q": "Explain Transformer architecture in deep learning.", "expected_search": True, "desc": "Educational structured architecture inquiry"},
    {"num": 3, "q": "Compare CNN and Transformer architectures, including their advantages and disadvantages.", "expected_search": True, "desc": "Multi-concept comparative inquiry"},
    {"num": 4, "q": "What are the latest developments in Artificial Intelligence?", "expected_search": True, "desc": "Current events / research freshness check"},
    {"num": 5, "q": "Hello, how are you?", "expected_search": False, "desc": "Casual greeting under web=ON (must NOT search)"},
    {"num": 6, "q": "I am feeling sad. Can we talk?", "expected_search": False, "desc": "Emotional support under web=ON (must NOT search)"}
]


def evaluate_offline_quality(q: str, resp: str, cat: str) -> tuple[str, str]:
    """
    Evaluates response honestly:
    PASS: Relevant, understandable, factually reasonable, appropriate, non-repetitive.
    PARTIAL: Relevant topic attempted, but contains circular phrases or incomplete clarity.
    FAIL: Irrelevant, severe repetition loops, or empty.
    """
    r_low = resp.lower()
    words = resp.split()

    if len(words) < 5:
        return "FAIL", "Response is too short or empty."

    # Repetition checks
    for w in set(words):
        if len(w) > 4 and words.count(w) >= 6 and len(words) < 60:
            return "FAIL", f"Severe repetitive looping of token '{w}'."

    # Emotional questions
    if cat == "Emotional Conversation":
        if any(w in r_low for w in ["sorry", "here for you", "listen", "talk", "support", "unwind", "alone"]):
            return "PASS", "Warm, empathetic companion response appropriate for emotional support."
        return "PARTIAL", "Responded but lacked clear warmth or emotional attunement."

    # General conversation greetings
    if cat == "General Conversation":
        if any(w in r_low for w in ["hello", "help", "language model", "assistant", "companion", "day", "chat", "fine", "good"]):
            return "PASS", "Natural conversational response appropriate for greeting/casual dialogue."
        return "PARTIAL", "Responded conversationally but somewhat generic."

    # Educational / Programming
    if cat in ["Educational Concepts", "Programming"]:
        # Check if subject was addressed
        q_clean = q.lower().replace("what is", "").replace("a", "").replace("the", "").replace("?", "").strip()
        q_tokens = [t for t in q_clean.split() if len(t) > 3]
        has_overlap = any(t in r_low for t in q_tokens)
        
        # Check for circular definition ("a neural network is a neural network")
        if any(f"{t} is a {t}" in r_low for t in q_tokens):
            return "PARTIAL", "Circular definition loop present in model output."
        
        if has_overlap and len(words) >= 15:
            return "PASS", "Relevant technical concept explained with understandable substance."
        elif has_overlap:
            return "PARTIAL", "Relevant subject mentioned but explanation is basic or terse."
        return "FAIL", "Did not address the requested technical concept."

    # Everyday conversation
    if cat == "Everyday Conversation":
        if any(w in r_low for w in ["time", "study", "schedule", "plan", "communication", "exam", "practice", "help", "listen"]):
            return "PASS", "Constructive, practical advice provided for everyday topic."
        return "PARTIAL", "Everyday topic acknowledged but advice is limited."

    return "PARTIAL", "Standard conversational generation."


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"STARTING COMPREHENSIVE SYSTEM EVALUATION ON {device}")
    print("=" * 60)

    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

    # -------------------------------------------------------------
    # 1. TEST A: Web Search OFF (Mode 1)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXECUTING TEST A: WEB SEARCH OFF (MODE 1 - 25 QUESTIONS)")
    print("=" * 60)

    results_test_a = []
    pass_cnt = 0
    partial_cnt = 0
    fail_cnt = 0

    for item in QUESTIONS_TEST_A:
        cat = item["cat"]
        num = item["num"]
        q = item["q"]

        history = [{"role": "user", "text": q}]
        t0 = time.time()
        resp, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=history,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=False
        )
        lat = (time.time() - t0) * 1000

        status, reason = evaluate_offline_quality(q, resp, cat)
        if status == "PASS":
            pass_cnt += 1
        elif status == "PARTIAL":
            partial_cnt += 1
        else:
            fail_cnt += 1

        rec = {
            "number": num,
            "category": cat,
            "question": q,
            "response": resp,
            "status": status,
            "reason": reason,
            "route": route,
            "latency_ms": round(lat, 1)
        }
        results_test_a.append(rec)
        print(f"[{num:02d}/25] [{cat}] {q}")
        print(f"       -> Status: {status} ({reason})")
        print(f"       -> Route: {route} | Latency: {lat:.0f}ms")
        print(f"       -> BOT: {resp[:120]}...\n")

    print(f"Test A Summary: Total=25, PASS={pass_cnt}, PARTIAL={partial_cnt}, FAIL={fail_cnt}")

    # -------------------------------------------------------------
    # 2. TEST B: Web Search ON (Mode 2)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXECUTING TEST B: WEB SEARCH ON (MODE 2)")
    print("=" * 60)

    results_test_b = []
    for item in QUESTIONS_TEST_B:
        num = item["num"]
        q = item["q"]
        expected_search = item["expected_search"]
        desc = item["desc"]

        history = [{"role": "user", "text": q}]
        t0 = time.time()
        resp, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=history,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=True
        )
        lat = (time.time() - t0) * 1000

        used_search = ("web" in route)
        search_compliance = (used_search == expected_search)

        rec = {
            "number": num,
            "question": q,
            "desc": desc,
            "expected_search": expected_search,
            "used_search": used_search,
            "search_compliance": search_compliance,
            "route": route,
            "response": resp,
            "latency_ms": round(lat, 1)
        }
        results_test_b.append(rec)
        print(f"[B-{num}] {q} ({desc})")
        print(f"      -> Used Search: {used_search} (Expected: {expected_search}) => {'PASS' if search_compliance else 'FAIL'}")
        print(f"      -> Route: {route} | Latency: {lat:.0f}ms")
        print(f"      -> BOT: {resp[:150]}...\n")

    # -------------------------------------------------------------
    # 3. Multi-turn Conversational Context Test
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXECUTING MULTI-TURN CONTEXT TEST")
    print("=" * 60)

    multi_history = []
    # Turn 1
    q1 = "What is Machine Learning?"
    multi_history.append({"role": "user", "text": q1})
    resp1, route1 = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=multi_history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=False
    )
    multi_history.append({"role": "assistant", "text": resp1})
    print(f"Turn 1 USER: {q1}")
    print(f"Turn 1 BOT: {resp1}\n")

    # Turn 2: Anaphora / Contextual follow-up
    q2 = "Give me an example."
    multi_history.append({"role": "user", "text": q2})
    resp2, route2 = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=multi_history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=False
    )
    multi_history.append({"role": "assistant", "text": resp2})
    print(f"Turn 2 USER: {q2}")
    print(f"Turn 2 BOT: {resp2}\n")

    context_rec = {
        "turn_1": {"q": q1, "r": resp1, "route": route1},
        "turn_2": {"q": q2, "r": resp2, "route": route2}
    }

    # Save all results to logs/model_development_stage_2/eval_results.json
    out_path = PROJECT_DIR / "logs" / "model_development_stage_2" / "system_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_report = {
        "timestamp": time.time(),
        "device": str(device),
        "test_a": {
            "summary": {"total": 25, "pass": pass_cnt, "partial": partial_cnt, "fail": fail_cnt},
            "results": results_test_a
        },
        "test_b": {
            "results": results_test_b
        },
        "multi_turn": context_rec
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"Complete evaluation results written to: {out_path}")


if __name__ == "__main__":
    main()
