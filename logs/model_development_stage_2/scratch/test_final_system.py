"""
test_final_system.py
====================
Comprehensive evaluation script for the finalized AI Companion Chatbot.
Evaluates all requirements from Section 14:
A. Casual conversation
B. Emotional rules
C. Offline knowledge (quality gate + fallback)
D. Web search (Punjab, Sidhu Moose Wala, Diljit Dosanjh, latest AI developments)
E. Multi-turn conversation (Transformer follow-up chain, pronoun resolution)
F. Spelling errors
G. Source display (hidden by default, shown upon 'Give me the links')
H. Mode switching (without restarting)
"""

import sys
import json
import time
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"LOADING FINALIZED AI COMPANION ON {device}")
    print("=" * 65)

    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

    all_test_results = {}

    # -----------------------------------------------------------------
    # A. Casual Conversation
    # -----------------------------------------------------------------
    print("\n--- TEST A: CASUAL CONVERSATION ---")
    casual_queries = [
        "Hello.",
        "How are you?",
        "What are you doing?",
        "Can we talk?",
        "How was your day?"
    ]
    results_a = []
    for q in casual_queries:
        history = [{"role": "user", "text": q}]
        t0 = time.time()
        reply, route = generate_chat_response(
            model, tokenizer, history, device, retrieval_engine,
            web_search_enabled=False
        )
        lat = (time.time() - t0) * 1000
        print(f"USER: {q}")
        print(f"BOT ({route}, {lat:.0f}ms): {reply}\n")
        results_a.append({"q": q, "reply": reply, "route": route, "latency_ms": round(lat, 1)})
    all_test_results["A_casual"] = results_a

    # -----------------------------------------------------------------
    # B. Emotional Rules
    # -----------------------------------------------------------------
    print("\n--- TEST B: EMOTIONAL RULES ---")
    emotional_queries = [
        "I passed my examination.",
        "I failed my examination.",
        "I am feeling sad.",
        "I am stressed about my future.",
        "I am confused.",
        "I feel lonely.",
        "I am angry because nothing is working."
    ]
    results_b = []
    for q in emotional_queries:
        history = [{"role": "user", "text": q}]
        t0 = time.time()
        reply, route = generate_chat_response(
            model, tokenizer, history, device, retrieval_engine,
            web_search_enabled=False
        )
        lat = (time.time() - t0) * 1000
        print(f"USER: {q}")
        print(f"BOT ({route}, {lat:.0f}ms): {reply}\n")
        results_b.append({"q": q, "reply": reply, "route": route, "latency_ms": round(lat, 1)})
    all_test_results["B_emotional"] = results_b

    # -----------------------------------------------------------------
    # C. Offline Knowledge (Quality Gate + Fallback)
    # -----------------------------------------------------------------
    print("\n--- TEST C: OFFLINE KNOWLEDGE ---")
    offline_queries = [
        "What is Python?",
        "What is Machine Learning?",
        "What is a CNN?",
        "What is a Transformer?",
        "What is supervised learning?",
        "What are the latest AI developments?"  # Expected to trigger Case B fallback
    ]
    results_c = []
    for q in offline_queries:
        history = [{"role": "user", "text": q}]
        t0 = time.time()
        reply, route = generate_chat_response(
            model, tokenizer, history, device, retrieval_engine,
            web_search_enabled=False
        )
        lat = (time.time() - t0) * 1000
        print(f"USER: {q}")
        print(f"BOT ({route}, {lat:.0f}ms): {reply}\n")
        results_c.append({"q": q, "reply": reply, "route": route, "latency_ms": round(lat, 1)})
    all_test_results["C_offline"] = results_c

    # -----------------------------------------------------------------
    # D. Web Search (Live Internet)
    # -----------------------------------------------------------------
    print("\n--- TEST D: WEB SEARCH (ONLINE MODE) ---")
    web_queries = [
        "What is Punjab?",
        "Who is Sidhu Moose Wala?",
        "Who is Diljit Dosanjh?",
        "What are the latest AI developments?",
        "What is Punjab and who is Sidhu Moose Wala?"  # Compound question
    ]
    results_d = []
    for q in web_queries:
        history = [{"role": "user", "text": q}]
        t0 = time.time()
        reply, route = generate_chat_response(
            model, tokenizer, history, device, retrieval_engine,
            web_search_enabled=True
        )
        lat = (time.time() - t0) * 1000
        print(f"USER: {q}")
        print(f"BOT ({route}, {lat:.0f}ms): {reply[:300]}...")
        # Verify sources are hidden by default
        has_sources_dump = ("Sources Consulted:" in reply or "http://" in reply or "https://" in reply)
        print(f"   -> Sources hidden by default: {not has_sources_dump}\n")
        results_d.append({
            "q": q,
            "reply": reply,
            "route": route,
            "sources_hidden_by_default": not has_sources_dump,
            "latency_ms": round(lat, 1)
        })
    all_test_results["D_web_search"] = results_d

    # -----------------------------------------------------------------
    # E. Multi-Turn Conversation
    # -----------------------------------------------------------------
    print("\n--- TEST E: MULTI-TURN CONVERSATION ---")
    multi_turns_1 = [
        "What is a Transformer?",
        "How does it work?",
        "Give me an example.",
        "Where is it used?"
    ]
    results_e1 = []
    history_e1 = []
    for q in multi_turns_1:
        history_e1.append({"role": "user", "text": q})
        t0 = time.time()
        reply, route = generate_chat_response(
            model, tokenizer, history_e1, device, retrieval_engine,
            web_search_enabled=True
        )
        lat = (time.time() - t0) * 1000
        print(f"USER: {q}")
        print(f"BOT ({route}, {lat:.0f}ms): {reply[:250]}...\n")
        history_e1.append({"role": "assistant", "text": reply})
        results_e1.append({"q": q, "reply": reply[:250], "route": route})

    # Pronoun Resolution Turn: Diljit Dosanjh -> "What about his songs?"
    multi_turns_2 = [
        "Who is Diljit Dosanjh?",
        "What about his songs?"
    ]
    results_e2 = []
    history_e2 = []
    for q in multi_turns_2:
        history_e2.append({"role": "user", "text": q})
        t0 = time.time()
        reply, route = generate_chat_response(
            model, tokenizer, history_e2, device, retrieval_engine,
            web_search_enabled=True
        )
        lat = (time.time() - t0) * 1000
        print(f"USER: {q}")
        print(f"BOT ({route}, {lat:.0f}ms): {reply[:250]}...\n")
        history_e2.append({"role": "assistant", "text": reply})
        results_e2.append({"q": q, "reply": reply[:250], "route": route})

    all_test_results["E_multi_turn"] = {
        "transformer_chain": results_e1,
        "pronoun_chain": results_e2
    }

    # -----------------------------------------------------------------
    # F. Spelling Errors
    # -----------------------------------------------------------------
    print("\n--- TEST F: SPELLING ERRORS ---")
    spelling_queries = [
        ("tell me hw are yuo", False),
        ("wat is mashine lerning", False),
        ("who is sidhumoosewala", True)
    ]
    results_f = []
    for q, web_on in spelling_queries:
        history = [{"role": "user", "text": q}]
        t0 = time.time()
        reply, route = generate_chat_response(
            model, tokenizer, history, device, retrieval_engine,
            web_search_enabled=web_on
        )
        lat = (time.time() - t0) * 1000
        print(f"USER: {q} (web={web_on})")
        print(f"BOT ({route}, {lat:.0f}ms): {reply[:250]}...\n")
        results_f.append({"q": q, "web_on": web_on, "reply": reply[:250], "route": route})
    all_test_results["F_spelling"] = results_f

    # -----------------------------------------------------------------
    # G. Source Display
    # -----------------------------------------------------------------
    print("\n--- TEST G: SOURCE DISPLAY ---")
    history_g = [{"role": "user", "text": "Who is Sidhu Moose Wala?"}]
    reply_g1, route_g1 = generate_chat_response(
        model, tokenizer, history_g, device, retrieval_engine,
        web_search_enabled=True
    )
    print("USER: Who is Sidhu Moose Wala?")
    print(f"BOT: {reply_g1[:200]}...")
    print(f"   -> Sources hidden in answer: {('Sources for the' not in reply_g1 and 'http' not in reply_g1)}")

    # User asks for links
    history_g.append({"role": "assistant", "text": reply_g1})
    history_g.append({"role": "user", "text": "Give me the links."})
    reply_g2, route_g2 = generate_chat_response(
        model, tokenizer, history_g, device, retrieval_engine,
        web_search_enabled=True
    )
    print("\nUSER: Give me the links.")
    print(f"BOT ({route_g2}):\n{reply_g2}\n")
    print(f"   -> Clickable links shown: {('http' in reply_g2)}")

    all_test_results["G_source_display"] = {
        "answer_hidden_sources": reply_g1[:200],
        "links_request_output": reply_g2
    }

    # -----------------------------------------------------------------
    # H. Mode Switching
    # -----------------------------------------------------------------
    print("\n--- TEST H: MODE SWITCHING ---")
    history_h = []
    # Turn 1: Offline
    history_h.append({"role": "user", "text": "What is Machine Learning?"})
    r_h1, route_h1 = generate_chat_response(
        model, tokenizer, history_h, device, retrieval_engine,
        web_search_enabled=False
    )
    history_h.append({"role": "assistant", "text": r_h1})
    print("TURN 1 [WEB=OFF] USER: What is Machine Learning?")
    print(f"BOT ({route_h1}): {r_h1[:150]}...\n")

    # Turn 2: Switch to Online
    history_h.append({"role": "user", "text": "What are the latest AI developments?"})
    r_h2, route_h2 = generate_chat_response(
        model, tokenizer, history_h, device, retrieval_engine,
        web_search_enabled=True
    )
    history_h.append({"role": "assistant", "text": r_h2})
    print("TURN 2 [SWITCHED WEB=ON] USER: What are the latest AI developments?")
    print(f"BOT ({route_h2}): {r_h2[:150]}...\n")

    # Turn 3: Switch back to Offline
    history_h.append({"role": "user", "text": "What are the latest AI developments?"})
    r_h3, route_h3 = generate_chat_response(
        model, tokenizer, history_h, device, retrieval_engine,
        web_search_enabled=False
    )
    history_h.append({"role": "assistant", "text": r_h3})
    print("TURN 3 [SWITCHED WEB=OFF] USER: What are the latest AI developments?")
    print(f"BOT ({route_h3}): {r_h3}\n")

    all_test_results["H_mode_switching"] = {
        "turn_1_offline": r_h1[:150],
        "turn_2_online": r_h2[:150],
        "turn_3_offline": r_h3
    }

    # Save to disk
    out_file = PROJECT_DIR / "logs" / "model_development_stage_2" / "final_system_evaluation_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_test_results, f, indent=2)
    print(f"\nAll final evaluation tests successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
