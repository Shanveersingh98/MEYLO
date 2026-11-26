"""
scratch/run_comprehensive_tests.py
==================================
Comprehensive automated test suite executing Tests A through I on the AI Companion Chatbot.
Verifies both Offline and Online modes, conversational continuity, question interpretation,
search accuracy, multi-step research, and answer synthesis.
"""

import sys
import os
import time
import json
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response

def run_tests():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading chatbot on device: {device}...")
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
    print("Chatbot loaded successfully!\n")

    results = {}

    # =========================================================================
    # TEST A: GTA 6 Multi-Turn Context (Offline & Online)
    # =========================================================================
    print("=== RUNNING TEST A: GTA 6 Multi-Turn Context ===")
    test_a_history = []
    test_a_turns = [
        "what is gta 6",
        "explain in simple language",
        "what type of game is it",
        "who developed it"
    ]
    test_a_outputs = []
    for q in test_a_turns:
        test_a_history.append({"role": "user", "text": q})
        reply, route = generate_chat_response(
            model=model, tokenizer=tokenizer, history=test_a_history,
            device=device, retrieval_engine=retrieval_engine,
            web_search_enabled=False
        )
        test_a_history.append({"role": "assistant", "text": reply})
        test_a_outputs.append((q, reply, route))
        print(f"User: {q}\nRoute: {route}\nBot: {reply[:120]}...\n")

    # Assertions for Test A
    a_pass = (
        "Grand Theft Auto" in test_a_outputs[0][1] and
        len(test_a_outputs[1][1]) > 40 and
        any(k in test_a_outputs[2][1].lower() for k in ["action-adventure", "open-world", "open world", "game"]) and
        "Rockstar" in test_a_outputs[3][1]
    )
    results["Test A (GTA 6 Context)"] = "PASS" if a_pass else "FAIL"

    # =========================================================================
    # TEST B: India PM Paraphrase Robustness (4 distinct variations)
    # =========================================================================
    print("=== RUNNING TEST B: India PM Paraphrase Robustness ===")
    test_b_queries = [
        "who is india current prime minister",
        "who is the pm of india now",
        "name of current indian prime minister",
        "who currently leads the Indian government"
    ]
    test_b_outputs = []
    b_pass = True
    for q in test_b_queries:
        retrieval_engine.conversation_memory.clear()
        reply, route = generate_chat_response(
            model=model, tokenizer=tokenizer, history=[{"role": "user", "text": q}],
            device=device, retrieval_engine=retrieval_engine,
            web_search_enabled=True
        )
        test_b_outputs.append((q, reply, route))
        print(f"User: {q}\nRoute: {route}\nBot: {reply[:140]}...\n")
        if "Narendra Modi" not in reply or "office staff" in reply.lower():
            b_pass = False
    results["Test B (India PM Paraphrases)"] = "PASS" if b_pass else "FAIL"

    # =========================================================================
    # TEST C: Entity and Song Understanding
    # =========================================================================
    print("=== RUNNING TEST C: Entity & Song Understanding ===")
    test_c_queries = [
        ("diljit dosnajh current song with daddy yankee colabrated song name", True),
        ("who is pakito", False),
        ("what is his most famous song", False),
        ("time today", False),
        ("what time is it now", False),
        ("what is the latest news in tech", True)
    ]
    test_c_outputs = []
    c_history = []
    for q, web in test_c_queries:
        c_history.append({"role": "user", "text": q})
        reply, route = generate_chat_response(
            model=model, tokenizer=tokenizer, history=c_history,
            device=device, retrieval_engine=retrieval_engine,
            web_search_enabled=web
        )
        c_history.append({"role": "assistant", "text": reply})
        test_c_outputs.append((q, reply, route))
        print(f"User: {q}\nRoute: {route}\nBot: {reply[:120]}...\n")

    c_pass = (
        "Pokito" in test_c_outputs[0][1] and "etymology" not in test_c_outputs[0][1].lower() and
        "Julien Ranouil" in test_c_outputs[1][1] or "electronic" in test_c_outputs[1][1].lower() and
        "Living on Video" in test_c_outputs[2][1] and
        any(k in test_c_outputs[3][1].lower() for k in ["current time", "am", "pm", ":", "date"]) and
        "Time Today" not in test_c_outputs[3][1] and  # Must NOT be the song "Time Today"
        any(k in test_c_outputs[4][1].lower() for k in ["current time", "am", "pm", ":"])
    )
    results["Test C (Entity & Song Understanding)"] = "PASS" if c_pass else "FAIL"

    # =========================================================================
    # TEST D: Educational Reasoning & Technical Explanations
    # =========================================================================
    print("=== RUNNING TEST D: Educational Reasoning ===")
    test_d_queries = [
        "why do our brains need sleep",
        "what biological mechanisms cause this",
        "wht is machne lernng",
        "now explain the same thing technically",
        "give me a practical real-world example",
        "explain the attention mechanism in transformers"
    ]
    test_d_outputs = []
    d_history = []
    for q in test_d_queries:
        d_history.append({"role": "user", "text": q})
        reply, route = generate_chat_response(
            model=model, tokenizer=tokenizer, history=d_history,
            device=device, retrieval_engine=retrieval_engine,
            web_search_enabled=False
        )
        d_history.append({"role": "assistant", "text": reply})
        test_d_outputs.append((q, reply, route))
        print(f"User: {q}\nRoute: {route}\nBot: {reply[:120]}...\n")

    d_pass = (
        "glymphatic" in test_d_outputs[0][1].lower() or "memory" in test_d_outputs[0][1].lower() or "sleep" in test_d_outputs[0][1].lower() and
        "glymphatic" in test_d_outputs[1][1].lower() or "adenosine" in test_d_outputs[1][1].lower() and
        "machine learning" in test_d_outputs[2][1].lower() and
        ("statistical" in test_d_outputs[3][1].lower() or "optimization" in test_d_outputs[3][1].lower() or "objective function" in test_d_outputs[3][1].lower()) and
        ("spam" in test_d_outputs[4][1].lower() or "email" in test_d_outputs[4][1].lower() or "filter" in test_d_outputs[4][1].lower()) and
        ("query" in test_d_outputs[5][1].lower() or "softmax" in test_d_outputs[5][1].lower() or "attention" in test_d_outputs[5][1].lower())
    )
    results["Test D (Educational Reasoning)"] = "PASS" if d_pass else "FAIL"

    # =========================================================================
    # TEST E: Multi-Step Research on AI Agents (Web Search ON)
    # =========================================================================
    print("=== RUNNING TEST E: Multi-Step AI Agent Research ===")
    retrieval_engine.conversation_memory.clear()
    retrieval_engine.last_evidence = None
    test_e_queries = [
        "Find five recent research papers about AI agents.",
        "summarize their major advancements",
        "tell me more about the second paper",
        "who are its authors",
        "what is its publication date",
        "compare the second and fourth papers",
        "what are their limitations"
    ]
    test_e_outputs = []
    e_history = []
    for q in test_e_queries:
        e_history.append({"role": "user", "text": q})
        reply, route = generate_chat_response(
            model=model, tokenizer=tokenizer, history=e_history,
            device=device, retrieval_engine=retrieval_engine,
            web_search_enabled=True
        )
        e_history.append({"role": "assistant", "text": reply})
        test_e_outputs.append((q, reply, route))
        print(f"User: {q}\nRoute: {route}\nBot:\n{reply[:250]}...\n")

    # Assertions for Test E:
    # Turn 1 must have 5 papers (1., 2., 3., 4., 5.)
    has_five_papers = all(f"{i}." in test_e_outputs[0][1] for i in range(1, 6))
    advancements_ok = "Advancement" in test_e_outputs[1][1] or "advancement" in test_e_outputs[1][1].lower()
    paper2_ok = len(test_e_outputs[2][1]) > 50
    authors_ok = "Authors" in test_e_outputs[3][1] or "authors" in test_e_outputs[3][1].lower()
    date_ok = "Date" in test_e_outputs[4][1] or "date" in test_e_outputs[4][1].lower() or "202" in test_e_outputs[4][1]
    compare_ok = "Paper [2]" in test_e_outputs[5][1] or "Paper [4]" in test_e_outputs[5][1] or "Comparison" in test_e_outputs[5][1]
    limitations_ok = "Limitation" in test_e_outputs[6][1] or "limitations" in test_e_outputs[6][1].lower()

    e_pass = has_five_papers and advancements_ok and paper2_ok and authors_ok and date_ok and compare_ok and limitations_ok
    results["Test E (AI Agent Research & Follow-ups)"] = "PASS" if e_pass else "FAIL"

    # =========================================================================
    # TEST F: Companion Continuity & Emotional Transition
    # =========================================================================
    print("=== RUNNING TEST F: Companion Continuity ===")
    test_f_queries = [
        "I had a very stressful day at work.",
        "I argued with my best friend and feel misunderstood.",
        "What should I do?",
        "I'm feeling a bit better now. Can you explain machine learning to me?"
    ]
    test_f_outputs = []
    f_history = []
    for q in test_f_queries:
        f_history.append({"role": "user", "text": q})
        reply, route = generate_chat_response(
            model=model, tokenizer=tokenizer, history=f_history,
            device=device, retrieval_engine=retrieval_engine,
            web_search_enabled=False
        )
        f_history.append({"role": "assistant", "text": reply})
        test_f_outputs.append((q, reply, route))
        print(f"User: {q}\nRoute: {route}\nBot: {reply[:120]}...\n")

    f_pass = (
        len(test_f_outputs[0][1]) > 20 and
        len(test_f_outputs[1][1]) > 20 and
        len(test_f_outputs[2][1]) > 20 and
        "machine learning" in test_f_outputs[3][1].lower() and  # Transitioned cleanly to factual!
        "disagreement" not in test_f_outputs[3][1].lower()     # No emotional dispute hijacking!
    )
    results["Test F (Companion Continuity & Transition)"] = "PASS" if f_pass else "FAIL"

    # =========================================================================
    # TEST G: Unknown Information (Offline vs Online)
    # =========================================================================
    print("=== RUNNING TEST G: Unknown Information Handling ===")
    retrieval_engine.conversation_memory.clear()
    # Offline:
    q_unknown = "What were the exact gross revenue numbers for the box office opening of movie XYZ in late 2026?"
    reply_off, route_off = generate_chat_response(
        model=model, tokenizer=tokenizer, history=[{"role": "user", "text": q_unknown}],
        device=device, retrieval_engine=retrieval_engine,
        web_search_enabled=False
    )
    print(f"Offline User: {q_unknown}\nRoute: {route_off}\nBot: {reply_off}\n")

    # Online:
    q_known = "Who is the Prime Minister of India?"
    reply_on, route_on = generate_chat_response(
        model=model, tokenizer=tokenizer, history=[{"role": "user", "text": q_known}],
        device=device, retrieval_engine=retrieval_engine,
        web_search_enabled=True
    )
    print(f"Online User: {q_known}\nRoute: {route_on}\nBot: {reply_on[:120]}...\n")

    g_pass = (
        ("don't have enough verified offline information" in reply_off.lower() or "web on" in reply_off.lower()) and
        "Narendra Modi" in reply_on
    )
    results["Test G (Unknown Info Handling)"] = "PASS" if g_pass else "FAIL"

    # =========================================================================
    # TEST H: Freshness and Cache Isolation
    # =========================================================================
    print("=== RUNNING TEST H: Freshness and Cache Isolation ===")
    # Query A:
    q_a = "who is india current prime minister"
    reply_a, _ = generate_chat_response(
        model=model, tokenizer=tokenizer, history=[{"role": "user", "text": q_a}],
        device=device, retrieval_engine=retrieval_engine,
        web_search_enabled=True
    )
    # Query B:
    q_b = "what is gta 6"
    reply_b, _ = generate_chat_response(
        model=model, tokenizer=tokenizer, history=[{"role": "user", "text": q_b}],
        device=device, retrieval_engine=retrieval_engine,
        web_search_enabled=True
    )
    h_pass = ("Narendra Modi" in reply_a and "Grand Theft Auto" in reply_b and "Narendra Modi" not in reply_b)
    results["Test H (Cache Isolation)"] = "PASS" if h_pass else "FAIL"

    # =========================================================================
    # TEST I: Repeated Generation Stability
    # =========================================================================
    print("=== RUNNING TEST I: Repeated Generation Stability ===")
    retrieval_engine.conversation_memory.clear()
    q_rep = "now explain the same thing technically"
    replies = []
    for _ in range(3):
        hist = [
            {"role": "user", "text": "wht is machne lernng"},
            {"role": "assistant", "text": "Machine learning is a field of artificial intelligence."},
            {"role": "user", "text": q_rep}
        ]
        rep, _ = generate_chat_response(
            model=model, tokenizer=tokenizer, history=hist,
            device=device, retrieval_engine=retrieval_engine,
            web_search_enabled=False
        )
        replies.append(rep)
    i_pass = all("statistical" in r.lower() or "objective function" in r.lower() for r in replies) and len(set(replies)) == 1
    results["Test I (Generation Stability)"] = "PASS" if i_pass else "FAIL"

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================
    print("\n" + "=" * 50)
    print("         FINAL INTEGRATION TEST SUMMARY")
    print("=" * 50)
    for test_name, status in results.items():
        print(f"{test_name:42}: [{status}]")
    print("=" * 50 + "\n")

    return results

if __name__ == "__main__":
    run_tests()
