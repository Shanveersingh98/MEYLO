"""
scratch/test_final_boss_directive_comprehensive.py
==================================================
Comprehensive real-world runtime test script for the Final Boss Directive.
Tests all 20 Offline Scenarios, 15 Online Scenarios, and 10 Regression Scenarios.
Captures actual responses, routes, search queries, and displayed sources.
"""

import sys
import json
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response

def run_tests():
    device = torch.device("cpu")
    print(f"Loading chatbot on device: {device}")
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
    print("Chatbot successfully loaded.\n")

    results = {
        "offline_tests": [],
        "online_tests": [],
        "regression_tests": []
    }

    # =========================================================================
    # PART 1: 20 OFFLINE TESTS (Web Search = OFF)
    # =========================================================================
    print("=" * 80)
    print("PART 1: 20 OFFLINE RUNTIME TESTS (Web Search = OFF)")
    print("=" * 80)

    # Note: Scenarios 5-8 form a multi-turn conversation
    offline_prompts = [
        # (id, prompt, is_multi_turn, expected_notes)
        (1, "Hey, how are you?", False, "Natural friendly greeting"),
        (2, "What are you doing?", False, "Conversational companion activity check"),
        (3, "I had a tiring day.", False, "Empathetic listening for exhausted day"),
        (4, "I feel lonely.", False, "Warm, understanding companionship"),
        # Multi-turn sequence:
        (5, "I had an argument with my best friend.", True, "Acknowledges friend conflict, asks what caused it"),
        (6, "He misunderstood me.", True, "Understands misunderstanding in ongoing conflict"),
        (7, "What should I do?", True, "Interpersonal advice tailored to situation"),
        (8, "What would you do in my situation?", True, "Thoughtful, gentle personal advice"),
        # Individual emotional & conversational:
        (9, "I am feeling helpless.", False, "Empathetic support for helplessness"),
        (10, "I am angry with my father.", False, "Family conflict empathy (not assuming a friend)"),
        (11, "I passed my exam!", False, "Warm celebration and excitement"),
        # Educational concepts:
        (12, "What is artificial intelligence?", False, "Core AI definition"),
        (13, "What is machine learning?", False, "Core ML definition (distinct from AI)"),
        (14, "What is deep learning?", False, "Core DL definition (hierarchical features)"),
        (15, "What is a chatbot?", False, "Chatbot definition"),
        (16, "What is education?", False, "Education definition"),
        # Honest offline limitations:
        (17, "What is the latest discovery announced today?", False, "Honest offline limitation (current info)"),
        (18, "Who is the current Prime Minister of India?", False, "Honest offline limitation (current officeholder)"),
        (19, "What is the gestation period of an African elephant?", False, "Honest limitation for unknown offline fact"),
        (20, "wats mashine lerning", False, "Spelling correction + educational answer")
    ]

    history_multi = []

    for test_id, prompt, is_multi, note in offline_prompts:
        if is_multi:
            history_multi.append({"role": "user", "text": prompt})
            active_hist = list(history_multi)
        else:
            active_hist = [{"role": "user", "text": prompt}]

        reply, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=active_hist,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=False
        )

        if is_multi:
            history_multi.append({"role": "assistant", "text": reply})

        print(f"\n[OFFLINE TEST {test_id}] User: '{prompt}'")
        print(f"  Route: {route}")
        print(f"  Bot: {reply[:200]}..." if len(reply) > 200 else f"  Bot: {reply}")
        print(f"  Note: {note}")

        results["offline_tests"].append({
            "id": test_id,
            "prompt": prompt,
            "route": route,
            "response": reply,
            "note": note
        })

    # =========================================================================
    # PART 2: 15 ONLINE TESTS (Web Search = ON)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: 15 ONLINE RUNTIME TESTS (Web Search = ON)")
    print("=" * 80)

    online_scenarios = [
        (1, "Hello, how are you?", False, "Conversational greeting - NO search"),
        (2, "I feel stressed.", False, "Empathetic emotional support - NO search"),
        (3, "What is machine learning?", False, "Educational answer - NO unnecessary search"),
        (4, "What is education?", False, "Educational answer - NO unnecessary search"),
        (5, "Explain quantum computing.", False, "Educational web search explanation"),
        (6, "Who is the current Prime Minister of India?", False, "Current officeholder - direct verified answer"),
        (7, "What are the latest developments in AI?", False, "Freshness query - live research with sources"),
        (8, "Why does the human brain need sleep?", False, "Broad universal search query - science"),
        (9, "Compare two smartphone processors.", False, "Comparison query - structured breakdown"),
        # Multi-turn sequence (10 -> 11):
        (10, "Find recent research about artificial intelligence.", True, "Research query with multiple papers/sources"),
        (11, "Explain the second result from your previous answer.", True, "Multi-turn context follow-up on source 2"),
        # Safety & general:
        (12, "Find an official government contact number.", False, "Emergency/helpline directory"),
        (13, "Why do flamingos stand on one leg?", False, "Universal search query outside every predefined KB"),
        (14, "Explain the differences between supervised learning and reinforcement learning with examples", False, "Multi-part comparative question"),
        (15, "can u explain quantam computing in simple english", False, "Typo + informal explanation")
    ]

    online_hist_multi = []

    for test_id, prompt, is_multi, note in online_scenarios:
        if is_multi:
            online_hist_multi.append({"role": "user", "text": prompt})
            active_hist = list(online_hist_multi)
        else:
            active_hist = [{"role": "user", "text": prompt}]

        reply, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=active_hist,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=True
        )

        if is_multi:
            online_hist_multi.append({"role": "assistant", "text": reply})

        last_srcs = getattr(retrieval_engine, "session_last_sources", [])
        src_urls = [getattr(s, "url", str(s)) for s in last_srcs[:2]]

        print(f"\n[ONLINE TEST {test_id}] User: '{prompt}'")
        print(f"  Route: {route}")
        print(f"  Bot: {reply[:200]}..." if len(reply) > 200 else f"  Bot: {reply}")
        if src_urls:
            print(f"  Sources: {src_urls}")
        print(f"  Note: {note}")

        results["online_tests"].append({
            "id": test_id,
            "prompt": prompt,
            "route": route,
            "response": reply,
            "sources": src_urls,
            "note": note
        })

    # =========================================================================
    # PART 3: 10 REGRESSION TESTS (Retesting Previous Failures)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: 10 REGRESSION TESTS (Specific Past Failure Cases)")
    print("=" * 80)

    regression_cases = [
        (1, "What is a chatbot?", False, "Must define chatbot clearly"),
        (2, "What is a machine?", False, "Must define machine, NOT machine learning"),
        (3, "What is machine learning?", False, "Must define ML, NOT generic AI"),
        (4, "Give me police number", False, "Verified emergency police helpline 112 / 100"),
        (5, "I am helpless", False, "Empathetic emotional support for helplessness"),
        (6, "I am foolish", False, "Empathetic support for feeling foolish/self-doubt"),
        (7, "I got injured", False, "Injury acknowledgment and first-aid/medical care guidance"),
        (8, "I beat up my friend", False, "Violence de-escalation, checking if anyone is hurt"),
        (9, "What is the latest discovery announced today?", False, "Honest offline limitation when Web=OFF"),
        (10, "Who is the current Prime Minister of India?", True, "Online: direct answer Narendra Modi")
    ]

    for test_id, prompt, web_mode, note in regression_cases:
        active_hist = [{"role": "user", "text": prompt}]
        reply, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=active_hist,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=web_mode
        )

        print(f"\n[REGRESSION {test_id}] User: '{prompt}' (Web Search={'ON' if web_mode else 'OFF'})")
        print(f"  Route: {route}")
        print(f"  Bot: {reply[:200]}..." if len(reply) > 200 else f"  Bot: {reply}")
        print(f"  Note: {note}")

        results["regression_tests"].append({
            "id": test_id,
            "prompt": prompt,
            "web_search": web_mode,
            "route": route,
            "response": reply,
            "note": note
        })

    # Save all captured results to a JSON file for the final report
    out_file = PROJECT_DIR / "scratch" / "final_boss_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAll test results saved to: {out_file}")

if __name__ == "__main__":
    run_tests()
