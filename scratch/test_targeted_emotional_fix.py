import sys
from pathlib import Path
import json

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response

def run_targeted_tests():
    device = torch.device("cpu")
    print("Loading chatbot...")
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
    print("Chatbot loaded successfully.\n")

    results = []
    all_passed = True

    # -------------------------------------------------------------------------
    # SEQUENCE 1: Factual questions after emotional conversation (Web Search = ON)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("SEQUENCE 1: Factual questions after emotional conversation (Web Search = ON)")
    print("=" * 80)
    
    seq1_turns = [
        ("I am feeling very stressed today.", ["companion_stressed", "companion_tired"], "Empathetic emotional support, no search"),
        ("Why does the human brain need sleep?", ["web_grounded_evidence"], "Factual research question with web sources, NOT emotional fallback"),
        ("Why do flamingos stand on one leg?", ["web_grounded_evidence"], "Factual research question with web sources, NOT emotional fallback")
    ]

    hist1 = []
    for prompt, expected_routes, desc in seq1_turns:
        hist1.append({"role": "user", "text": prompt})
        reply, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=list(hist1),
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=True
        )
        hist1.append({"role": "assistant", "text": reply})

        # Crucial check: A test must FAIL if a factual question produces an emotional fallback
        is_emotional_fallback = ("I'm right here with you" in reply or "Feel free to share whatever is on your mind" in reply or "Tell me what's on your heart" in reply) and route != "companion_stressed"
        passed = (route in expected_routes) and not is_emotional_fallback

        if not passed:
            all_passed = False

        print(f"\nUser: '{prompt}'")
        print(f"  Route: {route} (Expected one of: {expected_routes})")
        print(f"  Bot: {reply[:180]}..." if len(reply) > 180 else f"  Bot: {reply}")
        print(f"  Evaluation: {'PASS' if passed else 'FAIL - Misrouted to emotional fallback or wrong route'}")

        results.append({
            "sequence": 1,
            "prompt": prompt,
            "route": route,
            "expected_routes": expected_routes,
            "response": reply,
            "passed": passed,
            "description": desc
        })

    # -------------------------------------------------------------------------
    # SEQUENCE 2: Genuine emotional continuity (Web Search = ON / OFF)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SEQUENCE 2: Genuine emotional continuity (Interpersonal dialogue)")
    print("=" * 80)

    seq2_turns = [
        ("I had an argument with my best friend.", ["companion_emotional_hurt"], "Acknowledges friend conflict"),
        ("He misunderstood me.", ["companion_emotional_hurt"], "Empathy for misunderstanding in ongoing dispute"),
        ("What should I do?", ["companion_interpersonal_advice"], "Thoughtful personal advice without web search")
    ]

    hist2 = []
    for prompt, expected_routes, desc in seq2_turns:
        hist2.append({"role": "user", "text": prompt})
        reply, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=list(hist2),
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=True
        )
        hist2.append({"role": "assistant", "text": reply})

        passed = route in expected_routes
        if not passed:
            all_passed = False

        print(f"\nUser: '{prompt}'")
        print(f"  Route: {route} (Expected: {expected_routes})")
        print(f"  Bot: {reply[:180]}..." if len(reply) > 180 else f"  Bot: {reply}")
        print(f"  Evaluation: {'PASS' if passed else 'FAIL'}")

        results.append({
            "sequence": 2,
            "prompt": prompt,
            "route": route,
            "expected_routes": expected_routes,
            "response": reply,
            "passed": passed,
            "description": desc
        })

    # -------------------------------------------------------------------------
    # SEQUENCE 3: Other Questions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SEQUENCE 3: Other Specific Questions")
    print("=" * 80)

    seq3_cases = [
        ("What is machine learning?", False, ["offline_educational_grounded"], "Web OFF: Clean offline definition"),
        ("Who is the current Prime Minister of India?", False, ["offline_freshness_fallback"], "Web OFF: Honest offline limitation"),
        ("What are the latest AI developments in 2026?", True, ["web_grounded_evidence"], "Web ON: Live research retrieval with sources")
    ]

    for prompt, web_mode, expected_routes, desc in seq3_cases:
        hist3 = [{"role": "user", "text": prompt}]
        reply, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=hist3,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=web_mode
        )

        passed = route in expected_routes
        if not passed:
            all_passed = False

        print(f"\nUser: '{prompt}' (Web Search={'ON' if web_mode else 'OFF'})")
        print(f"  Route: {route} (Expected: {expected_routes})")
        print(f"  Bot: {reply[:180]}..." if len(reply) > 180 else f"  Bot: {reply}")
        print(f"  Evaluation: {'PASS' if passed else 'FAIL'}")

        results.append({
            "sequence": 3,
            "prompt": prompt,
            "web_search": web_mode,
            "route": route,
            "expected_routes": expected_routes,
            "response": reply,
            "passed": passed,
            "description": desc
        })

    print("\n" + "=" * 80)
    print(f"OVERALL TARGETED TEST SUITE: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80)

    out_file = PROJECT_DIR / "scratch" / "targeted_repair_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_file}")

    return all_passed

if __name__ == "__main__":
    success = run_targeted_tests()
    if not success:
        sys.exit(1)
