"""
scratch/test_final_master_directive.py
======================================
Targeted verification script for all Master Directive critical repair points.
"""

import sys
import torch
from pathlib import Path

# Add src to path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from chat import load_chatbot, generate_chat_response


def main():
    print("=" * 70)
    print("MASTER DIRECTIVE TARGETED RUNTIME VERIFICATION")
    print("=" * 70)

    device = torch.device("cpu")
    print(f"Loading ChatBot on {device}...")
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device=device)
    print("ChatBot loaded successfully.\n")

    def call_chat(prompt: str, history: list, web_search: bool = False):
        history.append({"role": "user", "text": prompt})
        resp, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=history,
            device=device,
            retrieval_engine=retrieval_engine,
            web_search_enabled=web_search,
            tool_assisted=True
        )
        history.append({"role": "assistant", "text": resp})
        return resp, route

    test_cases = [
        # 1. Chatbot definition
        ("What is a chatbot?", False, "chatbot", ["chatbot", "conversational", "natural language"]),
        # 2. Machine definition
        ("What is a machine?", False, "machine", ["machine", "energy", "work", "apparatus", "mechanism"]),
        # 3. Machine learning
        ("What is machine learning?", False, "machine learning", ["machine learning", "patterns", "data"]),
        # 4. Deep learning
        ("What is deep learning?", False, "deep learning", ["deep learning", "neural network", "layers"]),
        # 5. Public Emergency Information
        ("Give me police number", False, "police number", ["112", "100", "police", "emergency"]),
        ("police number in India", False, "police number in India", ["112", "100", "police"]),
        ("Give me ambulance number", False, "ambulance number", ["102", "108", "112", "ambulance"]),
        # 6. Safety-Sensitive Disclosure: Violence
        ("I beat up my friend", False, "violence deescalation", ["violence", "harm", "injured", "medical", "112"]),
        # 7. Helplessness & Feeling Useless
        ("I am helpless", False, "helpless", ["helpless", "with you", "listen", "weight", "carry"]),
        ("I feel so useless", False, "useless", ["useless", "worth", "breath", "listen", "with you"]),
        # 8. Photosynthesis
        ("What is photosynthesis?", False, "photosynthesis", ["photosynthesis", "glucose", "oxygen", "chlorophyll"]),
    ]

    all_passed = True

    for query, web_on, label, expected_keywords in test_cases:
        print(f"Testing [{label}] -> '{query}' (Web Search={'ON' if web_on else 'OFF'})...")
        hist = []
        if hasattr(retrieval_engine, "conversation_memory"):
            retrieval_engine.conversation_memory.clear()
        if hasattr(retrieval_engine, "session_last_sources"):
            retrieval_engine.session_last_sources = []

        reply, route = call_chat(query, hist, web_search=web_on)

        print(f"  Route: {route}")
        print(f"  Response: {reply[:120]}...\n")

        # Verify no generic bad fallback
        generic_fillers = [
            "Tell me what you'd like to work on, ask, or discuss",
            "I don't have enough verified offline information to answer that accurately. If you would like me to search the web"
        ]
        if any(gf in reply for gf in generic_fillers) and "What is" in query:
            print(f"  [FAIL] Generic fallback detected for query: '{query}'")
            all_passed = False
            continue

        # Check at least one expected keyword
        found = any(k.lower() in reply.lower() for k in expected_keywords)
        if not found:
            print(f"  [FAIL] Expected one of {expected_keywords} in response, but got:\n{reply}")
            all_passed = False
        else:
            print(f"  [PASS] Successfully verified '{label}'")

    # Multi-turn Context Test
    print("\n" + "-" * 70)
    print("Testing Multi-Turn Interpersonal Context Flow:")
    print("-" * 70)
    hist = []
    if hasattr(retrieval_engine, "conversation_memory"):
        retrieval_engine.conversation_memory.clear()

    multiturn_dialogue = [
        ("Hey! I had a pretty tiring day today.", ["tiring", "exhausting", "rough", "listen", "with you"]),
        ("I had a disagreement with my best friend.", ["disagreement", "friend", "close", "listen", "weighs"]),
        ("I don't know whether I should message them first.", ["message", "first", "reach out", "friendship", "conflict"]),
        ("What would you do in this situation?", ["shoes", "message", "friendship", "care", "settle"])
    ]

    for turn_idx, (user_msg, expected_keywords) in enumerate(multiturn_dialogue, 1):
        print(f"Turn {turn_idx} User: {user_msg}")
        reply, route = call_chat(user_msg, hist, web_search=False)
        print(f"  Route: {route}")
        print(f"  Bot: {reply[:140]}...\n")

        # Must not say it needs web search for interpersonal advice!
        if "enable web search" in reply.lower() or "search the web" in reply.lower():
            print(f"  [FAIL] Turn {turn_idx} requested web search for interpersonal advice!")
            all_passed = False

        found = any(k.lower() in reply.lower() for k in expected_keywords)
        if not found:
            print(f"  [FAIL] Expected keywords {expected_keywords} not found in Turn {turn_idx}")
            all_passed = False
        else:
            print(f"  [PASS] Turn {turn_idx} passed.")

    # Educational Roadmap Multi-Turn Flow
    print("\n" + "-" * 70)
    print("Testing Educational Roadmap Multi-Turn Flow:")
    print("-" * 70)
    hist = []
    if hasattr(retrieval_engine, "conversation_memory"):
        retrieval_engine.conversation_memory.clear()

    roadmap_dialogue = [
        ("I want to learn artificial intelligence.", ["roadmap", "python", "machine learning", "deep learning"]),
        ("Why did you put Python before machine learning?", ["python", "fundamentals", "data handling", "implementation"]),
        ("Explain the second step.", ["mathematics", "linear algebra", "calculus", "statistics"])
    ]

    for turn_idx, (user_msg, expected_keywords) in enumerate(roadmap_dialogue, 1):
        print(f"Roadmap Turn {turn_idx} User: {user_msg}")
        reply, route = call_chat(user_msg, hist, web_search=False)
        print(f"  Route: {route}")
        print(f"  Bot: {reply[:140]}...\n")

        found = any(k.lower() in reply.lower() for k in expected_keywords)
        if not found:
            print(f"  [FAIL] Expected keywords {expected_keywords} not found in Roadmap Turn {turn_idx}")
            all_passed = False
        else:
            print(f"  [PASS] Roadmap Turn {turn_idx} passed.")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TARGETED VERIFICATIONS PASSED SUCCESSFULLY!")
    else:
        print("SOME VERIFICATIONS FAILED. PLEASE CHECK LOGS ABOVE.")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
