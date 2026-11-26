"""
run_test_b_live.py
===================
Runs Test B (Mode 2: With Web Search) with live internet access.
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

QUESTIONS_TEST_B = [
    {"num": 1, "q": "What is the latest NVIDIA GPU?", "desc": "Hardware freshness check"},
    {"num": 2, "q": "Explain Transformer architecture in deep learning.", "desc": "Educational structured architecture inquiry"},
    {"num": 3, "q": "Compare CNN and Transformer architectures, including their advantages and disadvantages.", "desc": "Multi-concept comparative inquiry"},
    {"num": 4, "q": "What are the latest developments in Artificial Intelligence?", "desc": "Current events / research freshness check"},
    {"num": 5, "q": "Hello, how are you?", "desc": "Casual greeting under web=ON (must NOT search)"},
    {"num": 6, "q": "I am feeling sad. Can we talk?", "desc": "Emotional support under web=ON (must NOT search)"}
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading chatbot for live web test on {device}...")
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

    results_b = []
    print("\n" + "=" * 60)
    print("EXECUTING LIVE TEST B (WITH INTERNET ACCESS)")
    print("=" * 60)

    for item in QUESTIONS_TEST_B:
        num = item["num"]
        q = item["q"]
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
        print(f"\n[B-{num}] Question: {q}")
        print(f"     Description: {desc}")
        print(f"     Route: {route} | Used Search: {used_search} | Latency: {lat:.0f}ms")
        print(f"     BOT Response:\n{resp}\n")
        print("-" * 60)

        results_b.append({
            "number": num,
            "question": q,
            "desc": desc,
            "route": route,
            "used_search": used_search,
            "latency_ms": round(lat, 1),
            "response": resp
        })

    out_file = PROJECT_DIR / "logs" / "model_development_stage_2" / "test_b_live_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_b, f, indent=2)
    print(f"Test B live results saved to: {out_file}")

if __name__ == "__main__":
    main()
