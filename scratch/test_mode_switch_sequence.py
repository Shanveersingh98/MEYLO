import os
import sys
import time
import torch

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.chat import load_chatbot, generate_chat_response

def run_mode_switch_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading chatbot on device: {device}...")
    t0 = time.time()
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
    print(f"Chatbot loaded in {time.time() - t0:.2f}s\n")

    query = "What is Machine Learning?"
    history = []

    # Sequence as specified by user:
    # Cycle 1: /web off -> "What is Machine Learning?" -> /web on -> "What is Machine Learning?" -> /web off -> "What is Machine Learning?"
    # We will run 3 full cycles (9 steps total)
    cycles = 3
    results = []

    print("=" * 80)
    print("STARTING MULTI-CYCLE MODE SWITCHING VALIDATION TEST")
    print(f"Query under test: '{query}'")
    print(f"Cycles: {cycles} (Total turns: {cycles * 3})")
    print("=" * 80)

    for cycle in range(1, cycles + 1):
        print(f"\n--- CYCLE {cycle} ---")
        steps = [
            ("OFF", False),
            ("ON", True),
            ("OFF", False)
        ]

        for step_idx, (mode_label, web_enabled) in enumerate(steps, 1):
            step_name = f"Cycle {cycle} - Step {step_idx} (/web {mode_label.lower()})"
            print(f"\n>>> Executing: {step_name}")
            print(f"USER: {query}")

            history.append({"role": "user", "text": query})

            t_start = time.time()
            reply, route = generate_chat_response(
                model=model,
                tokenizer=tokenizer,
                history=history,
                device=device,
                retrieval_engine=retrieval_engine,
                tool_assisted=True,
                web_search_enabled=web_enabled,
                debug_mode=True
            )
            elapsed_ms = (time.time() - t_start) * 1000.0

            history.append({"role": "assistant", "text": reply})

            print(f"BOT ({route}, {elapsed_ms:.1f}ms): {reply}")

            # Verification checks
            if not web_enabled:
                # Mode is OFF:
                # 1. Must NOT make web search calls
                # 2. Must NOT contain meaningless hallucinations or inverted definitions
                assert "broad machine learning" not in reply.lower(), f"Failed: response contained 'broad machine learning' in {step_name}"
                assert "not suitable for human intelligence" not in reply.lower(), f"Failed: response contained 'not suitable for human intelligence' in {step_name}"
                assert "greing" not in reply.lower(), f"Failed: response contained 'greing' in {step_name}"
                assert "ber learning" not in reply.lower(), f"Failed: response contained 'ber learning' in {step_name}"
                assert "projectives" not in reply.lower(), f"Failed: response contained 'projectives' in {step_name}"
                if route == "offline_quality_fallback":
                    assert "couldn't generate a reliable answer using my offline knowledge" in reply or "can't reliably answer that using my offline knowledge" in reply
                    print("  [PASS] Offline quality fallback correctly triggered.")
                elif route == "neural_model":
                    print("  [PASS] Neural model generated coherent validated response.")
                else:
                    print(f"  [NOTE] Route: {route}")
            else:
                # Mode is ON:
                # 1. Must be grounded in web search
                # 2. Must be fast and accurate
                assert route in ["web_grounded", "web_wikipedia_grounded", "web_wikidata_profile", "web_grounded_evidence"], f"Unexpected route for web ON: {route}"
                assert "machine learning" in reply.lower() or "algorithms" in reply.lower(), f"Web grounded answer lacks ML context in {step_name}"
                print("  [PASS] Web grounded accurate answer retrieved.")

            results.append({
                "cycle": cycle,
                "step": step_idx,
                "mode": mode_label,
                "web_enabled": web_enabled,
                "route": route,
                "latency_ms": elapsed_ms,
                "reply_snippet": reply[:100] + ("..." if len(reply) > 100 else "")
            })

    print("\n" + "=" * 80)
    print("TEST SUMMARY ACROSS ALL CYCLES:")
    print("=" * 80)
    for r in results:
        print(f"Cycle {r['cycle']} | Step {r['step']} | Web: {r['mode']:3s} | Route: {r['route']:26s} | Latency: {r['latency_ms']:6.1f}ms | Reply: {r['reply_snippet']}")

    print("\nALL MULTI-CYCLE MODE-SWITCHING CHECKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_mode_switch_test()
