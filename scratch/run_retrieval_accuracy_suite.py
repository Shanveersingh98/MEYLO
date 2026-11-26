"""
Comprehensive Retrieval Accuracy and Concept Discrimination Test Suite
Executes the 10 distinct educational concepts, multi-turn follow-ups, and regressions.
Generates:
  - retrieval_test_results.json
  - source_relevance_results.json
  - regression_results.json
under logs/chatbot_retrieval_accuracy/
"""

import sys
import os
import json
import time
import re
import torch

sys.path.insert(0, os.path.abspath("."))

from src.chat import load_chatbot, generate_chat_response

LOG_DIR = os.path.abspath("logs/chatbot_retrieval_accuracy")
os.makedirs(LOG_DIR, exist_ok=True)

def evaluate_concept_correctness(query: str, primary_source: str, response: str, engine: str) -> dict:
    q_low = query.lower()
    r_low = response.lower()
    src_low = primary_source.lower()

    passed = False
    source_matches = False
    notes = ""

    # 1. Supervised learning
    if "difference between supervised and self-supervised" in q_low:
        source_matches = "supervised" in src_low
        passed = (
            "supervised learning" in r_low and
            "self-supervised learning" in r_low and
            ("label" in r_low or "annotat" in r_low) and
            ("unlabeled" in r_low or "mask" in r_low or "pseudo" in r_low or "pre-train" in r_low)
        )
        notes = "Explains distinction between human-labeled data and data-generated supervision."

    elif "self-supervised learning" in q_low:
        source_matches = "self-supervised" in src_low
        passed = (
            source_matches and
            ("unlabeled" in r_low or "itself" in r_low or "pretext" in r_low or "mask" in r_low or "contrastive" in r_low or "labels" in r_low) and
            not r_low.startswith("in machine learning, supervised learning (sl) is a type of machine learning paradigm where an algorithm learns to map input data to a specific output")
        )
        notes = "Explains self-supervision using the data itself to generate training signals."

    elif "unsupervised learning" in q_low:
        source_matches = "unsupervised" in src_low
        passed = (
            source_matches and
            ("unlabeled" in r_low or "cluster" in r_low or "patterns" in r_low or "without" in r_low or "density" in r_low)
        )
        notes = "Explains finding patterns in unlabeled data without target labels."

    elif "semi-supervised learning" in q_low:
        source_matches = "semi-supervised" in src_low or "weak supervision" in src_low
        passed = (
            source_matches and
            ("small amount of labeled" in r_low or "labeled" in r_low and "unlabeled" in r_low or "weak supervision" in r_low)
        )
        notes = "Explains combining labeled and unlabeled data (or weak supervision)."

    elif "reinforcement learning" in q_low:
        source_matches = "reinforcement" in src_low
        passed = (
            source_matches and
            ("reward" in r_low or "agent" in r_low or "environment" in r_low or "action" in r_low)
        )
        notes = "Explains learning by interacting with environment to maximize cumulative reward."

    elif "supervised learning" in q_low:
        source_matches = "supervised learning" in src_low and "self-supervised" not in src_low
        passed = (
            source_matches and
            ("label" in r_low or "input" in r_low or "output" in r_low or "supervisor" in r_low or "teacher" in r_low) and
            "self-supervised learning (ssl) is a paradigm" not in r_low
        )
        notes = "Prioritizes supervised learning with labeled training data and input-output mapping."

    elif "difference between ai" in q_low or "difference between artificial intelligence" in q_low:
        source_matches = True
        passed = "artificial intelligence" in r_low and "machine learning" in r_low and "deep learning" in r_low
        notes = "Hierarchical breakdown of AI, ML, and DL."

    elif "machine learning" in q_low:
        source_matches = "machine learning" in src_low
        passed = source_matches and ("data" in r_low or "algorithms" in r_low or "artificial intelligence" in r_low)
        notes = "Defines ML as a field of AI learning from data."

    elif "deep learning" in q_low:
        source_matches = "deep learning" in src_low and "software" not in src_low
        passed = source_matches and ("multilayer" in r_low or "layers" in r_low or "neural network" in r_low)
        notes = "Explains deep learning using multilayered neural networks."

    elif "neural network" in q_low:
        source_matches = "neural network" in src_low
        passed = source_matches and ("neuron" in r_low or "nodes" in r_low or "units" in r_low)
        notes = "Defines neural networks as interconnected artificial neurons."

    elif "chatgpt" in q_low:
        source_matches = "chatgpt" in src_low
        passed = source_matches and ("openai" in r_low or "chatbot" in r_low or "language model" in r_low)
        notes = "Defines ChatGPT as an AI chatbot developed by OpenAI."

    elif "hello" in q_low:
        source_matches = True
        passed = engine == "neural_model" and len(r_low) > 3
        notes = "Conversational greeting handled locally."

    elif "45 * 12" in q_low:
        source_matches = True
        passed = engine == "calculator" and "540" in r_low
        notes = "Arithmetic expression calculated deterministically."

    else:
        source_matches = bool(primary_source)
        passed = len(response) > 20
        notes = "General validation passed."

    return {
        "passed": passed,
        "source_matches_subject": source_matches,
        "factually_appropriate": passed,
        "notes": notes
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading chatbot onto {device}...")
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
    print("Chatbot loaded successfully.")

    # 1. Ten Core Concept Tests
    concept_tests = [
        {"id": "concept_1", "query": "What is supervised learning?", "web": True, "tool": True},
        {"id": "concept_2", "query": "What is self-supervised learning?", "web": True, "tool": True},
        {"id": "concept_3", "query": "What is unsupervised learning?", "web": True, "tool": True},
        {"id": "concept_4", "query": "What is semi-supervised learning?", "web": True, "tool": True},
        {"id": "concept_5", "query": "What is reinforcement learning?", "web": True, "tool": True},
        {"id": "concept_6", "query": "What is machine learning?", "web": True, "tool": True},
        {"id": "concept_7", "query": "What is deep learning?", "web": True, "tool": True},
        {"id": "concept_8", "query": "What is a neural network?", "web": True, "tool": True},
        {"id": "concept_9", "query": "What is ChatGPT?", "web": True, "tool": True},
        {"id": "concept_10", "query": "Explain the difference between supervised and self-supervised learning.", "web": True, "tool": True},
    ]

    # 2. Multi-turn Follow-up Tests
    multiturn_tests = [
        {"id": "turn_1", "query": "What is supervised learning?", "web": True, "tool": True},
        {"id": "turn_2", "query": "What are the main points from the second search result?", "web": True, "tool": True},
        {"id": "turn_3", "query": "Summarize the first source", "web": True, "tool": True},
    ]

    # 3. Regression Tests
    regression_tests = [
        {"id": "reg_dl_simple", "query": "Explain deep learning in simple words.", "web": True, "tool": True},
        {"id": "reg_chatgpt_search", "query": "/search What is ChatGPT", "web": True, "tool": True},
        {"id": "reg_ai_ml_dl", "query": "Explain the difference between AI, machine learning, and deep learning.", "web": True, "tool": True},
        {"id": "reg_greeting", "query": "Hello! How are you?", "web": True, "tool": True},
        {"id": "reg_math", "query": "What is 45 * 12?", "web": True, "tool": True},
        {"id": "reg_offline", "query": "What is supervised learning?", "web": False, "tool": True},
    ]

    retrieval_results = []
    source_relevance_records = []
    regression_records = []

    print("\n" + "=" * 70)
    print("PHASE 1: EXECUTING 10 TARGET CONCEPT TESTS")
    print("=" * 70)

    for tc in concept_tests:
        tid = tc["id"]
        q = tc["query"]
        norm_q = retrieval_engine.extract_search_query(q)

        print(f"\n[{tid}] Query: '{q}' (Normalized: '{norm_q}')")
        history = [{"role": "user", "text": q}]

        t0 = time.time()
        reply, engine = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=history,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=tc["tool"],
            web_search_enabled=tc["web"]
        )
        latency_ms = (time.time() - t0) * 1000

        # Extract sources
        sources = []
        source_titles = []
        if "Sources:" in reply:
            s_lines = reply.split("Sources:")[1].strip().split("\n")
            for line in s_lines:
                if line.strip():
                    sources.append(line.strip())
                    # Extract title between bracket and colon: e.g. "[1] Supervised learning: https://..."
                    t_match = re.search(r"\[\d+\]\s+([^:]+):", line)
                    if t_match:
                        source_titles.append(t_match.group(1).strip())
                    else:
                        source_titles.append(line.strip())

        primary_title = source_titles[0] if source_titles else "None"
        eval_res = evaluate_concept_correctness(q, primary_title, reply, engine)

        rec = {
            "test_id": tid,
            "original_query": q,
            "normalized_query": norm_q,
            "engine": engine,
            "latency_ms": round(latency_ms, 1),
            "primary_source_title": primary_title,
            "all_source_titles": source_titles,
            "source_urls": sources,
            "source_matches_subject": eval_res["source_matches_subject"],
            "factually_appropriate": eval_res["factually_appropriate"],
            "passed": eval_res["passed"],
            "notes": eval_res["notes"],
            "final_answer": reply
        }
        retrieval_results.append(rec)

        source_relevance_records.append({
            "test_id": tid,
            "query": q,
            "target_concept": norm_q,
            "primary_source": primary_title,
            "source_titles": source_titles,
            "is_relevant_to_subject": eval_res["source_matches_subject"]
        })

        print(f"    Engine: {engine} ({latency_ms:.0f}ms)")
        print(f"    Primary Source: {primary_title}")
        print(f"    Source Matches Subject: {eval_res['source_matches_subject']}")
        print(f"    Result: {'PASS' if eval_res['passed'] else 'FAIL'}")

    print("\n" + "=" * 70)
    print("PHASE 2: EXECUTING MULTI-TURN CACHE FOLLOW-UP TESTS")
    print("=" * 70)

    mt_history = []
    for tc in multiturn_tests:
        tid = tc["id"]
        q = tc["query"]
        print(f"\n[{tid}] Turn: '{q}'")
        mt_history.append({"role": "user", "text": q})

        t0 = time.time()
        reply, engine = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=mt_history,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=tc["tool"],
            web_search_enabled=tc["web"]
        )
        latency_ms = (time.time() - t0) * 1000
        mt_history.append({"role": "assistant", "text": reply})

        is_passed = len(reply.strip()) > 15
        if tid == "turn_2":
            is_passed = engine == "web_cache_followup" and ("Source [2]" in reply or "URL:" in reply)
        elif tid == "turn_3":
            is_passed = engine == "web_cache_followup" and ("Source [1]" in reply or "URL:" in reply)

        print(f"    Engine: {engine} ({latency_ms:.0f}ms)")
        print(f"    Result: {'PASS' if is_passed else 'FAIL'}")

        retrieval_results.append({
            "test_id": f"multiturn_{tid}",
            "original_query": q,
            "normalized_query": retrieval_engine.extract_search_query(q),
            "engine": engine,
            "latency_ms": round(latency_ms, 1),
            "primary_source_title": "Cached Follow-up",
            "all_source_titles": ["Cached Search Context"],
            "source_urls": [],
            "source_matches_subject": True,
            "factually_appropriate": is_passed,
            "passed": is_passed,
            "notes": "Multi-turn context follow-up",
            "final_answer": reply
        })

    print("\n" + "=" * 70)
    print("PHASE 3: EXECUTING REGRESSION SUITE")
    print("=" * 70)

    for tc in regression_tests:
        tid = tc["id"]
        q = tc["query"]
        print(f"\n[{tid}] Regression Query: '{q}'")
        history = [{"role": "user", "text": q}]

        t0 = time.time()
        reply, engine = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=history,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=tc["tool"],
            web_search_enabled=tc["web"]
        )
        latency_ms = (time.time() - t0) * 1000

        primary_title = "None"
        if "Sources:" in reply:
            s_lines = reply.split("Sources:")[1].strip().split("\n")
            for line in s_lines:
                if line.strip():
                    t_match = re.search(r"\[\d+\]\s+([^:]+):", line)
                    if t_match:
                        primary_title = t_match.group(1).strip()
                        break

        eval_res = evaluate_concept_correctness(q, primary_title, reply, engine)

        if tid == "reg_offline":
            # For offline mode, should be disclaimed
            eval_res["passed"] = engine == "neural_model_unverified" or "[Offline Mode Notice" in reply

        print(f"    Engine: {engine} ({latency_ms:.0f}ms)")
        print(f"    Result: {'PASS' if eval_res['passed'] else 'FAIL'}")

        regression_records.append({
            "test_id": tid,
            "query": q,
            "engine": engine,
            "latency_ms": round(latency_ms, 1),
            "passed": eval_res["passed"],
            "notes": eval_res["notes"],
            "response_preview": reply[:250].replace("\n", " ")
        })

    # Save deliverables
    retrieval_file = os.path.join(LOG_DIR, "retrieval_test_results.json")
    with open(retrieval_file, "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, indent=2)

    source_file = os.path.join(LOG_DIR, "source_relevance_results.json")
    with open(source_file, "w", encoding="utf-8") as f:
        json.dump(source_relevance_records, f, indent=2)

    regression_file = os.path.join(LOG_DIR, "regression_results.json")
    with open(regression_file, "w", encoding="utf-8") as f:
        json.dump(regression_records, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE! Deliverables saved to logs/chatbot_retrieval_accuracy/")
    print("=" * 70)

if __name__ == "__main__":
    main()
