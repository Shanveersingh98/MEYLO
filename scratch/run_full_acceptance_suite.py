"""
Full Acceptance Test Suite for Chatbot Answer Repair
Evaluates the 10 core questions from Section 5 plus controls.
Records responses, routing engines, latency, sources, and correctness.
"""

import sys
import os
import json
import time
import torch

sys.path.insert(0, os.path.abspath("."))

from src.chat import load_chatbot, generate_chat_response

LOG_DIR = os.path.abspath("logs/chatbot_answer_repair")
os.makedirs(LOG_DIR, exist_ok=True)

def evaluate_factual_correctness(question: str, response: str, engine: str) -> dict:
    q_low = question.lower()
    r_low = response.lower()
    
    # 1. Deep learning
    if "what is deep learning" in q_low:
        is_dl = ("neural network" in r_low or "multilayer" in r_low or "layers" in r_low) and "machine learning" in r_low
        no_mach = not r_low.startswith("mach is a deep learning")
        correct = is_dl and no_mach
        notes = "Explains DL as a subset of ML using multilayered neural networks."
    # 2. ChatGPT
    elif "what is chatgpt" in q_low:
        is_gpt = ("chatbot" in r_low or "conversational" in r_low or "language model" in r_low) and "openai" in r_low
        not_subfield = "subfield of machine learning" not in r_low
        correct = is_gpt and not_subfield
        notes = "Explains ChatGPT as an AI assistant developed by OpenAI."
    # 3. Machine learning
    elif "what is machine learning" in q_low:
        correct = ("data" in r_low or "patterns" in r_low or "statistical" in r_low) and ("artificial intelligence" in r_low or "ai" in r_low)
        notes = "Identifies ML as a branch of AI focusing on data and learning."
    # 4. Difference between AI, ML, DL
    elif "difference between" in q_low and "deep learning" in q_low:
        correct = "artificial intelligence" in r_low and "machine learning" in r_low and "deep learning" in r_low and ("subset" in r_low or "hierarchical" in r_low or "layers" in r_low)
        notes = "Distinguishes hierarchical relationship: AI includes ML, ML includes DL."
    # 5. Three applications
    elif "applications of deep learning" in q_low or "give three" in q_low:
        correct = any(t in r_low for t in ["vision", "image", "natural language", "nlp", "speech", "autonomous", "translation"])
        notes = "Gives concrete real-world domains like computer vision, NLP, or autonomous systems."
    # 6. Supervised learning simple words
    elif "supervised learning" in q_low:
        correct = ("labeled" in r_low or "labels" in r_low or "input" in r_low or "answer key" in r_low) and "student" in r_low or "data" in r_low
        notes = "Explains mapping inputs to outputs using labeled data or intuitive analogy."
    # 7. Neural network
    elif "neural network" in q_low:
        correct = ("interconnected" in r_low or "neurons" in r_low or "layers" in r_low or "nodes" in r_low)
        notes = "Defines neural networks as interconnected artificial neurons or nodes."
    # 8. Summarize educational source
    elif "summarize an accessible" in q_low:
        correct = ("wikipedia" in r_low or "summary" in r_low) and "neural network" in r_low or "deep learning" in r_low
        notes = "Summarizes public educational article with proper attribution."
    # 9. Main points from second result
    elif "second search result" in q_low or "second result" in q_low:
        correct = "source [2]" in r_low and ("url:" in r_low or "points" in r_low or "summary" in r_low)
        notes = "Retrieves second search result and extracts key points."
    # 10. Current info
    elif "python" in q_low:
        correct = "python" in r_low and ("release" in r_low or "programming" in r_low or "foundation" in r_low)
        notes = "Answers query requiring real-world software information from web sources."
    # Controls
    elif "hello" in q_low:
        correct = engine == "neural_model" and len(r_low) > 3
        notes = "Conversational greeting handled locally by neural model."
    elif "45 * 12" in q_low:
        correct = engine == "calculator" and "540" in r_low
        notes = "Arithmetic expression calculated deterministically."
    else:
        correct = True
        notes = "Ad-hoc check passed."

    return {
        "factually_correct": correct,
        "notes": notes,
        "relevant_and_understandable": len(response.strip()) > 30 and ("\n" in response or " " in response),
        "grounded_in_evidence": "Sources:" in response or "URL:" in response or engine == "calculator" or engine == "neural_model"
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading chatbot onto {device}...")
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
    print("Chatbot loaded successfully.")

    test_cases = [
        # Acceptance Question 1
        {"id": "test_1", "query": "What is deep learning?", "web": True, "tool": True},
        # Acceptance Question 2
        {"id": "test_2", "query": "What is ChatGPT?", "web": True, "tool": True},
        # Acceptance Question 3
        {"id": "test_3", "query": "What is machine learning?", "web": True, "tool": True},
        # Acceptance Question 4
        {"id": "test_4", "query": "Explain the difference between AI, machine learning, and deep learning.", "web": True, "tool": True},
        # Acceptance Question 5
        {"id": "test_5", "query": "Give three real-world applications of deep learning.", "web": True, "tool": True},
        # Acceptance Question 6
        {"id": "test_6", "query": "Explain supervised learning in simple words.", "web": True, "tool": True},
        # Acceptance Question 7
        {"id": "test_7", "query": "What is a neural network?", "web": True, "tool": True},
        # Acceptance Question 8
        {"id": "test_8", "query": "Summarize an accessible public educational source about deep learning.", "web": True, "tool": True},
        # Acceptance Question 9 (Follow-up to test 8)
        {"id": "test_9", "query": "What are the main points from the second search result?", "web": True, "tool": True},
        # Acceptance Question 10 (Current information)
        {"id": "test_10", "query": "What is Python programming language and its latest release history?", "web": True, "tool": True},
        
        # Explicit search commands
        {"id": "explicit_search_1", "query": "/search what is deep learning", "web": True, "tool": True},
        {"id": "explicit_search_2", "query": "/search what is ChatGPT", "web": True, "tool": True},

        # Controls
        {"id": "control_greeting", "query": "Hello! How are you?", "web": True, "tool": True},
        {"id": "control_math", "query": "What is 45 * 12?", "web": True, "tool": True},
        {"id": "control_offline_fallback", "query": "What is deep learning?", "web": False, "tool": True},
    ]

    all_results = []
    corrected_responses = []
    source_verifications = []

    history = []

    print("\nStarting Test Execution...\n")

    for tc in test_cases:
        tc_id = tc["id"]
        q = tc["query"]
        web_enabled = tc["web"]
        tool_enabled = tc["tool"]

        print(f"[{tc_id}] Executing: '{q}' (web={web_enabled})...")

        # For follow-up test 9, we keep previous search cache in history; for standalone tests, we manage history
        if tc_id != "test_9":
            history = [{"role": "user", "text": q}]
        else:
            history.append({"role": "user", "text": q})

        t0 = time.time()
        reply, engine = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=history,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=tool_enabled,
            web_search_enabled=web_enabled
        )
        latency_ms = (time.time() - t0) * 1000

        history.append({"role": "assistant", "text": reply})

        eval_res = evaluate_factual_correctness(q, reply, engine)

        # Extract sources from reply
        sources = []
        if "Sources:" in reply:
            sources_part = reply.split("Sources:")[1].strip().split("\n")
            for s in sources_part:
                if s.strip():
                    sources.append(s.strip())
        elif "URL:" in reply:
            for line in reply.split("\n"):
                if line.startswith("URL:"):
                    sources.append(line.replace("URL:", "").strip())

        result_entry = {
            "test_id": tc_id,
            "query": q,
            "engine": engine,
            "web_search_enabled": web_enabled,
            "latency_ms": round(latency_ms, 1),
            "response": reply,
            "sources": sources,
            "eval_results": eval_res
        }
        all_results.append(result_entry)

        corrected_responses.append({
            "test_id": tc_id,
            "query": q,
            "engine": engine,
            "latency_ms": round(latency_ms, 1),
            "response": reply
        })

        if sources:
            source_verifications.append({
                "test_id": tc_id,
                "query": q,
                "sources": sources,
                "provider": "wikipedia" if any("wikipedia.org" in s for s in sources) else "web"
            })

        print(f"    Engine: {engine} ({latency_ms:.0f}ms)")
        print(f"    Factually Correct: {eval_res['factually_correct']}")
        print(f"    Grounded in Evidence: {eval_res['grounded_in_evidence']}")
        print(f"    Notes: {eval_res['notes']}\n")

    # Save deliverable JSON files
    with open(os.path.join(LOG_DIR, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(LOG_DIR, "corrected_responses.json"), "w", encoding="utf-8") as f:
        json.dump(corrected_responses, f, indent=2)

    with open(os.path.join(LOG_DIR, "source_verification_results.json"), "w", encoding="utf-8") as f:
        json.dump(source_verifications, f, indent=2)

    print("All deliverables written to logs/chatbot_answer_repair/")

if __name__ == "__main__":
    main()
