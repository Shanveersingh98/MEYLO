"""
tests/evaluate_model_capabilities.py
====================================
Evaluates both models (Custom FromScratchLM & External Qwen2.5-1.5B)
across the 12 core capability categories in Offline and Online modes.
Outputs structured evaluation results to data/verified_model_capabilities.json
and generates an audit summary for the frontend Capability Index.
"""

import sys
import json
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from fastapi.testclient import TestClient
from src.api_server import app, state

CATEGORY_PROMPTS = [
    {
        "id": "general_conversation",
        "name": "General Conversation",
        "prompt": "How are you doing today, and what are you capable of helping me with?",
        "test_type": "offline_and_online"
    },
    {
        "id": "greetings_everyday",
        "name": "Greetings & Everyday Questions",
        "prompt": "Good morning! Can you suggest a healthy morning routine for high productivity?",
        "test_type": "offline"
    },
    {
        "id": "educational_explanations",
        "name": "Educational Explanations",
        "prompt": "Explain the concept of photosynthesis in simple terms.",
        "test_type": "offline"
    },
    {
        "id": "computer_science_programming",
        "name": "Computer Science & Programming Concepts",
        "prompt": "What is binary search, and what is its big-O time complexity?",
        "test_type": "offline"
    },
    {
        "id": "math_logical_reasoning",
        "name": "Mathematics & Logical Reasoning",
        "prompt": "If a car travels 180 kilometers in 3 hours, what is its average speed in km/h?",
        "test_type": "offline"
    },
    {
        "id": "emotional_support",
        "name": "Emotional & Supportive Conversations",
        "prompt": "I've had an exhausting and stressful day at work today.",
        "test_type": "offline"
    },
    {
        "id": "intent_understanding",
        "name": "Intent Understanding & Nuance",
        "prompt": "I want to learn to code. Where should a complete beginner start?",
        "test_type": "offline"
    },
    {
        "id": "writing_summarization",
        "name": "Writing, Rewriting & Summarization",
        "prompt": "Summarize this in one sentence: 'Due to the fact that inclement weather occurred, the outdoor event was postponed.'",
        "test_type": "offline"
    },
    {
        "id": "general_knowledge",
        "name": "General Knowledge",
        "prompt": "What is the capital of France and what river flows through it?",
        "test_type": "offline"
    },
    {
        "id": "current_events",
        "name": "Current Events & Recent Advances",
        "prompt": "What are recent major developments in generative artificial intelligence?",
        "test_type": "online_grounded"
    },
    {
        "id": "web_research",
        "name": "Web Research & Fact Verification",
        "prompt": "Who is the current CEO of Google and Alphabet?",
        "test_type": "online_grounded"
    },
    {
        "id": "multi_turn_context",
        "name": "Multi-turn Conversation & Context Retention",
        "prompt": "My favorite programming language is Python.",
        "follow_up": "What is my favorite programming language?",
        "test_type": "offline"
    }
]


def run_evaluation():
    print("=" * 65)
    print("Evaluating Model Capabilities across 12 Categories")
    print("=" * 65)

    client_context = TestClient(app)
    results = {
        "custom": {"offline": {}, "online": {}},
        "external": {"offline": {}, "online": {}}
    }

    with client_context as client:
        # 1. EVALUATE CUSTOM MODEL
        print("\n--- Evaluating Custom Model (FromScratchLM, CUDA) ---")
        client.post("/api/models/switch", json={"model": "custom"})

        # Custom Offline
        client.post("/api/web-search", json={"enabled": False})
        conv = client.post("/api/conversations", json={"title": "Eval Custom Offline", "model": "custom"}).json()
        conv_id = conv["conversation_id"]

        for item in CATEGORY_PROMPTS:
            print(f"  [Custom Offline] {item['name']}...")
            t0 = time.time()
            if item["id"] == "multi_turn_context":
                client.post("/api/chat", json={"conversation_id": conv_id, "message": item["prompt"]})
                r = client.post("/api/chat", json={"conversation_id": conv_id, "message": item["follow_up"]})
            else:
                r = client.post("/api/chat", json={"conversation_id": conv_id, "message": item["prompt"]})
            latency = (time.time() - t0) * 1000
            data = r.json()
            results["custom"]["offline"][item["id"]] = {
                "category": item["name"],
                "response": data.get("content", ""),
                "latency_ms": round(latency, 1),
                "model_used": data.get("model_used")
            }

        # Custom Online (Categories 10 & 11)
        print("  [Custom Online] Web Grounded Queries...")
        client.post("/api/web-search", json={"enabled": True})
        conv_on = client.post("/api/conversations", json={"title": "Eval Custom Online", "model": "custom"}).json()
        for cat_id in ["current_events", "web_research"]:
            item = next(p for p in CATEGORY_PROMPTS if p["id"] == cat_id)
            print(f"  [Custom Online] {item['name']}...")
            t0 = time.time()
            r = client.post("/api/chat", json={"conversation_id": conv_on["conversation_id"], "message": item["prompt"]})
            latency = (time.time() - t0) * 1000
            data = r.json()
            results["custom"]["online"][item["id"]] = {
                "category": item["name"],
                "response": data.get("content", ""),
                "sources_count": len(data.get("sources", [])),
                "latency_ms": round(latency, 1),
                "model_used": data.get("model_used")
            }

        # 2. EVALUATE EXTERNAL MODEL (Qwen2.5-1.5B, CPU)
        print("\n--- Evaluating External Model (Qwen2.5-1.5B-Instruct, CPU) ---")
        client.post("/api/models/switch", json={"model": "external"})

        # External Offline
        client.post("/api/web-search", json={"enabled": False})
        conv_q_off = client.post("/api/conversations", json={"title": "Eval Qwen Offline", "model": "external"}).json()
        conv_q_id = conv_q_off["conversation_id"]

        for item in CATEGORY_PROMPTS:
            print(f"  [Qwen Offline] {item['name']}...")
            t0 = time.time()
            if item["id"] == "multi_turn_context":
                client.post("/api/chat", json={"conversation_id": conv_q_id, "message": item["prompt"]})
                r = client.post("/api/chat", json={"conversation_id": conv_q_id, "message": item["follow_up"]})
            else:
                r = client.post("/api/chat", json={"conversation_id": conv_q_id, "message": item["prompt"]})
            latency = (time.time() - t0) * 1000
            data = r.json()
            results["external"]["offline"][item["id"]] = {
                "category": item["name"],
                "response": data.get("content", ""),
                "latency_ms": round(latency, 1),
                "model_used": data.get("model_used")
            }

        # External Online (Categories 10 & 11)
        print("  [Qwen Online] Web Grounded Queries...")
        client.post("/api/web-search", json={"enabled": True})
        conv_q_on = client.post("/api/conversations", json={"title": "Eval Qwen Online", "model": "external"}).json()
        for cat_id in ["current_events", "web_research"]:
            item = next(p for p in CATEGORY_PROMPTS if p["id"] == cat_id)
            print(f"  [Qwen Online] {item['name']}...")
            t0 = time.time()
            r = client.post("/api/chat", json={"conversation_id": conv_q_on["conversation_id"], "message": item["prompt"]})
            latency = (time.time() - t0) * 1000
            data = r.json()
            results["external"]["online"][item["id"]] = {
                "category": item["name"],
                "response": data.get("content", ""),
                "sources_count": len(data.get("sources", [])),
                "latency_ms": round(latency, 1),
                "model_used": data.get("model_used")
            }

    # Save results
    output_file = PROJECT_DIR / "data" / "verified_model_capabilities.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] Capability evaluation written to: {output_file}")


if __name__ == "__main__":
    run_evaluation()
