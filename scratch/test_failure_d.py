"""
scratch/test_failure_d.py
Verify multi-turn follow-up: "Now explain the same thing technically."
"""
import sys
import torch
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

from model_manager import ModelManager
from chat import generate_chat_response
from web_retrieval import WebRetrievalEngine

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrieval_engine = WebRetrievalEngine()
mgr = ModelManager(device=dev, default_model="custom")

print("--- Testing Custom Model Multi-turn ---")
history = []
history.append({"role": "user", "text": "What is machine learning?"})
resp1, route1 = generate_chat_response(
    model=mgr.custom_engine.model,
    tokenizer=mgr.custom_engine.tokenizer,
    history=history,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
history.append({"role": "assistant", "text": resp1})
print("Turn 1:", resp1[:80], f"[Route: {route1}]")

history.append({"role": "user", "text": "Now explain the same thing technically."})
resp2, route2 = generate_chat_response(
    model=mgr.custom_engine.model,
    tokenizer=mgr.custom_engine.tokenizer,
    history=history,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
print("Turn 2 (Technical follow-up):", resp2[:120], f"[Route: {route2}]")
history.append({"role": "assistant", "text": resp2})

print("\n--- Testing Qwen Model Multi-turn ---")
mgr.set_model("external")
history_q = []
history_q.append({"role": "user", "text": "What is machine learning in simple terms?"})
resp_q1, route_q1 = generate_chat_response(
    model=None,
    tokenizer=None,
    history=history_q,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
history_q.append({"role": "assistant", "text": resp_q1})
print("Qwen Turn 1:", resp_q1[:80], f"[Route: {route_q1}]")

history_q.append({"role": "user", "text": "Now explain the same thing technically."})
resp_q2, route_q2 = generate_chat_response(
    model=None,
    tokenizer=None,
    history=history_q,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=False,
    model_manager=mgr
)
print("Qwen Turn 2 (Technical follow-up):", resp_q2[:120], f"[Route: {route_q2}]")

mgr.shutdown()
print("\nMulti-turn technical follow-up test completed successfully!")
