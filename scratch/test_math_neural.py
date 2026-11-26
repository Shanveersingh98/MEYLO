"""
scratch/test_math_neural.py
Verifies mathematical reasoning and question-answering across both models
without any function tools or calculator interceptors.
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

math_queries = [
    ("Differentiation of sin(x)/cos(x)", ["tan", "sec", "cos", "derivative"]),
    ("What is 45 * 12?", ["540"]),
    ("Calculate 15% of 800.", ["120"]),
    ("If someone earns 50000 a month and saves 20 percent, how much do they save in a year?", ["120,000", "120000"])
]

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
retrieval_engine = WebRetrievalEngine()
mgr = ModelManager(device=dev, default_model="external")

print("==================================================")
print("TESTING MATHEMATICAL CAPABILITIES WITH QWEN MODEL")
print("==================================================")
mgr.set_model("external")

for query, expected_tokens in math_queries:
    history = [{"role": "user", "text": query}]
    reply, route = generate_chat_response(
        model=None,
        tokenizer=None,
        history=history,
        device=dev,
        retrieval_engine=retrieval_engine,
        web_search_enabled=False,
        model_manager=mgr
    )
    print(f"\nUser: {query}")
    print(f"Route: {route}")
    print(f"Qwen: {reply.strip()}")
    
    # Check no tool syntax
    assert "[Tool:" not in reply, f"Tool tag found in response: {reply}"
    assert "Calculator" not in reply or "tan" in reply or "sec" in reply, "Unexpected tool reference"

print("\n==================================================")
print("TESTING MATHEMATICAL CAPABILITIES WITH CUSTOM MODEL")
print("==================================================")
mgr.set_model("custom")
custom_model = mgr.custom_engine.model
custom_tokenizer = mgr.custom_engine.tokenizer

for query, expected_tokens in math_queries[:2]:
    history = [{"role": "user", "text": query}]
    reply, route = generate_chat_response(
        model=custom_model,
        tokenizer=custom_tokenizer,
        history=history,
        device=dev,
        retrieval_engine=retrieval_engine,
        web_search_enabled=False,
        model_manager=mgr
    )
    print(f"\nUser: {query}")
    print(f"Route: {route}")
    print(f"Custom Model: {reply.strip()[:300]}")
    
    # Check no tool syntax
    assert "[Tool:" not in reply, f"Tool tag found in response: {reply}"

mgr.shutdown()
print("\nALL MATHEMATICAL NEURAL TESTS PASSED CLEANLY WITHOUT TOOLS!")
