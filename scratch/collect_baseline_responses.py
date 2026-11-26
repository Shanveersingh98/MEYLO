import sys
import time
import json
import torch
from pathlib import Path

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from chat import load_chatbot, generate_chat_response
from web_retrieval import WebRetrievalEngine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

baseline_questions = [
    {"id": "q01", "prompt": "what is deep learning"},
    {"id": "q02", "prompt": "what is chatgpt"},
    {"id": "q03", "prompt": "/search what is deep learning"},
    {"id": "q04", "prompt": "/search what is ChatGPT"},
    {"id": "q05", "prompt": "What is machine learning?"},
    {"id": "q06", "prompt": "Explain the difference between AI, machine learning, and deep learning."},
    {"id": "q07", "prompt": "Give three real-world applications of deep learning."},
    {"id": "q08", "prompt": "Explain supervised learning in simple words."},
    {"id": "q09", "prompt": "What is a neural network?"},
    {"id": "q10", "prompt": "What are the latest release updates for Python 3.12?"}
]

baseline_records = []

for q in baseline_questions:
    prompt = q["prompt"]
    history = [{"role": "user", "text": prompt}]
    should_search, reason = retrieval_engine.should_search_web(prompt, web_mode_enabled=True)
    t0 = time.time()
    resp, engine = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=True
    )
    latency_ms = (time.time() - t0) * 1000
    
    baseline_records.append({
        "id": q["id"],
        "prompt": prompt,
        "routing_decision": {
            "should_search_web": should_search,
            "reason": reason
        },
        "engine_used": engine,
        "latency_ms": round(latency_ms, 2),
        "response": resp,
        "has_sources": "Sources:" in resp or "Source [" in resp,
        "verdict": "FAIL" if ("Mach is a" in resp or "Burgpt" in resp or "RUsearching" in resp or "Yes, here are the steps" in resp or not should_search) else "PASS"
    })
    print(f"[{q['id']}] {prompt} -> Engine: {engine} | Search: {should_search}")

out_path = PROJECT_DIR / "logs" / "chatbot_answer_repair" / "baseline_responses.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(baseline_records, f, indent=2)

print(f"Saved {len(baseline_records)} baseline records to {out_path}")
