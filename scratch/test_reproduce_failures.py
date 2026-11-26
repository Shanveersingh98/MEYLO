import sys
from pathlib import Path
import json

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading chatbot on {device}...")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

failures_to_test = [
    ("Failure 1", "Who is Shanveer Singh Khosa on LinkedIn?", None),
    ("Failure 2", "Explain how PyTorch's torch.nn.Linear works, with a simple Python example.", None),
    ("Failure 3", "Find the latest official Python release information and provide the source links.", None),
    ("Failure 4", "Explain the difference between supervised and reinforcement learning.", None),
    ("Failure 5 (Follow-up)", "Explain the second source in simpler words.", "followup_after_4")
]

results = []
history = []

for name, prompt, mode in failures_to_test:
    print(f"\n==========================================")
    print(f"Testing: {name}")
    print(f"Prompt: {prompt}")
    
    if mode != "followup_after_4":
        history = [{"role": "user", "text": prompt}]
    else:
        history.append({"role": "user", "text": prompt})
        
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=True
    )
    history.append({"role": "assistant", "text": reply})
    
    # Check last_evidence state
    evidence_items = []
    if retrieval_engine.last_evidence and retrieval_engine.last_evidence.items:
        evidence_items = [{"title": it.title, "url": it.url} for it in retrieval_engine.last_evidence.items]
    
    res = {
        "test": name,
        "prompt": prompt,
        "route": route,
        "response": reply,
        "evidence_count": len(evidence_items),
        "evidence_items": evidence_items
    }
    results.append(res)
    print(f"Route: {route}")
    print(f"Response Preview:\n{reply[:300]}")
    print(f"Evidence items count: {len(evidence_items)}")
    for i, it in enumerate(evidence_items, 1):
        print(f"  [{i}] {it['title']}: {it['url']}")

scratch_out = PROJECT_DIR / "scratch" / "reproduce_failures.json"
scratch_out.parent.mkdir(parents=True, exist_ok=True)
with open(scratch_out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nResults written to:", scratch_out)
