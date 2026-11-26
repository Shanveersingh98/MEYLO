"""
scratch/test_punjab_fix.py
Test Punjab India CM query with Qwen model (web ON).
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
mgr = ModelManager(device=dev, default_model="external")

query = "name of current cm of punjab india"
print(f"Testing query: '{query}' with External Qwen Model + Web Search ON...")

history = [{"role": "user", "text": query}]
reply, route = generate_chat_response(
    model=None,
    tokenizer=None,
    history=history,
    device=dev,
    retrieval_engine=retrieval_engine,
    web_search_enabled=True,
    model_manager=mgr
)

print(f"\nResponse Route: {route}")
print(f"BOT (Qwen): {reply}\n")

# Verify Bhagwant Mann is in response and not Pakistan
is_bhagwant = "bhagwant mann" in reply.lower() or "bhagwant" in reply.lower() or "mann" in reply.lower()
is_pakistan = "maryam nawaz" in reply.lower() or "pakistan" in reply.lower()

print(f"Mentions Bhagwant Mann (Correct India CM): {is_bhagwant}")
print(f"Confused with Pakistan (Should be False): {is_pakistan}")

mgr.shutdown()
if is_bhagwant and not is_pakistan:
    print("\nSUCCESS: Punjab India CM resolved correctly without crash!")
    sys.exit(0)
else:
    print("\nFAILED: Output did not match expectations.")
    sys.exit(1)
