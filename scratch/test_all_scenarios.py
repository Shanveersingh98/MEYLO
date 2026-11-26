import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response
from source_selection import SourceRecord, EvidenceBundle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, retrieval_engine, _ = load_chatbot(device)

def test_scenario(title, messages, web_search):
    print(f"\n=======================================================")
    print(f"--- {title} (web_search={web_search}) ---")
    print(f"=======================================================")
    if hasattr(retrieval_engine, "conversation_memory"):
        retrieval_engine.conversation_memory.clear()
    if hasattr(retrieval_engine, "session_last_sources"):
        retrieval_engine.session_last_sources = []
    
    hist = []
    for msg in messages:
        hist.append({"role": "user", "text": msg})
        reply, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=hist,
            device=device,
            retrieval_engine=retrieval_engine,
            web_search_enabled=web_search
        )
        hist.append({"role": "assistant", "text": reply})
        print(f"USER: {msg}")
        print(f"ROUTE: {route}")
        print(f"BOT: {reply[:250]}...\n" if len(reply) > 250 else f"BOT: {reply}\n")

# Scenario 4: Emotional support with web search OFF
test_scenario("Scenario 4: Emotional Support (Web OFF)", [
    "I'm feeling really stressed about my work and life balance.",
    "I had an argument with someone close to me today."
], web_search=False)

# Scenario 5: Basic educational questions with web search ON
test_scenario("Scenario 5: Basic Educational Questions (Web ON)", [
    "What is deep learning?",
    "What is the difference between AI and machine learning?"
], web_search=True)

# Scenario 6: Current information requiring web search (mocked item to test pipeline)
mock_item = SourceRecord(
    title="Recent AI Research Developments in 2026",
    url="https://arxiv.org/abs/2601.12345",
    snippet="Recent developments in artificial intelligence feature reasoning models and architectural scaling.",
    full_text="Recent developments in artificial intelligence in 2026 feature reasoning models.",
    source_provider="openalex",
    rank=1
)
mock_bundle = EvidenceBundle(query="latest AI news", items=[mock_item], has_results=True)
original_search = retrieval_engine.search_and_retrieve
retrieval_engine.search_and_retrieve = lambda *args, **kwargs: mock_bundle
try:
    test_scenario("Scenario 6: Current Information (Web ON)", [
        "What is the latest AI news?",
        "Give me the sources"
    ], web_search=True)
finally:
    retrieval_engine.search_and_retrieve = original_search
