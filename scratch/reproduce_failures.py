import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "src/ultimate_improvement"))

import torch
from chat import load_chatbot, generate_chat_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, retrieval_engine, _ = load_chatbot(device)

def run_dialogue(name, messages, web_search=False):
    print(f"\n=======================================================")
    print(f"--- {name} (web_search={web_search}) ---")
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
        print(f"BOT: {reply}\n")

# Conversation 1: Interpersonal / Emotional Support
run_dialogue("Conversation 1: Interpersonal Support", [
    "Hey! I had a pretty tiring day today.",
    "I had a disagreement with my best friend.",
    "I don't know whether I should message them first.",
    "What would you do in this situation?"
], web_search=False)

# Conversation 2: Educational AI Roadmap & Follow-up
run_dialogue("Conversation 2: AI Roadmap & Follow-up", [
    "I want to learn artificial intelligence.",
    "Why did you put Python before machine learning?",
    "Give me an example.",
    "What should I learn next?"
], web_search=False)

# Conversation 3: Python Calculator
run_dialogue("Conversation 3: Python Calculator", [
    "Write a Python calculator",
    "Explain the code",
    "Add multiplication and division"
], web_search=False)
