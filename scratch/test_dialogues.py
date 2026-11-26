import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
import torch
from chat import load_chatbot, generate_chat_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

def run_dialog(turns):
    history = []
    print("\n" + "="*60)
    print("STARTING DIALOGUE SESSION")
    print("="*60)
    for q in turns:
        print(f"\nUSER: {q}")
        history.append({"role": "user", "text": q})
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
        print(f"BOT ({route}):\n{reply}")

# Dialogue 1: Educational + Follow-ups
run_dialog([
    "What is supervised learning?",
    "Explain that in simpler words.",
    "Give me a shorter summary.",
    "What did the second source say?"
])

# Dialogue 2: Public Profile Search
run_dialog([
    "tell me who is the shanveer singh khosa on the linkedin",
    "Who is Geoffrey Hinton?",
    "Who is the CEO of Microsoft?"
])

# Dialogue 3: Health & Urgent Symptoms
run_dialog([
    "i have pain in my heart",
    "What are the common causes of chest pain?"
])

# Dialogue 4: Calculator & Ambiguous
run_dialog([
    "45 * 12",
    "stuff"
])
