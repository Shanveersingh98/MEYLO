import sys
sys.path.insert(0, 'src')
import torch
from chat import load_chatbot, generate_chat_response, parse_command

print("Testing command parser...")
assert parse_command("/web on") == ("web_on", None)
assert parse_command("/web  on") == ("web_on", None)
assert parse_command("/WEB ON") == ("web_on", None)
assert parse_command("/web off") == ("web_off", None)
assert parse_command("/web   off") == ("web_off", None)
assert parse_command("/WEB OFF") == ("web_off", None)
assert parse_command("/help") == ("help", None)
assert parse_command("/exit") == ("exit", None)
assert parse_command("/quit") == ("exit", None)
assert parse_command("Hello there") == (None, None)
act, msg = parse_command("/unknown")
assert act == "unknown_command"
print("Command parser tests passed!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading chatbot on device: {device}")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

test_prompts = [
    "hello what are you doing",
    "what's your name",
    "hey i need help",
    "what is machine learning",
    "what is deep learning in detail",
    "pls",
    "help",
    "explain",
    "pls explain",
    "what iks machine learning",
    "explain deep lernng",
    "hii how r u",
    "i ned help",
    "I am sad today.",
    "I am happy today."
]

for p in test_prompts:
    hist = [{"role": "user", "text": p}]
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=hist,
        device=device,
        retrieval_engine=retrieval_engine,
        web_search_enabled=False
    )
    print(f"PROMPT: {p}")
    print(f"ROUTE : {route}")
    print(f"REPLY : {reply[:100]}...")
    print("-" * 50)
