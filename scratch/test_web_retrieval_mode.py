import sys
sys.path.insert(0, 'src')
import torch
from chat import load_chatbot, generate_chat_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

web_prompts = [
    "what is deep learning in detail",
    "what is machine learning in detail",
    "what is artificial intelligence",
    "what is the difference between machine learning and deep learning"
]

for p in web_prompts:
    hist = [{"role": "user", "text": p}]
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=hist,
        device=device,
        retrieval_engine=retrieval_engine,
        web_search_enabled=True
    )
    print(f"PROMPT: {p}")
    print(f"ROUTE : {route}")
    print(f"REPLY :\n{reply[:250]}...")
    print("=" * 60)
