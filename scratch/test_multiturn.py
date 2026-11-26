import sys
sys.path.insert(0, 'src')
import torch
from chat import load_chatbot, generate_chat_response

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)

turns = [
    "I am learning machine learning.",
    "Deep learning.",
    "Explain it."
]

history = []
for t in turns:
    history.append({"role": "user", "text": t})
    reply, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        web_search_enabled=False
    )
    print(f"USER: {t}")
    print(f"ROUTE: {route}")
    print(f"BOT:\n{reply}")
    print("-" * 60)
    history.append({"role": "assistant", "text": reply})
