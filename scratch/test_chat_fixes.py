import sys, os
sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import torch
from src.chat import load_chatbot, generate_chat_response

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading chatbot on device: {device}...")
    model, tokenizer, retrieval_engine, ckpt = load_chatbot(device)
    print("Chatbot successfully loaded!")
    
    test_queries = [
        "tell me who is the shanveer singh khosa on the linkedin",
        "i have pain in my heart",
        "What is supervised learning?",
        "what is 25 * 4",
        "hello! how are you?",
        "Find the LinkedIn profile of PersonDefinitelyDoesNotExistXYZ999"
    ]
    
    history = []
    for q in test_queries:
        print("\n" + "=" * 70)
        print(f"USER: {q}")
        history.append({"role": "user", "text": q})
        resp, route = generate_chat_response(
            model=model,
            tokenizer=tokenizer,
            history=history,
            device=device,
            retrieval_engine=retrieval_engine,
            tool_assisted=True,
            web_search_enabled=True
        )
        print(f"ROUTE: {route}")
        print(f"BOT:\n{resp}")
        history.append({"role": "assistant", "text": resp})
        
    # Test follow-up
    followup_q = "What did the first source say?"
    print("\n" + "=" * 70)
    print(f"USER: {followup_q}")
    history.append({"role": "user", "text": followup_q})
    resp, route = generate_chat_response(
        model=model,
        tokenizer=tokenizer,
        history=history,
        device=device,
        retrieval_engine=retrieval_engine,
        tool_assisted=True,
        web_search_enabled=True
    )
    print(f"ROUTE: {route}")
    print(f"BOT:\n{resp}")

if __name__ == "__main__":
    main()
