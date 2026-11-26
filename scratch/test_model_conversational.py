import torch
from pathlib import Path
from tokenizers import Tokenizer
import sys

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))
from model import FromScratchLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file(str(PROJECT_DIR / "tokenizer" / "tokenizer.json"))
checkpoint = torch.load(PROJECT_DIR / "checkpoints" / "ultimate_improvement" / "best.pt", map_location=device, weights_only=False)

cfg = checkpoint["config"]
model = FromScratchLM(
    vocab_size=cfg["vocab_size"],
    context_length=cfg["context_length"],
    d_model=cfg["d_model"],
    num_heads=cfg["num_heads"],
    num_layers=cfg["num_layers"],
    d_ff=cfg["d_ff"],
    dropout=0.0
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate(prompt, temp=0.5):
    formatted = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(formatted)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    out_tokens = []
    
    for _ in range(80):
        with torch.no_grad():
            logits = model(input_ids[:, -1024:])
        logits = logits[:, -1, :].float()
        
        # apply temperature
        if temp <= 0.05:
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temp, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            
        tid = next_tok.item()
        if tid in {0, 2, 3, 4, 5}: # stop tokens
            break
        out_tokens.append(tid)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        
    return tokenizer.decode(out_tokens)

test_prompts = [
    "Hello! How are you doing today?",
    "I had a great day today, I got a promotion at work!",
    "I'm feeling really lonely lately.",
    "Tell me a fun fact about the world."
]

for p in test_prompts:
    print(f"\nPROMPT: {p}")
    for temp in [0.2, 0.5]:
        resp = generate(p, temp=temp)
        print(f"  [temp={temp}]: {repr(resp)}")
