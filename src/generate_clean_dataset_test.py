import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import torch
from tokenizers import Tokenizer

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Generation Device:", device)

tokenizer_path = PROJECT_DIR / "tokenizer" / "tokenizer.json"
tokenizer = Tokenizer.from_file(str(tokenizer_path))

checkpoint_path = PROJECT_DIR / "checkpoints" / "clean_dataset_test_best.pt"
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint["config"]

print(f"Model step:            {checkpoint['step']}")
print(f"Validation loss:       {checkpoint['validation_loss']:.4f}")
print(f"Best validation loss:  {checkpoint['best_validation_loss']:.4f}")

model = FromScratchLM(
    vocab_size=config["vocab_size"],
    context_length=config["context_length"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    d_ff=config["d_ff"],
    dropout=config["dropout"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

def generate_response(prompt, max_new_tokens=100, temperature=0.8, top_k=50, repetition_penalty=1.15):
    # Formatted prompt
    formatted = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(formatted)
    prompt_length = len(encoded.ids)

    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("[EOS]")

    for _ in range(max_new_tokens):
        input_context = input_ids[:, -config["context_length"]:]

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(input_context)

        logits = logits[:, -1, :]

        # Repetition penalty applied strictly to tokens generated AFTER the prompt
        generated_tokens = set(input_ids[0, prompt_length:].tolist())
        for token_id in generated_tokens:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        # Temperature scaling
        logits = logits / temperature

        # Top-k filtering
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        filtered_logits = torch.full_like(logits, float("-inf"))
        filtered_logits.scatter_(1, indices, values)

        # Softmax and sampling
        probabilities = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    generated_ids = input_ids[0, prompt_length:].tolist()
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response

test_prompts = [
    "Hello",
    "What is artificial intelligence?",
    "What is machine learning?",
    "Explain Python.",
    "What is 2 + 2?"
]

print("\n" + "=" * 70)
print("1,000-STEP CLEAN DATASET GENERATION TESTS")
print("Settings: max_new_tokens=100, temperature=0.8, top_k=50, rep_penalty=1.15")
print("=" * 70)

for idx, p in enumerate(test_prompts, start=1):
    print(f"\n--- PROMPT {idx}: '{p}' ---")
    resp = generate_response(p)
    print("ASSISTANT:")
    print(resp if resp else "[EMPTY RESPONSE / EOS IMMEDIATELY]")
