import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import argparse
import torch
from tokenizers import Tokenizer

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FromScratchLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "clean_50k_best.pt"
TOKENIZER_PATH = PROJECT_DIR / "tokenizer" / "tokenizer.json"

TEST_PROMPTS = [
    "Hello",
    "What is artificial intelligence?",
    "What is machine learning?",
    "Explain Python in simple words.",
    "What is 2 + 2?",
    "Explain what a function is in Python.",
    "What is the difference between AI and machine learning?",
    "Explain monopsony in simple words."
]

def load_clean_model(checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FromScratchLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        dropout=config.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint

def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    repetition_penalty=1.15,
    do_sample=False
):
    formatted = f"[BOS]\n<|user|>\n{prompt}\n<|assistant|>\n"
    encoded = tokenizer.encode(formatted)
    prompt_length = len(encoded.ids)

    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    eos_id = tokenizer.token_to_id("[EOS]")

    for _ in range(max_new_tokens):
        input_context = input_ids[:, -1024:]

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(input_context)

        logits = logits[:, -1, :]

        if not do_sample:
            # Deterministic / Greedy Argmax
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Repetition penalty strictly applied to tokens generated AFTER the prompt
            generated_tokens = set(input_ids[0, prompt_length:].tolist())
            for token_id in generated_tokens:
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

            logits = logits / temperature
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered_logits = torch.full_like(logits, float("-inf"))
            filtered_logits.scatter_(1, indices, values)

            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if next_token.item() == eos_id:
            break

    generated_ids = input_ids[0, prompt_length:].tolist()
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def run_eval_suite(model, tokenizer, do_sample=False):
    mode_str = "SAMPLING (T=0.8, Top-K=50, RepPenalty=1.15)" if do_sample else "GREEDY (Deterministic Argmax)"
    results = []

    print("\n" + "=" * 70)
    print(f"EVALUATION SUITE: {mode_str}")
    print("=" * 70)

    for i, p in enumerate(TEST_PROMPTS, start=1):
        resp = generate_text(model, tokenizer, p, do_sample=do_sample)
        print(f"\n[{i}/8] PROMPT: '{p}'")
        print(f"RESPONSE:\n{resp if resp else '[EMPTY RESPONSE / EOS IMMEDIATELY]'}")
        results.append({"prompt": p, "response": resp})

    return results

def main():
    parser = argparse.ArgumentParser(description="Generate responses from clean 50k experiment checkpoints")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of greedy decoding")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    print(f"Loading model from: {ckpt_path}")
    model, ckpt = load_clean_model(ckpt_path)
    print(f"Checkpoint Step:     {ckpt.get('step')}")
    print(f"Validation Loss:     {ckpt.get('validation_loss')}")
    print(f"Best Val Loss:       {ckpt.get('best_validation_loss')}")

    run_eval_suite(model, tokenizer, do_sample=args.sample)

if __name__ == "__main__":
    main()
