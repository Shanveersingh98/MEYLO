
import torch
from tokenizers import Tokenizer

from model import FromScratchLM


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 2. Load tokenizer
# --------------------------------------------------

tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)


# --------------------------------------------------
# 3. Load 50,000-step checkpoint
# --------------------------------------------------

checkpoint = torch.load(
    "checkpoints/masked_best.pt",
    map_location=device,
    weights_only=False
)

config = checkpoint["config"]


# --------------------------------------------------
# 4. Recreate the trained model
# --------------------------------------------------

model = FromScratchLM(
    vocab_size=config["vocab_size"],
    context_length=config["context_length"],
    d_model=config["d_model"],
    num_heads=config["num_heads"],
    num_layers=config["num_layers"],
    d_ff=config["d_ff"],
    dropout=config["dropout"],
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


print("Model loaded successfully.")
print("Training step:", checkpoint["step"])
print("Validation loss:", checkpoint["validation_loss"])


# --------------------------------------------------
# 5. Generation function
# --------------------------------------------------

def generate(
    prompt,
    max_new_tokens=100,
):

    # Format the prompt exactly like the training format
    text = (
        "[BOS]"
        "<|user|>"
        + prompt
        + "<|assistant|>"
    )

    encoded = tokenizer.encode(text)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device
    )

    # Get the assistant special-token ID
    assistant_id = tokenizer.token_to_id(
        "<|assistant|>"
    )

    # Get the EOS special-token ID
    eos_id = tokenizer.token_to_id(
        "[EOS]"
    )

    for _ in range(max_new_tokens):

        # Keep only the latest context window
        input_context = input_ids[
            :, -config["context_length"]:
        ]

        with torch.no_grad():

            logits = model(input_context)

        # Get predictions for the final position
        logits = logits[:, -1, :]

        # Sampling-based decoding
        temperature = 0.8
        top_k = 50
        repetition_penalty = 1.15

        # Penalize tokens that already appeared
        for token_id in set(input_ids[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

        # Apply temperature
        logits = logits / temperature

        # Keep only the top-k candidates
        values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
        filtered_logits = torch.full_like(logits, float('-inf'))
        filtered_logits.scatter_(1, indices, values)

        # Convert scores to probabilities and sample
        probabilities = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        # Add the predicted token to the sequence
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

        # Stop when EOS is generated
        if next_token.item() == eos_id:
            break

    generated_ids = input_ids[0].tolist()

    # Find the assistant marker
    assistant_start = 0

    for i, token_id in enumerate(generated_ids):

        if token_id == assistant_id:

            assistant_start = i + 1
            break

    # Keep only the response after <|assistant|>
    assistant_ids = generated_ids[
        assistant_start:
    ]

    # Remove special tokens from the displayed response
    response = tokenizer.decode(
        assistant_ids,
        skip_special_tokens=True
    ).strip()

    return response


# --------------------------------------------------
# 6. Test prompts
# --------------------------------------------------

prompts = [
    "Hello! How are you today?",
    "What is artificial intelligence?",
    "Can you explain machine learning simply?",
    "What is Python?",
    "Write a simple Python function to add two numbers.",
    "What is the difference between AI and machine learning?",
]


# --------------------------------------------------
# 7. Generate responses
# --------------------------------------------------

for prompt in prompts:

    print("\n" + "=" * 70)

    print("USER:", prompt)

    response = generate(
        prompt,
        max_new_tokens=100,
    )

    print("MODEL:", response)

