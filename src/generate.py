
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
# 3. Load latest verified checkpoint
# --------------------------------------------------

from pathlib import Path

ckpt_path = Path("checkpoints/ultimate_improvement/best.pt")
if not ckpt_path.exists():
    ckpt_path = Path("checkpoints/experiment_2/best.pt")

print(f"Loading checkpoint from: {ckpt_path}")

checkpoint = torch.load(
    str(ckpt_path),
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
    dropout=config.get("dropout", 0.0),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


print("Model loaded successfully.")
print("Training step:", checkpoint.get("optimizer_update", checkpoint.get("step", "N/A")))
print("Validation loss:", checkpoint.get("validation_loss", "N/A"))


# --------------------------------------------------
# 5. Generation function
# --------------------------------------------------

def generate(
    prompt,
    max_new_tokens=100,
):

    # Format the prompt exactly like the training format
    text = (
    "[BOS]\n"
    "<|user|>\n"
    + prompt
    + "\n<|assistant|>\n"
)

    encoded = tokenizer.encode(text)

    prompt_length = len(encoded.ids)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device
    )

    # Special stop tokens: PAD (0), BOS (2), EOS (3), USER (4), ASSISTANT (5)
    stop_token_ids = {0, 2, 3, 4, 5}
    assistant_id = tokenizer.token_to_id("<|assistant|>")

    for _ in range(max_new_tokens):

        # Keep only the latest context window
        input_context = input_ids[
            :, -config["context_length"]:
        ]

        with torch.no_grad():
            logits = model(input_context)

        # Get predictions for the final position
        logits = logits[:, -1, :].clone()

        # Penalize tokens that already appeared in generated response
        generated_tokens = input_ids[0, prompt_length:].tolist()
        if generated_tokens:
            for token_id in set(generated_tokens):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= 1.15
                else:
                    logits[0, token_id] *= 1.15

        # 3-gram repetition blocking
        if len(generated_tokens) >= 2:
            curr_prefix = tuple(generated_tokens[-2:])
            banned = set()
            for i in range(len(generated_tokens) - 2):
                if tuple(generated_tokens[i:i+2]) == curr_prefix:
                    banned.add(generated_tokens[i+2])
            for b_tok in banned:
                logits[0, b_tok] = -float("inf")

        # Calibrated nucleus sampling
        temperature = 0.35
        top_k = 20
        top_p = 0.85
        logits = logits / max(temperature, 1e-4)

        # Top-k filtering
        if top_k > 0:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits.scatter_(1, indices, values)
            logits = filtered_logits

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float("inf"))

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)

        # Add the predicted token to the sequence
        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

        # Stop when special token is generated
        if next_token.item() in stop_token_ids:
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
    'Can you write a short introduction about the relevance of the term "monopsony" in economics? Please use examples related to potential monopsonies in the labour market and cite relevant research.',
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

