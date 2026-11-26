import torch
from tokenizers import Tokenizer

from model import FromScratchLM


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)


checkpoint = torch.load(
    "checkpoints/best.pt",
    map_location=device,
    weights_only=False
)

config = checkpoint["config"]


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


def generate(
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_k=50,
):

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

    for _ in range(max_new_tokens):

        input_context = input_ids[
            :, -config["context_length"] :
        ]

        with torch.no_grad():

            logits = model(input_context)

        logits = logits[:, -1, :]

        logits = logits / temperature

        values, indices = torch.topk(
            logits,
            top_k
        )

        probabilities = torch.softmax(
            values,
            dim=-1
        )

        next_token_index = torch.multinomial(
            probabilities,
            num_samples=1
        )

        next_token = indices.gather(
            -1,
            next_token_index
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1
        )

        if next_token.item() == 3:
            break

    generated_ids = input_ids[0].tolist()

    assistant_id = tokenizer.token_to_id(
        "<|assistant|>"
    )

    assistant_start = 0

    for i, token_id in enumerate(generated_ids):

        if token_id == assistant_id:

            assistant_start = i + 1

            break

    assistant_ids = generated_ids[
        assistant_start:
    ]

    return tokenizer.decode(
        assistant_ids
    ).strip()


tests = [
    (
        "Conversation",
        "Hello! How are you today?"
    ),
    (
        "Knowledge",
        "What is artificial intelligence?"
    ),
    (
        "Explanation",
        "Explain machine learning simply."
    ),
    (
        "Programming",
        "What is Python used for?"
    ),
    (
        "Reasoning",
        "Why does a computer need memory?"
    ),
    (
        "Instruction",
        "Give me three benefits of learning Python."
    ),
]


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


for category, prompt in tests:

    print("\n" + "-" * 70)
    print("CATEGORY:", category)
    print("PROMPT:", prompt)

    response = generate(
        prompt,
        max_new_tokens=80,
        temperature=0.8,
        top_k=50,
    )

    print("RESPONSE:", response)