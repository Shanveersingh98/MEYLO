import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        context_length,
        dropout=0.1,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Create Q, K and V in one projection.
        self.qkv = nn.Linear(
            d_model,
            3 * d_model
        )

        # Projects concatenated attention outputs.
        self.out_proj = nn.Linear(
            d_model,
            d_model
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask:
        # tokens can only attend to themselves
        # and tokens before them.
        mask = torch.tril(
            torch.ones(
                context_length,
                context_length
            )
        )

        self.register_buffer(
            "causal_mask",
            mask.view(
                1,
                1,
                context_length,
                context_length
            )
        )

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention.
        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        # Apply causal mask.
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[
                :, :, :sequence_length, :sequence_length
            ] == 0,
            float("-inf")
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.attn_dropout(
            attention_weights
        )

        attention_output = attention_weights @ v

        attention_output = attention_output.transpose(
            1, 2
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.d_model
        )

        output = self.out_proj(
            attention_output
        )

        output = self.resid_dropout(output)

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        context_length,
        dropout=0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)

        self.attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

    def forward(self, x):

        # Attention + residual connection.
        x = x + self.attention(
            self.norm1(x)
        )

        # Feed-forward + residual connection.
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class FromScratchLM(nn.Module):
    def __init__(
        self,
        vocab_size=8000,
        context_length=1024,
        d_model=384,
        num_heads=6,
        num_layers=6,
        d_ff=1536,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        # Token embedding.
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # Learned positional embedding.
        self.position_embedding = nn.Embedding(
            context_length,
            d_model
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    context_length=context_length,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

        # Weight tying:
        # output projection uses token_embedding weights.
        self.output_bias = nn.Parameter(
            torch.zeros(vocab_size)
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids):

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} "
                f"exceeds context length "
                f"{self.context_length}."
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        token_embeddings = self.token_embedding(
            input_ids
        )

        position_embeddings = self.position_embedding(
            positions
        )

        x = token_embeddings + position_embeddings

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        # Weight-tied language-model head.
        logits = F.linear(
            x,
            self.token_embedding.weight,
            self.output_bias
        )

        return logits