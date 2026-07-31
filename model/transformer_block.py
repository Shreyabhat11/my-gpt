"""
Transformer Block

Implements a Pre-LayerNorm GPT-2 style Transformer block.

Architecture

x
│
├───────────────┐
│               │
▼               │
LayerNorm       │
│               │
▼               │
MultiHeadAttention
│               │
▼               │
Residual Add ◄──┘
│
├───────────────┐
│               │
▼               │
LayerNorm       │
│               │
▼               │
FeedForward
│               │
▼               │
Residual Add ◄──┘
│
▼
Output
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import MultiHeadSelfAttention
from model.feedforward import FeedForward

from configs.model_config import GPTConfig

class TransformerBlock(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )

        self.attn = MultiHeadSelfAttention(config)

        self.ln2 = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )

        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------
        # Attention Block
        # --------------------------

        x = x + self.attn(
            self.ln1(x)
        )

        # --------------------------
        # Feed Forward Block
        # --------------------------

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


if __name__ == "__main__":

    torch.manual_seed(42)

    model = TransformerBlock(
        embed_dim=128,
        num_heads=8,
        context_length=32,
    )

    x = torch.randn(
        4,
        32,
        128,
    )

    out = model(x)

    print("=" * 60)

    print("Input Shape :", x.shape)

    print("Output Shape:", out.shape)

    print("=" * 60)