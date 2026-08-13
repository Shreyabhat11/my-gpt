"""
Transformer Block.

Pre-LayerNorm decoder block used by the GPT model.

Architecture:

    x
    │
    ├───────────────┐
    │               │
    ▼               │
 LayerNorm          │
    │               │
    ▼               │
 Attention          │
    │               │
    └────── + ◄─────┘
           │
           ▼
    LayerNorm
           │
           ▼
      FeedForward
           │
           └────── + ◄───── residual
           │
           ▼
        output

Author: Shreya Bhat
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.model_config import GPTConfig
from model.attention import MultiHeadSelfAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Single decoder Transformer block.

    Uses Pre-LayerNorm architecture.
    """

    def __init__(
        self,
        config: GPTConfig,
    ) -> None:

        super().__init__()

        # ======================================================
        # First LayerNorm
        # ======================================================

        self.ln1 = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )

        # ======================================================
        # Multi-Head Causal Self-Attention
        # ======================================================

        self.attn = MultiHeadSelfAttention(
            config
        )

        # ======================================================
        # Second LayerNorm
        # ======================================================

        self.ln2 = nn.LayerNorm(
            config.embed_dim,
            eps=config.layer_norm_eps,
        )

        # ======================================================
        # Feed Forward Network
        # ======================================================

        self.ffn = FeedForward(
            config
        )

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ------------------------------------------------------
        # Attention sub-layer
        #
        # Pre-LN:
        #
        # x -> LayerNorm -> Attention -> Residual Add
        # ------------------------------------------------------

        x = x + self.attn(
            self.ln1(x)
        )

        # ------------------------------------------------------
        # Feed Forward sub-layer
        #
        # Pre-LN:
        #
        # x -> LayerNorm -> FFN -> Residual Add
        # ------------------------------------------------------

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    config = GPTConfig(
        vocab_size=10_000,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    )

    model = TransformerBlock(
        config
    )

    x = torch.randn(
        4,
        32,
        128,
    )

    output = model(x)

    print("=" * 60)

    print(
        "Input Shape  :",
        x.shape,
    )

    print(
        "Output Shape :",
        output.shape,
    )

    print(
        "Parameters   :",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    print("=" * 60)