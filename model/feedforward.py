"""
Position-wise Feed Forward Network.

GPT-style MLP used inside each Transformer block.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.model_config import GPTConfig


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Architecture:

        Input
          │
          ▼
        Linear
          │
          ▼
        GELU
          │
          ▼
        Linear
          │
          ▼
        Dropout
          │
          ▼
        Output

    Shapes:

        Input:
            (B, T, C)

        Output:
            (B, T, C)
    """

    def __init__(
        self,
        config: GPTConfig,
    ) -> None:

        super().__init__()

        self.embed_dim = config.embed_dim

        self.hidden_dim = (
            config.embed_dim
            * config.expansion_factor
        )

        # ======================================================
        # Feed Forward Network
        # ======================================================

        self.fc1 = nn.Linear(
            config.embed_dim,
            self.hidden_dim,
            bias=config.bias,
        )

        self.activation = nn.GELU()

        self.fc2 = nn.Linear(
            self.hidden_dim,
            config.embed_dim,
            bias=config.bias,
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape: (B, T, C)

        Returns
        -------
        torch.Tensor
            Shape: (B, T, C)
        """

        # ------------------------------------------------------
        # Expand
        # ------------------------------------------------------

        x = self.fc1(x)

        # ------------------------------------------------------
        # Non-linearity
        # ------------------------------------------------------

        x = self.activation(x)

        # ------------------------------------------------------
        # Project back
        # ------------------------------------------------------

        x = self.fc2(x)

        # ------------------------------------------------------
        # Dropout
        # ------------------------------------------------------

        x = self.dropout(x)

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
        expansion_factor=4,
        dropout=0.1,
    )

    model = FeedForward(config)

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
        "Hidden Dim   :",
        model.hidden_dim,
    )

    print("=" * 60)