"""
Feed Forward Network (MLP)

Implements the position-wise feed-forward network
used in GPT-2 style Transformer blocks.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.

    Architecture:
        Linear(embed_dim -> 4 * embed_dim)
            ↓
        GELU
            ↓
        Linear(4 * embed_dim -> embed_dim)
            ↓
        Dropout
    """

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        hidden_dim = 4 * embed_dim

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                Shape (batch_size, seq_length, embed_dim)

        Returns:
            Tensor of shape
            (batch_size, seq_length, embed_dim)
        """
        return self.net(x)


if __name__ == "__main__":

    torch.manual_seed(42)

    model = FeedForward(
        embed_dim=128,
        dropout=0.1,
    )

    x = torch.randn(4, 32, 128)

    out = model(x)

    print("=" * 50)
    print("Input Shape :", x.shape)
    print("Output Shape:", out.shape)
    print("=" * 50)