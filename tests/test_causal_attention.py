"""
Test causal masking in GPT attention.

Author: Shreya Bhat
"""

import torch

from configs.model_config import GPTConfig
from model.attention import MultiHeadSelfAttention


def test_causal_attention():

    torch.manual_seed(42)

    config = GPTConfig(
        vocab_size=100,
        context_length=8,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )

    attention = MultiHeadSelfAttention(config)

    attention.eval()

    x = torch.randn(
        1,
        8,
        32,
    )

    _, weights = attention(
        x,
        return_attention=True,
    )

    # weights:
    #
    # (B, H, T, T)
    #

    upper_triangle = torch.triu(
        weights,
        diagonal=1,
    )

    assert torch.allclose(
        upper_triangle,
        torch.zeros_like(upper_triangle),
    )

    print(
        "Causal attention test passed!"
    )


if __name__ == "__main__":

    test_causal_attention()