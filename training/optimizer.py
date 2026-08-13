"""
Optimizer Factory for GPT.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch

from configs.model_config import GPTConfig
from model.gpt import GPT


def create_optimizer(
    model: GPT,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer for GPT.

    Parameters
    ----------
    model:
        GPT model.

    learning_rate:
        Initial learning rate.

    weight_decay:
        Weight decay coefficient.

    betas:
        Adam beta parameters.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

    return optimizer


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    config = GPTConfig(
        vocab_size=256,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    )

    model = GPT(config)

    optimizer = create_optimizer(
        model=model,
        learning_rate=3e-4,
        weight_decay=0.1,
    )

    print("=" * 60)

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        ),
    )

    print(
        "Optimizer:",
        type(optimizer).__name__,
    )

    print(
        "Learning rate:",
        optimizer.param_groups[0]["lr"],
    )

    print(
        "Weight decay:",
        optimizer.param_groups[0]["weight_decay"],
    )

    print("=" * 60)