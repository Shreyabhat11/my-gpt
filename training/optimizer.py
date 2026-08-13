"""
Optimizer Factory

Creates optimizers for GPT training.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch
import torch.nn as nn


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer.

    Parameters
    ----------
    model : nn.Module
        GPT model.

    learning_rate : float
        Learning rate.

    weight_decay : float
        Weight decay.

    betas : tuple
        Adam beta values.

    Returns
    -------
    Optimizer
    """

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

if __name__ == "__main__":

    from configs.model_config import GPTConfig
    from model.gpt import GPT

    config = GPTConfig()

    model = GPT(config)

    optimizer = create_optimizer(model)

    print(optimizer)