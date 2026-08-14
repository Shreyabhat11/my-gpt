"""
Perplexity Evaluation Utilities.

Author: Shreya Bhat
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> float:
    """
    Calculate average cross-entropy loss.

    Args:
        model:
            GPT model.

        data_loader:
            Validation DataLoader.

        device:
            CPU or CUDA device.

    Returns:
        Average validation loss.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in data_loader:

        input_ids = input_ids.to(
            device,
            non_blocking=True,
        )

        target_ids = target_ids.to(
            device,
            non_blocking=True,
        )

        # ------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------

        output = model(
            input_ids,
            targets=target_ids,
        )

        # GPT returns:
        #
        #     logits, loss
        #
        # We only need loss here.

        if isinstance(output, tuple):

            _, loss = output

        else:

            logits = output

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                target_ids.reshape(-1),
            )

        total_loss += loss.item()

        total_batches += 1

    if total_batches == 0:

        raise RuntimeError(
            "Validation DataLoader is empty."
        )

    return total_loss / total_batches


def calculate_perplexity(
    loss: float,
) -> float:
    """
    Calculate perplexity from cross-entropy loss.

    Perplexity = exp(loss)
    """

    if loss < 0:

        raise ValueError(
            "Loss cannot be negative."
        )

    # Prevent overflow for pathological values.
    if loss > 20:

        return float("inf")

    return math.exp(loss)