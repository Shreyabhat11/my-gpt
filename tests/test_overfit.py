"""
Tiny-Batch Overfitting Test

Tests whether GPT can memorize a very small dataset.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch

from configs.model_config import GPTConfig
from model.gpt import GPT


def test_overfit():

    torch.manual_seed(42)

    # ==========================================================
    # Small GPT
    # ==========================================================

    config = GPTConfig(
        vocab_size=65,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.0,
    )

    model = GPT(config)

    model.train()

    # ==========================================================
    # Tiny Fixed Dataset
    # ==========================================================

    batch_size = 4
    sequence_length = 32

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (
            batch_size,
            sequence_length,
        ),
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (
            batch_size,
            sequence_length,
        ),
    )

    # ==========================================================
    # Optimizer
    # ==========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.0,
    )

    # ==========================================================
    # Training
    # ==========================================================

    num_steps = 1000

    initial_loss = None
    final_loss = None

    for step in range(num_steps):

        optimizer.zero_grad(
            set_to_none=True
        )

        # ------------------------------------------------------
        # Forward
        # ------------------------------------------------------

        logits, loss = model(
            input_ids,
            targets,
        )

        if initial_loss is None:
            initial_loss = loss.item()

        # ------------------------------------------------------
        # Backward
        # ------------------------------------------------------

        loss.backward()

        # ------------------------------------------------------
        # Gradient Clipping
        # ------------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        # ------------------------------------------------------
        # Update
        # ------------------------------------------------------

        optimizer.step()

        final_loss = loss.item()

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------

        if step % 100 == 0:

            print(
                f"Step {step:4d} | "
                f"Loss {loss.item():.4f}"
            )

    # ==========================================================
    # Results
    # ==========================================================

    print("=" * 60)

    print(
        f"Initial Loss: {initial_loss:.4f}"
    )

    print(
        f"Final Loss  : {final_loss:.4f}"
    )

    print("=" * 60)

    # ==========================================================
    # Sanity Check
    # ==========================================================

    assert final_loss < initial_loss, (
        "Loss did not decrease."
    )

    print(
        "Tiny-batch overfitting test passed!"
    )


if __name__ == "__main__":

    test_overfit()