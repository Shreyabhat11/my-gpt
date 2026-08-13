"""
Gradient Flow Test

Verifies that gradients propagate through the complete GPT model.

Author: Shreya Bhat
"""

import torch

from configs.model_config import GPTConfig
from model.gpt import GPT


def test_gradient_flow():

    torch.manual_seed(42)

    # ----------------------------------------------------------
    # Small model for testing
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Dummy input
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Forward
    # ----------------------------------------------------------

    logits, loss = model(
        input_ids,
        targets,
    )

    # ----------------------------------------------------------
    # Backward
    # ----------------------------------------------------------

    loss.backward()

    # ----------------------------------------------------------
    # Verify gradients
    # ----------------------------------------------------------

    parameters_with_gradients = 0
    parameters_without_gradients = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if parameter.grad is None:

            parameters_without_gradients.append(
                name
            )

        else:

            parameters_with_gradients += 1

    # ----------------------------------------------------------
    # Results
    # ----------------------------------------------------------

    print("=" * 60)

    print(
        "Loss:",
        loss.item(),
    )

    print(
        "Parameters with gradients:",
        parameters_with_gradients,
    )

    print(
        "Parameters without gradients:",
        len(parameters_without_gradients),
    )

    if parameters_without_gradients:

        print(
            "\nMissing gradients:"
        )

        for name in parameters_without_gradients:
            print(
                " -",
                name,
            )

    print("=" * 60)

    # ----------------------------------------------------------
    # Assertions
    # ----------------------------------------------------------

    assert loss.requires_grad, (
        "Loss does not require gradients."
    )

    assert len(parameters_without_gradients) == 0, (
        "Some trainable parameters did not receive gradients."
    )

    assert parameters_with_gradients > 0, (
        "No gradients were produced."
    )

    print(
        "Gradient flow test passed!"
    )


if __name__ == "__main__":

    test_gradient_flow()