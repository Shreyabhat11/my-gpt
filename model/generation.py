"""
Autoregressive Text Generation.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.

    Args:
        model:
            GPT model.

        input_ids:
            Tensor of shape (B, T).

        max_new_tokens:
            Number of tokens to generate.

        context_length:
            Maximum context window supported by model.

        temperature:
            Controls randomness.

        top_k:
            If provided, only sample from the top-k tokens.

    Returns:
        Tensor containing original + generated tokens.
    """

    # ==========================================================
    # Validation
    # ==========================================================

    if temperature <= 0:

        raise ValueError(
            "temperature must be greater than 0."
        )

    if max_new_tokens < 0:

        raise ValueError(
            "max_new_tokens must be >= 0."
        )

    # ==========================================================
    # Evaluation Mode
    # ==========================================================

    model.eval()

    # ==========================================================
    # Autoregressive Generation
    # ==========================================================

    for _ in range(max_new_tokens):

        # ------------------------------------------------------
        # Keep only the latest context window
        # ------------------------------------------------------

        idx_cond = input_ids[
            :, -context_length:
        ]

        # ------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------

        output = model(
            idx_cond
        )

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        if logits.ndim != 3:

            raise RuntimeError(
                "GPT model must return logits with shape "
                "(batch_size, sequence_length, vocab_size). "
                f"Received shape: {logits.shape}"
            )
        # ------------------------------------------------------
        # Get logits for final token
        # ------------------------------------------------------

        logits = logits[:, -1, :]

        # ------------------------------------------------------
        # Temperature
        # ------------------------------------------------------

        logits = logits / temperature

        # ------------------------------------------------------
        # Top-k filtering
        # ------------------------------------------------------

        if top_k is not None:

            if top_k <= 0:

                raise ValueError(
                    "top_k must be greater than 0."
                )

            top_k = min(
                top_k,
                logits.size(-1),
            )

            values, _ = torch.topk(
                logits,
                top_k,
            )

            minimum_value = values[
                :, -1
            ].unsqueeze(-1)

            logits = torch.where(
                logits < minimum_value,
                torch.full_like(
                    logits,
                    float("-inf"),
                ),
                logits,
            )

        # ------------------------------------------------------
        # Convert logits → probabilities
        # ------------------------------------------------------

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        # ------------------------------------------------------
        # Sample next token
        # ------------------------------------------------------

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # ------------------------------------------------------
        # Append token
        # ------------------------------------------------------

        input_ids = torch.cat(
            (
                input_ids,
                next_token,
            ),
            dim=1,
        )

    return input_ids