"""
Production-Oriented Autoregressive Text Generation.

Supports:

- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p / nucleus sampling
- Repetition penalty

Author: Shreya Bhat
"""

from __future__ import annotations

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty to tokens already generated.

    Args:
        logits:
            Shape (B, vocab_size)

        input_ids:
            Shape (B, T)

        penalty:
            Must be >= 1.0.

    Returns:
        Modified logits.
    """

    if penalty < 1.0:

        raise ValueError(
            "repetition_penalty must be >= 1.0"
        )

    if penalty == 1.0:
        return logits

    # Tokens already present in each sequence.
    for batch_idx in range(
        input_ids.size(0)
    ):

        previous_tokens = input_ids[
            batch_idx
        ].unique()

        token_logits = logits[
            batch_idx,
            previous_tokens,
        ]

        # Standard repetition penalty:
        #
        # positive logit → divide
        # negative logit → multiply

        token_logits = torch.where(
            token_logits > 0,
            token_logits / penalty,
            token_logits * penalty,
        )

        logits[
            batch_idx,
            previous_tokens,
        ] = token_logits

    return logits


def apply_top_k(
    logits: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """
    Keep only the top-k logits.
    """

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
        dim=-1,
    )

    threshold = values[
        :, -1
    ].unsqueeze(-1)

    logits = torch.where(
        logits < threshold,
        torch.full_like(
            logits,
            float("-inf"),
        ),
        logits,
    )

    return logits


def apply_top_p(
    logits: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """
    Apply nucleus / top-p sampling.

    Keeps the smallest set of tokens whose
    cumulative probability exceeds top_p.
    """

    if not 0.0 < top_p <= 1.0:

        raise ValueError(
            "top_p must be in the range (0, 1]."
        )

    if top_p == 1.0:
        return logits

    # Sort logits from highest probability to lowest.
    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
        dim=-1,
    )

    # Convert to probabilities.
    sorted_probs = torch.softmax(
        sorted_logits,
        dim=-1,
    )

    # Cumulative probability.
    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=-1,
    )

    # Remove tokens after probability exceeds top_p.
    sorted_mask = (
        cumulative_probs > top_p
    )

    # Keep the first token above the threshold.
    sorted_mask[:, 1:] = (
        sorted_mask[:, :-1].clone()
    )

    sorted_mask[:, 0] = False

    # Apply mask.
    sorted_logits = sorted_logits.masked_fill(
        sorted_mask,
        float("-inf"),
    )

    # Restore original vocabulary order.
    logits = torch.full_like(
        logits,
        float("-inf"),
    )

    logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=sorted_logits,
    )

    return logits


@torch.no_grad()
@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    context_length: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = 40,
    top_p: float | None = 0.9,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Autoregressive text generation.

    Supports:
        - temperature sampling
        - top-k sampling
        - top-p / nucleus sampling
        - repetition penalty
        - context window limiting
    """

    model.eval()

    # ==========================================================
    # Validate parameters
    # ==========================================================

    if max_new_tokens < 0:
        raise ValueError(
            "max_new_tokens must be >= 0."
        )

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than 0."
        )

    if top_k is not None and top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0."
        )

    if top_p is not None and not (
        0 < top_p <= 1
    ):
        raise ValueError(
            "top_p must be between 0 and 1."
        )

    if repetition_penalty <= 0:
        raise ValueError(
            "repetition_penalty must be greater than 0."
        )

    # ==========================================================
    # Context length
    # ==========================================================

    if context_length is None:

        context_length = model.config.context_length

    # ==========================================================
    # Generation loop
    # ==========================================================

    for _ in range(max_new_tokens):

        # ------------------------------------------------------
        # Keep only the most recent context
        # ------------------------------------------------------

        input_context = input_ids[
            :, -context_length:
        ]

        # ------------------------------------------------------
        # Forward pass
        # ------------------------------------------------------

        output = model(
            input_context
        )

        # ------------------------------------------------------
        # Handle GPT output
        #
        # Model may return:
        #
        #     logits
        #
        # or:
        #
        #     logits, loss
        # ------------------------------------------------------

        if isinstance(output, tuple):

            logits = output[0]

        else:

            logits = output

        # ------------------------------------------------------
        # Select logits for the final token
        #
        # (B, T, V)
        #       ↓
        # (B, V)
        # ------------------------------------------------------

        logits = logits[:, -1, :]

        # ======================================================
        # Repetition Penalty
        # ======================================================

        if repetition_penalty != 1.0:

            for token_id in set(
                input_ids[0].tolist()
            ):

                if logits[0, token_id] < 0:

                    logits[0, token_id] *= (
                        repetition_penalty
                    )

                else:

                    logits[0, token_id] /= (
                        repetition_penalty
                    )

        # ======================================================
        # Temperature
        # ======================================================

        logits = logits / temperature

        # ======================================================
        # Top-K Filtering
        # ======================================================

        if top_k is not None:

            k = min(
                top_k,
                logits.size(-1),
            )

            values, _ = torch.topk(
                logits,
                k,
                dim=-1,
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

        # ======================================================
        # Top-P / Nucleus Sampling
        # ======================================================

        if top_p is not None:

            sorted_logits, sorted_indices = torch.sort(
                logits,
                descending=True,
                dim=-1,
            )

            sorted_probabilities = torch.softmax(
                sorted_logits,
                dim=-1,
            )

            cumulative_probabilities = torch.cumsum(
                sorted_probabilities,
                dim=-1,
            )

            remove_mask = (
                cumulative_probabilities > top_p
            )

            # Always keep the most probable token
            remove_mask[:, 0] = False

            sorted_logits = sorted_logits.masked_fill(
                remove_mask,
                float("-inf"),
            )

            logits = torch.full_like(
                logits,
                float("-inf"),
            )

            logits.scatter_(
                dim=-1,
                index=sorted_indices,
                src=sorted_logits,
            )

        # ======================================================
        # Convert logits to probabilities
        # ======================================================

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        # ======================================================
        # Sample next token
        # ======================================================

        next_token = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        # ======================================================
        # Append token
        # ======================================================

        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    return input_ids