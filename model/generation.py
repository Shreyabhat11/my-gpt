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
def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Autoregressively generate tokens.

    Args:
        model:
            GPT model.

        input_ids:
            Shape (B, T).

        max_new_tokens:
            Number of tokens to generate.

        context_length:
            Maximum model context length.

        temperature:
            Controls randomness.

            temperature < 1:
                More deterministic.

            temperature > 1:
                More random.

        top_k:
            Restrict sampling to top-k tokens.

        top_p:
            Nucleus sampling threshold.

        repetition_penalty:
            Penalize previously generated tokens.

    Returns:
        Generated token IDs.
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

    if not 0.0 < top_p <= 1.0:

        raise ValueError(
            "top_p must be in the range (0, 1]."
        )

    # ==========================================================
    # Preserve Original Training State
    # ==========================================================

    was_training = model.training

    model.eval()

    try:

        # ======================================================
        # Generation Loop
        # ======================================================

        for _ in range(max_new_tokens):

            # --------------------------------------------------
            # Context Window
            # --------------------------------------------------

            idx_cond = input_ids[
                :, -context_length:
            ]

            # --------------------------------------------------
            # Forward Pass
            # --------------------------------------------------

            output = model(
                idx_cond
            )

            # GPT may return:
            #
            # logits
            #
            # or:
            #
            # logits, loss

            if isinstance(
                output,
                tuple,
            ):

                logits = output[0]

            else:

                logits = output

            # --------------------------------------------------
            # Validate Model Output
            # --------------------------------------------------

            if logits.ndim != 3:

                raise RuntimeError(
                    "GPT must return logits with "
                    "shape (B, T, vocab_size). "
                    f"Received: {logits.shape}"
                )

            # --------------------------------------------------
            # Last Token
            # --------------------------------------------------

            logits = logits[:, -1, :]

            # --------------------------------------------------
            # Repetition Penalty
            # --------------------------------------------------

            logits = apply_repetition_penalty(
                logits=logits,
                input_ids=input_ids,
                penalty=repetition_penalty,
            )

            # --------------------------------------------------
            # Temperature
            # --------------------------------------------------

            logits = logits / temperature

            # --------------------------------------------------
            # Top-k
            # --------------------------------------------------

            if top_k is not None:

                logits = apply_top_k(
                    logits,
                    top_k,
                )

            # --------------------------------------------------
            # Top-p
            # --------------------------------------------------

            if top_p < 1.0:

                logits = apply_top_p(
                    logits,
                    top_p,
                )

            # --------------------------------------------------
            # Probabilities
            # --------------------------------------------------

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            # --------------------------------------------------
            # Numerical Safety
            # --------------------------------------------------

            if not torch.isfinite(
                probabilities
            ).all():

                raise RuntimeError(
                    "Generation produced invalid "
                    "probabilities."
                )

            # --------------------------------------------------
            # Sample Next Token
            # --------------------------------------------------

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            # --------------------------------------------------
            # Append
            # --------------------------------------------------

            input_ids = torch.cat(
                (
                    input_ids,
                    next_token,
                ),
                dim=1,
            )

    finally:

        # Restore original model state.
        if was_training:

            model.train()

    return input_ids