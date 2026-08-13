"""
Generation configuration for GPT.

Controls text generation / inference behavior.

Author: Shreya Bhat
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.base_config import BaseConfig


@dataclass
class GenerationConfig(BaseConfig):
    """
    Configuration for autoregressive text generation.
    """

    # ==========================================================
    # Generation Length
    # ==========================================================

    max_new_tokens: int = 100

    # ==========================================================
    # Sampling
    # ==========================================================

    do_sample: bool = True

    temperature: float = 1.0

    # ==========================================================
    # Top-K Sampling
    # ==========================================================

    top_k: int | None = 50

    # ==========================================================
    # Top-P / Nucleus Sampling
    # ==========================================================

    top_p: float | None = 0.95

    # ==========================================================
    # Special Tokens
    # ==========================================================

    eos_token_id: int | None = 2

    pad_token_id: int | None = 0

    # ==========================================================
    # Repetition
    # ==========================================================

    repetition_penalty: float = 1.0

    # ==========================================================
    # Validation
    # ==========================================================

    def validate(self) -> None:
        """
        Validate generation configuration.
        """

        if self.max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be greater than zero."
            )

        if self.temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        if self.top_k is not None:

            if self.top_k <= 0:
                raise ValueError(
                    "top_k must be greater than zero."
                )

        if self.top_p is not None:

            if not 0.0 < self.top_p <= 1.0:
                raise ValueError(
                    "top_p must be in the range (0, 1]."
                )

        if self.repetition_penalty <= 0:
            raise ValueError(
                "repetition_penalty must be greater than zero."
            )

        if self.eos_token_id is not None:

            if self.eos_token_id < 0:
                raise ValueError(
                    "eos_token_id cannot be negative."
                )

        if self.pad_token_id is not None:

            if self.pad_token_id < 0:
                raise ValueError(
                    "pad_token_id cannot be negative."
                )

        # Greedy generation doesn't need sampling parameters.
        if not self.do_sample:

            self.top_k = None
            self.top_p = None
            self.temperature = 1.0