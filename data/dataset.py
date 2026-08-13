"""
GPT Language Modeling Dataset.

Creates input/target sequences for
autoregressive next-token prediction.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from configs.model_config import GPTConfig


class GPTDataset(Dataset):
    """
    Dataset for decoder-only GPT training.

    Given a sequence of token IDs:

        [t0, t1, t2, t3, t4, ...]

    creates:

        input:
        [t0, t1, t2, t3, ...]

        target:
        [t1, t2, t3, t4, ...]
    """

    def __init__(
        self,
        token_ids: list[int] | torch.Tensor,
        config: GPTConfig,
    ) -> None:

        super().__init__()

        # ======================================================
        # Convert to Tensor
        # ======================================================

        if isinstance(token_ids, list):

            token_ids = torch.tensor(
                token_ids,
                dtype=torch.long,
            )

        elif not isinstance(
            token_ids,
            torch.Tensor,
        ):

            raise TypeError(
                "token_ids must be a list[int] "
                "or torch.Tensor."
            )

        # ======================================================
        # Validate dtype
        # ======================================================

        if token_ids.dtype != torch.long:

            token_ids = token_ids.long()

        # ======================================================
        # Validate dimensions
        # ======================================================

        if token_ids.ndim != 1:

            raise ValueError(
                "token_ids must be a 1-dimensional sequence."
            )

        # ======================================================
        # Validate length
        # ======================================================

        if len(token_ids) <= config.context_length:

            raise ValueError(
                "Token sequence must contain more tokens "
                "than context_length."
            )

        # ======================================================
        # Store
        # ======================================================

        self.token_ids = token_ids

        self.context_length = config.context_length

    # ==========================================================
    # Number of training examples
    # ==========================================================

    def __len__(self) -> int:

        return (
            len(self.token_ids)
            - self.context_length
        )

    # ==========================================================
    # Get Training Example
    # ==========================================================

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        start = index

        end = (
            start
            + self.context_length
        )

        # ------------------------------------------------------
        # Input
        # ------------------------------------------------------

        input_ids = self.token_ids[
            start:end
        ]

        # ------------------------------------------------------
        # Target
        #
        # Shifted by one token
        # ------------------------------------------------------

        target_ids = self.token_ids[
            start + 1:end + 1
        ]

        return input_ids, target_ids


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    config = GPTConfig(
        vocab_size=65,
        context_length=8,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
    )

    # Simple artificial token sequence
    tokens = list(
        range(20)
    )

    dataset = GPTDataset(
        tokens,
        config,
    )

    print("=" * 60)

    print(
        "Total Tokens :",
        len(tokens),
    )

    print(
        "Dataset Size :",
        len(dataset),
    )

    input_ids, target_ids = dataset[0]

    print(
        "Input IDs    :",
        input_ids.tolist(),
    )

    print(
        "Target IDs   :",
        target_ids.tolist(),
    )

    print(
        "Input Shape  :",
        input_ids.shape,
    )

    print(
        "Target Shape :",
        target_ids.shape,
    )

    print("=" * 60)