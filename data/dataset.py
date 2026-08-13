"""
GPT Dataset

Creates training samples for next-token prediction.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class GPTDataset(Dataset):
    """
    Dataset for autoregressive language modeling.
    """

    def __init__(
        self,
        token_ids: list[int],
        context_length: int,
    ):
        """
        Parameters
        ----------
        token_ids : list[int]
            Entire tokenized corpus.

        context_length : int
            Maximum sequence length.
        """

        if len(token_ids) <= context_length:
            raise ValueError(
                "Corpus must be longer than context_length."
            )

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        """
        Number of training examples.
        """

        return len(self.token_ids) - self.context_length

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one (input, target) pair.
        """

        x = self.token_ids[
            idx : idx + self.context_length
        ]

        y = self.token_ids[
            idx + 1 : idx + self.context_length + 1
        ]

        return (
            torch.tensor(
                x,
                dtype=torch.long,
            ),
            torch.tensor(
                y,
                dtype=torch.long,
            ),
        )


if __name__ == "__main__":

    tokens = list(range(10))

    dataset = GPTDataset(
        token_ids=tokens,
        context_length=4,
    )

    print("Dataset size:", len(dataset))

    for i in range(len(dataset)):
        x, y = dataset[i]

        print("-" * 40)
        print("Input :", x.tolist())
        print("Target:", y.tolist())