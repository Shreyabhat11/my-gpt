"""
DataLoader utilities.

Author: Shreya Bhat
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from data.dataset import GPTDataset


def create_dataloader(
    dataset: GPTDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.

    Parameters
    ----------
    dataset : GPTDataset
        Training dataset.

    batch_size : int
        Number of samples per batch.

    shuffle : bool, default=True
        Shuffle dataset every epoch.

    num_workers : int, default=0
        Number of worker processes.

    pin_memory : bool, default=False
        Pin memory for faster GPU transfer.

    Returns
    -------
    DataLoader
    """

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

if __name__ == "__main__":

    tokens = list(range(100))

    dataset = GPTDataset(
        token_ids=tokens,
        context_length=8,
        stride=4,
    )

    loader = create_dataloader(
        dataset,
        batch_size=4,
    )

    for x, y in loader:

        print(x.shape)

        print(y.shape)

        break