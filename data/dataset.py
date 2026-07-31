"""
dataset.py

Creates GPT training samples using a sliding window over token IDs.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharacterTokenizer


class GPTDataset(Dataset):
    """
    Dataset for next-token prediction.
    """

    def __init__(
        self,
        token_ids,
        context_length: int,
    ):

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):

        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx):

        x = self.token_ids[idx : idx + self.context_length]

        y = self.token_ids[
            idx + 1 : idx + self.context_length + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


if __name__ == "__main__":

    data_path = Path("data/processed/shakespeare_clean.txt")

    vocab_path = Path("data/processed/vocab.json")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharacterTokenizer()

    tokenizer.load_vocab(vocab_path)

    token_ids = tokenizer.encode(text)

    dataset = GPTDataset(
        token_ids=token_ids,
        context_length=16,
    )

    print("Dataset Size:", len(dataset))

    x, y = dataset[0]

    print("\nInput IDs")

    print(x)

    print("\nTarget IDs")

    print(y)

    print("\nDecoded Input")

    print(tokenizer.decode(x.tolist()))

    print("\nDecoded Target")

    print(tokenizer.decode(y.tolist()))

    batch_size = 4

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    x_batch, y_batch = next(iter(loader))

    print("\nBatch Shape")

    print(x_batch.shape)

    print(y_batch.shape)