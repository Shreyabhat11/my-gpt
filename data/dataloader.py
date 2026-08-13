"""
GPT DataLoader and Train/Validation Split.

Author: Shreya Bhat
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from configs.model_config import GPTConfig
from configs.training_config import TrainingConfig
from data.dataset import GPTDataset


@dataclass
class DataLoaders:
    """
    Container for training and validation DataLoaders.
    """

    train_loader: DataLoader
    val_loader: DataLoader


def split_token_ids(
    token_ids: list[int] | torch.Tensor,
    validation_split: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split token IDs into training and validation streams.

    The split happens BEFORE creating GPTDataset objects.

    Parameters
    ----------
    token_ids:
        Complete tokenized corpus.

    validation_split:
        Fraction of tokens reserved for validation.

    Returns
    -------
    train_tokens:
        Training portion of the corpus.

    val_tokens:
        Validation portion of the corpus.
    """

    # ==========================================================
    # Convert to Tensor
    # ==========================================================

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

    # ==========================================================
    # Validate
    # ==========================================================

    if token_ids.ndim != 1:

        raise ValueError(
            "token_ids must be a 1-dimensional tensor."
        )

    if not 0.0 < validation_split < 1.0:

        raise ValueError(
            "validation_split must be between 0 and 1."
        )

    # ==========================================================
    # Calculate split point
    # ==========================================================

    split_index = int(
        len(token_ids)
        * (1.0 - validation_split)
    )

    # ==========================================================
    # Prevent invalid dataset sizes
    # ==========================================================

    if split_index <= 0:

        raise ValueError(
            "Training split contains no tokens."
        )

    if split_index >= len(token_ids):

        raise ValueError(
            "Validation split contains no tokens."
        )

    # ==========================================================
    # Split sequentially
    # ==========================================================

    train_tokens = token_ids[
        :split_index
    ]

    val_tokens = token_ids[
        split_index:
    ]

    return train_tokens, val_tokens


def create_dataloaders(
    token_ids: list[int] | torch.Tensor,
    model_config: GPTConfig,
    training_config: TrainingConfig,
    validation_split: float = 0.1,
) -> DataLoaders:
    """
    Create training and validation DataLoaders.

    Parameters
    ----------
    token_ids:
        Complete tokenized corpus.

    model_config:
        GPT architecture configuration.

    training_config:
        Training configuration.

    validation_split:
        Fraction of corpus used for validation.

    Returns
    -------
    DataLoaders
        Training and validation DataLoaders.
    """

    # ==========================================================
    # Split Corpus
    # ==========================================================

    train_tokens, val_tokens = split_token_ids(
        token_ids=token_ids,
        validation_split=validation_split,
    )

    # ==========================================================
    # Create Datasets
    # ==========================================================

    train_dataset = GPTDataset(
        token_ids=train_tokens,
        config=model_config,
    )

    val_dataset = GPTDataset(
        token_ids=val_tokens,
        config=model_config,
    )

    # ==========================================================
    # Create Training DataLoader
    # ==========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=training_config.drop_last,
    )

    # ==========================================================
    # Create Validation DataLoader
    # ==========================================================

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=training_config.pin_memory,
        drop_last=False,
    )

    return DataLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
    )


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    # ----------------------------------------------------------
    # Model configuration
    # ----------------------------------------------------------

    model_config = GPTConfig(
        vocab_size=65,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
    )

    # ----------------------------------------------------------
    # Training configuration
    # ----------------------------------------------------------

    training_config = TrainingConfig(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    # ----------------------------------------------------------
    # Fake corpus
    #
    # In the real pipeline this will come from the tokenizer.
    # ----------------------------------------------------------

    token_ids = torch.randint(
        0,
        model_config.vocab_size,
        (10_000,),
        dtype=torch.long,
    )

    # ----------------------------------------------------------
    # Create DataLoaders
    # ----------------------------------------------------------

    loaders = create_dataloaders(
        token_ids=token_ids,
        model_config=model_config,
        training_config=training_config,
        validation_split=0.1,
    )

    train_loader = loaders.train_loader
    val_loader = loaders.val_loader

    # ----------------------------------------------------------
    # Inspect one training batch
    # ----------------------------------------------------------

    train_inputs, train_targets = next(
        iter(train_loader)
    )

    # ----------------------------------------------------------
    # Inspect one validation batch
    # ----------------------------------------------------------

    val_inputs, val_targets = next(
        iter(val_loader)
    )

    print("=" * 60)

    print(
        "Total Tokens       :",
        len(token_ids),
    )

    print(
        "Train Dataset Size :",
        len(train_loader.dataset),
    )

    print(
        "Val Dataset Size   :",
        len(val_loader.dataset),
    )

    print()

    print(
        "Train Input Shape  :",
        train_inputs.shape,
    )

    print(
        "Train Target Shape :",
        train_targets.shape,
    )

    print()

    print(
        "Val Input Shape    :",
        val_inputs.shape,
    )

    print(
        "Val Target Shape   :",
        val_targets.shape,
    )

    print("=" * 60)