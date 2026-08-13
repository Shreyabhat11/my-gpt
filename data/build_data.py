"""
Build GPT training and validation DataLoaders
from a trained BPE tokenizer and text corpus.

Author: Shreya Bhat
"""

from __future__ import annotations

from pathlib import Path

from configs.model_config import GPTConfig
from configs.training_config import TrainingConfig

from data.dataloader import create_dataloaders

from tokenizer.bpe import BPETokenizer


CORPUS_PATH = Path(
    "data/corpus.txt"
)

TOKENIZER_PATH = Path(
    "artifacts/tokenizer.json"
)


def main():

    # ==========================================================
    # Configuration
    # ==========================================================

    model_config = GPTConfig(
        vocab_size=256,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    )

    training_config = TrainingConfig(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    # ==========================================================
    # Load Corpus
    # ==========================================================

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        text = file.read()

    # ==========================================================
    # Load Tokenizer
    # ==========================================================

    tokenizer = BPETokenizer()

    tokenizer.load(
        TOKENIZER_PATH
    )

    print("=" * 60)

    print(
        "Tokenizer vocabulary:",
        tokenizer.vocab_size,
    )

    print(
        "Model vocabulary:",
        model_config.vocab_size,
    )

    # ==========================================================
    # Verify Vocabulary Compatibility
    # ==========================================================

    if tokenizer.vocab_size != model_config.vocab_size:

        raise ValueError(
            "Tokenizer vocabulary size and model vocabulary "
            "size do not match.\n"
            f"Tokenizer: {tokenizer.vocab_size}\n"
            f"Model:     {model_config.vocab_size}"
        )

    # ==========================================================
    # Encode Corpus
    # ==========================================================

    token_ids = tokenizer.encode(
        text
    )

    print(
        "Total tokens:",
        len(token_ids),
    )

    # ==========================================================
    # Create DataLoaders
    # ==========================================================

    loaders = create_dataloaders(
        token_ids=token_ids,
        model_config=model_config,
        training_config=training_config,
        validation_split=0.1,
    )

    # ==========================================================
    # Inspect Training Batch
    # ==========================================================

    input_ids, target_ids = next(
        iter(
            loaders.train_loader
        )
    )

    print()

    print(
        "Train input shape:",
        input_ids.shape,
    )

    print(
        "Train target shape:",
        target_ids.shape,
    )

    # ==========================================================
    # Decode Example
    # ==========================================================

    decoded_input = tokenizer.decode(
        input_ids[0].tolist()
    )

    decoded_target = tokenizer.decode(
        target_ids[0].tolist()
    )

    print()

    print(
        "Input text:"
    )

    print(
        repr(decoded_input)
    )

    print()

    print(
        "Target text:"
    )

    print(
        repr(decoded_target)
    )

    print("=" * 60)


if __name__ == "__main__":

    main()