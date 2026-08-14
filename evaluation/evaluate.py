"""
GPT Evaluation Script.

Evaluates a trained GPT checkpoint using
validation loss and perplexity.

Author: Shreya Bhat
"""

from __future__ import annotations

from pathlib import Path

import torch

from configs.model_config import GPTConfig
from configs.training_config import TrainingConfig

from tokenizer.bpe import BPETokenizer

from data.dataloader import create_dataloaders

from model.gpt import GPT

from evaluation.perplexity import (
    evaluate_loss,
    calculate_perplexity,
)


# ==============================================================
# Paths
# ==============================================================

CORPUS_PATH = Path(
    "data/corpus.txt"
)

TOKENIZER_PATH = Path(
    "artifacts/tokenizer.json"
)

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/latest.pt"
)


# ==============================================================
# Main
# ==============================================================

def main() -> None:

    # ==========================================================
    # Device
    # ==========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("GPT Evaluation")
    print("=" * 60)

    print(
        "Device:",
        device,
    )

    # ==========================================================
    # Validate Files
    # ==========================================================

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}"
        )

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # ==========================================================
    # Model Configuration
    # ==========================================================

    model_config = GPTConfig(
        vocab_size=256,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        bias=False,
    )

    # ==========================================================
    # Evaluation Data Configuration
    # ==========================================================

    training_config = TrainingConfig(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
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

    print(
        "Corpus characters:",
        len(text),
    )

    # ==========================================================
    # Load Tokenizer
    # ==========================================================

    tokenizer = BPETokenizer()

    tokenizer.load(
        TOKENIZER_PATH
    )

    print(
        "Tokenizer vocabulary:",
        tokenizer.vocab_size,
    )

    # ==========================================================
    # Vocabulary Check
    # ==========================================================

    if (
        tokenizer.vocab_size
        != model_config.vocab_size
    ):

        raise ValueError(
            "Tokenizer vocabulary and model vocabulary "
            "do not match.\n"
            f"Tokenizer: {tokenizer.vocab_size}\n"
            f"Model: {model_config.vocab_size}"
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
    # Build DataLoaders
    # ==========================================================

    loaders = create_dataloaders(
        token_ids=token_ids,
        model_config=model_config,
        training_config=training_config,
        validation_split=0.1,
    )

    # ==========================================================
    # Create Model
    # ==========================================================

    model = GPT(
        model_config
    )

    # ==========================================================
    # Load Checkpoint
    # ==========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    # ----------------------------------------------------------
    # Support checkpoint containing model_state_dict
    # ----------------------------------------------------------

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        # Support raw state_dict checkpoints
        model.load_state_dict(
            checkpoint
        )

    model.to(device)

    model.eval()

    # ==========================================================
    # Evaluate
    # ==========================================================

    validation_loss = evaluate_loss(
        model=model,
        data_loader=loaders.val_loader,
        device=device,
    )

    perplexity = calculate_perplexity(
        validation_loss
    )

    # ==========================================================
    # Report
    # ==========================================================

    print()
    print("=" * 60)

    print(
        "Evaluation Results"
    )

    print("=" * 60)

    print(
        f"Validation Loss : {validation_loss:.4f}"
    )

    print(
        f"Perplexity      : {perplexity:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()