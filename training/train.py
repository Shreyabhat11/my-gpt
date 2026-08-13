"""
Main GPT Training Script.

Pipeline:

    Corpus
       ↓
    Trained BPE Tokenizer
       ↓
    Token IDs
       ↓
    Train / Validation Split
       ↓
    DataLoaders
       ↓
    GPT
       ↓
    Trainer
       ↓
    Checkpoints

Author: Shreya Bhat
"""

from __future__ import annotations

from pathlib import Path

import torch

from configs.model_config import GPTConfig
from configs.training_config import TrainingConfig

from data.dataloader import create_dataloaders

from tokenizer.bpe import BPETokenizer

from model.gpt import GPT

from training.trainer import Trainer


# ==============================================================
# Paths
# ==============================================================

CORPUS_PATH = Path(
    "data/corpus.txt"
)

TOKENIZER_PATH = Path(
    "artifacts/tokenizer.json"
)


# ==============================================================
# Main
# ==============================================================

def main() -> None:

    # ==========================================================
    # Reproducibility
    # ==========================================================

    torch.manual_seed(42)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(42)

    # ==========================================================
    # Device
    # ==========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)

    print(
        "Device:",
        device,
    )

    print("=" * 60)

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
    # Training Configuration
    # ==========================================================

    training_config = TrainingConfig(

        # Batch
        batch_size=4,

        # Training
        max_steps=1000,

        # Optimizer
        learning_rate=3e-4,
        min_learning_rate=3e-5,

        weight_decay=0.1,

        beta1=0.9,
        beta2=0.95,

        # Gradient clipping
        grad_clip=1.0,

        # Scheduler
        warmup_steps=100,

        # Validation
        validation_interval=100,
        validation_steps=20,

        # Checkpoint
        checkpoint_interval=500,

        # DataLoader
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    # ==========================================================
    # Validate Corpus
    # ==========================================================

    if not CORPUS_PATH.exists():

        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}"
        )

    # ==========================================================
    # Validate Tokenizer
    # ==========================================================

    if not TOKENIZER_PATH.exists():

        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}\n"
            "Run:\n"
            "python -m data.tokenize_corpus"
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
    # Load Trained Tokenizer
    # ==========================================================

    tokenizer = BPETokenizer()

    tokenizer.load(
        TOKENIZER_PATH
    )

    print(
        "Tokenizer vocabulary:",
        tokenizer.vocab_size,
    )

    print(
        "Model vocabulary:",
        model_config.vocab_size,
    )

    # ==========================================================
    # Vocabulary Compatibility Check
    # ==========================================================

    if (
        tokenizer.vocab_size
        != model_config.vocab_size
    ):

        raise ValueError(
            "Tokenizer vocabulary size and "
            "model vocabulary size do not match.\n"
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
    # Build DataLoaders
    # ==========================================================

    loaders = create_dataloaders(
        token_ids=token_ids,

        model_config=model_config,

        training_config=training_config,

        validation_split=0.1,
    )

    print(
        "Train batches:",
        len(loaders.train_loader),
    )

    print(
        "Validation batches:",
        len(loaders.val_loader),
    )

    # ==========================================================
    # Inspect One Batch
    # ==========================================================

    input_ids, target_ids = next(
        iter(
            loaders.train_loader
        )
    )

    print()
    print(
        "Input shape:",
        input_ids.shape,
    )

    print(
        "Target shape:",
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
        "Input example:"
    )

    print(
        repr(decoded_input)
    )

    print()
    print(
        "Target example:"
    )

    print(
        repr(decoded_target)
    )

    # ==========================================================
    # Create GPT
    # ==========================================================

    model = GPT(
        model_config
    )

    # ==========================================================
    # Create Trainer
    # ==========================================================

    trainer = Trainer(
        model=model,

        train_loader=loaders.train_loader,

        val_loader=loaders.val_loader,

        config=training_config,

        device=device,
    )

    # ==========================================================
    # Training Information
    # ==========================================================

    print()
    print("=" * 60)

    print(
        "Starting GPT training..."
    )

    print(
        "Maximum steps:",
        training_config.max_steps,
    )

    print(
        "Batch size:",
        training_config.batch_size,
    )

    print(
        "Context length:",
        model_config.context_length,
    )

    print(
        "Learning rate:",
        training_config.learning_rate,
    )

    print("=" * 60)

    # ==========================================================
    # Train
    # ==========================================================

    history = trainer.train()

    # ==========================================================
    # Final Results
    # ==========================================================

    print()
    print("=" * 60)

    print(
        "Training completed."
    )

    print(
        "Total steps:",
        trainer.global_step,
    )

    print(
        "Final train loss:",
        history[-1].train_loss,
    )

    if history[-1].val_loss is not None:

        print(
            "Final validation loss:",
            history[-1].val_loss,
        )

    print("=" * 60)


# ==============================================================
# Entry Point
# ==============================================================

if __name__ == "__main__":

    main()