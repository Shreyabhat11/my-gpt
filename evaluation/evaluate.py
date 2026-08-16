"""
GPT Model Evaluation

Evaluates a trained GPT model on the validation dataset.

Metrics:
    - Validation Loss
    - Perplexity

Author: Shreya Bhat
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from configs.model_config import GPTConfig
from configs.training_config import TrainingConfig

from data.build_data import (
    CORPUS_PATH,
    TOKENIZER_PATH,
)

from data.dataloader import create_dataloaders

from tokenizer.bpe import BPETokenizer

from model.gpt import GPT

from training.checkpoint import CheckpointManager

import json
from pathlib import Path

# ==============================================================
# Evaluation
# ==============================================================


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> float:
    """
    Evaluate model on a dataset.

    Returns:
        Average cross-entropy loss.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    for input_ids, target_ids in data_loader:

        input_ids = input_ids.to(
            device
        )

        target_ids = target_ids.to(
            device
        )

        output = model(
            input_ids,
            targets=target_ids,
        )

        # ------------------------------------------------------
        # GPT returns:
        #
        # logits, loss
        # ------------------------------------------------------

        if isinstance(output, tuple):

            _, loss = output

        else:

            logits = output

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                target_ids.reshape(-1),
            )

        total_loss += loss.item()

        total_batches += 1

    if total_batches == 0:

        raise RuntimeError(
            "Validation DataLoader is empty."
        )

    return total_loss / total_batches


# ==============================================================
# Main
# ==============================================================


def main():

    # ==========================================================
    # Device
    # ==========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
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
    )

    # ==========================================================
    # Training Configuration
    # ==========================================================

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

    # ==========================================================
    # Encode Corpus
    # ==========================================================

    token_ids = tokenizer.encode(
        text
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
    # Create Model
    # ==========================================================

    model = GPT(
        model_config
    )

    model.to(device)

    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Initialized GPT with "
        f"{num_parameters:,} parameters."
    )

    # ==========================================================
    # Load Best Checkpoint
    # ==========================================================

    checkpoint_manager = CheckpointManager(
        "artifacts/checkpoints"
    )

    checkpoint_path = (
        "artifacts/checkpoints/best.pt"
    )

    checkpoint_manager.load(
        path=checkpoint_path,
        model=model,
        device=device,
    )

    print(
        "Loaded checkpoint:",
        checkpoint_path,
    )

    # ==========================================================
    # Evaluate
    # ==========================================================

    validation_loss = evaluate(
        model=model,
        data_loader=loaders.val_loader,
        device=device,
    )

    # ==========================================================
    # Perplexity
    # ==========================================================

    perplexity = math.exp(
        validation_loss
    )

    # ==========================================================
    # Save Evaluation Report
    # ==========================================================

    evaluation_dir = Path(
        "artifacts/evaluation"
    )

    evaluation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "checkpoint": checkpoint_path,
        "parameters": num_parameters,
        "validation_loss": validation_loss,
        "perplexity": perplexity,
        "device": str(device),
    }

    report_path = (
        evaluation_dir
        / "evaluation.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        "Evaluation report saved:",
        report_path,
    )
    # ==========================================================
    # Results
    # ==========================================================

    print()
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)

    print(
        f"Validation Loss : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Perplexity      : "
        f"{perplexity:.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":

    main()