"""
Checkpoint Management for GPT Training.

Stores:
    - Model state
    - Optimizer state
    - Scheduler state
    - Training step
    - Configuration

Author: Shreya Bhat
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:
    """
    Handles saving and loading GPT training checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "artifacts/checkpoints",
    ) -> None:

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # Save
    # ==========================================================

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        config: Any,
        filename: str = "latest.pt",
    ) -> Path:
        """
        Save complete training state.
        """

        checkpoint = {
            "step": step,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "config":
                config.__dict__,
        }

        path = (
            self.checkpoint_dir
            / filename
        )

        # ------------------------------------------------------
        # Save atomically
        # ------------------------------------------------------

        temporary_path = (
            self.checkpoint_dir
            / f"{filename}.tmp"
        )

        torch.save(
            checkpoint,
            temporary_path,
        )

        temporary_path.replace(
            path
        )

        return path

    # ==========================================================
    # Load
    # ==========================================================

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: torch.device | str = "cpu",
    ) -> int:
        """
        Load checkpoint.

        Returns:
            Training step stored in checkpoint.
        """

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        # ------------------------------------------------------
        # Model
        # ------------------------------------------------------

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        # ------------------------------------------------------
        # Optimizer
        # ------------------------------------------------------

        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        # ------------------------------------------------------
        # Scheduler
        # ------------------------------------------------------

        if scheduler is not None:

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        return checkpoint["step"]


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    from configs.model_config import GPTConfig
    from training.optimizer import create_optimizer
    from training.scheduler import (
        CosineWarmupScheduler,
    )

    torch.manual_seed(42)

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    config = GPTConfig(
        vocab_size=256,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    )

    model = torch.nn.Linear(
        128,
        128,
    )

    # ----------------------------------------------------------
    # Optimizer
    # ----------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    # ----------------------------------------------------------
    # Scheduler
    # ----------------------------------------------------------

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=10,
        max_steps=100,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
    )

    # ----------------------------------------------------------
    # Checkpoint manager
    # ----------------------------------------------------------

    manager = CheckpointManager(
        "artifacts/test_checkpoints"
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=42,
        config=config,
    )

    print("=" * 60)

    print(
        "Checkpoint saved:"
    )

    print(path)

    # ----------------------------------------------------------
    # New model
    # ----------------------------------------------------------

    new_model = torch.nn.Linear(
        128,
        128,
    )

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=3e-4,
    )

    new_scheduler = CosineWarmupScheduler(
        optimizer=new_optimizer,
        warmup_steps=10,
        max_steps=100,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
    )

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------

    step = manager.load(
        path=path,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
    )

    print(
        "Loaded step:",
        step,
    )

    print("=" * 60)