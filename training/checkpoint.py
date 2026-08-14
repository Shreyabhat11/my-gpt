"""
Checkpoint Management for GPT Training.

Stores:
    - Model state
    - Optimizer state
    - Scheduler state
    - Training step
    - Best validation loss
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
        best_val_loss: float | None = None,
    ) -> Path:
        """
        Save complete training state.

        Args:
            model:
                GPT model.

            optimizer:
                Optimizer state.

            scheduler:
                Learning-rate scheduler.

            step:
                Current training step.

            config:
                GPT configuration.

            filename:
                Checkpoint filename.

            best_val_loss:
                Best validation loss observed so far.
        """

        checkpoint = {
            "step": step,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                (
                    scheduler.state_dict()
                    if scheduler is not None
                    else None
                ),

            "config":
                config.__dict__,

            "best_val_loss":
                best_val_loss,
        }

        path = (
            self.checkpoint_dir
            / filename
        )

        # ------------------------------------------------------
        # Atomic save
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
    ) -> dict:
        """
        Load checkpoint.

        Returns:
            Complete checkpoint metadata.
        """

        path = Path(path)

        if not path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
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

        if (
            optimizer is not None
            and checkpoint.get(
                "optimizer_state_dict"
            ) is not None
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        # ------------------------------------------------------
        # Scheduler
        # ------------------------------------------------------

        if (
            scheduler is not None
            and checkpoint.get(
                "scheduler_state_dict"
            ) is not None
        ):

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        return checkpoint

    # ==========================================================
    # Save Latest
    # ==========================================================

    def save_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        config: Any,
        best_val_loss: float | None = None,
    ) -> Path:

        return self.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=config,
            filename="latest.pt",
            best_val_loss=best_val_loss,
        )

    # ==========================================================
    # Save Best
    # ==========================================================

    def save_best(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        step: int,
        config: Any,
        best_val_loss: float,
    ) -> Path:

        return self.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            config=config,
            filename="best.pt",
            best_val_loss=best_val_loss,
        )


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    from configs.model_config import GPTConfig
    from training.scheduler import (
        CosineWarmupScheduler,
    )

    torch.manual_seed(42)

    # ----------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------

    config = GPTConfig(
        vocab_size=256,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

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
    # Manager
    # ----------------------------------------------------------

    manager = CheckpointManager(
        "artifacts/test_checkpoints"
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    latest_path = manager.save_latest(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=42,
        config=config,
        best_val_loss=2.5,
    )

    best_path = manager.save_best(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=42,
        config=config,
        best_val_loss=2.5,
    )

    print("=" * 60)

    print(
        "Latest checkpoint:",
        latest_path,
    )

    print(
        "Best checkpoint:",
        best_path,
    )

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

    checkpoint = manager.load(
        path=latest_path,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
    )

    print(
        "Loaded step:",
        checkpoint["step"],
    )

    print(
        "Best validation loss:",
        checkpoint["best_val_loss"],
    )

    print("=" * 60)