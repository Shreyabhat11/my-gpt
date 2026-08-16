"""
GPT Training Engine.

Handles:

    - Training steps
    - Validation
    - Gradient clipping
    - Optimizer updates
    - Learning-rate scheduling
    - Training statistics

Author: Shreya Bhat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader

from configs.training_config import TrainingConfig
from model.gpt import GPT
from training.optimizer import create_optimizer
from training.scheduler import CosineWarmupScheduler
from training.checkpoint import CheckpointManager

import json
from pathlib import Path
@dataclass
class TrainingMetrics:
    """
    Stores training metrics.
    """

    step: int
    train_loss: float
    learning_rate: float
    val_loss: Optional[float] = None


class Trainer:

    def __init__(
        self,
        model: GPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
    ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.global_step = 0
        self.checkpoint_manager = CheckpointManager(
            "artifacts/checkpoints"
        )
        self.optimizer = None
        self.scheduler = None
        self.best_val_loss = float("inf")
        # ======================================================
        # Move model to device
        # ======================================================

        self.model.to(self.device)

        # ======================================================
        # Optimizer
        # ======================================================

        self.optimizer = create_optimizer(
            model=self.model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(
                config.beta1,
                config.beta2,
            ),
        )

        # ======================================================
        # Learning Rate Scheduler
        # ======================================================

        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            min_learning_rate=config.min_learning_rate,
        )

        self.history = {
            "steps": [],
            "train_loss": [],
            "val_steps": [],
            "val_loss": [],
            "learning_rates": [],
        }


    # ==========================================================
    # Train Step
    # ==========================================================

    def train_step(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> float:

        self.model.train()

        # ------------------------------------------------------
        # Move data to device
        # ------------------------------------------------------

        input_ids = input_ids.to(
            self.device,
            non_blocking=True,
        )

        target_ids = target_ids.to(
            self.device,
            non_blocking=True,
        )

        # ------------------------------------------------------
        # Clear gradients
        # ------------------------------------------------------

        self.optimizer.zero_grad(
            set_to_none=True
        )

        # ------------------------------------------------------
        # Forward
        # ------------------------------------------------------

        _, loss = self.model(
            input_ids,
            target_ids,
        )

        self.history["steps"].append(
            self.global_step
        )

        self.history["train_loss"].append(
            loss.item()
        )

        self.history["learning_rates"].append(
            self.optimizer.param_groups[0]["lr"]
        )

        # ------------------------------------------------------
        # Backward
        # ------------------------------------------------------

        loss.backward()

        # ------------------------------------------------------
        # Gradient Clipping
        # ------------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config.grad_clip,
        )

        # ------------------------------------------------------
        # Optimizer Update
        # ------------------------------------------------------

        self.optimizer.step()

        # ------------------------------------------------------
        # Scheduler Update
        # ------------------------------------------------------

        self.scheduler.step()

        return loss.item()

    # ==========================================================
    # Validation
    # ==========================================================

    @torch.no_grad()
    def evaluate(self) -> float:

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, (
            input_ids,
            target_ids,
        ) in enumerate(self.val_loader):

            # --------------------------------------------------
            # Limit validation batches
            # --------------------------------------------------

            if (
                batch_idx
                >= self.config.validation_steps
            ):
                break

            # --------------------------------------------------
            # Move to device
            # --------------------------------------------------

            input_ids = input_ids.to(
                self.device,
                non_blocking=True,
            )

            target_ids = target_ids.to(
                self.device,
                non_blocking=True,
            )

            # --------------------------------------------------
            # Forward
            # --------------------------------------------------

            _, loss = self.model(
                input_ids,
                target_ids,
            )

            total_loss += loss.item()

            num_batches += 1

        # ------------------------------------------------------
        # Avoid division by zero
        # ------------------------------------------------------

        if num_batches == 0:

            raise RuntimeError(
                "Validation DataLoader produced no batches."
            )

        return (
            total_loss
            / num_batches
        )

    # ==========================================================
    # Training Loop
    # ==========================================================

    def train(self) -> list[TrainingMetrics]:

        history: list[TrainingMetrics] = []

        train_iterator = iter(
            self.train_loader
        )

        while (
            self.global_step
            < self.config.max_steps
        ):

            # --------------------------------------------------
            # Get next batch
            # --------------------------------------------------

            try:

                input_ids, target_ids = next(
                    train_iterator
                )

            except StopIteration:

                train_iterator = iter(
                    self.train_loader
                )

                input_ids, target_ids = next(
                    train_iterator
                )

            # --------------------------------------------------
            # Training step
            # --------------------------------------------------

            train_loss = self.train_step(
                input_ids,
                target_ids,
            )

            self.global_step += 1

            # --------------------------------------------------
            # Current learning rate
            # --------------------------------------------------

            learning_rate = self.scheduler.get_lr()

            # --------------------------------------------------
            # Validation
            # --------------------------------------------------

            val_loss = None

            if (
                self.global_step
                % self.config.validation_interval
                == 0
            ):

                val_loss = self.evaluate()

                self.history["val_steps"].append(
                    self.global_step
                )

                self.history["val_loss"].append(
                    val_loss
                )

                # --------------------------------------------------
                # Best checkpoint
                # --------------------------------------------------

                if val_loss < self.best_val_loss:

                    self.best_val_loss = val_loss

                    self.save_checkpoint(
                        filename="best.pt"
                    )

                    print(
                        f"New best checkpoint saved | "
                        f"Val Loss {val_loss:.4f}"
                    )

            # --------------------------------------------------
            # Latest checkpoint
            # --------------------------------------------------

            if (
                self.global_step
                % self.config.checkpoint_interval
                == 0
            ):

                self.save_checkpoint(
                    filename="latest.pt"
                )

            # --------------------------------------------------
            # Store metrics
            # --------------------------------------------------

            metrics = TrainingMetrics(
                step=self.global_step,
                train_loss=train_loss,
                learning_rate=learning_rate,
                val_loss=val_loss,
            )

            history.append(
                metrics
            )

            # --------------------------------------------------
            # Logging
            # --------------------------------------------------

            if val_loss is not None:

                print(
                    f"Step {self.global_step:5d} | "
                    f"Train Loss {train_loss:.4f} | "
                    f"Val Loss {val_loss:.4f} | "
                    f"LR {learning_rate:.6e}"
                )

            else:

                print(
                    f"Step {self.global_step:5d} | "
                    f"Train Loss {train_loss:.4f} | "
                    f"LR {learning_rate:.6e}"
                )
        # ==========================================================
        # Save Training History
        # ==========================================================

        history_path = Path(
            "artifacts/training_history.json"
        )

        history_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.history,
                file,
                indent=4,
            )

        print(
            "Training history saved:",
            history_path,
        )

        return history

    def save_checkpoint(
        self,
        filename: str = "latest.pt",
    ) -> None:

        path = self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            config=self.config,
            filename=filename,
            best_val_loss=self.best_val_loss,
        )

        print(
            f"Checkpoint saved: {path}"
        )

    def load_checkpoint(
        self,
        path: str,
    ) -> None:

        checkpoint = self.checkpoint_manager.load(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
        )

        self.global_step = checkpoint["step"]

        self.best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf"),
        )

        print(
            f"Checkpoint loaded: {path}"
        )

        print(
            f"Resuming from step: "
            f"{self.global_step}"
        )

        print(
            f"Best validation loss: "
            f"{self.best_val_loss}"
        )


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    from configs.model_config import GPTConfig
    from data.dataloader import create_dataloaders

    torch.manual_seed(42)

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
    # Model Config
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
    # Training Config
    # ==========================================================

    training_config = TrainingConfig(
        batch_size=4,
        max_steps=100,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
        warmup_steps=10,
        validation_interval=20,
        validation_steps=5,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    # ==========================================================
    # Fake Dataset
    # ==========================================================

    token_ids = torch.randint(
        0,
        model_config.vocab_size,
        (10_000,),
        dtype=torch.long,
    )

    # ==========================================================
    # DataLoaders
    # ==========================================================

    loaders = create_dataloaders(
        token_ids=token_ids,
        model_config=model_config,
        training_config=training_config,
        validation_split=0.1,
    )

    # ==========================================================
    # Model
    # ==========================================================

    model = GPT(
        model_config
    )

    # ==========================================================
    # Trainer
    # ==========================================================

    trainer = Trainer(
        model=model,
        train_loader=loaders.train_loader,
        val_loader=loaders.val_loader,
        config=training_config,
        device=device,
    )

    # ==========================================================
    # Train
    # ==========================================================

    history = trainer.train()

    # ==========================================================
    # Final Result
    # ==========================================================

    print("=" * 60)
    print("Training completed.")

    print(
        f"Total steps: {trainer.global_step}"
    )

    if history:

        print(
            f"Final train loss: "
            f"{history[-1].train_loss}"
        )

    else:

        print(
            "No new training steps were executed."
        )

        print(
            "Using metrics from the existing checkpoint/evaluation."
        )

    print("=" * 60)