"""
Learning Rate Scheduler.

Implements:

    Linear Warmup
          ↓
    Cosine Decay
          ↓
    Minimum Learning Rate

Author: Shreya Bhat
"""

from __future__ import annotations

import math

import torch


class CosineWarmupScheduler:
    """
    Learning-rate scheduler for GPT training.

    Schedule:

        warmup:
            min_lr → max_lr

        decay:
            max_lr → min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        learning_rate: float,
        min_learning_rate: float,
    ) -> None:

        if warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative."
            )

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        if warmup_steps > max_steps:
            raise ValueError(
                "warmup_steps cannot exceed max_steps."
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if min_learning_rate < 0:
            raise ValueError(
                "min_learning_rate cannot be negative."
            )

        if min_learning_rate > learning_rate:
            raise ValueError(
                "min_learning_rate cannot exceed learning_rate."
            )

        self.optimizer = optimizer

        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate

        self.step_count = 0

        # Initialize optimizer LR
        self._set_learning_rate(
            self._get_learning_rate(0)
        )

    # ==========================================================
    # Calculate LR
    # ==========================================================

    def _get_learning_rate(
        self,
        step: int,
    ) -> float:

        # ------------------------------------------------------
        # Warmup
        # ------------------------------------------------------

        if self.warmup_steps > 0 and step < self.warmup_steps:

            warmup_progress = (
                step + 1
            ) / self.warmup_steps

            return (
                self.min_learning_rate
                + (
                    self.learning_rate
                    - self.min_learning_rate
                )
                * warmup_progress
            )

        # ------------------------------------------------------
        # After training
        # ------------------------------------------------------

        if step >= self.max_steps:

            return self.min_learning_rate

        # ------------------------------------------------------
        # Cosine Decay
        # ------------------------------------------------------

        decay_steps = (
            self.max_steps
            - self.warmup_steps
        )

        current_step = (
            step
            - self.warmup_steps
        )

        progress = (
            current_step
            / decay_steps
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine_decay = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * progress
                )
            )
        )

        return (
            self.min_learning_rate
            + (
                self.learning_rate
                - self.min_learning_rate
            )
            * cosine_decay
        )

    # ==========================================================
    # Set Optimizer LR
    # ==========================================================

    def _set_learning_rate(
        self,
        learning_rate: float,
    ) -> None:

        for param_group in self.optimizer.param_groups:

            param_group["lr"] = learning_rate

    # ==========================================================
    # Step Scheduler
    # ==========================================================

    def step(self) -> None:

        self.step_count += 1

        learning_rate = self._get_learning_rate(
            self.step_count
        )

        self._set_learning_rate(
            learning_rate
        )

    # ==========================================================
    # Current LR
    # ==========================================================

    def get_lr(self) -> float:

        return self.optimizer.param_groups[0]["lr"]

    # ==========================================================
    # State Dict
    # ==========================================================

    def state_dict(self) -> dict:

        return {
            "step_count": self.step_count,
        }

    # ==========================================================
    # Load State Dict
    # ==========================================================

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:

        self.step_count = state_dict[
            "step_count"
        ]

        learning_rate = self._get_learning_rate(
            self.step_count
        )

        self._set_learning_rate(
            learning_rate
        )


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    model = torch.nn.Linear(
        10,
        10,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scheduler = CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=100,
        max_steps=1000,
        learning_rate=3e-4,
        min_learning_rate=3e-5,
    )

    learning_rates = []

    for step in range(1000):

        learning_rates.append(
            scheduler.get_lr()
        )

        scheduler.step()

    print("=" * 60)

    print(
        "Initial LR:",
        learning_rates[0],
    )

    print(
        "Peak LR:",
        max(learning_rates),
    )

    print(
        "Final LR:",
        learning_rates[-1],
    )

    print("=" * 60)

    plt.plot(
        learning_rates
    )

    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title(
        "GPT Learning Rate Schedule"
    )

    plt.show()