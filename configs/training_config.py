"""
Training Configuration.

Author: Shreya Bhat
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:

    # ==========================================================
    # Batch Configuration
    # ==========================================================

    batch_size: int = 4

    # ==========================================================
    # Training Duration
    # ==========================================================

    max_steps: int = 1000

    # ==========================================================
    # Optimization
    # ==========================================================

    learning_rate: float = 3e-4

    min_learning_rate: float = 3e-5

    weight_decay: float = 0.1

    beta1: float = 0.9

    beta2: float = 0.95

    grad_clip: float = 1.0

    # ==========================================================
    # Learning Rate Schedule
    # ==========================================================

    warmup_steps: int = 100

    # ==========================================================
    # DataLoader
    # ==========================================================

    num_workers: int = 0

    pin_memory: bool = False

    drop_last: bool = True

    # ==========================================================
    # Validation
    # ==========================================================

    validation_interval: int = 100

    validation_steps: int = 20

    # ==========================================================
    # Checkpointing
    # ==========================================================

    checkpoint_interval: int = 500

    def __post_init__(self):

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        if self.max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero."
            )

        if self.min_learning_rate < 0:
            raise ValueError(
                "min_learning_rate cannot be negative."
            )

        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "min_learning_rate cannot be greater "
                "than learning_rate."
            )

        if self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative."
            )

        if self.warmup_steps > self.max_steps:
            raise ValueError(
                "warmup_steps cannot exceed max_steps."
            )

        if self.grad_clip <= 0:
            raise ValueError(
                "grad_clip must be greater than zero."
            )