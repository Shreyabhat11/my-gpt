"""
Plot GPT training metrics.

Author: Shreya Bhat
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


HISTORY_PATH = Path(
    "artifacts/training_history.json"
)

OUTPUT_DIR = Path(
    "artifacts/plots"
)


def main():

    if not HISTORY_PATH.exists():

        raise FileNotFoundError(
            f"Training history not found: {HISTORY_PATH}"
        )

    with open(
        HISTORY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        history = json.load(file)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==========================================================
    # Loss Curve
    # ==========================================================

    steps = history["steps"]
    train_loss = history["train_loss"]
    val_steps = history.get("val_steps", [])
    val_loss = history.get("val_loss", [])

    plt.figure(figsize=(10, 6))

    plt.plot(
        steps,
        train_loss,
        label="Train Loss",
    )

    if val_loss:

        plt.plot(
            val_steps,
            val_loss,
            label="Validation Loss",
        )

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("GPT Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    loss_path = (
        OUTPUT_DIR
        / "loss_curve.png"
    )

    plt.savefig(
        loss_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # ==========================================================
    # Learning Rate Curve
    # ==========================================================

    learning_rates = history.get(
        "learning_rates",
        [],
    )

    if learning_rates:

        plt.figure(figsize=(10, 6))

        plt.plot(
            steps,
            learning_rates,
        )

        plt.xlabel("Training Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3)

        lr_path = (
            OUTPUT_DIR
            / "learning_rate.png"
        )

        plt.savefig(
            lr_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "Learning-rate plot saved:",
            lr_path,
        )

    print(
        "Loss plot saved:",
        loss_path,
    )


if __name__ == "__main__":

    main()