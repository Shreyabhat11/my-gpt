"""
GPT Trainer

Handles model training.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        device: torch.device,
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.train_loader = train_loader

        self.device = device

    def train_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> float:

        self.model.train()

        x, y = batch

        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        _, loss = self.model(
            input_ids=x,
            targets=y,
        )

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns
        -------
        float
            Average training loss.
        """

        total_loss = 0.0

        for batch in self.train_loader:

            loss = self.train_step(batch)

            total_loss += loss

        average_loss = total_loss / len(self.train_loader)

        return average_loss

    def train(
        self,
        epochs: int,
    ) -> None:
        """
        Train the model.

        Parameters
        ----------
        epochs : int
            Number of epochs.
        """

        for epoch in range(epochs):

            loss = self.train_epoch()

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Loss: {loss:.4f}"
            )