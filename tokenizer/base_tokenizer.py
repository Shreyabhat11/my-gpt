"""
Base Tokenizer Interface

Defines the common API for all tokenizers.

Author: Shreya Bhat
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTokenizer(ABC):

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        pass

    @abstractmethod
    def train(
        self,
        text: str,
        vocab_size: int,
    ) -> None:
        pass

    @abstractmethod
    def encode(
        self,
        text: str,
    ) -> list[int]:
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        pass

    @abstractmethod
    def save(
        self,
        path: str | Path,
    ) -> None:
        pass

    @abstractmethod
    def load(
        self,
        path: str | Path,
    ) -> None:
        pass