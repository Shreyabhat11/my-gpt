"""
Base configuration utilities.

Provides common validation, serialization, and persistence
functionality for all project configurations.

Author: Shreya Bhat
"""

from __future__ import annotations

import json
from abc import ABC
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T", bound="BaseConfig")


class BaseConfig(ABC):
    """
    Base class for all configuration objects.
    """

    def validate(self) -> None:
        """
        Validate configuration values.

        Subclasses can override this method to add
        configuration-specific validation.
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to a dictionary.
        """

        self.validate()

        return asdict(self)

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save configuration to a JSON file.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.to_dict(),
                file,
                indent=4,
            )

    @classmethod
    def load(
        cls: type[T],
        path: str | Path,
    ) -> T:
        """
        Load configuration from a JSON file.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        config = cls(**data)

        config.validate()

        return config

    def display(self) -> None:
        """
        Pretty-print configuration.
        """

        print("=" * 60)
        print(
            f"{self.__class__.__name__}"
        )
        print("=" * 60)

        for field in fields(self):

            value = getattr(
                self,
                field.name,
            )

            print(
                f"{field.name:<25}: {value}"
            )

        print("=" * 60)