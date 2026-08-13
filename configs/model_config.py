"""
Configuration for the GPT model.

Author: Shreya Bhat
"""

from __future__ import annotations

from dataclasses import dataclass

from configs.base_config import BaseConfig


@dataclass
class GPTConfig(BaseConfig):
    """
    Configuration for GPT model architecture.
    """

    # ==========================================================
    # Vocabulary
    # ==========================================================

    vocab_size: int = 50257

    # ==========================================================
    # Context Window
    # ==========================================================

    context_length: int = 1024

    # ==========================================================
    # Transformer Architecture
    # ==========================================================

    embed_dim: int = 768

    num_heads: int = 12

    num_layers: int = 12

    # ==========================================================
    # Regularization
    # ==========================================================

    dropout: float = 0.1

    bias: bool = False

    # ==========================================================
    # Feed Forward Network
    # ==========================================================

    expansion_factor: int = 4

    # ==========================================================
    # LayerNorm
    # ==========================================================

    layer_norm_eps: float = 1e-5

    # ==========================================================
    # Initialization
    # ==========================================================

    initializer_range: float = 0.02

    # ==========================================================
    # Special Tokens
    # ==========================================================

    pad_token_id: int = 0

    bos_token_id: int = 1

    eos_token_id: int = 2

    # ==========================================================
    # Derived Properties
    # ==========================================================

    @property
    def head_dim(self) -> int:
        """
        Dimension of each attention head.
        """

        return self.embed_dim // self.num_heads

    # ==========================================================
    # Validation
    # ==========================================================

    def validate(self) -> None:
        """
        Validate model configuration.
        """

        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than zero."
            )

        if self.context_length <= 0:
            raise ValueError(
                "context_length must be greater than zero."
            )

        if self.embed_dim <= 0:
            raise ValueError(
                "embed_dim must be greater than zero."
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be greater than zero."
            )

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero."
            )

        if self.dropout < 0.0 or self.dropout > 1.0:
            raise ValueError(
                "dropout must be between 0 and 1."
            )

        if self.expansion_factor <= 0:
            raise ValueError(
                "expansion_factor must be greater than zero."
            )

        if self.layer_norm_eps <= 0:
            raise ValueError(
                "layer_norm_eps must be greater than zero."
            )

        if self.initializer_range <= 0:
            raise ValueError(
                "initializer_range must be greater than zero."
            )

if __name__ == "__main__":

    config = GPTConfig(
        vocab_size=10000,
        context_length=256,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
    )

    config.display()

    config.save(
        "artifacts/model_config.json"
    )

    loaded_config = GPTConfig.load(
        "artifacts/model_config.json"
    )

    print("\nLoaded configuration:")
    loaded_config.display()

    print(
        "\nHead dimension:",
        loaded_config.head_dim,
    )