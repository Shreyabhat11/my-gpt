"""
Configuration class for the GPT model.

Author: Shreya Bhat
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration for GPT model architecture.
    """

    # ==========================================================
    # Vocabulary
    # ==========================================================
    vocab_size: int = 50257          # GPT-2 vocabulary size

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

    @property
    def head_dim(self) -> int:
        """
        Dimension of each attention head.
        """
        return self.embed_dim // self.num_heads

    def __post_init__(self):
        """
        Validate configuration.
        """

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )

        if self.context_length <= 0:
            raise ValueError(
                "context_length must be greater than zero."
            )

        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be greater than zero."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero."
            )

        if self.dropout < 0 or self.dropout > 1:
            raise ValueError(
                "dropout must be between 0 and 1."
            )