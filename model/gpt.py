"""
GPT Model

Decoder-only Transformer inspired by GPT-2.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch
import torch.nn as nn

from configs.model_config import GPTConfig
from model.transformer_block import TransformerBlock

import torch.nn.functional as F
from typing import Optional

class GPT(nn.Module):
    """
    GPT Decoder-only Language Model.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # --------------------------------------------------
        # Token Embedding
        # --------------------------------------------------

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
        )

        # --------------------------------------------------
        # Positional Embedding
        # --------------------------------------------------

        self.position_embedding = nn.Embedding(
            config.context_length,
            config.embed_dim,
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        # --------------------------------------------------
        # Transformer Blocks
        # --------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.num_layers)
            ]
        )

        # --------------------------------------------------
        # Final LayerNorm
        # --------------------------------------------------

        self.final_layer_norm = nn.LayerNorm(
            config.embed_dim
        )

        # --------------------------------------------------
        # Language Modeling Head
        # --------------------------------------------------

        self.lm_head = nn.Linear(
            config.embed_dim,
            config.vocab_size,
            bias=False,
        )

        # --------------------------------------------------
        # Weight Tying
        #
        # GPT-2 shares embedding and output weights.
        # --------------------------------------------------

        self.lm_head.weight = self.token_embedding.weight

        # --------------------------------------------------
        # Initialize parameters
        # --------------------------------------------------

        self.apply(self._init_weights)

        print(
            f"Initialized GPT with "
            f"{self.get_num_parameters():,} parameters."
        )

    def _init_weights(self, module: nn.Module) -> None:
        """
        GPT-2 weight initialization.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_parameters(self) -> int:
        """
        Returns total trainable parameters.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the GPT model.

        Args:
            input_ids:
                Tensor of shape (batch_size, seq_length)
                containing token IDs.

            targets:
                Optional tensor of shape (batch_size, seq_length)
                containing target token IDs for language modeling.

        Returns:
            logits:
                Tensor of shape (batch_size, seq_length, vocab_size)

            loss:
                Cross-entropy loss if targets are provided,
                otherwise None.
        """

        # -----------------------------------------------------
        # Input Dimensions
        # -----------------------------------------------------

        batch_size, seq_length = input_ids.shape

        # -----------------------------------------------------
        # Ensure sequence length does not exceed context length
        # -----------------------------------------------------

        if seq_length > self.config.context_length:
            raise ValueError(
                f"Input sequence length ({seq_length}) exceeds "
                f"maximum context length ({self.config.context_length})."
            )

        # -----------------------------------------------------
        # Create Position IDs
        #
        # Example:
        # [0, 1, 2, 3, ..., T-1]
        # -----------------------------------------------------

        position_ids = torch.arange(
            seq_length,
            device=input_ids.device,
        )

        # -----------------------------------------------------
        # Token Embeddings
        # Shape:
        # (B, T) -> (B, T, C)
        # -----------------------------------------------------

        token_embeddings = self.token_embedding(input_ids)

        # -----------------------------------------------------
        # Position Embeddings
        # Shape:
        # (T,) -> (T, C)
        #
        # Broadcasting automatically expands to
        # (B, T, C) during addition.
        # -----------------------------------------------------

        position_embeddings = self.position_embedding(position_ids)

        # -----------------------------------------------------
        # Combine Token + Position Embeddings
        # -----------------------------------------------------

        x = token_embeddings + position_embeddings

        x = self.embedding_dropout(x)

        # -----------------------------------------------------
        # Transformer Decoder Blocks
        # -----------------------------------------------------

        for block in self.blocks:
            x = block(x)

        # -----------------------------------------------------
        # Final Layer Normalization
        # -----------------------------------------------------

        x = self.final_layer_norm(x)

        # -----------------------------------------------------
        # Vocabulary Projection
        #
        # Shape:
        # (B, T, C)
        #
        # ->
        #
        # (B, T, vocab_size)
        # -----------------------------------------------------

        logits = self.lm_head(x)

        # -----------------------------------------------------
        # Compute Loss (Training)
        # -----------------------------------------------------

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        Args:
            input_ids:
                Tensor of shape (B, T)

            max_new_tokens:
                Number of tokens to generate.

            temperature:
                Controls randomness.

            top_k:
                Restrict sampling to top-k tokens.

        Returns:
            Generated token IDs.
        """

        self.eval()

        for _ in range(max_new_tokens):

            # ------------------------------
            # Crop context if too long
            # ------------------------------

            idx = input_ids[:, -self.config.context_length :]

            # ------------------------------
            # Forward Pass
            # ------------------------------

            logits, _ = self(idx)

            # ------------------------------
            # Last token only
            # Shape:
            # (B,V)
            # ------------------------------

            logits = logits[:, -1, :]

            # ------------------------------
            # Temperature
            # ------------------------------

            logits = logits / temperature

            # ------------------------------
            # Top-k Sampling
            # ------------------------------

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    top_k,
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            # ------------------------------
            # Convert to probabilities
            # ------------------------------

            probs = F.softmax(
                logits,
                dim=-1,
            )

            # ------------------------------
            # Sample next token
            # ------------------------------

            next_token = torch.multinomial(
                probs,
                num_samples=1,
            )

            # ------------------------------
            # Append
            # ------------------------------

            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1,
            )

        return input_ids

if __name__ == "__main__":
    from configs.model_config import GPTConfig

    torch.manual_seed(42)

    config = GPTConfig(
        vocab_size=65,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
    )

    model = GPT(config)

    x = torch.randint(
        0,
        config.vocab_size,
        (4, 32),
    )

    y = torch.randint(
        0,
        config.vocab_size,
        (4, 32),
    )

    logits, loss = model(x, y)

    print("=" * 60)
    print("Input Shape :", x.shape)
    print("Logits Shape:", logits.shape)
    print("Loss:", loss.item())
    print("=" * 60)