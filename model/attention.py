"""
Multi-Head Causal Self-Attention.

Production-oriented implementation for a decoder-only GPT.

Author: Shreya Bhat
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.model_config import GPTConfig


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Input:
        x -> (batch_size, sequence_length, embed_dim)

    Output:
        out -> (batch_size, sequence_length, embed_dim)
    """

    def __init__(
        self,
        config: GPTConfig,
    ) -> None:

        super().__init__()

        # ======================================================
        # Configuration
        # ======================================================

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.dropout = config.dropout

        # Scaling factor:
        #
        # attention = QK^T / sqrt(head_dim)
        #
        self.scale = self.head_dim ** -0.5

        # ======================================================
        # Validation
        # ======================================================

        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible "
                f"by num_heads ({self.num_heads})"
            )

        # ======================================================
        # QKV Projection
        # ======================================================

        self.qkv = nn.Linear(
            self.embed_dim,
            3 * self.embed_dim,
            bias=config.bias,
        )

        # ======================================================
        # Output Projection
        # ======================================================

        self.out_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=config.bias,
        )

        # ======================================================
        # Dropout
        # ======================================================

        self.attn_dropout = nn.Dropout(
            config.dropout
        )

        self.resid_dropout = nn.Dropout(
            config.dropout
        )

        # ======================================================
        # Causal Mask
        # ======================================================

        self.register_buffer(
            "causal_mask",
            torch.tril(
                torch.ones(
                    config.context_length,
                    config.context_length,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

    # ==========================================================
    # Forward
    # ==========================================================

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        # ------------------------------------------------------
        # Input
        # ------------------------------------------------------

        batch_size, sequence_length, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, "
                f"got {embed_dim}."
            )

        if sequence_length > self.causal_mask.size(0):
            raise ValueError(
                f"Sequence length ({sequence_length}) exceeds "
                f"context length ({self.causal_mask.size(0)})."
            )

        # ------------------------------------------------------
        # QKV Projection
        #
        # (B, T, C)
        #
        # ->
        #
        # (B, T, 3C)
        # ------------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(
            3,
            dim=-1,
        )

        # ------------------------------------------------------
        # Split into Attention Heads
        #
        # (B, T, C)
        #
        # ->
        #
        # (B, T, H, D)
        #
        # ->
        #
        # (B, H, T, D)
        # ------------------------------------------------------

        q = (
            q.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        k = (
            k.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        v = (
            v.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        # ======================================================
        # Attention
        # ======================================================

        if return_attention:

            # --------------------------------------------------
            # Explicit implementation.
            #
            # Useful when we actually want the attention matrix.
            # --------------------------------------------------

            scores = (
                q @ k.transpose(-2, -1)
            ) * self.scale

            causal_mask = self.causal_mask[
                :sequence_length,
                :sequence_length,
            ]

            scores = scores.masked_fill(
                ~causal_mask,
                torch.finfo(scores.dtype).min,
            )

            attention = F.softmax(
                scores,
                dim=-1,
            )

            attention = self.attn_dropout(
                attention
            )

            out = attention @ v

        else:

            # --------------------------------------------------
            # Optimized PyTorch attention.
            #
            # PyTorch can dispatch this to optimized kernels
            # such as Flash Attention when supported.
            # --------------------------------------------------

            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=(
                    self.dropout
                    if self.training
                    else 0.0
                ),
                is_causal=True,
            )

            attention = None

        # ======================================================
        # Merge Heads
        #
        # (B, H, T, D)
        #
        # ->
        #
        # (B, T, H, D)
        #
        # ->
        #
        # (B, T, C)
        # ======================================================

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                sequence_length,
                self.embed_dim,
            )
        )

        # ======================================================
        # Output Projection
        # ======================================================

        out = self.out_proj(out)

        out = self.resid_dropout(out)

        # ======================================================
        # Return
        # ======================================================

        if return_attention:

            return out, attention

        return out


# ==============================================================
# Test
# ==============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    config = GPTConfig(
        vocab_size=10_000,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
    )

    model = MultiHeadSelfAttention(config)

    x = torch.randn(
        4,
        32,
        128,
    )

    # ----------------------------------------------------------
    # Normal forward
    # ----------------------------------------------------------

    output = model(x)

    print("=" * 60)

    print(
        "Input Shape  :",
        x.shape,
    )

    print(
        "Output Shape :",
        output.shape,
    )

    # ----------------------------------------------------------
    # Attention inspection
    # ----------------------------------------------------------

    output, attention = model(
        x,
        return_attention=True,
    )

    print(
        "Attention Shape:",
        attention.shape,
    )

    print("=" * 60)