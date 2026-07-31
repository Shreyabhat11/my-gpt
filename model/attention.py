"""
Multi-Head Causal Self Attention

Production-style implementation inspired by GPT-2/NanoGPT.

Author: Shreya Bhat
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Causal Self Attention.

    Input Shape:
        (batch_size, seq_length, embed_dim)

    Output Shape:
        (batch_size, seq_length, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.1,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # One projection for QKV
        self.qkv = nn.Linear(
            embed_dim,
            3 * embed_dim,
            bias=bias,
        )

        # Output projection
        self.out_proj = nn.Linear(
            embed_dim,
            embed_dim,
            bias=bias,
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular causal mask
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length,
                    dtype=torch.bool,
                )
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_length, embed_dim = x.shape

        # ---------------------------------------------------
        # Compute Query, Key and Value
        # ---------------------------------------------------

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        # ---------------------------------------------------
        # Split into heads
        #
        # (B,T,C)
        # ->
        # (B,T,H,D)
        # ->
        # (B,H,T,D)
        # ---------------------------------------------------

        q = (
            q.view(
                batch_size,
                seq_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        k = (
            k.view(
                batch_size,
                seq_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        v = (
            v.view(
                batch_size,
                seq_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        # ---------------------------------------------------
        # Attention Scores
        # (B,H,T,D) x (B,H,D,T)
        # ->
        # (B,H,T,T)
        # ---------------------------------------------------

        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        causal_mask = self.mask[:seq_length, :seq_length]

        scores = scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        attention = F.softmax(scores, dim=-1)

        attention = self.attn_dropout(attention)

        # ---------------------------------------------------
        # Weighted Sum
        # ---------------------------------------------------

        out = attention @ v

        # ---------------------------------------------------
        # Merge Heads
        #
        # (B,H,T,D)
        # ->
        # (B,T,H,D)
        # ->
        # (B,T,C)
        # ---------------------------------------------------

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_length,
                embed_dim,
            )
        )

        out = self.out_proj(out)

        out = self.resid_dropout(out)

        if return_attention:
            return out, attention

        return out


if __name__ == "__main__":

    torch.manual_seed(42)

    model = MultiHeadSelfAttention(
        embed_dim=128,
        num_heads=8,
        context_length=32,
    )

    x = torch.randn(4, 32, 128)

    output, attention = model(
        x,
        return_attention=True,
    )

    print("=" * 60)
    print("Input Shape      :", x.shape)
    print("Output Shape     :", output.shape)
    print("Attention Shape  :", attention.shape)
    print("=" * 60)