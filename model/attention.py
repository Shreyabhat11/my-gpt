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

from configs.model_config import GPTConfig

from configs.model_config import GPTConfig

class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        config: GPTConfig,
    ) -> None:

        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(
            config.embed_dim,
            3 * config.embed_dim,
            bias=config.bias,
        )

        self.out_proj = nn.Linear(
            config.embed_dim,
            config.embed_dim,
            bias=config.bias,
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    config.context_length,
                    config.context_length,
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

    from configs.model_config import GPTConfig

    config = GPTConfig(
        embed_dim=128,
        num_heads=8,
        context_length=32,
    )

    model = MultiHeadSelfAttention(config)

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