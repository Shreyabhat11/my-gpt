"""
embedding.py

Implements token embeddings and positional embeddings
for the GPT model.
"""

import torch
import torch.nn as nn


class GPTEmbedding(nn.Module):
    """
    Combines token embeddings and positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, input_ids: torch.Tensor):

        batch_size, seq_length = input_ids.shape

        positions = torch.arange(
            seq_length,
            device=input_ids.device,
        )

        token_embeddings = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(
            positions
        )

        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings