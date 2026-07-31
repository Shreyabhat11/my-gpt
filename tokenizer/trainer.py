"""
BPE Trainer

Learns Byte Pair Encoding merge rules from text.

Author: Shreya Bhat
"""

from __future__ import annotations

from tokenizer.utils import (
    text_to_bytes,
    get_pair_frequencies,
    merge_pair,
)


class BPETrainer:
    """
    Learns BPE merge rules.
    """

    def __init__(self):

        # pair -> new_token_id
        self.merges = {}

        # token_id -> bytes
        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }

    def train(
        self,
        text: str,
        vocab_size: int,
    ):
        """
        Learn merge rules until vocabulary reaches vocab_size.
        """

        if vocab_size < 256:
            raise ValueError(
                "Vocabulary size must be at least 256."
            )

        token_ids = text_to_bytes(text)

        next_token = 256

        while next_token < vocab_size:

            pair_counts = get_pair_frequencies(token_ids)

            if not pair_counts:
                break

            best_pair = max(
                pair_counts,
                key=pair_counts.get,
            )

            # Save merge
            self.merges[best_pair] = next_token

            # Build new vocabulary entry
            self.vocab[next_token] = (
                self.vocab[best_pair[0]]
                + self.vocab[best_pair[1]]
            )

            # Merge sequence
            token_ids = merge_pair(
                token_ids,
                best_pair,
                next_token,
            )

            print(
                f"Learned token {next_token}: "
                f"{best_pair} "
                f"freq={pair_counts[best_pair]}"
            )

            next_token += 1

        return token_ids

if __name__ == "__main__":

    trainer = BPETrainer()

    final_tokens = trainer.train(
        text="helloo alll....",
        vocab_size=270,
    )

    print("\nFinal Tokens")
    print(final_tokens)

    print("\nNumber of merges")
    print(len(trainer.merges))

    print("\nVocabulary Size")
    print(len(trainer.vocab))