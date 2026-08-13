"""
Byte Pair Encoding Tokenizer

Author: Shreya Bhat
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.base_tokenizer import BaseTokenizer
from tokenizer.trainer import BPETrainer
from tokenizer.utils import (
    text_to_bytes,
    bytes_to_text,
    merge_pair,
)

import json

class BPETokenizer(BaseTokenizer):

    def __init__(self):

        self.trainer = BPETrainer()

        self.merges = self.trainer.merges

        self.vocab = self.trainer.vocab

    def train(
        self,
        text: str,
        vocab_size: int,
    ) -> None:

        self.trainer.train(
            text,
            vocab_size,
        )

        self.merges = self.trainer.merges
        self.vocab = self.trainer.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        byte_sequence = b"".join(
            self.vocab[token]
            for token in token_ids
        )

        return bytes_to_text(
            list(byte_sequence)
        )
    
    def encode(
        self,
        text: str,
    ) -> list[int]:
        """
        Encode text into BPE token IDs.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        list[int]
            Encoded token IDs.
        """
        token_ids = text_to_bytes(text)

        for pair, new_token in self.merges.items():

            token_ids = merge_pair(
                token_ids,
                pair,
                new_token,
            )

        return token_ids

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save tokenizer to a JSON file.
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "merges": {
                f"{pair[0]},{pair[1]}": token
                for pair, token in self.merges.items()
            },
            "vocab": {
                str(token): list(byte_seq)
                for token, byte_seq in self.vocab.items()
            },
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=4,
            )


    def load(
        self,
        path: str | Path,
    ) -> None:
        """
        Load tokenizer from a JSON file.
        """

        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.merges = {
            tuple(map(int, key.split(","))): value
            for key, value in data["merges"].items()
        }

        self.vocab = {
            int(token): bytes(byte_seq)
            for token, byte_seq in data["vocab"].items()
        }

        # Keep trainer in sync
        self.trainer.merges = self.merges
        self.trainer.vocab = self.vocab
    
if __name__ == "__main__":

    corpus = """
    banana banana bandana
    banana banana
    """

    tokenizer = BPETokenizer()

    tokenizer.train(
        corpus,
        vocab_size=270,
    )

    tokenizer.save("artifacts/tokenizer.json")

    print("Tokenizer saved!")

    tokenizer2 = BPETokenizer()

    tokenizer2.load("artifacts/tokenizer.json")

    text = "banana"

    ids = tokenizer2.encode(text)

    recovered = tokenizer2.decode(ids)

    print("=" * 60)
    print("Original :", text)
    print("Encoded  :", ids)
    print("Decoded  :", recovered)
    print("=" * 60)