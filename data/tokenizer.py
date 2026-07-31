"""
Character-level tokenizer.

This tokenizer converts individual characters into integer token IDs
and vice versa. It serves as the simplest tokenizer for understanding
LLM training pipelines.
"""

import json
from pathlib import Path
from typing import Dict, List


class CharacterTokenizer:
    """
    Character-level tokenizer.
    """

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

    def build_vocab(self, text: str) -> None:
        """
        Build vocabulary from text.
        """

        unique_chars = sorted(list(set(text)))

        self.char_to_id = {
            ch: idx
            for idx, ch in enumerate(unique_chars)
        }

        self.id_to_char = {
            idx: ch
            for ch, idx in self.char_to_id.items()
        }

        print(f"Vocabulary Size: {self.vocab_size}")

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text into token IDs.
        """

        return [self.char_to_id[ch] for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back into text.
        """

        return "".join(
            self.id_to_char[idx]
            for idx in token_ids
        )

    def save_vocab(self, save_path: Path) -> None:
        """
        Save vocabulary as JSON.
        """

        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                self.char_to_id,
                f,
                indent=4,
                ensure_ascii=False
            )

        print(f"Vocabulary saved to {save_path}")

    def load_vocab(self, vocab_path: Path) -> None:
        """
        Load vocabulary from JSON.
        """

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.char_to_id = json.load(f)

        self.id_to_char = {
            idx: ch
            for ch, idx in self.char_to_id.items()
        }

        print(f"Vocabulary loaded from {vocab_path}")


if __name__ == "__main__":

    data_path = Path("data/processed/shakespeare_clean.txt")

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharacterTokenizer()

    tokenizer.build_vocab(text)

    encoded = tokenizer.encode("Hello")

    print("\nEncoded:")
    print(encoded)

    decoded = tokenizer.decode(encoded)

    print("\nDecoded:")
    print(decoded)

    tokenizer.save_vocab(
        Path("data/processed/vocab.json")
    )

    print("\nFirst 20 vocabulary entries:")

    for i, (char, idx) in enumerate(tokenizer.char_to_id.items()):
        if i == 20:
            break
        print(repr(char), "->", idx)