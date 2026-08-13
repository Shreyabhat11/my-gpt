"""
Tokenize training corpus using the custom BPE tokenizer.

Author: Shreya Bhat
"""

from __future__ import annotations

from pathlib import Path

from tokenizer.bpe import BPETokenizer


# ==============================================================
# Paths
# ==============================================================

CORPUS_PATH = Path(
    "data/corpus.txt"
)

TOKENIZER_PATH = Path(
    "artifacts/tokenizer.json"
)


def load_corpus(
    path: Path,
) -> str:
    """
    Load text corpus from disk.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Corpus not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return file.read()


def main():

    # ----------------------------------------------------------
    # Load corpus
    # ----------------------------------------------------------

    text = load_corpus(
        CORPUS_PATH
    )

    print("=" * 60)

    print(
        "Corpus path:",
        CORPUS_PATH,
    )

    print(
        "Characters:",
        len(text),
    )

    # ----------------------------------------------------------
    # Create tokenizer
    # ----------------------------------------------------------

    tokenizer = BPETokenizer()

    # ----------------------------------------------------------
    # Train tokenizer
    # ----------------------------------------------------------

    tokenizer.train(
        text=text,
        vocab_size=256,
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    # ----------------------------------------------------------
    # Encode corpus
    # ----------------------------------------------------------

    token_ids = tokenizer.encode(
        text
    )

    print(
        "Token count:",
        len(token_ids),
    )

    # ----------------------------------------------------------
    # Save tokenizer
    # ----------------------------------------------------------

    TOKENIZER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        TOKENIZER_PATH
    )

    print(
        "Tokenizer saved:",
        TOKENIZER_PATH,
    )

    # ----------------------------------------------------------
    # Test round trip
    # ----------------------------------------------------------

    sample = text[:100]

    encoded = tokenizer.encode(
        sample
    )

    decoded = tokenizer.decode(
        encoded
    )

    print()
    print(
        "Original:",
        repr(sample),
    )

    print(
        "Decoded :",
        repr(decoded),
    )

    print("=" * 60)


if __name__ == "__main__":

    main()