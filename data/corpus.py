"""
Corpus Loader

Utilities for loading raw text corpora.

Author: Shreya Bhat
"""

from __future__ import annotations

from pathlib import Path


class CorpusLoader:
    """
    Load text corpus from disk.
    """

    @staticmethod
    def load(path: str | Path) -> str:
        """
        Load a UTF-8 text file.

        Parameters
        ----------
        path : str | Path
            Path to the text file.

        Returns
        -------
        str
            File contents.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Corpus not found: {path}"
            )

        return path.read_text(
            encoding="utf-8"
        )

if __name__ == "__main__":

    text = CorpusLoader.load(
        "data/sample.txt"
    )

    print(text[:500])