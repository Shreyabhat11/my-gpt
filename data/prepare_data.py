"""
prepare_data.py

Loads the raw text dataset, performs basic cleaning,
and saves a processed version for tokenization.
"""

from pathlib import Path


class DataPreprocessor:
    """
    Handles loading, cleaning, and saving text datasets.
    """

    def __init__(self,
                 raw_path: Path,
                 processed_path: Path):
        self.raw_path = raw_path
        self.processed_path = processed_path

    def load_text(self) -> str:
        """
        Reads the raw dataset.
        """
        print(f"Loading dataset from: {self.raw_path}")

        with open(self.raw_path, "r", encoding="utf-8") as f:
            text = f.read()

        print(f"Loaded {len(text):,} characters")

        return text

    def clean_text(self, text: str) -> str:
        """
        Performs very basic cleaning.

        We intentionally keep punctuation,
        capitalization, and formatting because
        GPT models learn from them.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        return text.strip()

    def save_text(self, text: str):
        """
        Saves processed text.
        """

        self.processed_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.processed_path,
                  "w",
                  encoding="utf-8") as f:
            f.write(text)

        print(f"Saved processed dataset to:")
        print(self.processed_path)

    def process(self):

        text = self.load_text()

        text = self.clean_text(text)

        self.save_text(text)

        print("\nProcessing Complete!")
        print(f"Final length : {len(text):,} characters")


if __name__ == "__main__":

    raw_file = Path("data/raw/tiny_shakespeare.txt")

    processed_file = Path("data/processed/shakespeare_clean.txt")

    processor = DataPreprocessor(
        raw_file,
        processed_file
    )

    processor.process()