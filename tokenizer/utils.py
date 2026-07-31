"""
Utility functions for Byte Pair Encoding (BPE).

Author: Shreya Bhat
"""

from __future__ import annotations

from collections import Counter


def text_to_bytes(text: str) -> list[int]:
    """
    Convert text into UTF-8 byte IDs.

    Example
    -------
    "Hi"

    ->

    [72, 105]
    """

    return list(text.encode("utf-8"))

def bytes_to_text(byte_ids: list[int]) -> str:
    """
    Convert UTF-8 byte IDs back to text.
    """

    return bytes(byte_ids).decode(
        "utf-8",
        errors="replace",
    )

def get_pair_frequencies(
    token_ids: list[int],
) -> Counter[tuple[int, int]]:
    """
    Count the frequency of every adjacent token pair.

    Example
    -------
    Input:
        [1, 2, 3, 2, 3]

    Output:
        {
            (1, 2): 1,
            (2, 3): 2,
            (3, 2): 1
        }
    """

    pair_counts = Counter()

    for i in range(len(token_ids) - 1):
        pair = (
            token_ids[i],
            token_ids[i + 1],
        )

        pair_counts[pair] += 1

    return pair_counts

def merge_pair(
    token_ids: list[int],
    pair: tuple[int, int],
    new_token: int,
) -> list[int]:
    """
    Replace every occurrence of 'pair' with 'new_token'.

    Example
    -------
    Input:
        token_ids = [1, 2, 3, 2, 3]
        pair = (2, 3)
        new_token = 256

    Output:
        [1, 256, 256]
    """

    merged = []

    i = 0

    while i < len(token_ids):

        # Check if the current pair matches
        if (
            i < len(token_ids) - 1
            and token_ids[i] == pair[0]
            and token_ids[i + 1] == pair[1]
        ):
            merged.append(new_token)
            i += 2

        else:
            merged.append(token_ids[i])
            i += 1

    return merged

if __name__ == "__main__":

    text = "Hello GPT 😊"

    byte_ids = text_to_bytes(text)

    print(text)

    print(byte_ids)

    recovered = bytes_to_text(byte_ids)

    print(recovered)

    text = "banana"

    byte_ids = text_to_bytes(text)

    print("\nByte IDs")
    print(byte_ids)

    pair_counts = get_pair_frequencies(byte_ids)

    print("\nPair Frequencies")

    for pair, count in pair_counts.items():
        print(pair, "->", count)

    text = "banana"

    tokens = text_to_bytes(text)

    print("Original")
    print(tokens)

    pair_counts = get_pair_frequencies(tokens)

    print("\nPair Counts")
    print(pair_counts)

    best_pair = max(
        pair_counts,
        key=pair_counts.get,
    )

    print("\nBest Pair")
    print(best_pair)

    merged = merge_pair(
        tokens,
        best_pair,
        256,
    )

    print("\nMerged")
    print(merged)

    tokens = text_to_bytes(text)

    print("Original")
    print(tokens)

    pair_counts = get_pair_frequencies(tokens)

    print("\nPair Counts")
    print(pair_counts)

    best_pair = max(
        pair_counts,
        key=pair_counts.get,
    )

    print("\nBest Pair")
    print(best_pair)

    merged = merge_pair(
        tokens,
        best_pair,
        256,
    )

    print("\nMerged")
    print(merged)