"""
GPT Text Generation CLI.

Author: Shreya Bhat
"""

from __future__ import annotations

import argparse

import torch

from configs.model_config import GPTConfig
from tokenizer.bpe import BPETokenizer
from model.gpt import GPT
from model.generation import generate


TOKENIZER_PATH = "artifacts/tokenizer.json"

CHECKPOINT_PATH = (
    "artifacts/checkpoints/latest.pt"
)


def parse_args():
    """
    Parse command-line generation arguments.
    """

    parser = argparse.ArgumentParser(
        description="Generate text using the trained GPT model."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="The world",
        help="Text prompt used for generation.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling value.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling value.",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalty applied to previously generated tokens.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ==========================================================
    # Device
    # ==========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ==========================================================
    # Tokenizer
    # ==========================================================

    tokenizer = BPETokenizer()

    tokenizer.load(
        TOKENIZER_PATH
    )

    # ==========================================================
    # Model Configuration
    # ==========================================================

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        bias=False,
    )

    # ==========================================================
    # Model
    # ==========================================================

    model = GPT(
        config
    )

    # ==========================================================
    # Load Checkpoint
    # ==========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)

    model.eval()

    # ==========================================================
    # Prompt
    # ==========================================================

    prompt = args.prompt

    # ==========================================================
    # Encode
    # ==========================================================

    input_ids = tokenizer.encode(
        prompt
    )

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    # ==========================================================
    # Generate
    # ==========================================================

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # ==========================================================
    # Decode
    # ==========================================================

    generated_text = tokenizer.decode(
        output_ids[0].tolist()
    )

    # ==========================================================
    # Output
    # ==========================================================

    print("=" * 60)

    print(
        "Prompt:"
    )

    print(prompt)

    print()

    print(
        "Generated:"
    )

    print(generated_text)

    print("=" * 60)


if __name__ == "__main__":

    main()