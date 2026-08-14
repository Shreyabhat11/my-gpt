"""
GPT Text Generation Demo.

Author: Shreya Bhat
"""

from __future__ import annotations

import torch

from configs.model_config import GPTConfig

from tokenizer.bpe import BPETokenizer

from model.gpt import GPT

from model.generation import generate


CORPUS_PATH = "data/corpus.txt"

TOKENIZER_PATH = (
    "artifacts/tokenizer.json"
)

CHECKPOINT_PATH = (
    "artifacts/checkpoints/latest.pt"
)


def main():

    # ==========================================================
    # Device
    # ==========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    prompt = (
        "The world"
    )

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
        max_new_tokens=100,
        context_length=config.context_length,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
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