"""
End-to-end smoke test for the GPT pipeline.

Author: Shreya Bhat
"""

from pathlib import Path

import torch

from configs.model_config import GPTConfig
from tokenizer.bpe import BPETokenizer
from model.gpt import GPT
from model.generation import generate


TOKENIZER_PATH = Path(
    "artifacts/tokenizer.json"
)

CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/best.pt"
)


def main():

    print("=" * 60)
    print("GPT END-TO-END TEST")
    print("=" * 60)

    # ----------------------------------------------------------
    # Verify artifacts
    # ----------------------------------------------------------

    assert TOKENIZER_PATH.exists(), (
        "Tokenizer artifact missing."
    )

    assert CHECKPOINT_PATH.exists(), (
        "Best checkpoint missing."
    )

    # ----------------------------------------------------------
    # Device
    # ----------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ----------------------------------------------------------
    # Tokenizer
    # ----------------------------------------------------------

    tokenizer = BPETokenizer()

    tokenizer.load(
        TOKENIZER_PATH
    )

    assert tokenizer.vocab_size == 256

    print(
        "Tokenizer: PASS"
    )

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=32,
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        bias=False,
    )

    model = GPT(
        config
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print(
        "Checkpoint: PASS"
    )

    # ----------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------

    prompt = "The world"

    input_ids = tokenizer.encode(
        prompt
    )

    input_ids = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():

        logits, loss = model(
            input_ids
        )

    assert logits.shape[0] == 1
    assert logits.shape[-1] == config.vocab_size

    print(
        "Forward pass: PASS"
    )

    # ----------------------------------------------------------
    # Generation
    # ----------------------------------------------------------

    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=20,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    assert output_ids.shape[0] == 1
    assert output_ids.shape[1] > input_ids.shape[1]

    generated_text = tokenizer.decode(
        output_ids[0].tolist()
    )

    print(
        "Generation: PASS"
    )

    # ----------------------------------------------------------
    # Result
    # ----------------------------------------------------------

    print()
    print("Prompt:")
    print(prompt)

    print()
    print("Generated:")
    print(generated_text)

    print()
    print("=" * 60)
    print("ALL END-TO-END TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()