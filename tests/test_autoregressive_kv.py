
import sys
from pathlib import Path

import torch

# Add the project root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import RCTModel


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)

    model = RCTModel(
        vocab_size=256,
        hidden_size=64,
        layers=2,
        heads=4,
        max_seq_len=64
    )

    model.eval()

    # Disable FFN caching to isolate KV-cache correctness.
    for block in model.blocks:
        block.ffn.train()

    prompt = torch.tensor(
        [[10, 20, 30, 40, 50, 60, 70, 80]],
        dtype=torch.long
    )

    generated_tokens = [90, 91, 92, 93, 94]

    print("\n===== RCT 1.3 AUTOREGRESSIVE KV TEST =====")

    with torch.inference_mode():
        cached_logits, cache = model.create_generation_cache(prompt)

        current_sequence = prompt.clone()

        print("Initial cache length:", cache["length"])

        for index, token in enumerate(generated_tokens):
            new_token = torch.tensor(
                [[token]],
                dtype=torch.long
            )

            current_sequence = torch.cat(
                [current_sequence, new_token],
                dim=1
            )

            cached_logits, cache = model.decode_step(
                new_token,
                cache
            )

            full_logits = model(current_sequence)
            reference_logits = full_logits[:, -1, :]

            max_difference = (
                cached_logits - reference_logits
            ).abs().max().item()

            correct = torch.allclose(
                cached_logits,
                reference_logits,
                rtol=1e-4,
                atol=1e-4
            )

            print(
                f"Step {index + 1}: "
                f"max difference={max_difference:.8f}, "
                f"correct={correct}, "
                f"cache length={cache['length']}"
            )

            assert correct, (
                f"Incremental logits mismatch at step {index + 1}"
            )

    print("\n===== GENERATION TEST =====")

    torch.manual_seed(100)

    generated = model.generate_cached(
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        top_k=10
    )

    print("Input shape:", tuple(prompt.shape))
    print("Generated shape:", tuple(generated.shape))
    print("Generated tokens:", generated.tolist())

    assert generated.shape == (1, 13)

    print("\nAUTOREGRESSIVE KV CACHE TEST PASSED")


if __name__ == "__main__":
    main()
