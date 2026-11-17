
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model.transformer import RCTModel


def main():

    torch.manual_seed(42)
    torch.set_num_threads(4)

    model = RCTModel(
        vocab_size=1000,
        hidden_size=64,
        layers=2,
        heads=4,
        max_seq_len=128
    )

    model.eval()

    prompt_a = torch.tensor([[10, 20, 30, 40, 50, 60]])
    prompt_b = torch.tensor([[10, 20, 30, 40, 70, 80]])

    prefix_len = 4

    with torch.inference_mode():

        print("\n===== RCT 0.6 KV CACHE TEST =====")

        cached_a = model.forward_cached(
            prompt_a,
            prefix_len
        )

        cached_b = model.forward_cached(
            prompt_b,
            prefix_len
        )

        # Capture statistics before invalidating the cache.
        stats = model.prefix_cache_statistics()

        # Standard Transformer reference outputs.
        reference_a = model(prompt_a)
        reference_b = model(prompt_b)

        difference_a = (
            cached_a - reference_a
        ).abs().max().item()

        difference_b = (
            cached_b - reference_b
        ).abs().max().item()

        print("\n===== NUMERICAL CORRECTNESS =====")

        print("Prompt A maximum difference:", difference_a)
        print("Prompt B maximum difference:", difference_b)

        print(
            "Prompt A numerically close:",
            torch.allclose(
                cached_a,
                reference_a,
                rtol=1e-4,
                atol=1e-5
            )
        )

        print(
            "Prompt B numerically close:",
            torch.allclose(
                cached_b,
                reference_b,
                rtol=1e-4,
                atol=1e-5
            )
        )

        print("\n===== PREFIX CACHE STATISTICS =====")
        print(stats)

        print("\n===== OUTPUT SHAPES =====")
        print("Cached A:", tuple(cached_a.shape))
        print("Cached B:", tuple(cached_b.shape))

    print("\n===== TEST COMPLETE =====")


if __name__ == "__main__":
    main()
