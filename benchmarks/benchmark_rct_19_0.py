
import os
import sys
import time
import random
import statistics

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from model.reusable_block import ReusableFFN


def measure(function, trials=100):

    samples = []

    for _ in range(trials):

        start = time.perf_counter()

        function()

        samples.append(
            (time.perf_counter() - start) * 1000
        )

    return statistics.median(samples)


def main():

    torch.manual_seed(42)
    random.seed(42)

    torch.set_num_threads(4)

    sequence_length = 32
    hidden_size = 256

    ffn = ReusableFFN(
        hidden_size=hidden_size
    )

    ffn.eval()

    unique_tokens = torch.randn(
        1,
        4,
        hidden_size
    )

    indices = (
        torch.arange(sequence_length) % 4
    )

    x = unique_tokens.index_select(
        1,
        indices
    )

    sample_indices = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28]
    )

    sample = x[0].index_select(
        0,
        sample_indices
    )

    feature_indices = torch.tensor(
        [0, 32, 64, 96, 128, 160, 192, 224]
    )

    print("=" * 75)
    print("RCT 1.9.0 SAMPLE OVERHEAD BENCHMARK")
    print("=" * 75)

    print("Sequence length:", sequence_length)
    print("Hidden size:", hidden_size)
    print("Sample size:", len(sample_indices))
    print("Selected features:", len(feature_indices))

    with torch.inference_mode():

        operations = {

            "Standard FFN": lambda:
                ffn.network(x),

            "Full torch.unique": lambda:
                torch.unique(x[0], dim=0),

            "Full vector sample comparison": lambda:
                torch.unique(sample, dim=0),

            "Selected-feature sample": lambda:
                torch.unique(
                    sample.index_select(
                        1,
                        feature_indices
                    ),
                    dim=0
                ),

            "Selected-feature equality": lambda:
                (
                    sample[:, None, feature_indices]
                    ==
                    sample[None, :, feature_indices]
                ).all(dim=-1)
        }

        results = {}

        for name, function in operations.items():

            # Warmup
            for _ in range(20):
                function()

            results[name] = measure(
                function,
                trials=100
            )

        baseline = results["Standard FFN"]

        for name, latency in results.items():

            print(
                f"{name:35s}: "
                f"{latency:.6f} ms | "
                f"Relative time: {latency / baseline:.3f}x"
            )

    print("\n" + "=" * 75)
    print("RCT 1.9.0 BENCHMARK COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
