
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


def standard(ffn, x):
    return ffn.network(x)


def unique_dedup(ffn, x):
    unique_tokens, inverse = torch.unique(
        x,
        dim=1,
        return_inverse=True
    )

    output = ffn.network(unique_tokens)

    return output.index_select(1, inverse)


def sort_dedup(ffn, x):
    batch_size, sequence_length, hidden_size = x.shape

    outputs = []

    for batch_index in range(batch_size):

        tokens = x[batch_index]

        # Sort token vectors lexicographically by converting
        # each vector into a stable scalar key.
        # This is an experimental implementation.
        sorted_indices = torch.arange(
            sequence_length,
            device=x.device
        )

        for feature_index in range(hidden_size - 1, -1, -1):

            order = torch.argsort(
                tokens[sorted_indices, feature_index],
                stable=True
            )

            sorted_indices = sorted_indices[order]

        sorted_tokens = tokens[sorted_indices]

        equal_previous = torch.zeros(
            sequence_length,
            dtype=torch.bool,
            device=x.device
        )

        if sequence_length > 1:
            equal_previous[1:] = (
                sorted_tokens[1:] == sorted_tokens[:-1]
            ).all(dim=-1)

        group_ids = torch.cumsum(
            (~equal_previous).to(torch.long),
            dim=0
        ) - 1

        unique_positions = torch.nonzero(
            ~equal_previous,
            as_tuple=False
        ).flatten()

        unique_tokens = sorted_tokens.index_select(
            0,
            unique_positions
        )

        unique_output = ffn.network(
            unique_tokens.unsqueeze(0)
        ).squeeze(0)

        sorted_output = unique_output.index_select(
            0,
            group_ids
        )

        restored = torch.empty_like(sorted_output)

        restored[sorted_indices] = sorted_output

        outputs.append(restored)

    return torch.stack(outputs, dim=0)


def measure(function, ffn, x):
    start = time.perf_counter()
    function(ffn, x)
    return (time.perf_counter() - start) * 1000


def main():

    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    sequence_length = 32

    unique_counts = [4, 8, 12, 16, 24, 32]

    trials = 60

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4
    )

    ffn.eval()

    methods = {
        "Standard": standard,
        "Torch unique": unique_dedup,
        "Sort dedup": sort_dedup
    }

    print("=" * 75)
    print("RCT 1.8.8 DEDUPLICATION ALGORITHM COMPARISON")
    print("=" * 75)

    print("Sequence length:", sequence_length)
    print("Hidden size:", hidden_size)
    print("Trials:", trials)
    print("CPU threads:", torch.get_num_threads())

    with torch.inference_mode():

        for unique_count in unique_counts:

            unique_tokens = torch.randn(
                1,
                unique_count,
                hidden_size
            )

            indices = (
                torch.arange(sequence_length)
                % unique_count
            )

            x = unique_tokens.index_select(
                1,
                indices
            )

            reference = standard(ffn, x)

            print("\n" + "-" * 75)
            print(
                f"UNIQUE TOKENS: {unique_count}/{sequence_length}"
            )
            print("-" * 75)

            timings = {
                name: []
                for name in methods
            }

            for name, function in methods.items():

                output = function(ffn, x)

                correct = torch.allclose(
                    reference,
                    output,
                    rtol=1e-5,
                    atol=1e-6
                )

                print(
                    f"{name} correctness: {correct}"
                )

                if not correct:
                    raise AssertionError(
                        f"{name} produced incorrect output"
                    )

                for _ in range(10):
                    function(ffn, x)

            for _ in range(trials):

                order = list(methods.items())
                random.shuffle(order)

                for name, function in order:

                    timings[name].append(
                        measure(function, ffn, x)
                    )

            standard_median = statistics.median(
                timings["Standard"]
            )

            for name in methods:

                median = statistics.median(
                    timings[name]
                )

                speedup = standard_median / median

                print(
                    f"{name:15s}: "
                    f"{median:.6f} ms | "
                    f"Speedup: {speedup:.3f}x"
                )

    print("\n" + "=" * 75)
    print("RCT 1.8.8 BENCHMARK COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
