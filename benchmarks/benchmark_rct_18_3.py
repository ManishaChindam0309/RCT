
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


def deduplicated(ffn, x):
    unique_tokens, inverse = torch.unique(
        x,
        dim=1,
        return_inverse=True
    )

    unique_outputs = ffn.network(unique_tokens)

    return unique_outputs.index_select(1, inverse)


def make_input(sequence_length, hidden_size, unique_count):
    unique_tokens = torch.randn(
        1, unique_count, hidden_size
    )

    indices = torch.arange(sequence_length) % unique_count

    return unique_tokens.index_select(1, indices)


def measure(fn, ffn, x):
    start = time.perf_counter()
    fn(ffn, x)
    return (time.perf_counter() - start) * 1000


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    sequence_lengths = [16, 32, 64, 128]

    duplicate_targets = [0.50, 0.625, 0.75, 0.875]

    trials = 100

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=256
    )

    ffn.eval()

    print("=" * 75)
    print("RCT 1.8.3 SEQUENCE-LENGTH ADAPTIVE REUSE BENCHMARK")
    print("=" * 75)

    print("Hidden size:", hidden_size)
    print("Sequence lengths:", sequence_lengths)
    print("Duplicate targets:", duplicate_targets)
    print("Paired trials:", trials)
    print("CPU threads:", torch.get_num_threads())

    with torch.inference_mode():

        for sequence_length in sequence_lengths:

            print("\n" + "#" * 75)
            print("SEQUENCE LENGTH:", sequence_length)
            print("#" * 75)

            for duplicate_target in duplicate_targets:

                unique_count = max(
                    1,
                    round(sequence_length * (1 - duplicate_target))
                )

                unique_count = min(
                    unique_count,
                    sequence_length
                )

                x = make_input(
                    sequence_length,
                    hidden_size,
                    unique_count
                )

                expected = standard(ffn, x)
                actual = deduplicated(ffn, x)

                correct = torch.allclose(
                    expected,
                    actual,
                    rtol=1e-5,
                    atol=1e-6
                )

                if not correct:
                    raise AssertionError(
                        "Numerical mismatch"
                    )

                for _ in range(20):
                    standard(ffn, x)
                    deduplicated(ffn, x)

                standard_times = []
                deduplicated_times = []
                paired_speedups = []

                for _ in range(trials):

                    if random.random() < 0.5:

                        standard_time = measure(
                            standard, ffn, x
                        )

                        deduplicated_time = measure(
                            deduplicated, ffn, x
                        )

                    else:

                        deduplicated_time = measure(
                            deduplicated, ffn, x
                        )

                        standard_time = measure(
                            standard, ffn, x
                        )

                    standard_times.append(standard_time)
                    deduplicated_times.append(deduplicated_time)

                    paired_speedups.append(
                        standard_time / deduplicated_time
                    )

                speedup = statistics.median(
                    paired_speedups
                )

                duplicate_ratio = (
                    (sequence_length - unique_count)
                    / sequence_length
                ) * 100

                print(
                    f"\nUnique: {unique_count:3d}/{sequence_length:3d} | "
                    f"Duplicates: {duplicate_ratio:6.2f}%"
                )

                print(
                    f"Standard median: "
                    f"{statistics.median(standard_times):.6f} ms"
                )

                print(
                    f"Deduplicated median: "
                    f"{statistics.median(deduplicated_times):.6f} ms"
                )

                print(
                    f"Paired speedup: {speedup:.3f}x"
                )

                print(
                    "Result:",
                    "DEDUP FASTER" if speedup > 1.01
                    else "STANDARD FASTER" if speedup < 0.99
                    else "APPROXIMATELY EQUAL"
                )

                print("Numerical correctness:", correct)

    print("\n" + "=" * 75)
    print("RCT 1.8.3 BENCHMARK COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
