
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


def percentile_95(values):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[index]


def run_standard(ffn, x):
    return ffn.network(x)


def run_deduplicated(ffn, x):
    unique_tokens, inverse_indices = torch.unique(
        x,
        dim=1,
        return_inverse=True
    )

    unique_outputs = ffn.network(unique_tokens)

    return unique_outputs.index_select(
        dim=1,
        index=inverse_indices
    )


def create_workload(sequence_length, hidden_size, unique_count):
    unique_tokens = torch.randn(
        1, unique_count, hidden_size
    )

    indices = torch.arange(sequence_length) % unique_count

    return unique_tokens.index_select(1, indices)


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    sequence_length = 32
    trials = 100

    unique_counts = [8, 9, 10, 11, 12]

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=64
    )

    ffn.eval()

    results = []

    print("=" * 70)
    print("RCT 1.8.1 BREAK-EVEN BOUNDARY BENCHMARK")
    print("=" * 70)

    print("Sequence length:", sequence_length)
    print("Hidden size:", hidden_size)
    print("Trials:", trials)
    print("CPU threads:", torch.get_num_threads())

    with torch.inference_mode():

        for unique_count in unique_counts:

            x = create_workload(
                sequence_length,
                hidden_size,
                unique_count
            )

            standard_output = run_standard(ffn, x)
            deduplicated_output = run_deduplicated(ffn, x)

            correct = torch.allclose(
                standard_output,
                deduplicated_output,
                rtol=1e-5,
                atol=1e-6
            )

            if not correct:
                raise AssertionError(
                    f"Numerical mismatch at {unique_count} unique tokens"
                )

            # Warm up both execution paths.
            for _ in range(15):
                run_standard(ffn, x)
                run_deduplicated(ffn, x)

            standard_times = []
            deduplicated_times = []

            # Alternate order to reduce systematic measurement bias.
            for trial in range(trials):

                methods = [
                    run_standard,
                    run_deduplicated
                ]

                if trial % 2:
                    methods.reverse()

                for method in methods:
                    start = time.perf_counter()
                    method(ffn, x)
                    elapsed = (
                        time.perf_counter() - start
                    ) * 1000

                    if method == run_standard:
                        standard_times.append(elapsed)
                    else:
                        deduplicated_times.append(elapsed)

            standard_median = statistics.median(standard_times)
            deduplicated_median = statistics.median(
                deduplicated_times
            )

            speedup = standard_median / deduplicated_median

            duplicate_ratio = (
                (sequence_length - unique_count)
                / sequence_length
            ) * 100

            reduction = (
                (standard_median - deduplicated_median)
                / standard_median
            ) * 100

            results.append({
                "unique_count": unique_count,
                "duplicate_ratio": duplicate_ratio,
                "standard": standard_median,
                "deduplicated": deduplicated_median,
                "speedup": speedup
            })

            print("\n" + "-" * 70)
            print(f"UNIQUE TOKENS: {unique_count}/{sequence_length}")
            print("-" * 70)

            print(f"Duplicate ratio: {duplicate_ratio:.2f}%")
            print("Numerical correctness:", correct)

            print(
                f"Standard median: {standard_median:.6f} ms"
            )
            print(
                f"Standard P95: {percentile_95(standard_times):.6f} ms"
            )

            print(
                f"Deduplicated median: {deduplicated_median:.6f} ms"
            )
            print(
                f"Deduplicated P95: {percentile_95(deduplicated_times):.6f} ms"
            )

            print(f"Speedup: {speedup:.3f}x")
            print(f"Latency reduction: {reduction:.2f}%")

    print("\n" + "=" * 70)
    print("BREAK-EVEN SUMMARY")
    print("=" * 70)

    for result in results:
        status = (
            "DEDUPLICATION FASTER"
            if result["speedup"] > 1.0
            else "STANDARD FASTER"
        )

        print(
            f"{result['unique_count']:2d} unique | "
            f"{result['duplicate_ratio']:6.2f}% duplicates | "
            f"{result['speedup']:.3f}x | {status}"
        )

    print("\nAll correctness checks passed.")
    print("RCT 1.8.1 BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
