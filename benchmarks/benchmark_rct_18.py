
import os
import sys
import time
import statistics

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from model.reusable_block import ReusableFFN


def median(values):
    return statistics.median(values)


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
    """
    Create a sequence containing exactly unique_count
    distinct token representations.
    """
    unique_tokens = torch.randn(
        1,
        unique_count,
        hidden_size
    )

    indices = torch.arange(sequence_length) % unique_count

    return unique_tokens.index_select(
        dim=1,
        index=indices
    )


def measure(function, ffn, x, trials):
    timings = []

    for _ in range(trials):
        start = time.perf_counter()

        function(ffn, x)

        elapsed = (time.perf_counter() - start) * 1000

        timings.append(elapsed)

    return median(timings)


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    sequence_length = 32
    trials = 60

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=64
    )

    ffn.eval()

    # Unique token counts determine duplicate ratios.
    unique_counts = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]

    results = []

    print("=" * 70)
    print("RCT 1.8.0 ADAPTIVE REUSE BENCHMARK")
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

            # Correctness validation.
            standard_output = run_standard(ffn, x)
            deduplicated_output = run_deduplicated(ffn, x)

            correct = torch.allclose(
                standard_output,
                deduplicated_output,
                rtol=1e-5,
                atol=1e-6
            )

            max_difference = (
                standard_output - deduplicated_output
            ).abs().max().item()

            if not correct:
                raise AssertionError(
                    f"Numerical mismatch at {unique_count} unique tokens"
                )

            # Warm-up both execution methods.
            for _ in range(10):
                run_standard(ffn, x)
                run_deduplicated(ffn, x)

            # Alternate measurement order.
            standard_times = []
            deduplicated_times = []

            for trial in range(trials):

                if trial % 2 == 0:

                    start = time.perf_counter()
                    run_standard(ffn, x)
                    standard_times.append(
                        (time.perf_counter() - start) * 1000
                    )

                    start = time.perf_counter()
                    run_deduplicated(ffn, x)
                    deduplicated_times.append(
                        (time.perf_counter() - start) * 1000
                    )

                else:

                    start = time.perf_counter()
                    run_deduplicated(ffn, x)
                    deduplicated_times.append(
                        (time.perf_counter() - start) * 1000
                    )

                    start = time.perf_counter()
                    run_standard(ffn, x)
                    standard_times.append(
                        (time.perf_counter() - start) * 1000
                    )

            standard_median = median(standard_times)
            deduplicated_median = median(deduplicated_times)

            speedup = standard_median / deduplicated_median

            reduction = (
                (standard_median - deduplicated_median)
                / standard_median
            ) * 100

            duplicate_ratio = (
                (sequence_length - unique_count)
                / sequence_length
            ) * 100

            results.append({
                "unique_count": unique_count,
                "duplicate_ratio": duplicate_ratio,
                "standard": standard_median,
                "deduplicated": deduplicated_median,
                "speedup": speedup,
                "reduction": reduction
            })

            print("\n" + "-" * 70)
            print(
                f"UNIQUE TOKENS: {unique_count}/{sequence_length}"
            )
            print("-" * 70)

            print(f"Duplicate ratio: {duplicate_ratio:.2f}%")
            print("Numerically equivalent:", correct)
            print(f"Maximum difference: {max_difference:.10f}")

            print(
                f"Standard median: {standard_median:.6f} ms"
            )

            print(
                f"Deduplicated median: {deduplicated_median:.6f} ms"
            )

            print(f"Speedup: {speedup:.3f}x")
            print(f"Latency reduction: {reduction:.2f}%")

    # Determine the lowest duplicate ratio that produced
    # a speedup greater than 1.0x in this experiment.
    successful = [
        result
        for result in results
        if result["speedup"] > 1.0
    ]

    print("\n" + "=" * 70)
    print("RCT 1.8.0 SUMMARY")
    print("=" * 70)

    if successful:
        lowest_threshold = min(
            successful,
            key=lambda item: item["duplicate_ratio"]
        )

        print(
            "Lowest tested duplicate ratio with speedup:",
            f"{lowest_threshold['duplicate_ratio']:.2f}%"
        )

        print(
            "Corresponding speedup:",
            f"{lowest_threshold['speedup']:.3f}x"
        )
    else:
        print("No tested workload achieved a speedup.")

    print("\nAll numerical correctness tests passed.")
    print("RCT 1.8.0 BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
