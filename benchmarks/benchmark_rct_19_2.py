
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


def standard_ffn(ffn, x):
    return ffn.network(x)


def deduplicated_ffn(ffn, x):

    unique_tokens, inverse = torch.unique(
        x,
        dim=1,
        return_inverse=True
    )

    unique_output = ffn.network(unique_tokens)

    return unique_output.index_select(1, inverse)


def measure(function, trials=60):

    timings = []

    for _ in range(trials):

        start = time.perf_counter()

        function()

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        timings.append(elapsed)

    return statistics.median(timings)


def main():

    torch.manual_seed(42)
    random.seed(42)

    torch.set_num_threads(4)

    hidden_size = 256
    expansion = 4

    sequence_lengths = [64, 128, 256]

    duplicate_ratios = [
        0.50,
        0.625,
        0.75,
        0.875
    ]

    trials = 60

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=expansion
    )

    ffn.eval()

    print("=" * 80)
    print("RCT 1.9.2 SEQUENCE-LENGTH SCALABILITY BENCHMARK")
    print("=" * 80)

    print("Hidden size:", hidden_size)
    print("FFN expansion:", expansion)
    print("CPU threads:", torch.get_num_threads())
    print("Trials:", trials)

    results = []

    with torch.inference_mode():

        for sequence_length in sequence_lengths:

            for duplicate_ratio in duplicate_ratios:

                unique_count = max(
                    1,
                    round(
                        sequence_length *
                        (1.0 - duplicate_ratio)
                    )
                )

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

                reference = standard_ffn(ffn, x)

                deduplicated = deduplicated_ffn(
                    ffn,
                    x
                )

                correct = torch.allclose(
                    reference,
                    deduplicated,
                    rtol=1e-5,
                    atol=1e-6
                )

                if not correct:
                    raise AssertionError(
                        "Numerical correctness failed"
                    )

                # Warmup
                for _ in range(10):
                    standard_ffn(ffn, x)
                    deduplicated_ffn(ffn, x)

                standard_times = []
                deduplicated_times = []

                # Randomize measurement order.
                for _ in range(trials):

                    order = [
                        "standard",
                        "deduplicated"
                    ]

                    random.shuffle(order)

                    for method in order:

                        if method == "standard":

                            elapsed = measure(
                                lambda: standard_ffn(ffn, x),
                                trials=1
                            )

                            standard_times.append(elapsed)

                        else:

                            elapsed = measure(
                                lambda: deduplicated_ffn(ffn, x),
                                trials=1
                            )

                            deduplicated_times.append(elapsed)

                standard_median = statistics.median(
                    standard_times
                )

                deduplicated_median = statistics.median(
                    deduplicated_times
                )

                speedup = (
                    standard_median /
                    deduplicated_median
                )

                reduction = (
                    1.0 -
                    deduplicated_median /
                    standard_median
                ) * 100

                results.append({
                    "sequence_length": sequence_length,
                    "unique_count": unique_count,
                    "duplicate_ratio": duplicate_ratio,
                    "standard_ms": standard_median,
                    "deduplicated_ms": deduplicated_median,
                    "speedup": speedup,
                    "reduction": reduction
                })

                print("\n" + "-" * 80)

                print(
                    f"Sequence: {sequence_length} | "
                    f"Unique: {unique_count} | "
                    f"Duplicates: {duplicate_ratio * 100:.1f}%"
                )

                print(
                    f"Standard FFN: "
                    f"{standard_median:.6f} ms"
                )

                print(
                    f"Deduplicated FFN: "
                    f"{deduplicated_median:.6f} ms"
                )

                print(
                    f"Speedup: {speedup:.3f}x"
                )

                print(
                    f"Latency reduction: {reduction:.2f}%"
                )

                print(
                    "Numerical correctness:",
                    "PASSED"
                )

    print("\n" + "=" * 80)
    print("RCT 1.9.2 SUMMARY")
    print("=" * 80)

    for sequence_length in sequence_lengths:

        subset = [
            result
            for result in results
            if result["sequence_length"] == sequence_length
        ]

        average_speedup = statistics.mean(
            result["speedup"]
            for result in subset
        )

        faster_cases = sum(
            result["speedup"] > 1.0
            for result in subset
        )

        print(
            f"Sequence {sequence_length}: "
            f"Average speedup {average_speedup:.3f}x | "
            f"Faster cases {faster_cases}/{len(subset)}"
        )

    print("\nRCT 1.9.2 BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
