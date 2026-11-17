
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


def measure_once(function):

    start = time.perf_counter()

    function()

    return (
        time.perf_counter() - start
    ) * 1000


def make_input(sequence_length, unique_count, hidden_size):

    unique_tokens = torch.randn(
        1,
        unique_count,
        hidden_size
    )

    indices = (
        torch.arange(sequence_length) % unique_count
    )

    # Randomize the positions of repeated tokens.
    permutation = torch.randperm(sequence_length)

    indices = indices[permutation]

    return unique_tokens.index_select(
        1,
        indices
    )


def main():

    torch.manual_seed(42)
    random.seed(42)

    torch.set_num_threads(4)

    hidden_size = 256

    sequence_lengths = [64, 128, 256]

    duplicate_ratios = [
        0.50,
        0.625,
        0.75,
        0.875
    ]

    rounds = 5
    trials_per_round = 40

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4
    )

    ffn.eval()

    all_results = []

    print("=" * 80)
    print("RCT 1.9.3 REPEATED RANDOMIZED VALIDATION")
    print("=" * 80)

    print("Hidden size:", hidden_size)
    print("CPU threads:", torch.get_num_threads())
    print("Independent rounds:", rounds)
    print("Paired trials per round:", trials_per_round)

    with torch.inference_mode():

        for sequence_length in sequence_lengths:

            for target_ratio in duplicate_ratios:

                unique_count = max(
                    1,
                    round(
                        sequence_length *
                        (1.0 - target_ratio)
                    )
                )

                round_results = []

                for round_index in range(rounds):

                    x = make_input(
                        sequence_length,
                        unique_count,
                        hidden_size
                    )

                    reference = standard_ffn(ffn, x)

                    deduplicated = deduplicated_ffn(
                        ffn,
                        x
                    )

                    if not torch.allclose(
                        reference,
                        deduplicated,
                        rtol=1e-5,
                        atol=1e-6
                    ):
                        raise AssertionError(
                            "Numerical correctness failed"
                        )

                    # Warm up both execution paths.
                    for _ in range(10):
                        standard_ffn(ffn, x)
                        deduplicated_ffn(ffn, x)

                    standard_times = []
                    deduplicated_times = []

                    for _ in range(trials_per_round):

                        order = [
                            "standard",
                            "deduplicated"
                        ]

                        random.shuffle(order)

                        for method in order:

                            if method == "standard":

                                elapsed = measure_once(
                                    lambda: standard_ffn(ffn, x)
                                )

                                standard_times.append(elapsed)

                            else:

                                elapsed = measure_once(
                                    lambda: deduplicated_ffn(ffn, x)
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

                    round_results.append(speedup)

                    print(
                        f"\nSeq {sequence_length} | "
                        f"Unique {unique_count} | "
                        f"Round {round_index + 1}: "
                        f"Standard {standard_median:.5f} ms | "
                        f"Deduplicated {deduplicated_median:.5f} ms | "
                        f"Speedup {speedup:.3f}x | "
                        f"Reduction {reduction:.2f}%"
                    )

                median_speedup = statistics.median(
                    round_results
                )

                faster_rounds = sum(
                    speedup > 1.0
                    for speedup in round_results
                )

                all_results.append({
                    "sequence_length": sequence_length,
                    "unique_count": unique_count,
                    "duplicate_ratio": target_ratio,
                    "median_speedup": median_speedup,
                    "faster_rounds": faster_rounds
                })

                print(
                    f"\nSUMMARY: Sequence {sequence_length}, "
                    f"duplicates {target_ratio * 100:.1f}%"
                )

                print(
                    f"Median speedup across rounds: "
                    f"{median_speedup:.3f}x"
                )

                print(
                    f"Faster rounds: {faster_rounds}/{rounds}"
                )

    print("\n" + "=" * 80)
    print("RCT 1.9.3 FINAL SUMMARY")
    print("=" * 80)

    for sequence_length in sequence_lengths:

        subset = [
            result
            for result in all_results
            if result["sequence_length"] == sequence_length
        ]

        average_speedup = statistics.mean(
            result["median_speedup"]
            for result in subset
        )

        consistently_faster = sum(
            result["faster_rounds"] == rounds
            for result in subset
        )

        print(
            f"Sequence {sequence_length}: "
            f"Average speedup {average_speedup:.3f}x | "
            f"Consistently faster workloads "
            f"{consistently_faster}/{len(subset)}"
        )

    print("\nRCT 1.9.3 BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
