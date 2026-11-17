
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


def estimate_duplicate_ratio(
    x,
    sample_size=8,
    feature_sample_size=8
):

    sequence_length = x.shape[1]
    hidden_size = x.shape[2]

    token_indices = torch.linspace(
        0,
        sequence_length - 1,
        steps=min(sample_size, sequence_length)
    ).long()

    feature_indices = torch.linspace(
        0,
        hidden_size - 1,
        steps=min(feature_sample_size, hidden_size)
    ).long()

    sampled = x.index_select(
        1,
        token_indices
    )

    sampled = sampled.index_select(
        2,
        feature_indices
    )

    sampled = sampled.reshape(
        -1,
        sampled.shape[-1]
    )

    unique_sample = torch.unique(
        sampled,
        dim=0
    )

    duplicate_ratio = (
        sampled.shape[0] - unique_sample.shape[0]
    ) / sampled.shape[0]

    return duplicate_ratio


def adaptive_ffn(
    ffn,
    x,
    threshold=0.625
):

    estimated_ratio = estimate_duplicate_ratio(x)

    # Reject reuse when the sample does not meet the threshold.
    if estimated_ratio < threshold:

        return (
            standard_ffn(ffn, x),
            "standard",
            estimated_ratio
        )

    # Sample accepted: perform exact full-vector deduplication.
    unique_tokens, inverse = torch.unique(
        x,
        dim=1,
        return_inverse=True
    )

    actual_ratio = (
        x.shape[1] - unique_tokens.shape[1]
    ) / x.shape[1]

    # Verify the actual duplication ratio before reusing.
    if actual_ratio < threshold:

        return (
            standard_ffn(ffn, x),
            "standard_after_check",
            estimated_ratio
        )

    unique_output = ffn.network(unique_tokens)

    output = unique_output.index_select(
        1,
        inverse
    )

    return (
        output,
        "deduplicated",
        estimated_ratio
    )


def measure_once(function):

    start = time.perf_counter()

    function()

    return (
        time.perf_counter() - start
    ) * 1000


def median_measurement(function, trials=30):

    timings = []

    for _ in range(trials):
        timings.append(
            measure_once(function)
        )

    return statistics.median(timings)


def make_input(
    sequence_length,
    unique_count,
    hidden_size
):

    unique_tokens = torch.randn(
        1,
        unique_count,
        hidden_size
    )

    indices = (
        torch.arange(sequence_length) % unique_count
    )

    # Randomize token positions.
    indices = indices[
        torch.randperm(sequence_length)
    ]

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

    threshold = 0.625

    rounds = 5
    trials_per_round = 30

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4
    )

    ffn.eval()

    print("=" * 85)
    print("RCT 1.9.4 COMPLETE ADAPTIVE REUSE BENCHMARK")
    print("=" * 85)

    print("Hidden size:", hidden_size)
    print("CPU threads:", torch.get_num_threads())
    print("Adaptive threshold:", threshold)
    print("Independent rounds:", rounds)
    print("Paired trials per round:", trials_per_round)

    all_results = []

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

                    reference = standard_ffn(
                        ffn,
                        x
                    )

                    deduplicated = deduplicated_ffn(
                        ffn,
                        x
                    )

                    adaptive_output, adaptive_mode, estimate = (
                        adaptive_ffn(
                            ffn,
                            x,
                            threshold
                        )
                    )

                    if not torch.allclose(
                        reference,
                        deduplicated,
                        rtol=1e-5,
                        atol=1e-6
                    ):
                        raise AssertionError(
                            "Deduplication correctness failed"
                        )

                    if not torch.allclose(
                        reference,
                        adaptive_output,
                        rtol=1e-5,
                        atol=1e-6
                    ):
                        raise AssertionError(
                            "Adaptive correctness failed"
                        )

                    # Warmup all methods.
                    for _ in range(10):
                        standard_ffn(ffn, x)
                        deduplicated_ffn(ffn, x)
                        adaptive_ffn(ffn, x, threshold)

                    standard_times = []
                    deduplicated_times = []
                    adaptive_times = []

                    for _ in range(trials_per_round):

                        order = [
                            "standard",
                            "deduplicated",
                            "adaptive"
                        ]

                        random.shuffle(order)

                        for method in order:

                            if method == "standard":

                                elapsed = measure_once(
                                    lambda: standard_ffn(ffn, x)
                                )

                                standard_times.append(elapsed)

                            elif method == "deduplicated":

                                elapsed = measure_once(
                                    lambda: deduplicated_ffn(ffn, x)
                                )

                                deduplicated_times.append(elapsed)

                            else:

                                elapsed = measure_once(
                                    lambda: adaptive_ffn(
                                        ffn,
                                        x,
                                        threshold
                                    )
                                )

                                adaptive_times.append(elapsed)

                    standard_ms = statistics.median(
                        standard_times
                    )

                    deduplicated_ms = statistics.median(
                        deduplicated_times
                    )

                    adaptive_ms = statistics.median(
                        adaptive_times
                    )

                    dedup_speedup = (
                        standard_ms / deduplicated_ms
                    )

                    adaptive_speedup = (
                        standard_ms / adaptive_ms
                    )

                    adaptive_reduction = (
                        1.0 - adaptive_ms / standard_ms
                    ) * 100

                    round_results.append({
                        "dedup_speedup": dedup_speedup,
                        "adaptive_speedup": adaptive_speedup
                    })

                    print("\n" + "-" * 85)

                    print(
                        f"Sequence: {sequence_length} | "
                        f"Unique: {unique_count} | "
                        f"Duplicates: {target_ratio * 100:.1f}% | "
                        f"Round: {round_index + 1}"
                    )

                    print(
                        f"Sample estimated duplicates: "
                        f"{estimate * 100:.1f}%"
                    )

                    print(
                        f"Adaptive decision: {adaptive_mode}"
                    )

                    print(
                        f"Standard: {standard_ms:.5f} ms"
                    )

                    print(
                        f"Always deduplicate: "
                        f"{deduplicated_ms:.5f} ms"
                    )

                    print(
                        f"Adaptive complete: "
                        f"{adaptive_ms:.5f} ms"
                    )

                    print(
                        f"Dedup speedup: "
                        f"{dedup_speedup:.3f}x"
                    )

                    print(
                        f"Adaptive speedup: "
                        f"{adaptive_speedup:.3f}x"
                    )

                    print(
                        f"Adaptive latency reduction: "
                        f"{adaptive_reduction:.2f}%"
                    )

                    print(
                        "Numerical correctness: PASSED"
                    )

                median_dedup_speedup = statistics.median(
                    result["dedup_speedup"]
                    for result in round_results
                )

                median_adaptive_speedup = statistics.median(
                    result["adaptive_speedup"]
                    for result in round_results
                )

                adaptive_faster_rounds = sum(
                    result["adaptive_speedup"] > 1.0
                    for result in round_results
                )

                all_results.append({
                    "sequence_length": sequence_length,
                    "duplicate_ratio": target_ratio,
                    "dedup_speedup": median_dedup_speedup,
                    "adaptive_speedup": median_adaptive_speedup,
                    "adaptive_faster_rounds": adaptive_faster_rounds
                })

                print("\nWORKLOAD SUMMARY")

                print(
                    f"Median dedup speedup: "
                    f"{median_dedup_speedup:.3f}x"
                )

                print(
                    f"Median adaptive speedup: "
                    f"{median_adaptive_speedup:.3f}x"
                )

                print(
                    f"Adaptive faster rounds: "
                    f"{adaptive_faster_rounds}/{rounds}"
                )

    print("\n" + "=" * 85)
    print("RCT 1.9.4 FINAL SUMMARY")
    print("=" * 85)

    for sequence_length in sequence_lengths:

        subset = [
            result
            for result in all_results
            if result["sequence_length"] == sequence_length
        ]

        average_adaptive_speedup = statistics.mean(
            result["adaptive_speedup"]
            for result in subset
        )

        successful_workloads = sum(
            result["adaptive_faster_rounds"] == rounds
            for result in subset
        )

        print(
            f"Sequence {sequence_length}: "
            f"Average adaptive speedup "
            f"{average_adaptive_speedup:.3f}x | "
            f"Consistently faster workloads "
            f"{successful_workloads}/{len(subset)}"
        )

    print("\nRCT 1.9.4 BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
