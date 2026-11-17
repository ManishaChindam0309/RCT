
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


def make_input(sequence_length, unique_count, hidden_size):

    unique_tokens = torch.randn(
        unique_count,
        hidden_size
    )

    indices = (
        torch.arange(sequence_length) % unique_count
    )

    indices = indices[
        torch.randperm(sequence_length)
    ]

    return unique_tokens.index_select(0, indices)


def estimate_duplicates(x, sample_size):

    sequence_length = x.shape[0]

    sample_indices = torch.randperm(
        sequence_length
    )[:sample_size]

    sample = x.index_select(
        0,
        sample_indices
    )

    # Compare complete token vectors.
    equal_matrix = (
        sample[:, None, :] == sample[None, :, :]
    ).all(dim=-1)

    # Exclude diagonal self-matches.
    sample_count = sample.shape[0]

    collision_count = (
        equal_matrix.sum().item() - sample_count
    )

    pair_count = sample_count * (sample_count - 1)

    collision_probability = (
        collision_count / pair_count
    )

    if collision_probability == 0:
        estimated_unique = sequence_length
    else:
        estimated_unique = round(
            1.0 / collision_probability
        )

    estimated_unique = max(
        1,
        min(sequence_length, estimated_unique)
    )

    estimated_duplicate_ratio = (
        sequence_length - estimated_unique
    ) / sequence_length

    return estimated_duplicate_ratio


def measure_detector(x, sample_size, trials=30):

    timings = []
    estimates = []

    for _ in range(trials):

        start = time.perf_counter()

        estimate = estimate_duplicates(
            x,
            sample_size
        )

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        timings.append(elapsed)
        estimates.append(estimate)

    return (
        statistics.median(timings),
        statistics.median(estimates)
    )


def main():

    torch.manual_seed(42)
    torch.set_num_threads(4)

    hidden_size = 256

    sequence_lengths = [64, 128, 256]

    duplicate_ratios = [
        0.50,
        0.625,
        0.75,
        0.875
    ]

    sample_sizes = [8, 16, 32, 64]

    rounds = 5
    trials = 30

    print("=" * 85)
    print("RCT 1.9.5 DUPLICATE DETECTOR CALIBRATION")
    print("=" * 85)

    print("Hidden size:", hidden_size)
    print("CPU threads:", torch.get_num_threads())
    print("Rounds:", rounds)
    print("Detector trials per round:", trials)

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

                x = make_input(
                    sequence_length,
                    unique_count,
                    hidden_size
                )

                exact_unique = torch.unique(
                    x,
                    dim=0
                ).shape[0]

                actual_ratio = (
                    sequence_length - exact_unique
                ) / sequence_length

                print("\n" + "=" * 85)

                print(
                    f"Sequence: {sequence_length} | "
                    f"Actual unique: {exact_unique} | "
                    f"Actual duplicates: "
                    f"{actual_ratio * 100:.1f}%"
                )

                for sample_size in sample_sizes:

                    if sample_size > sequence_length:
                        continue

                    round_estimates = []
                    round_timings = []

                    for _ in range(rounds):

                        median_ms, estimated_ratio = (
                            measure_detector(
                                x,
                                sample_size,
                                trials
                            )
                        )

                        round_estimates.append(
                            estimated_ratio
                        )

                        round_timings.append(
                            median_ms
                        )

                    median_estimate = statistics.median(
                        round_estimates
                    )

                    median_time = statistics.median(
                        round_timings
                    )

                    estimation_error = abs(
                        median_estimate - actual_ratio
                    ) * 100

                    accepted_rounds = sum(
                        estimate >= 0.625
                        for estimate in round_estimates
                    )

                    print(
                        f"\nSample size: {sample_size}"
                    )

                    print(
                        f"Estimated duplicates: "
                        f"{median_estimate * 100:.1f}%"
                    )

                    print(
                        f"Actual duplicates: "
                        f"{actual_ratio * 100:.1f}%"
                    )

                    print(
                        f"Absolute estimation error: "
                        f"{estimation_error:.2f} percentage points"
                    )

                    print(
                        f"Median detector time: "
                        f"{median_time:.5f} ms"
                    )

                    print(
                        f"Threshold acceptance: "
                        f"{accepted_rounds}/{rounds} rounds"
                    )

    print("\n" + "=" * 85)
    print("RCT 1.9.5 CALIBRATION COMPLETED")
    print("=" * 85)


if __name__ == "__main__":
    main()
