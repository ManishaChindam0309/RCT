
import copy
import random
import statistics
import time

import torch
import torch.nn as nn


SEED = 197
HIDDEN_SIZE = 256
EXPANSION = 4
CPU_THREADS = 4
ROUNDS = 5
TRIALS = 40

SEQUENCE_LENGTHS = [64, 128, 256]
DUPLICATE_RATIOS = [0.50, 0.625, 0.75, 0.875]


class StandardFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * EXPANSION),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE * EXPANSION, HIDDEN_SIZE)
        )

    def forward(self, x):
        return self.network(x)


class DeduplicatedFFN(nn.Module):
    def __init__(self, standard):
        super().__init__()
        self.network = copy.deepcopy(standard.network)

    def forward(self, x):
        batch, sequence, hidden = x.shape

        unique, inverse = torch.unique(
            x.reshape(batch * sequence, hidden),
            dim=0,
            return_inverse=True
        )

        result = self.network(unique)

        return result.index_select(
            0, inverse
        ).reshape(batch, sequence, hidden)


def make_random_input(sequence_length, duplicate_ratio, seed):
    generator = torch.Generator().manual_seed(seed)

    unique_count = max(
        1,
        round(sequence_length * (1.0 - duplicate_ratio))
    )

    unique_tokens = torch.randn(
        unique_count,
        HIDDEN_SIZE,
        generator=generator
    )

    indices = torch.randint(
        0,
        unique_count,
        (sequence_length,),
        generator=generator
    )

    # Ensure every unique token occurs at least once.
    indices[:unique_count] = torch.arange(unique_count)

    # Randomize positions rather than using a periodic pattern.
    permutation = torch.randperm(
        sequence_length,
        generator=generator
    )

    indices = indices[permutation]

    x = unique_tokens[indices].unsqueeze(0)

    actual_unique = torch.unique(
        x.reshape(-1, HIDDEN_SIZE),
        dim=0
    ).shape[0]

    actual_duplicate_ratio = (
        sequence_length - actual_unique
    ) / sequence_length

    return x, actual_unique, actual_duplicate_ratio


def measure_ms(function):
    start = time.perf_counter()
    function()
    return (time.perf_counter() - start) * 1000


def paired_benchmark(standard, deduplicated, x):
    standard_times = []
    dedup_times = []

    with torch.inference_mode():
        # Warmup
        for _ in range(10):
            standard(x)
            deduplicated(x)

        for trial in range(TRIALS):
            if trial % 2 == 0:
                standard_times.append(
                    measure_ms(lambda: standard(x))
                )
                dedup_times.append(
                    measure_ms(lambda: deduplicated(x))
                )
            else:
                dedup_times.append(
                    measure_ms(lambda: deduplicated(x))
                )
                standard_times.append(
                    measure_ms(lambda: standard(x))
                )

    standard_median = statistics.median(standard_times)
    dedup_median = statistics.median(dedup_times)

    return standard_median, dedup_median


def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(CPU_THREADS)

    standard = StandardFFN().eval()
    deduplicated = DeduplicatedFFN(standard).eval()

    print("=" * 85)
    print("RCT 1.9.7 RANDOMIZED INPUT-PATTERN VALIDATION")
    print("=" * 85)
    print("Hidden size:", HIDDEN_SIZE)
    print("CPU threads:", CPU_THREADS)
    print("Independent rounds:", ROUNDS)
    print("Paired trials per round:", TRIALS)

    all_results = []

    for sequence_length in SEQUENCE_LENGTHS:
        for duplicate_ratio in DUPLICATE_RATIOS:

            round_speedups = []
            round_reductions = []

            print("\n" + "-" * 85)
            print(
                f"Sequence {sequence_length} | "
                f"Target duplicates {duplicate_ratio:.1%}"
            )

            for round_number in range(ROUNDS):
                x, unique_count, actual_ratio = make_random_input(
                    sequence_length,
                    duplicate_ratio,
                    SEED + sequence_length * 100
                    + round_number * 10
                    + int(duplicate_ratio * 1000)
                )

                with torch.inference_mode():
                    expected = standard(x)
                    actual = deduplicated(x)

                correct = torch.allclose(
                    expected,
                    actual,
                    rtol=1e-4,
                    atol=1e-5
                )

                if not correct:
                    raise RuntimeError(
                        "Numerical correctness failed"
                    )

                standard_ms, dedup_ms = paired_benchmark(
                    standard,
                    deduplicated,
                    x
                )

                speedup = standard_ms / dedup_ms
                reduction = (
                    1 - dedup_ms / standard_ms
                ) * 100

                round_speedups.append(speedup)
                round_reductions.append(reduction)

                print(
                    f"Round {round_number + 1}: "
                    f"Unique {unique_count} | "
                    f"Actual duplicates {actual_ratio:.1%} | "
                    f"Standard {standard_ms:.5f} ms | "
                    f"Dedup {dedup_ms:.5f} ms | "
                    f"Speedup {speedup:.3f}x | "
                    f"Reduction {reduction:.2f}% | "
                    f"Correct {correct}"
                )

            median_speedup = statistics.median(
                round_speedups
            )

            median_reduction = statistics.median(
                round_reductions
            )

            faster_rounds = sum(
                value > 1.0 for value in round_speedups
            )

            all_results.append({
                "sequence": sequence_length,
                "duplicates": duplicate_ratio,
                "speedup": median_speedup,
                "reduction": median_reduction,
                "faster_rounds": faster_rounds
            })

            print("SUMMARY:")
            print(
                f"Median speedup: {median_speedup:.3f}x"
            )
            print(
                f"Median latency reduction: "
                f"{median_reduction:.2f}%"
            )
            print(
                f"Faster rounds: {faster_rounds}/{ROUNDS}"
            )

    print("\n" + "=" * 85)
    print("RCT 1.9.7 FINAL SUMMARY")
    print("=" * 85)

    for sequence_length in SEQUENCE_LENGTHS:
        selected = [
            item for item in all_results
            if item["sequence"] == sequence_length
        ]

        average_speedup = statistics.mean(
            item["speedup"] for item in selected
        )

        consistently_faster = sum(
            item["faster_rounds"] == ROUNDS
            for item in selected
        )

        print(
            f"Sequence {sequence_length}: "
            f"Average median speedup {average_speedup:.3f}x | "
            f"Consistently faster workloads "
            f"{consistently_faster}/{len(selected)}"
        )

    print("\nRCT 1.9.7 BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
