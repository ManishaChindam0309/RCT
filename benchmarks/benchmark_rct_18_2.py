
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


def measure_once(function, ffn, x):
    start = time.perf_counter()
    function(ffn, x)
    return (time.perf_counter() - start) * 1000


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    sequence_length = 32

    unique_counts = [8, 9, 10, 11, 12]

    rounds = 5
    pairs_per_round = 100

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=64
    )

    ffn.eval()

    print("=" * 70)
    print("RCT 1.8.2 REPEATED BREAK-EVEN VALIDATION")
    print("=" * 70)

    print("Sequence length:", sequence_length)
    print("Hidden size:", hidden_size)
    print("Rounds:", rounds)
    print("Paired measurements per round:", pairs_per_round)
    print("CPU threads:", torch.get_num_threads())

    with torch.inference_mode():

        for unique_count in unique_counts:

            x = create_workload(
                sequence_length,
                hidden_size,
                unique_count
            )

            # Verify numerical equivalence.
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
                    f"Numerical mismatch: {unique_count}"
                )

            # Warm up both implementations.
            for _ in range(20):
                run_standard(ffn, x)
                run_deduplicated(ffn, x)

            all_standard = []
            all_deduplicated = []
            all_paired_speedups = []
            round_speedups = []

            for round_number in range(rounds):

                standard_times = []
                deduplicated_times = []
                paired_speedups = []

                for _ in range(pairs_per_round):

                    # Randomize measurement order.
                    if random.random() < 0.5:

                        standard_time = measure_once(
                            run_standard, ffn, x
                        )

                        deduplicated_time = measure_once(
                            run_deduplicated, ffn, x
                        )

                    else:

                        deduplicated_time = measure_once(
                            run_deduplicated, ffn, x
                        )

                        standard_time = measure_once(
                            run_standard, ffn, x
                        )

                    standard_times.append(standard_time)
                    deduplicated_times.append(deduplicated_time)

                    paired_speedups.append(
                        standard_time / deduplicated_time
                    )

                standard_median = statistics.median(
                    standard_times
                )

                deduplicated_median = statistics.median(
                    deduplicated_times
                )

                paired_speedup = statistics.median(
                    paired_speedups
                )

                round_speedups.append(paired_speedup)

                all_standard.extend(standard_times)
                all_deduplicated.extend(deduplicated_times)
                all_paired_speedups.extend(paired_speedups)

                print(
                    f"\nUnique {unique_count} | "
                    f"Round {round_number + 1}/{rounds}"
                )

                print(
                    f"Standard: {standard_median:.6f} ms | "
                    f"Deduplicated: {deduplicated_median:.6f} ms | "
                    f"Paired speedup: {paired_speedup:.3f}x"
                )

            overall_speedup = statistics.median(
                all_paired_speedups
            )

            duplicate_ratio = (
                (sequence_length - unique_count)
                / sequence_length
            ) * 100

            print("\n" + "-" * 70)
            print(f"SUMMARY: {unique_count} UNIQUE TOKENS")
            print("-" * 70)

            print(f"Duplicate ratio: {duplicate_ratio:.2f}%")

            print(
                f"Overall standard median: "
                f"{statistics.median(all_standard):.6f} ms"
            )

            print(
                f"Overall deduplicated median: "
                f"{statistics.median(all_deduplicated):.6f} ms"
            )

            print(
                f"Median paired speedup: {overall_speedup:.3f}x"
            )

            print(
                f"Round speedups: "
                f"{[round(v, 3) for v in round_speedups]}"
            )

            print("Numerical correctness:", correct)

    print("\n" + "=" * 70)
    print("RCT 1.8.2 VALIDATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
