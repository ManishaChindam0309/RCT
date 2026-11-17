
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
    # Find unique token representations along sequence dimension.
    unique_tokens, inverse_indices = torch.unique(
        x,
        dim=1,
        return_inverse=True
    )

    # Execute the FFN on unique representations only.
    unique_outputs = ffn.network(unique_tokens)

    # Reconstruct the original sequence order.
    return unique_outputs.index_select(
        dim=1,
        index=inverse_indices
    )


def benchmark():
    torch.manual_seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    sequence_length = 32
    trials = 50

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=64
    )

    ffn.eval()

    workloads = {
        "All identical": torch.randn(
            1, 1, hidden_size
        ).expand(1, sequence_length, hidden_size).clone(),

        "Half repeated": torch.cat([
            torch.randn(1, 1, hidden_size).expand(
                1, sequence_length // 2, hidden_size
            ).clone(),
            torch.randn(
                1, sequence_length // 2, hidden_size
            )
        ], dim=1),

        "All unique": torch.randn(
            1, sequence_length, hidden_size
        )
    }

    print("=" * 65)
    print("RCT 1.7.2 VECTORIZED COMPUTATION REUSE")
    print("=" * 65)

    print("Hidden size:", hidden_size)
    print("Sequence length:", sequence_length)
    print("Trials:", trials)
    print("CPU threads:", torch.get_num_threads())

    with torch.inference_mode():

        for name, x in workloads.items():

            # Numerical correctness.
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
                    f"Numerical mismatch: {name}"
                )

            unique_tokens, _ = torch.unique(
                x,
                dim=1,
                return_inverse=True
            )

            unique_count = unique_tokens.size(1)

            # Warm-up.
            for _ in range(10):
                run_standard(ffn, x)
                run_deduplicated(ffn, x)

            standard_times = []
            deduplicated_times = []

            # Alternate measurement order to reduce ordering bias.
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

            print("\n" + "-" * 65)
            print("WORKLOAD:", name)
            print("-" * 65)

            print("Sequence tokens:", sequence_length)
            print("Unique tokens:", unique_count)
            print("FFN computations saved:",
                  sequence_length - unique_count)

            print("Numerically equivalent:", correct)
            print(f"Maximum difference: {max_difference:.10f}")

            print(
                f"Standard median: "
                f"{standard_median:.6f} ms"
            )

            print(
                f"Deduplicated median: "
                f"{deduplicated_median:.6f} ms"
            )

            print(f"Speedup: {speedup:.3f}x")
            print(f"Latency reduction: {reduction:.2f}%")

    print("\n" + "=" * 65)
    print("RCT 1.7.2 BENCHMARK COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    benchmark()
