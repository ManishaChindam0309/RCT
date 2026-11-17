
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


def run_rct(ffn, x):
    outputs = []

    for position in range(x.size(1)):
        token = x[:, position:position + 1, :]
        outputs.append(ffn(token))

    return torch.cat(outputs, dim=1)


def benchmark():
    torch.manual_seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    sequence_length = 32
    trials = 30

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=64
    )

    ffn.eval()

    print("=" * 65)
    print("RCT 1.7.1 TOKEN-LEVEL EXECUTION REUSE")
    print("=" * 65)

    print("Hidden size:", hidden_size)
    print("Sequence length:", sequence_length)
    print("Trials:", trials)
    print("CPU threads:", torch.get_num_threads())

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

    with torch.inference_mode():

        for workload_name, x in workloads.items():

            ffn.memory.clear()
            ffn.reset_execution_statistics()

            # Correctness check.
            standard_output = run_standard(ffn, x)

            ffn.memory.clear()

            rct_output = run_rct(ffn, x)

            max_difference = (
                standard_output - rct_output
            ).abs().max().item()

            correct = torch.allclose(
                standard_output,
                rct_output,
                rtol=1e-5,
                atol=1e-6
            )

            if not correct:
                raise AssertionError(
                    f"Numerical mismatch: {workload_name}"
                )

            # Warm-up.
            for _ in range(5):
                run_standard(ffn, x)

            ffn.memory.clear()

            for _ in range(5):
                run_rct(ffn, x)

            # Standard measurements.
            standard_times = []

            for _ in range(trials):
                start = time.perf_counter()
                run_standard(ffn, x)
                standard_times.append(
                    (time.perf_counter() - start) * 1000
                )

            # RCT measurements.
            ffn.memory.clear()
            rct_times = []

            for _ in range(trials):
                ffn.memory.clear()

                start = time.perf_counter()
                run_rct(ffn, x)
                rct_times.append(
                    (time.perf_counter() - start) * 1000
                )

            stats = ffn.cache_statistics()

            standard_median = median(standard_times)
            rct_median = median(rct_times)

            speedup = standard_median / rct_median

            reduction = (
                (standard_median - rct_median)
                / standard_median
            ) * 100

            print("\n" + "-" * 65)
            print("WORKLOAD:", workload_name)
            print("-" * 65)

            print("Numerically equivalent:", correct)
            print(f"Maximum difference: {max_difference:.10f}")

            print(
                f"Standard median: {standard_median:.6f} ms"
            )

            print(
                f"RCT token-wise median: {rct_median:.6f} ms"
            )

            print(f"Speedup: {speedup:.3f}x")
            print(f"Latency reduction: {reduction:.2f}%")

            print("Network executions:", stats["network_executions"])
            print("Cache hits:", stats["cache_hits"])
            print("Cache misses:", stats["cache_misses"])

    print("\n" + "=" * 65)
    print("RCT 1.7.1 BENCHMARK COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    benchmark()
