
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


def percentile_95(values):
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        int(0.95 * len(ordered))
    )
    return ordered[index]


def measure_standard(ffn, token, repetitions):
    start = time.perf_counter()

    outputs = []

    for _ in range(repetitions):
        outputs.append(ffn.network(token))

    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, outputs


def measure_rct(ffn, token, repetitions):
    start = time.perf_counter()

    outputs = []

    for _ in range(repetitions):
        outputs.append(ffn(token))

    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, outputs


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    repetitions = 8
    trials = 30

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=32
    )

    ffn.eval()

    token = torch.randn(1, 1, hidden_size)

    print("=" * 65)
    print("RCT 1.7 COMPUTATION REUSE BENCHMARK")
    print("=" * 65)

    print("Device:", token.device)
    print("Hidden size:", hidden_size)
    print("Repeated inputs per trial:", repetitions)
    print("Measurement trials:", trials)
    print("CPU threads:", torch.get_num_threads())

    standard_times = []
    cold_times = []
    warm_times = []

    # --------------------------------------------------
    # Numerical correctness
    # --------------------------------------------------

    with torch.inference_mode():

        reference = ffn.network(token)

        ffn.memory.clear()

        rct_output = ffn(token)

        if not torch.allclose(
            reference,
            rct_output,
            rtol=1e-5,
            atol=1e-6
        ):
            raise AssertionError(
                "RCT output does not match standard FFN."
            )

        print("\nNumerical correctness: PASSED")

        # --------------------------------------------------
        # Warm-up
        # --------------------------------------------------

        for _ in range(10):
            ffn.network(token)

        ffn.memory.clear()

        for _ in range(10):
            ffn(token)

        # --------------------------------------------------
        # Standard FFN benchmark
        # --------------------------------------------------

        for _ in range(trials):
            elapsed, _ = measure_standard(
                ffn,
                token,
                repetitions
            )

            standard_times.append(elapsed)

        # --------------------------------------------------
        # Cold RCT benchmark
        # One computation followed by cache hits.
        # Cache is cleared before every trial.
        # --------------------------------------------------

        for _ in range(trials):
            ffn.memory.clear()
            ffn.reset_execution_statistics()

            elapsed, _ = measure_rct(
                ffn,
                token,
                repetitions
            )

            cold_times.append(elapsed)

        cold_stats = ffn.cache_statistics()

        # --------------------------------------------------
        # Warm RCT benchmark
        # Populate cache before measuring.
        # --------------------------------------------------

        ffn.memory.clear()
        ffn(token)

        ffn.reset_execution_statistics()

        for _ in range(trials):
            elapsed, _ = measure_rct(
                ffn,
                token,
                repetitions
            )

            warm_times.append(elapsed)

        warm_stats = ffn.cache_statistics()

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    standard_median = median(standard_times)
    cold_median = median(cold_times)
    warm_median = median(warm_times)

    speedup = standard_median / warm_median

    latency_reduction = (
        (standard_median - warm_median)
        / standard_median
    ) * 100

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS")
    print("=" * 65)

    print(
        f"Standard FFN median: "
        f"{standard_median:.6f} ms"
    )

    print(
        f"Standard FFN P95:    "
        f"{percentile_95(standard_times):.6f} ms"
    )

    print(
        f"RCT cold median:     "
        f"{cold_median:.6f} ms"
    )

    print(
        f"RCT warm median:     "
        f"{warm_median:.6f} ms"
    )

    print(
        f"RCT warm P95:        "
        f"{percentile_95(warm_times):.6f} ms"
    )

    print(f"\nWarm speedup:        {speedup:.3f}x")

    print(
        f"Latency reduction:   "
        f"{latency_reduction:.2f}%"
    )

    print("\n" + "=" * 65)
    print("EXECUTION STATISTICS")
    print("=" * 65)

    print("Cold trial final statistics:")
    print(cold_stats)

    print("\nWarm benchmark statistics:")
    print(warm_stats)

    expected_warm_hits = trials * repetitions

    assert warm_stats["network_executions"] == 0
    assert warm_stats["cache_hits"] == expected_warm_hits

    print("\nWarm cache execution verification: PASSED")

    if speedup > 1:
        print("Measured warm latency improvement: YES")
    else:
        print("Measured warm latency improvement: NO")

    print("\nRCT 1.7 BENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
