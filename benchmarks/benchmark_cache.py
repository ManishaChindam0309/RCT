
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import statistics
import time

import torch

from model.reusable_block import ReusableFFN


def measure_latency(function, repeats=100, warmup=15):
    """Measure execution latency in milliseconds."""

    for _ in range(warmup):
        function()

    measurements = []

    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        end = time.perf_counter_ns()

        elapsed_ms = (end - start) / 1_000_000
        measurements.append(elapsed_ms)

    return {
        "median_ms": statistics.median(measurements),
        "mean_ms": statistics.mean(measurements),
        "min_ms": min(measurements),
    }


def main():
    # CPU-only benchmark.
    torch.set_num_threads(4)
    torch.manual_seed(42)

    hidden_size = 256
    sequence_length = 32
    batch_size = 1

    # Create RCT reusable FFN.
    rct = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=256
    )

    rct.eval()

    # Create a standard FFN with identical weights.
    baseline = copy.deepcopy(rct.network)
    baseline.eval()

    # Generate fixed input.
    x = torch.randn(
        batch_size,
        sequence_length,
        hidden_size
    )

    print("\n========== RCT CPU BENCHMARK ==========")
    print("Device: CPU")
    print("Hidden size:", hidden_size)
    print("Sequence length:", sequence_length)
    print("Batch size:", batch_size)

    with torch.inference_mode():

        # Check numerical correctness.
        baseline_output = baseline(x)
        rct_output = rct(x)

        print(
            "\nNumerically equivalent:",
            torch.equal(baseline_output, rct_output)
        )

        # 1. Standard FFN benchmark.
        baseline_result = measure_latency(
            lambda: baseline(x),
            repeats=100,
            warmup=15
        )

        # 2. RCT cold-cache benchmark.
        # Clear the cache before every execution.
        def cold_execution():
            rct.memory.clear()
            return rct(x)

        cold_result = measure_latency(
            cold_execution,
            repeats=30,
            warmup=0
        )

        # 3. RCT cached benchmark.
        rct.memory.clear()
        rct(x)  # Populate cache.

        hit_result = measure_latency(
            lambda: rct(x),
            repeats=100,
            warmup=15
        )

    # Calculate performance.
    baseline_ms = baseline_result["median_ms"]
    cold_ms = cold_result["median_ms"]
    cached_ms = hit_result["median_ms"]

    speedup = baseline_ms / cached_ms

    improvement = (
        (baseline_ms - cached_ms) / baseline_ms
    ) * 100

    print("\n========== PERFORMANCE RESULTS ==========")

    print(
        "Standard FFN median:",
        round(baseline_ms, 4),
        "ms"
    )

    print(
        "RCT cold execution median:",
        round(cold_ms, 4),
        "ms"
    )

    print(
        "RCT cached execution median:",
        round(cached_ms, 4),
        "ms"
    )

    print("\n========== SPEED ANALYSIS ==========")

    print("Speedup:", round(speedup, 3), "x")

    print(
        "Latency reduction:",
        round(improvement, 2),
        "%"
    )

    print("\n========== CACHE STATISTICS ==========")

    print(rct.cache_statistics())

    print("\n========== BENCHMARK COMPLETE ==========")


if __name__ == "__main__":
    main()
