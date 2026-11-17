
import sys
from pathlib import Path

# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import statistics
import time

import torch

from model.transformer import RCTModel


def measure_latency(function, repeats=30, warmup=5):
    """Measure inference latency in milliseconds."""

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


def clear_model_cache(model):
    """Clear all reusable FFN caches."""

    for block in model.blocks:
        block.ffn.memory.clear()


def print_cache_statistics(model):
    """Print cache statistics for every Transformer layer."""

    statistics_by_layer = model.cache_statistics()

    total_hits = 0
    total_misses = 0

    for layer_name, stats in statistics_by_layer.items():
        print(f"{layer_name}: {stats}")

        total_hits += stats["hits"]
        total_misses += stats["misses"]

    total_requests = total_hits + total_misses

    hit_rate = (
        total_hits / total_requests
        if total_requests > 0
        else 0.0
    )

    print("\nTotal hits:", total_hits)
    print("Total misses:", total_misses)
    print("Overall hit rate:", round(hit_rate, 4))


def main():
    # CPU configuration.
    torch.set_num_threads(4)
    torch.manual_seed(42)

    # Model configuration.
    vocab_size = 8000
    hidden_size = 256
    layers = 4
    heads = 4
    sequence_length = 32
    batch_size = 1

    print("\n========== RCT 0.4 FULL TRANSFORMER BENCHMARK ==========")

    # Create RCT model.
    rct = RCTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        max_seq_len=512,
        expansion=4
    )

    # Create independent standard model.
    baseline = RCTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        max_seq_len=512,
        expansion=4
    )

    # Copy identical weights without copying cache locks.
    baseline.load_state_dict(rct.state_dict())

    rct.eval()
    baseline.eval()

    # Generate a fixed input.
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, sequence_length)
    )

    print("\nModel configuration:")
    print("Vocabulary:", vocab_size)
    print("Hidden size:", hidden_size)
    print("Layers:", layers)
    print("Attention heads:", heads)
    print("Sequence length:", sequence_length)
    print("Batch size:", batch_size)

    with torch.inference_mode():

        # ---------------------------------
        # 1. Numerical correctness
        # ---------------------------------

        baseline_output = baseline(tokens)
        rct_output = rct(tokens)

        exact_match = torch.equal(
            baseline_output,
            rct_output
        )

        max_difference = (
            baseline_output - rct_output
        ).abs().max().item()

        print("\n========== CORRECTNESS ==========")
        print("Exact output equality:", exact_match)
        print("Maximum difference:", max_difference)

        # ---------------------------------
        # 2. Standard Transformer
        # ---------------------------------

        baseline_result = measure_latency(
            lambda: baseline(tokens),
            repeats=30,
            warmup=5
        )

        # ---------------------------------
        # 3. RCT cold-cache execution
        # ---------------------------------

        def cold_execution():
            clear_model_cache(rct)
            return rct(tokens)

        cold_result = measure_latency(
            cold_execution,
            repeats=30,
            warmup=0
        )

        # ---------------------------------
        # 4. RCT cached execution
        # ---------------------------------

        clear_model_cache(rct)

        # Populate cache.
        rct(tokens)

        cached_result = measure_latency(
            lambda: rct(tokens),
            repeats=30,
            warmup=5
        )

    # ---------------------------------
    # 5. Performance calculations
    # ---------------------------------

    baseline_ms = baseline_result["median_ms"]
    cold_ms = cold_result["median_ms"]
    cached_ms = cached_result["median_ms"]

    speedup = baseline_ms / cached_ms

    improvement = (
        (baseline_ms - cached_ms) / baseline_ms
    ) * 100

    print("\n========== LATENCY RESULTS ==========")

    print(
        "Standard Transformer:",
        round(baseline_ms, 4),
        "ms"
    )

    print(
        "RCT cold execution:",
        round(cold_ms, 4),
        "ms"
    )

    print(
        "RCT cached execution:",
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

    # ---------------------------------
    # 6. Cache statistics
    # ---------------------------------

    print("\n========== CACHE STATISTICS ==========")

    print_cache_statistics(rct)

    print("\n========== BENCHMARK COMPLETE ==========")


if __name__ == "__main__":
    main()
