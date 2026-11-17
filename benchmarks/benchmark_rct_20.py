
import os
import sys
import time
import random
import statistics
import argparse

import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.transformer import RCTModel


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 2020

VOCAB_SIZE = 2000
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
MAX_SEQ_LEN = 256

WARMUP_RUNS = 5
BENCHMARK_RUNS = 30

CPU_THREADS = 4

SEQUENCE_LENGTHS = [64, 128, 256]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def setup():

    random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(CPU_THREADS)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print("=" * 65)
    print("RCT 2.0.1 TRANSFORMER PERFORMANCE BENCHMARK")
    print("=" * 65)

    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())
    print("Device: CPU")


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=MAX_SEQ_LEN
    )

    model.eval()

    return model


# ============================================================
# DISABLE FFN CACHE EFFECTS
# ============================================================

def clear_ffn_caches(model):

    for block in model.blocks:
        block.ffn.invalidate_cache()


# ============================================================
# SINGLE INFERENCE
# ============================================================

@torch.inference_mode()
def run_inference(model, tokens, mode):

    clear_ffn_caches(model)

    start = time.perf_counter()

    output = model(
        tokens,
        ffn_mode=mode
    )

    elapsed = time.perf_counter() - start

    return output, elapsed


# ============================================================
# OPTIONAL STANDALONE MODE BENCHMARK
# ============================================================

def benchmark_mode(
    model,
    tokens,
    mode,
    runs=BENCHMARK_RUNS
):

    for _ in range(WARMUP_RUNS):

        run_inference(
            model,
            tokens,
            mode
        )

    measurements = []

    for _ in range(runs):

        _, elapsed = run_inference(
            model,
            tokens,
            mode
        )

        measurements.append(
            elapsed * 1000
        )

    return {
        "median_ms": statistics.median(measurements),
        "p95_ms": sorted(measurements)[
            int(0.95 * (len(measurements) - 1))
        ],
        "minimum_ms": min(measurements),
        "maximum_ms": max(measurements)
    }


# ============================================================
# NUMERICAL CORRECTNESS
# ============================================================

@torch.inference_mode()
def check_correctness(model, tokens):

    standard, _ = run_inference(
        model,
        tokens,
        "standard"
    )

    deduplicated, _ = run_inference(
        model,
        tokens,
        "deduplicated"
    )

    max_difference = (
        standard - deduplicated
    ).abs().max().item()

    correct = torch.allclose(
        standard,
        deduplicated,
        rtol=1e-4,
        atol=1e-5
    )

    return correct, max_difference


# ============================================================
# LAYER STATISTICS
# ============================================================

@torch.inference_mode()
def collect_dedup_statistics(model, tokens):

    model.reset_dedup_statistics()

    clear_ffn_caches(model)

    model(
        tokens,
        ffn_mode="deduplicated"
    )

    return model.dedup_statistics()


# ============================================================
# SINGLE WORKLOAD
# ============================================================

def run_workload(
    model,
    sequence_length,
    runs=BENCHMARK_RUNS
):

    print("\n" + "-" * 65)

    print(
        "Sequence length:",
        sequence_length
    )

    tokens = torch.randint(
        0,
        VOCAB_SIZE,
        (1, sequence_length)
    )

    # --------------------------------------------------------
    # NUMERICAL CORRECTNESS
    # --------------------------------------------------------

    correct, max_difference = check_correctness(
        model,
        tokens
    )

    print(
        "Numerical correctness:",
        correct
    )

    print(
        "Maximum output difference:",
        f"{max_difference:.10f}"
    )

    if not correct:

        raise RuntimeError(
            "Numerical correctness check failed."
        )

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    for mode in (
        "standard",
        "deduplicated"
    ):

        for _ in range(WARMUP_RUNS):

            run_inference(
                model,
                tokens,
                mode
            )

    # --------------------------------------------------------
    # RANDOMIZED PAIRED MEASUREMENTS
    # --------------------------------------------------------

    results = {
        "standard": [],
        "deduplicated": []
    }

    for trial in range(runs):

        modes = [
            "standard",
            "deduplicated"
        ]

        random.shuffle(modes)

        for mode in modes:

            _, elapsed = run_inference(
                model,
                tokens,
                mode
            )

            results[mode].append(
                elapsed * 1000
            )

    # --------------------------------------------------------
    # LATENCY STATISTICS
    # --------------------------------------------------------

    standard_median = statistics.median(
        results["standard"]
    )

    dedup_median = statistics.median(
        results["deduplicated"]
    )

    standard_p95 = sorted(
        results["standard"]
    )[
        int(0.95 * (runs - 1))
    ]

    dedup_p95 = sorted(
        results["deduplicated"]
    )[
        int(0.95 * (runs - 1))
    ]

    speedup = (
        standard_median / dedup_median
    )

    reduction = (
        1 - dedup_median / standard_median
    ) * 100

    # --------------------------------------------------------
    # DUPLICATION STATISTICS
    # --------------------------------------------------------

    layer_stats = collect_dedup_statistics(
        model,
        tokens
    )

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\nLatency results:")

    print(
        f"Standard median:      {standard_median:.4f} ms"
    )

    print(
        f"Deduplicated median:  {dedup_median:.4f} ms"
    )

    print(
        f"Standard P95:         {standard_p95:.4f} ms"
    )

    print(
        f"Deduplicated P95:     {dedup_p95:.4f} ms"
    )

    print(
        f"Speedup:              {speedup:.3f}x"
    )

    print(
        f"Latency reduction:    {reduction:.2f}%"
    )

    print("\nLayer duplication statistics:")

    for layer, stats in layer_stats.items():

        print(
            f"{layer}: "
            f"inputs={stats['total_inputs']}, "
            f"unique={stats['unique_inputs']}, "
            f"reuse={stats['reuse_ratio']:.2%}"
        )

    return {
        "sequence_length": sequence_length,
        "standard_median_ms": standard_median,
        "dedup_median_ms": dedup_median,
        "standard_p95_ms": standard_p95,
        "dedup_p95_ms": dedup_p95,
        "speedup": speedup,
        "latency_reduction": reduction,
        "layer_statistics": layer_stats
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="RCT 2.0.1 Transformer benchmark"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=BENCHMARK_RUNS
    )

    parser.add_argument(
        "--quick",
        action="store_true"
    )

    args = parser.parse_args()

    if args.runs < 1:

        parser.error(
            "--runs must be positive"
        )

    runs = (
        5 if args.quick
        else args.runs
    )

    setup()

    model = create_model()

    print("\nModel configuration:")

    print("Vocabulary:", VOCAB_SIZE)
    print("Hidden size:", HIDDEN_SIZE)
    print("Layers:", NUM_LAYERS)
    print("Heads:", NUM_HEADS)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        "Parameters:",
        f"{parameter_count:,}"
    )

    print(
        "Benchmark runs per mode:",
        runs
    )

    all_results = []

    for sequence_length in SEQUENCE_LENGTHS:

        result = run_workload(
            model,
            sequence_length,
            runs=runs
        )

        all_results.append(result)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("RCT 2.0.1 FINAL SUMMARY")
    print("=" * 65)

    print(
        f"{'Length':>8} "
        f"{'Standard':>12} "
        f"{'Dedup':>12} "
        f"{'Speedup':>10} "
        f"{'Reduction':>12}"
    )

    for result in all_results:

        print(
            f"{result['sequence_length']:>8} "
            f"{result['standard_median_ms']:>11.4f} "
            f"{result['dedup_median_ms']:>11.4f} "
            f"{result['speedup']:>9.3f}x "
            f"{result['latency_reduction']:>10.2f}%"
        )

    average_speedup = statistics.mean(
        result["speedup"]
        for result in all_results
    )

    average_reduction = statistics.mean(
        result["latency_reduction"]
        for result in all_results
    )

    print(
        "\nAverage speedup:",
        f"{average_speedup:.3f}x"
    )

    print(
        "Average latency reduction:",
        f"{average_reduction:.2f}%"
    )

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
