
import os
import sys
import time
import random
import math
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

SEED = 20202

VOCAB_SIZE = 2000
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
MAX_SEQ_LEN = 256

CPU_THREADS = 4

WARMUP_RUNS = 5
BENCHMARK_RUNS = 30

BATCH_SIZES = [1, 2, 4, 8]
SEQUENCE_LENGTHS = [64, 128]

PATTERNS = [
    "identical",
    "paired",
    "unique"
]


# ============================================================
# SETUP
# ============================================================

def setup():

    random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(CPU_THREADS)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print("=" * 75)
    print("RCT 2.0.2 CROSS-PROMPT DEDUPLICATION BENCHMARK")
    print("=" * 75)

    print("PyTorch:", torch.__version__)
    print("Device: CPU")
    print("CPU threads:", torch.get_num_threads())


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
# CACHE CONTROL
# ============================================================

def clear_ffn_caches(model):

    for block in model.blocks:
        block.ffn.invalidate_cache()


# ============================================================
# CREATE WORKLOADS
# ============================================================

def create_batch(batch_size, sequence_length, pattern):

    base_prompt = torch.randint(
        0,
        VOCAB_SIZE,
        (1, sequence_length)
    )

    if pattern == "identical":

        return base_prompt.repeat(
            batch_size,
            1
        )

    if pattern == "paired":

        unique_count = math.ceil(
            batch_size / 2
        )

        prompts = torch.randint(
            0,
            VOCAB_SIZE,
            (unique_count, sequence_length)
        )

        rows = []

        for index in range(batch_size):

            prompt_index = index // 2

            rows.append(
                prompts[prompt_index]
            )

        return torch.stack(rows)

    if pattern == "unique":

        return torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, sequence_length)
        )

    raise ValueError(
        f"Unknown pattern: {pattern}"
    )


# ============================================================
# INFERENCE
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
# CORRECTNESS
# ============================================================

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
# DEDUPLICATION STATISTICS
# ============================================================

@torch.inference_mode()
def collect_statistics(model, tokens):

    model.reset_dedup_statistics()

    clear_ffn_caches(model)

    model(
        tokens,
        ffn_mode="deduplicated"
    )

    return model.dedup_statistics()


# ============================================================
# LATENCY MEASUREMENT
# ============================================================

def measure_mode(model, tokens, mode, runs):

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

    return measurements


# ============================================================
# ONE EXPERIMENT
# ============================================================

def run_experiment(
    model,
    batch_size,
    sequence_length,
    pattern,
    runs
):

    tokens = create_batch(
        batch_size,
        sequence_length,
        pattern
    )

    # --------------------------------------------------------
    # CORRECTNESS
    # --------------------------------------------------------

    correct, max_difference = check_correctness(
        model,
        tokens
    )

    if not correct:

        raise RuntimeError(
            "Numerical correctness failed: "
            f"batch={batch_size}, "
            f"sequence={sequence_length}, "
            f"pattern={pattern}"
        )

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    for mode in ("standard", "deduplicated"):

        for _ in range(WARMUP_RUNS):

            run_inference(
                model,
                tokens,
                mode
            )

    # --------------------------------------------------------
    # RANDOMIZED PAIRED BENCHMARK
    # --------------------------------------------------------

    results = {
        "standard": [],
        "deduplicated": []
    }

    for _ in range(runs):

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

    standard_sorted = sorted(
        results["standard"]
    )

    dedup_sorted = sorted(
        results["deduplicated"]
    )

    p95_index = math.ceil(
        0.95 * runs
    ) - 1

    standard_p95 = standard_sorted[p95_index]
    dedup_p95 = dedup_sorted[p95_index]

    speedup = (
        standard_median / dedup_median
    )

    reduction = (
        1 - dedup_median / standard_median
    ) * 100

    # --------------------------------------------------------
    # LAYER DUPLICATION STATISTICS
    # --------------------------------------------------------

    layer_stats = collect_statistics(
        model,
        tokens
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "pattern": pattern,
        "correct": correct,
        "max_difference": max_difference,
        "standard_median": standard_median,
        "dedup_median": dedup_median,
        "standard_p95": standard_p95,
        "dedup_p95": dedup_p95,
        "speedup": speedup,
        "reduction": reduction,
        "layer_stats": layer_stats
    }


# ============================================================
# DISPLAY ONE RESULT
# ============================================================

def display_result(result):

    print(
        f"\nBatch={result['batch_size']} "
        f"Sequence={result['sequence_length']} "
        f"Pattern={result['pattern']}"
    )

    print(
        "Correctness:",
        result["correct"]
    )

    print(
        "Maximum difference:",
        f"{result['max_difference']:.10f}"
    )

    print(
        f"Standard:     "
        f"{result['standard_median']:.4f} ms"
    )

    print(
        f"Deduplicated: "
        f"{result['dedup_median']:.4f} ms"
    )

    print(
        f"Speedup:      "
        f"{result['speedup']:.3f}x"
    )

    print(
        f"Reduction:    "
        f"{result['reduction']:.2f}%"
    )

    print("Layer duplication:")

    for layer, stats in result["layer_stats"].items():

        print(
            f"  {layer}: "
            f"{stats['total_inputs']} inputs, "
            f"{stats['unique_inputs']} unique, "
            f"{stats['reuse_ratio']:.2%} reuse"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

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

    runs = 5 if args.quick else args.runs

    setup()

    model = create_model()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nModel configuration:")
    print("Vocabulary:", VOCAB_SIZE)
    print("Hidden size:", HIDDEN_SIZE)
    print("Layers:", NUM_LAYERS)
    print("Heads:", NUM_HEADS)
    print("Parameters:", f"{parameter_count:,}")
    print("Benchmark runs:", runs)

    all_results = []

    total_experiments = (
        len(BATCH_SIZES)
        * len(SEQUENCE_LENGTHS)
        * len(PATTERNS)
    )

    experiment_number = 0

    for sequence_length in SEQUENCE_LENGTHS:

        for batch_size in BATCH_SIZES:

            for pattern in PATTERNS:

                experiment_number += 1

                print("\n" + "=" * 75)

                print(
                    f"Experiment {experiment_number}/"
                    f"{total_experiments}"
                )

                result = run_experiment(
                    model=model,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    pattern=pattern,
                    runs=runs
                )

                display_result(result)

                all_results.append(result)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("RCT 2.0.2 FINAL SUMMARY")
    print("=" * 90)

    print(
        f"{'Batch':>6} "
        f"{'Seq':>6} "
        f"{'Pattern':>12} "
        f"{'Standard':>12} "
        f"{'Dedup':>12} "
        f"{'Speedup':>10} "
        f"{'Reduction':>12}"
    )

    for result in all_results:

        print(
            f"{result['batch_size']:>6} "
            f"{result['sequence_length']:>6} "
            f"{result['pattern']:>12} "
            f"{result['standard_median']:>11.4f} "
            f"{result['dedup_median']:>11.4f} "
            f"{result['speedup']:>9.3f}x "
            f"{result['reduction']:>10.2f}%"
        )

    # --------------------------------------------------------
    # GROUPED SUMMARY
    # --------------------------------------------------------

    print("\nAverage speedup by workload pattern:")

    for pattern in PATTERNS:

        matching = [
            result
            for result in all_results
            if result["pattern"] == pattern
        ]

        average_speedup = statistics.mean(
            result["speedup"]
            for result in matching
        )

        print(
            f"{pattern:>12}: "
            f"{average_speedup:.3f}x"
        )

    correct_count = sum(
        result["correct"]
        for result in all_results
    )

    print("\nCorrect experiments:")
    print(
        f"{correct_count}/{len(all_results)}"
    )

    print("\nRCT 2.0.2 benchmark completed.")


if __name__ == "__main__":
    main()
