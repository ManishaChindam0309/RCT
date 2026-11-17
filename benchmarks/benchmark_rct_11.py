
import sys
import time
import csv
import math
import random
import statistics

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import RCTModel


# ==========================================
# CONFIGURATION
# ==========================================

VOCAB_SIZE = 8000
HIDDEN_SIZE = 256
LAYERS = 4
HEADS = 4

MAX_SEQ_LEN = 512
CACHE_CAPACITY = 8

PREFIX_LENGTHS = [32, 64, 96, 128, 192]
SUFFIX_LENGTHS = [16, 32, 64]

INDEPENDENT_RUNS = 5
SAMPLES_PER_RUN = 20

WARMUP = 5

NUM_THREADS = 4

OUTPUT_FILE = (
    PROJECT_ROOT
    / "benchmarks"
    / "rct_11_results.csv"
)


# ==========================================
# STATISTICS
# ==========================================

def percentile(values, percentage):

    ordered = sorted(values)

    index = math.ceil(
        (percentage / 100) * len(ordered)
    ) - 1

    index = max(0, min(index, len(ordered) - 1))

    return ordered[index]


def median_ms(values):

    return statistics.median(values)


def measure_once(function):

    start = time.perf_counter()

    function()

    end = time.perf_counter()

    return (end - start) * 1000


# ==========================================
# MODEL CREATION
# ==========================================

def create_models():

    baseline = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        layers=LAYERS,
        heads=HEADS,
        max_seq_len=MAX_SEQ_LEN
    )

    rct = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        layers=LAYERS,
        heads=HEADS,
        max_seq_len=MAX_SEQ_LEN,
        prefix_cache_capacity=CACHE_CAPACITY
    )

    rct.load_state_dict(baseline.state_dict())

    baseline.eval()
    rct.eval()

    # Disable FFN cache to isolate prefix/KV caching.
    for block in baseline.blocks:
        block.ffn.train()

    for block in rct.blocks:
        block.ffn.train()

    return baseline, rct


# ==========================================
# PROMPT CREATION
# ==========================================

def create_prompts(prefix_length, suffix_length):

    prefix = torch.randint(
        0,
        VOCAB_SIZE,
        (1, prefix_length)
    )

    suffix_a = torch.randint(
        0,
        VOCAB_SIZE,
        (1, suffix_length)
    )

    suffix_b = torch.randint(
        0,
        VOCAB_SIZE,
        (1, suffix_length)
    )

    prompt_a = torch.cat(
        [prefix, suffix_a],
        dim=1
    )

    prompt_b = torch.cat(
        [prefix, suffix_b],
        dim=1
    )

    return prompt_a, prompt_b


# ==========================================
# CACHE MEMORY
# ==========================================

def calculate_cache_memory(model):

    total_bytes = 0

    for entry in model._prefix_cache.values():

        for record in entry["records"]:

            for name in ("prefix_key", "prefix_value"):

                tensor = record[name]

                total_bytes += (
                    tensor.numel()
                    * tensor.element_size()
                )

        tensor = entry["final_prefix"]

        total_bytes += (
            tensor.numel()
            * tensor.element_size()
        )

    return total_bytes


# ==========================================
# SINGLE WORKLOAD
# ==========================================

def benchmark_case(
    baseline,
    rct,
    prefix_length,
    suffix_length,
    run_number
):

    prompt_a, prompt_b = create_prompts(
        prefix_length,
        suffix_length
    )

    rct.invalidate_caches()

    with torch.inference_mode():

        reference = baseline(prompt_b)

        # Warm up standard inference.
        for _ in range(WARMUP):
            baseline(prompt_b)

        # Build the prefix cache.
        first_request_time = measure_once(
            lambda: rct.forward_cached(
                prompt_a,
                prefix_length
            )
        )

        # Verify cached output.
        cached_output = rct.forward_cached(
            prompt_b,
            prefix_length
        )

        max_difference = (
            cached_output - reference
        ).abs().max().item()

        correct = torch.allclose(
            cached_output,
            reference,
            rtol=1e-4,
            atol=1e-4
        )

        # Warm up cached inference.
        for _ in range(WARMUP):
            rct.forward_cached(
                prompt_b,
                prefix_length
            )

        standard_samples = []
        cached_samples = []

        # Randomize which implementation is measured first.
        for _ in range(SAMPLES_PER_RUN):

            order = ["standard", "cached"]

            random.shuffle(order)

            for implementation in order:

                if implementation == "standard":

                    elapsed = measure_once(
                        lambda: baseline(prompt_b)
                    )

                    standard_samples.append(elapsed)

                else:

                    elapsed = measure_once(
                        lambda: rct.forward_cached(
                            prompt_b,
                            prefix_length
                        )
                    )

                    cached_samples.append(elapsed)

        standard_median = median_ms(standard_samples)
        cached_median = median_ms(cached_samples)

        standard_p95 = percentile(
            standard_samples,
            95
        )

        cached_p95 = percentile(
            cached_samples,
            95
        )

        speedup = (
            standard_median / cached_median
        )

        reduction = (
            (standard_median - cached_median)
            / standard_median
        ) * 100

        extra_cost = max(
            0,
            first_request_time - standard_median
        )

        saving = (
            standard_median - cached_median
        )

        break_even = (
            extra_cost / saving
            if saving > 0
            else float("inf")
        )

        cache_memory = calculate_cache_memory(rct)

        return {
            "run": run_number,
            "prefix": prefix_length,
            "suffix": suffix_length,
            "total": prefix_length + suffix_length,
            "standard_median": standard_median,
            "standard_p95": standard_p95,
            "cached_median": cached_median,
            "cached_p95": cached_p95,
            "speedup": speedup,
            "reduction": reduction,
            "first_request": first_request_time,
            "break_even": break_even,
            "cache_memory_bytes": cache_memory,
            "max_difference": max_difference,
            "correct": correct
        }


# ==========================================
# FULL CACHE MEMORY TEST
# ==========================================

def benchmark_full_cache_memory():

    print("\n===== FULL CACHE MEMORY TEST =====")

    _, rct = create_models()

    prefix_length = 192
    suffix_length = 16

    with torch.inference_mode():

        for index in range(CACHE_CAPACITY):

            prefix = torch.randint(
                0,
                VOCAB_SIZE,
                (1, prefix_length)
            )

            suffix = torch.randint(
                0,
                VOCAB_SIZE,
                (1, suffix_length)
            )

            prompt = torch.cat(
                [prefix, suffix],
                dim=1
            )

            rct.forward_cached(
                prompt,
                prefix_length
            )

    memory_bytes = calculate_cache_memory(rct)

    print("Cache entries:", len(rct._prefix_cache))
    print("Cache capacity:", CACHE_CAPACITY)

    print(
        f"Total tensor memory: "
        f"{memory_bytes / (1024 ** 2):.3f} MB"
    )

    print(
        f"Total tensor memory: {memory_bytes:,} bytes"
    )

    assert len(rct._prefix_cache) == CACHE_CAPACITY

    print("Full cache test: PASSED")


# ==========================================
# MAIN BENCHMARK
# ==========================================

def main():

    torch.manual_seed(42)
    torch.set_num_threads(NUM_THREADS)

    random.seed(42)

    print("===== RCT 1.1 REPRODUCIBILITY TEST =====")

    print("Device: CPU")
    print("Threads:", NUM_THREADS)
    print("Independent runs:", INDEPENDENT_RUNS)
    print("Samples per implementation:", SAMPLES_PER_RUN)
    print("Cache capacity:", CACHE_CAPACITY)

    print(
        "FFN caching: disabled in both models"
    )

    baseline, rct = create_models()

    results = []

    total_cases = (
        len(PREFIX_LENGTHS)
        * len(SUFFIX_LENGTHS)
        * INDEPENDENT_RUNS
    )

    current_case = 0

    for run_number in range(1, INDEPENDENT_RUNS + 1):

        print(
            f"\n========== INDEPENDENT RUN "
            f"{run_number}/{INDEPENDENT_RUNS} =========="
        )

        for prefix_length in PREFIX_LENGTHS:

            for suffix_length in SUFFIX_LENGTHS:

                current_case += 1

                print(
                    f"\n[{current_case}/{total_cases}] "
                    f"Prefix={prefix_length}, "
                    f"Suffix={suffix_length}"
                )

                result = benchmark_case(
                    baseline,
                    rct,
                    prefix_length,
                    suffix_length,
                    run_number
                )

                results.append(result)

                print(
                    f"Standard median: "
                    f"{result['standard_median']:.3f} ms"
                )

                print(
                    f"Cached median: "
                    f"{result['cached_median']:.3f} ms"
                )

                print(
                    f"Standard P95: "
                    f"{result['standard_p95']:.3f} ms"
                )

                print(
                    f"Cached P95: "
                    f"{result['cached_p95']:.3f} ms"
                )

                print(
                    f"Speedup: "
                    f"{result['speedup']:.2f}x"
                )

                print(
                    f"Correct: {result['correct']}"
                )

    # ======================================
    # SAVE CSV
    # ======================================

    fieldnames = list(results[0].keys())

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nCSV saved to:")
    print(OUTPUT_FILE)

    # ======================================
    # SUMMARY
    # ======================================

    print("\n===== RCT 1.1 SUMMARY =====")

    correct_results = [
        result for result in results
        if result["correct"]
    ]

    faster_results = [
        result for result in correct_results
        if result["cached_median"]
        < result["standard_median"]
    ]

    print("Total experiments:", len(results))
    print("Correct experiments:", len(correct_results))
    print("Faster cached experiments:", len(faster_results))

    if correct_results:

        speedups = [
            result["speedup"]
            for result in correct_results
        ]

        reductions = [
            result["reduction"]
            for result in correct_results
        ]

        print(
            "Median speedup across experiments:",
            f"{statistics.median(speedups):.3f}x"
        )

        print(
            "Median latency reduction:",
            f"{statistics.median(reductions):.2f}%"
        )

        print(
            "Minimum speedup:",
            f"{min(speedups):.3f}x"
        )

        print(
            "Maximum speedup:",
            f"{max(speedups):.3f}x"
        )

    if len(correct_results) != len(results):

        raise AssertionError(
            "Numerical correctness failure detected"
        )

    benchmark_full_cache_memory()

    print("\n===== RCT 1.1 COMPLETE =====")


if __name__ == "__main__":
    main()
