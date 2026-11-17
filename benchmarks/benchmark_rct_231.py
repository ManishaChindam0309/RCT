
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import torch

# --------------------------------------------------
# Project imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV


# --------------------------------------------------
# Configuration
# --------------------------------------------------

NUM_THREADS = 4
REPEATS = 7
WARMUP_RUNS = 1
BASE_SEED = 231

torch.set_num_threads(NUM_THREADS)
torch.manual_seed(BASE_SEED)
random.seed(BASE_SEED)

print("=" * 75)
print("RCT 2.3.1 - 100M KV CACHE PERFORMANCE BENCHMARK")
print("=" * 75)

# --------------------------------------------------
# Load model
# --------------------------------------------------

print("\nLoading RCT 100M model...")

base_model = RCT100M()
base_model.eval()

kv_model = RCT100MKV(base_model)
kv_model.eval()

print(f"Parameters: {base_model.parameter_count():,}")
print(f"CPU threads: {NUM_THREADS}")
print(f"Benchmark repetitions: {REPEATS}")


# --------------------------------------------------
# Workloads
# --------------------------------------------------

workloads = [
    {"prompt": 16, "new_tokens": 8},
    {"prompt": 64, "new_tokens": 8},
    {"prompt": 64, "new_tokens": 16},
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def create_prompt(length):

    return torch.randint(
        0,
        base_model.config.vocab_size,
        (1, length),
        dtype=torch.long
    )


def run_standard(prompt, new_tokens):

    with torch.inference_mode():

        return base_model.generate(
            prompt,
            max_new_tokens=new_tokens,
            temperature=1.0
        )


def run_cached(prompt, new_tokens):

    with torch.inference_mode():

        return kv_model.generate_cached(
            prompt,
            max_new_tokens=new_tokens,
            temperature=1.0
        )


def measure(function, prompt, new_tokens, seed):

    # Reset RNG outside the timed section.
    torch.manual_seed(seed)

    start = time.perf_counter()

    output = function(prompt, new_tokens)

    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, output


# --------------------------------------------------
# KV cache memory estimate
# --------------------------------------------------

cache = kv_model.create_cache(batch_size=1)

cache_memory_bytes = cache.memory_bytes()
cache_memory_mib = cache_memory_bytes / (1024 ** 2)

print(
    f"\nKV cache allocated memory: "
    f"{cache_memory_mib:.2f} MiB"
)

del cache


# --------------------------------------------------
# Benchmark
# --------------------------------------------------

results = []

for index, workload in enumerate(workloads, start=1):

    prompt_length = workload["prompt"]
    new_tokens = workload["new_tokens"]

    prompt = create_prompt(prompt_length)

    print("\n" + "-" * 75)

    print(
        f"Workload {index}/{len(workloads)} | "
        f"Prompt={prompt_length} | "
        f"Generated tokens={new_tokens}"
    )

    # ----------------------------------------------
    # Correctness test
    # ----------------------------------------------

    standard_output_time, standard_output = measure(
        run_standard,
        prompt,
        new_tokens,
        999
    )

    cached_output_time, cached_output = measure(
        run_cached,
        prompt,
        new_tokens,
        999
    )

    correctness = torch.equal(
        standard_output,
        cached_output
    )

    print("Output correctness:", correctness)

    if not correctness:

        raise RuntimeError(
            "Generation mismatch for workload "
            f"{index}"
        )

    del standard_output
    del cached_output

    # ----------------------------------------------
    # Warm-up
    # ----------------------------------------------

    for warmup in range(WARMUP_RUNS):

        seed = 5000 + warmup

        measure(
            run_standard,
            prompt,
            new_tokens,
            seed
        )

        measure(
            run_cached,
            prompt,
            new_tokens,
            seed
        )

    # ----------------------------------------------
    # Paired measurements
    # ----------------------------------------------

    standard_times = []
    cached_times = []
    paired_speedups = []

    for repeat in range(REPEATS):

        order = ["standard", "cached"]
        random.shuffle(order)

        pair = {}

        seed = BASE_SEED + repeat + index * 100

        for mode in order:

            if mode == "standard":

                elapsed, _ = measure(
                    run_standard,
                    prompt,
                    new_tokens,
                    seed
                )

            else:

                elapsed, _ = measure(
                    run_cached,
                    prompt,
                    new_tokens,
                    seed
                )

            pair[mode] = elapsed

        standard_times.append(pair["standard"])
        cached_times.append(pair["cached"])

        paired_speedups.append(
            pair["standard"] / pair["cached"]
        )

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    standard_median = statistics.median(
        standard_times
    )

    cached_median = statistics.median(
        cached_times
    )

    speedup = standard_median / cached_median

    paired_speedup = statistics.median(
        paired_speedups
    )

    reduction = (
        1 - cached_median / standard_median
    ) * 100

    result = {

        "prompt_length": prompt_length,
        "generated_tokens": new_tokens,

        "correctness": correctness,

        "standard_median_ms": standard_median,
        "cached_median_ms": cached_median,

        "standard_min_ms": min(standard_times),
        "cached_min_ms": min(cached_times),

        "standard_max_ms": max(standard_times),
        "cached_max_ms": max(cached_times),

        "speedup": speedup,
        "paired_speedup": paired_speedup,

        "latency_reduction_percent": reduction,

        "kv_cache_memory_mib": cache_memory_mib
    }

    results.append(result)

    print(
        f"Standard median: {standard_median:.2f} ms"
    )

    print(
        f"Cached median: {cached_median:.2f} ms"
    )

    print(
        f"Speedup: {speedup:.3f}x"
    )

    print(
        f"Paired speedup: {paired_speedup:.3f}x"
    )

    print(
        f"Latency reduction: {reduction:.2f}%"
    )


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

output_path = (
    PROJECT_ROOT
    / "benchmarks"
    / "rct_231_results.csv"
)

with output_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 75)
print("RCT 2.3.1 FINAL SUMMARY")
print("=" * 75)

print(f"Workloads completed: {len(results)}")

print(
    "Correctness:",
    sum(r["correctness"] for r in results),
    "/",
    len(results)
)

print(
    "Average speedup:",
    f"{statistics.mean(r['speedup'] for r in results):.3f}x"
)

print(
    "Median speedup:",
    f"{statistics.median(r['speedup'] for r in results):.3f}x"
)

print(
    "Average latency reduction:",
    f"{statistics.mean(r['latency_reduction_percent'] for r in results):.2f}%"
)

print(f"\nResults saved to: {output_path}")

print("\nRCT 2.3.1 BENCHMARK COMPLETED")
