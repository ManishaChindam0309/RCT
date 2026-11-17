
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import torch

# --------------------------------------------------
# Project import path fix
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.rct_100m import RCT100M


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEQUENCE_LENGTH = 32
REPEATS = 20
WARMUP_RUNS = 2
NUM_THREADS = 4
RANDOM_SEED = 221

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.set_num_threads(NUM_THREADS)

device = torch.device("cpu")

print("=" * 75)
print("RCT 2.2.1 - RELIABLE 100M PROMPT DEDUPLICATION BENCHMARK")
print("=" * 75)

print("\nProject root:", PROJECT_ROOT)
print("Loading RCT 100M model...")

model = RCT100M().to(device)
model.eval()

print(f"Parameters: {model.parameter_count():,}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Paired repetitions: {REPEATS}")
print(f"CPU threads: {NUM_THREADS}")


# --------------------------------------------------
# Workload generation
# --------------------------------------------------

def make_tokens(batch_size, unique_count):

    unique_tokens = torch.randint(
        0,
        model.config.vocab_size,
        (unique_count, SEQUENCE_LENGTH),
        dtype=torch.long
    )

    # Ensure each unique prompt is different.
    unique_tokens[:, 0] = torch.arange(
        100,
        100 + unique_count
    )

    indices = [
        i % unique_count
        for i in range(batch_size)
    ]

    return unique_tokens[indices].clone()


workloads = []

for batch_size in [2, 4, 8, 16]:

    requested_rates = [0.0, 0.50, 0.75, 0.875]

    for requested_rate in requested_rates:

        unique_count = round(
            batch_size * (1.0 - requested_rate)
        )

        unique_count = max(1, unique_count)

        actual_rate = 1.0 - unique_count / batch_size

        # Only include exactly representable duplicate rates.
        if abs(actual_rate - requested_rate) > 1e-9:
            continue

        workloads.append({
            "batch": batch_size,
            "unique": unique_count,
            "requested_rate": requested_rate,
            "actual_rate": actual_rate,
            "tokens": make_tokens(batch_size, unique_count)
        })


# --------------------------------------------------
# Timing functions
# --------------------------------------------------

def run_model(tokens, mode):

    with torch.inference_mode():
        return model(tokens, prompt_mode=mode)


def measure_once(tokens, mode):

    start = time.perf_counter()

    run_model(tokens, mode)

    return (time.perf_counter() - start) * 1000


def percentile_95(values):

    return statistics.quantiles(
        values,
        n=20,
        method="inclusive"
    )[18]


# --------------------------------------------------
# Benchmark execution
# --------------------------------------------------

all_results = []

for workload_number, workload in enumerate(
    workloads,
    start=1
):

    batch = workload["batch"]
    unique = workload["unique"]
    duplicate_rate = workload["actual_rate"]
    tokens = workload["tokens"]

    print("\n" + "-" * 75)

    print(
        f"Workload {workload_number}/{len(workloads)} | "
        f"Batch={batch} | "
        f"Unique={unique} | "
        f"Duplicates={duplicate_rate * 100:.1f}%"
    )

    # ----------------------------------------------
    # Correctness test
    # ----------------------------------------------

    with torch.inference_mode():

        standard_output = run_model(
            tokens,
            "standard"
        )

        adaptive_output = run_model(
            tokens,
            "adaptive"
        )

    max_difference = (
        standard_output - adaptive_output
    ).abs().max().item()

    correctness = torch.allclose(
        standard_output,
        adaptive_output,
        atol=1e-5,
        rtol=1e-5
    )

    del standard_output
    del adaptive_output

    if not correctness:

        raise RuntimeError(
            f"Correctness failed: batch={batch}, "
            f"duplicate_rate={duplicate_rate}"
        )

    # ----------------------------------------------
    # Warm-up
    # ----------------------------------------------

    for _ in range(WARMUP_RUNS):

        run_model(tokens, "standard")
        run_model(tokens, "adaptive")

    # ----------------------------------------------
    # Randomized paired measurements
    # ----------------------------------------------

    standard_times = []
    adaptive_times = []
    paired_speedups = []

    for repeat in range(REPEATS):

        order = [
            "standard",
            "adaptive"
        ]

        random.shuffle(order)

        pair_times = {}

        for mode in order:

            elapsed = measure_once(
                tokens,
                mode
            )

            pair_times[mode] = elapsed

        standard_times.append(
            pair_times["standard"]
        )

        adaptive_times.append(
            pair_times["adaptive"]
        )

        paired_speedups.append(
            pair_times["standard"]
            / pair_times["adaptive"]
        )

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    standard_median = statistics.median(
        standard_times
    )

    adaptive_median = statistics.median(
        adaptive_times
    )

    speedup = (
        standard_median / adaptive_median
    )

    paired_speedup = statistics.median(
        paired_speedups
    )

    reduction = (
        1 - adaptive_median / standard_median
    ) * 100

    result = {

        "batch": batch,
        "sequence": SEQUENCE_LENGTH,
        "unique_prompts": unique,
        "actual_duplicate_rate": duplicate_rate,

        "correctness": correctness,
        "max_difference": max_difference,

        "standard_median_ms": standard_median,
        "adaptive_median_ms": adaptive_median,

        "standard_p95_ms": percentile_95(
            standard_times
        ),

        "adaptive_p95_ms": percentile_95(
            adaptive_times
        ),

        "speedup": speedup,
        "paired_speedup": paired_speedup,
        "latency_reduction_percent": reduction
    }

    all_results.append(result)

    # ----------------------------------------------
    # Print workload results
    # ----------------------------------------------

    print(
        f"Standard median: {standard_median:.3f} ms"
    )

    print(
        f"Adaptive median: {adaptive_median:.3f} ms"
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

    print(
        f"Correctness: {correctness}"
    )

    print(
        f"Maximum difference: {max_difference:.8f}"
    )


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

output_path = (
    PROJECT_ROOT
    / "benchmarks"
    / "rct_221_results.csv"
)

fieldnames = list(all_results[0].keys())

with output_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 75)
print("RCT 2.2.1 FINAL SUMMARY")
print("=" * 75)

print(
    f"Total workloads: {len(all_results)}"
)

print(
    f"Correctness passed: "
    f"{sum(row['correctness'] for row in all_results)}"
    f"/{len(all_results)}"
)

print(
    "Average speedup: "
    f"{statistics.mean(row['speedup'] for row in all_results):.3f}x"
)

print(
    "Median workload speedup: "
    f"{statistics.median(row['speedup'] for row in all_results):.3f}x"
)

print(
    "Average paired speedup: "
    f"{statistics.mean(row['paired_speedup'] for row in all_results):.3f}x"
)

print(
    "Average latency reduction: "
    f"{statistics.mean(row['latency_reduction_percent'] for row in all_results):.2f}%"
)

print("\nCSV saved to:")
print(output_path)

print("\nRCT 2.2.1 BENCHMARK COMPLETED")
