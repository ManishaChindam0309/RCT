
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined


# --------------------------------------------------
# Configuration
# --------------------------------------------------

THREADS = 4
REPEATS = 7
WARMUP_RUNS = 1
NEW_TOKENS = 8
BASE_SEED = 241

torch.set_num_threads(THREADS)
torch.manual_seed(BASE_SEED)
random.seed(BASE_SEED)

print("=" * 75)
print("RCT 2.4.1 - THREE-WAY PERFORMANCE BENCHMARK")
print("=" * 75)


# --------------------------------------------------
# Load shared model
# --------------------------------------------------

print("\nLoading RCT 100M...")

base_model = RCT100M()
base_model.eval()

kv_model = RCT100MKV(base_model)
kv_model.eval()

combined_model = RCT100MCombined(base_model)
combined_model.eval()

print("Parameters:", f"{base_model.parameter_count():,}")
print("CPU threads:", THREADS)
print("Repeats:", REPEATS)


# --------------------------------------------------
# Workloads
# --------------------------------------------------

workloads = [
    {"batch": 1, "sequence": 16, "unique": 1},
    {"batch": 4, "sequence": 16, "unique": 4},
    {"batch": 4, "sequence": 16, "unique": 2},
    {"batch": 8, "sequence": 16, "unique": 2},
    {"batch": 4, "sequence": 64, "unique": 4},
    {"batch": 4, "sequence": 64, "unique": 2},
    {"batch": 8, "sequence": 64, "unique": 2},
]


# --------------------------------------------------
# Prompt creation
# --------------------------------------------------

def create_prompts(batch_size, sequence_length, unique_count):

    unique_prompts = torch.randint(
        0,
        base_model.config.vocab_size,
        (unique_count, sequence_length),
        dtype=torch.long
    )

    indices = torch.arange(batch_size) % unique_count

    # Shuffle ordering so duplicates are not always adjacent.
    indices = indices[torch.randperm(batch_size)]

    return unique_prompts[indices].contiguous()


# --------------------------------------------------
# Generation functions
# --------------------------------------------------

def run_standard(prompts):

    with torch.inference_mode():

        return base_model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0
        )


def run_kv(prompts):

    with torch.inference_mode():

        return kv_model.generate_cached(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0
        )


def run_combined(prompts):

    with torch.inference_mode():

        return combined_model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0,
            deduplicate=True
        )


# --------------------------------------------------
# Timing
# --------------------------------------------------

def measure(function, prompts, seed):

    torch.manual_seed(seed)

    start = time.perf_counter()

    output = function(prompts)

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return elapsed, output


# --------------------------------------------------
# Correctness validation
# --------------------------------------------------

def validate_output(output, prompts):

    expected_shape = (
        prompts.size(0),
        prompts.size(1) + NEW_TOKENS
    )

    if output.shape != expected_shape:
        return False

    if not torch.equal(
        output[:, :prompts.size(1)],
        prompts
    ):
        return False

    return True


def validate_duplicate_outputs(output, prompts):

    for i in range(prompts.size(0)):

        for j in range(i + 1, prompts.size(0)):

            if torch.equal(prompts[i], prompts[j]):

                if not torch.equal(output[i], output[j]):
                    return False

    return True


# --------------------------------------------------
# Cache memory
# --------------------------------------------------

cache = kv_model.create_cache(batch_size=1)

cache_memory_mib = (
    cache.memory_bytes() / (1024 ** 2)
)

del cache

print(
    f"\nKV cache memory per sequence: "
    f"{cache_memory_mib:.2f} MiB"
)


# --------------------------------------------------
# Benchmark loop
# --------------------------------------------------

results = []

methods = {
    "standard": run_standard,
    "kv": run_kv,
    "combined": run_combined
}

for workload_index, workload in enumerate(workloads, start=1):

    batch_size = workload["batch"]
    sequence_length = workload["sequence"]
    unique_count = workload["unique"]

    duplicate_rate = (
        1 - unique_count / batch_size
    )

    prompts = create_prompts(
        batch_size,
        sequence_length,
        unique_count
    )

    print("\n" + "-" * 75)

    print(
        f"Workload {workload_index}/{len(workloads)}"
    )

    print(
        f"Batch={batch_size}, "
        f"Sequence={sequence_length}, "
        f"Unique={unique_count}, "
        f"Duplicates={duplicate_rate * 100:.1f}%"
    )

    # ----------------------------------------------
    # Correctness checks
    # ----------------------------------------------

    outputs = {}

    for method_index, (name, function) in enumerate(methods.items()):

        _, output = measure(
            function,
            prompts,
            seed=9000 + method_index
        )

        outputs[name] = output

        if not validate_output(output, prompts):

            raise RuntimeError(
                f"{name} output validation failed"
            )

    # Standard and KV should reproduce identical
    # sampling with the same seed.
    if unique_count == batch_size:

        torch.manual_seed(777)

        standard_check = run_standard(prompts)

        torch.manual_seed(777)

        kv_check = run_kv(prompts)

        if not torch.equal(standard_check, kv_check):

            raise RuntimeError(
                "Standard and KV generation mismatch"
            )

        torch.manual_seed(777)

        combined_check = run_combined(prompts)

        if not torch.equal(standard_check, combined_check):

            raise RuntimeError(
                "Combined and standard generation mismatch "
                "for unique prompts"
            )

        correctness = True

    else:

        # For duplicate prompts, combined generation
        # deliberately samples once and expands the result.
        correctness = validate_duplicate_outputs(
            outputs["combined"],
            prompts
        )

        if not correctness:

            raise RuntimeError(
                "Combined duplicate output validation failed"
            )

    print("Output validation:", correctness)

    # ----------------------------------------------
    # Warm-up
    # ----------------------------------------------

    for warmup in range(WARMUP_RUNS):

        seed = 10000 + workload_index + warmup

        for function in methods.values():

            measure(function, prompts, seed)

    # ----------------------------------------------
    # Randomized paired measurements
    # ----------------------------------------------

    timings = {
        "standard": [],
        "kv": [],
        "combined": []
    }

    paired_speedups = {
        "kv": [],
        "combined": []
    }

    for repeat in range(REPEATS):

        order = list(methods.keys())
        random.shuffle(order)

        pair = {}

        seed = BASE_SEED + workload_index * 100 + repeat

        for name in order:

            elapsed, _ = measure(
                methods[name],
                prompts,
                seed
            )

            timings[name].append(elapsed)
            pair[name] = elapsed

        paired_speedups["kv"].append(
            pair["standard"] / pair["kv"]
        )

        paired_speedups["combined"].append(
            pair["standard"] / pair["combined"]
        )

    # ----------------------------------------------
    # Statistics
    # ----------------------------------------------

    medians = {
        name: statistics.median(values)
        for name, values in timings.items()
    }

    kv_speedup = (
        medians["standard"] / medians["kv"]
    )

    combined_speedup = (
        medians["standard"] / medians["combined"]
    )

    combined_vs_kv = (
        medians["kv"] / medians["combined"]
    )

    kv_reduction = (
        1 - medians["kv"] / medians["standard"]
    ) * 100

    combined_reduction = (
        1 - medians["combined"] / medians["standard"]
    ) * 100

    result = {

        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "unique_prompts": unique_count,
        "duplicate_rate_percent": duplicate_rate * 100,

        "correctness": correctness,

        "standard_median_ms": medians["standard"],
        "kv_median_ms": medians["kv"],
        "combined_median_ms": medians["combined"],

        "kv_speedup": kv_speedup,
        "combined_speedup": combined_speedup,
        "combined_vs_kv_speedup": combined_vs_kv,

        "kv_latency_reduction_percent": kv_reduction,
        "combined_latency_reduction_percent": combined_reduction,

        "paired_kv_speedup": statistics.median(
            paired_speedups["kv"]
        ),

        "paired_combined_speedup": statistics.median(
            paired_speedups["combined"]
        ),

        "kv_cache_memory_mib_per_sequence": cache_memory_mib
    }

    results.append(result)

    print(
        f"Standard: {medians['standard']:.2f} ms"
    )

    print(
        f"KV only:  {medians['kv']:.2f} ms"
    )

    print(
        f"Combined: {medians['combined']:.2f} ms"
    )

    print(
        f"KV speedup: {kv_speedup:.3f}x"
    )

    print(
        f"Combined speedup: {combined_speedup:.3f}x"
    )

    print(
        f"Combined vs KV: {combined_vs_kv:.3f}x"
    )


# --------------------------------------------------
# Save CSV
# --------------------------------------------------

output_path = (
    PROJECT_ROOT
    / "benchmarks"
    / "rct_241_results.csv"
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
print("RCT 2.4.1 FINAL SUMMARY")
print("=" * 75)

print("Workloads:", len(results))

print(
    "Correct:",
    sum(r["correctness"] for r in results),
    "/",
    len(results)
)

print(
    "Average KV speedup:",
    f"{statistics.mean(r['kv_speedup'] for r in results):.3f}x"
)

print(
    "Average combined speedup:",
    f"{statistics.mean(r['combined_speedup'] for r in results):.3f}x"
)

print(
    "Average combined vs KV:",
    f"{statistics.mean(r['combined_vs_kv_speedup'] for r in results):.3f}x"
)

print(
    "Average combined latency reduction:",
    f"{statistics.mean(r['combined_latency_reduction_percent'] for r in results):.2f}%"
)

print("\nSaved:", output_path)

print("\nRCT 2.4.1 BENCHMARK COMPLETED")
