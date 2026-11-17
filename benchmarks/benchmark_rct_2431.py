
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import torch


# ==================================================
# PROJECT IMPORTS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined
from model.rct_100m_adaptive import RCT100MAdaptive


# ==================================================
# CONFIGURATION
# ==================================================

THREADS = 4
REPEATS = 7
WARMUP_RUNS = 1
NEW_TOKENS = 8
SEED = 243
THRESHOLD = 0.50

torch.set_num_threads(THREADS)
torch.manual_seed(SEED)
random.seed(SEED)

print("=" * 75)
print("RCT 2.4.3.1 - ADAPTIVE DEDUPLICATION BENCHMARK")
print("=" * 75)


# ==================================================
# LOAD MODELS
# ==================================================

print("\nLoading RCT 100M...")

base_model = RCT100M()
base_model.eval()

kv_model = RCT100MKV(base_model)
kv_model.eval()

combined_model = RCT100MCombined(base_model)
combined_model.eval()

adaptive_model = RCT100MAdaptive(
    base_model,
    duplicate_threshold=THRESHOLD
)
adaptive_model.eval()

print("Parameters:", f"{base_model.parameter_count():,}")
print("CPU threads:", THREADS)
print("Threshold:", f"{THRESHOLD * 100:.0f}%")
print("Repetitions:", REPEATS)


# ==================================================
# WORKLOAD MATRIX
# ==================================================

workloads = [
    {"batch": 1, "seq": 16, "unique": 1},

    {"batch": 2, "seq": 16, "unique": 2},
    {"batch": 2, "seq": 16, "unique": 1},

    {"batch": 4, "seq": 16, "unique": 4},
    {"batch": 4, "seq": 16, "unique": 3},
    {"batch": 4, "seq": 16, "unique": 2},
    {"batch": 4, "seq": 16, "unique": 1},

    {"batch": 8, "seq": 64, "unique": 8},
    {"batch": 8, "seq": 64, "unique": 4},
    {"batch": 8, "seq": 64, "unique": 2},
    {"batch": 8, "seq": 64, "unique": 1},

    {"batch": 16, "seq": 128, "unique": 16},
    {"batch": 16, "seq": 128, "unique": 8},
    {"batch": 16, "seq": 128, "unique": 4},
    {"batch": 16, "seq": 128, "unique": 2},
]


# ==================================================
# CSV SETUP
# ==================================================

output_path = (
    PROJECT_ROOT
    / "benchmarks"
    / "rct_2431_results.csv"
)

fieldnames = [
    "batch_size",
    "sequence_length",
    "unique_prompts",
    "duplicate_rate_percent",
    "adaptive_decision",
    "correctness",

    "kv_median_ms",
    "combined_median_ms",
    "adaptive_median_ms",

    "combined_vs_kv_speedup",
    "adaptive_vs_kv_speedup",
    "adaptive_vs_combined_speedup",

    "combined_latency_reduction_percent",
    "adaptive_latency_reduction_percent",

    "paired_adaptive_vs_kv_speedup",
    "paired_adaptive_vs_combined_speedup",

    "kv_cache_mib",
    "deduplicated_cache_mib"
]

results = []


def save_results():
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)


# Create an empty CSV before benchmarking.
save_results()


# ==================================================
# PROMPT GENERATION
# ==================================================

def create_prompts(batch, sequence, unique_count):

    unique_prompts = torch.randint(
        0,
        base_model.config.vocab_size,
        (unique_count, sequence),
        dtype=torch.long
    )

    indices = (
        torch.arange(batch) % unique_count
    )

    indices = indices[torch.randperm(batch)]

    return unique_prompts[indices].contiguous()


# ==================================================
# GENERATION METHODS
# ==================================================

def run_kv(prompts):

    return kv_model.generate_cached(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0
    )


def run_combined(prompts):

    return combined_model.generate(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0,
        deduplicate=True
    )


def run_adaptive(prompts):

    return adaptive_model.generate(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0
    )


methods = {
    "kv": run_kv,
    "combined": run_combined,
    "adaptive": run_adaptive
}


# ==================================================
# TIMING
# ==================================================

def measure(function, prompts, seed):

    # Reset sampling RNG so all methods receive
    # the same initial random-number-generator state.
    torch.manual_seed(seed)

    start = time.perf_counter()

    with torch.inference_mode():
        output = function(prompts)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return elapsed_ms, output


# ==================================================
# VALIDATION HELPERS
# ==================================================

def validate_shape(output, prompts):

    expected_shape = (
        prompts.size(0),
        prompts.size(1) + NEW_TOKENS
    )

    return output.shape == expected_shape


def validate_prompt_preservation(output, prompts):

    return torch.equal(
        output[:, :prompts.size(1)],
        prompts
    )


def validate_duplicate_consistency(output, prompts):

    for i in range(prompts.size(0)):

        for j in range(i + 1, prompts.size(0)):

            if torch.equal(prompts[i], prompts[j]):

                if not torch.equal(
                    output[i],
                    output[j]
                ):
                    return False

    return True


# ==================================================
# EXPECTED ADAPTIVE DECISION
# ==================================================

def expected_decision(batch_size, unique_count):

    duplicate_count = batch_size - unique_count

    required_duplicates = (
        __import__("math").ceil(
            THRESHOLD * batch_size
        )
    )

    should_deduplicate = (
        duplicate_count >= required_duplicates
        and duplicate_count > 0
    )

    return (
        "deduplicated"
        if should_deduplicate
        else "kv_only"
    )


# ==================================================
# CACHE MEMORY ESTIMATION
# ==================================================

def cache_memory_mib(sequence_count):

    config = base_model.config

    total_bytes = (
        2
        * config.num_layers
        * sequence_count
        * config.max_seq_len
        * config.hidden_size
        * 4
    )

    return total_bytes / (1024 ** 2)


# ==================================================
# BENCHMARK LOOP
# ==================================================

for workload_id, workload in enumerate(
    workloads,
    start=1
):

    batch = workload["batch"]
    sequence = workload["seq"]
    unique_count = workload["unique"]

    duplicate_count = batch - unique_count

    duplicate_rate = (
        duplicate_count / batch
    ) * 100.0

    print("\n" + "-" * 75)
    print(f"Workload {workload_id}/{len(workloads)}")

    print(
        f"Batch={batch}, "
        f"Sequence={sequence}, "
        f"Unique={unique_count}, "
        f"Duplicates={duplicate_rate:.1f}%"
    )

    # ----------------------------------------------
    # Generate synthetic prompts
    # ----------------------------------------------

    prompts = create_prompts(
        batch,
        sequence,
        unique_count
    )

    # Stable first-occurrence ordering.
    unique_prompts, inverse_indices = (
        adaptive_model.stable_deduplicate(prompts)
    )

    detected_unique_count = unique_prompts.size(0)

    if detected_unique_count != unique_count:
        raise RuntimeError(
            "Generated workload has an unexpected unique count"
        )

    # ----------------------------------------------
    # Determine expected adaptive behavior
    # ----------------------------------------------

    decision = expected_decision(
        batch,
        detected_unique_count
    )

    dedup_expected = (
        decision == "deduplicated"
    )

    if dedup_expected:
        reference_prompts = unique_prompts
    else:
        reference_prompts = prompts

    # ----------------------------------------------
    # KV reference generation
    # ----------------------------------------------

    correctness_seed = 50000 + workload_id

    torch.manual_seed(correctness_seed)

    with torch.inference_mode():
        reference_output = run_kv(reference_prompts)

    if dedup_expected:

        expected_adaptive_output = (
            reference_output.index_select(
                0,
                inverse_indices
            )
        )

    else:

        expected_adaptive_output = reference_output

    # ----------------------------------------------
    # Adaptive correctness
    # ----------------------------------------------

    torch.manual_seed(correctness_seed)

    with torch.inference_mode():
        adaptive_output = run_adaptive(prompts)

    adaptive_matches_reference = torch.equal(
        adaptive_output,
        expected_adaptive_output
    )

    adaptive_shape_correct = validate_shape(
        adaptive_output,
        prompts
    )

    adaptive_prefix_correct = (
        validate_prompt_preservation(
            adaptive_output,
            prompts
        )
    )

    # Read the decision actually made by the model.
    adaptive_stats = adaptive_model.statistics()

    actual_decision = adaptive_stats["last_decision"]

    decision_correct = (
        actual_decision == decision
    )

    # ----------------------------------------------
    # Combined correctness
    # ----------------------------------------------

    torch.manual_seed(correctness_seed)

    with torch.inference_mode():
        combined_output = run_combined(prompts)

    combined_shape_correct = validate_shape(
        combined_output,
        prompts
    )

    combined_prefix_correct = (
        validate_prompt_preservation(
            combined_output,
            prompts
        )
    )

    combined_duplicates_correct = (
        validate_duplicate_consistency(
            combined_output,
            prompts
        )
    )

    # ----------------------------------------------
    # Final correctness
    # ----------------------------------------------

    correctness = (
        adaptive_matches_reference
        and adaptive_shape_correct
        and adaptive_prefix_correct
        and decision_correct
        and combined_shape_correct
        and combined_prefix_correct
        and combined_duplicates_correct
    )

    print(
        "Adaptive matches KV reference:",
        adaptive_matches_reference
    )

    print(
        "Adaptive shape:",
        adaptive_shape_correct
    )

    print(
        "Adaptive prompt preservation:",
        adaptive_prefix_correct
    )

    print(
        "Combined shape:",
        combined_shape_correct
    )

    print(
        "Combined prompt preservation:",
        combined_prefix_correct
    )

    print(
        "Combined duplicate consistency:",
        combined_duplicates_correct
    )

    print(
        "Expected adaptive decision:",
        decision
    )

    print(
        "Actual adaptive decision:",
        actual_decision
    )

    print("Correctness:", correctness)

    del reference_output
    del expected_adaptive_output
    del adaptive_output
    del combined_output

    if not correctness:

        raise RuntimeError(
            f"Correctness failed in workload {workload_id}"
        )

    # ----------------------------------------------
    # Warm-up
    # ----------------------------------------------

    for warmup in range(WARMUP_RUNS):

        warmup_seed = (
            60000 + workload_id + warmup
        )

        for function in methods.values():

            _, warmup_output = measure(
                function,
                prompts,
                warmup_seed
            )

            del warmup_output

    # ----------------------------------------------
    # Randomized paired timing
    # ----------------------------------------------

    timings = {
        "kv": [],
        "combined": [],
        "adaptive": []
    }

    paired_adaptive_kv = []
    paired_adaptive_combined = []

    for repeat in range(REPEATS):

        order = list(methods.keys())
        random.shuffle(order)

        pair = {}

        current_seed = (
            SEED
            + workload_id * 1000
            + repeat
        )

        for name in order:

            elapsed, timed_output = measure(
                methods[name],
                prompts,
                current_seed
            )

            timings[name].append(elapsed)
            pair[name] = elapsed

            del timed_output

        paired_adaptive_kv.append(
            pair["kv"] / pair["adaptive"]
        )

        paired_adaptive_combined.append(
            pair["combined"] / pair["adaptive"]
        )

    # ----------------------------------------------
    # Calculate medians and speedups
    # ----------------------------------------------

    medians = {
        name: statistics.median(values)
        for name, values in timings.items()
    }

    combined_vs_kv = (
        medians["kv"] / medians["combined"]
    )

    adaptive_vs_kv = (
        medians["kv"] / medians["adaptive"]
    )

    adaptive_vs_combined = (
        medians["combined"] / medians["adaptive"]
    )

    combined_reduction = (
        1.0 - medians["combined"] / medians["kv"]
    ) * 100.0

    adaptive_reduction = (
        1.0 - medians["adaptive"] / medians["kv"]
    ) * 100.0

    # ----------------------------------------------
    # Cache memory estimates
    # ----------------------------------------------

    kv_memory = cache_memory_mib(batch)

    deduplicated_memory = cache_memory_mib(
        detected_unique_count
    )

    # ----------------------------------------------
    # Save result
    # ----------------------------------------------

    result = {
        "batch_size": batch,
        "sequence_length": sequence,
        "unique_prompts": detected_unique_count,
        "duplicate_rate_percent": duplicate_rate,
        "adaptive_decision": actual_decision,
        "correctness": correctness,

        "kv_median_ms": medians["kv"],
        "combined_median_ms": medians["combined"],
        "adaptive_median_ms": medians["adaptive"],

        "combined_vs_kv_speedup": combined_vs_kv,
        "adaptive_vs_kv_speedup": adaptive_vs_kv,
        "adaptive_vs_combined_speedup": adaptive_vs_combined,

        "combined_latency_reduction_percent": combined_reduction,
        "adaptive_latency_reduction_percent": adaptive_reduction,

        "paired_adaptive_vs_kv_speedup": statistics.median(
            paired_adaptive_kv
        ),

        "paired_adaptive_vs_combined_speedup": statistics.median(
            paired_adaptive_combined
        ),

        "kv_cache_mib": kv_memory,
        "deduplicated_cache_mib": deduplicated_memory
    }

    results.append(result)

    # Save after every completed workload.
    save_results()

    # ----------------------------------------------
    # Display results
    # ----------------------------------------------

    print(f"KV-only: {medians['kv']:.2f} ms")
    print(f"Combined: {medians['combined']:.2f} ms")
    print(f"Adaptive: {medians['adaptive']:.2f} ms")

    print(
        f"Combined vs KV: {combined_vs_kv:.3f}x"
    )

    print(
        f"Adaptive vs KV: {adaptive_vs_kv:.3f}x"
    )

    print(
        f"Adaptive vs Combined: {adaptive_vs_combined:.3f}x"
    )

    print(
        f"Adaptive latency change vs KV: "
        f"{adaptive_reduction:.2f}%"
    )

    print("Saved completed workload to CSV.")


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 75)
print("RCT 2.4.3.1 FINAL SUMMARY")
print("=" * 75)

print("Completed workloads:", len(results))

print(
    "Correctness:",
    sum(row["correctness"] for row in results),
    "/",
    len(results)
)

print(
    "Average combined vs KV:",
    f"{statistics.mean(r['combined_vs_kv_speedup'] for r in results):.3f}x"
)

print(
    "Average adaptive vs KV:",
    f"{statistics.mean(r['adaptive_vs_kv_speedup'] for r in results):.3f}x"
)

print(
    "Average adaptive vs combined:",
    f"{statistics.mean(r['adaptive_vs_combined_speedup'] for r in results):.3f}x"
)

print(
    "Average adaptive latency change vs KV:",
    f"{statistics.mean(r['adaptive_latency_reduction_percent'] for r in results):.2f}%"
)

print("\nResults saved to:")
print(output_path)

print("\nRCT 2.4.3.1 BENCHMARK COMPLETED")
