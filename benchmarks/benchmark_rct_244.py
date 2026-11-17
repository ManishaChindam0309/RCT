
import csv
import math
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
SEED = 244

THRESHOLDS = [
    0.25,
    0.375,
    0.50,
    0.625,
    0.75
]

DETECTION_REPEATS = 30

torch.set_num_threads(THREADS)
torch.manual_seed(SEED)
random.seed(SEED)

print("=" * 80)
print("RCT 2.4.4 - ADAPTIVE THRESHOLD OPTIMIZATION")
print("=" * 80)


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

adaptive_models = {
    threshold: RCT100MAdaptive(
        base_model,
        duplicate_threshold=threshold
    )
    for threshold in THRESHOLDS
}

for model in adaptive_models.values():
    model.eval()

print("Parameters:", f"{base_model.parameter_count():,}")
print("CPU threads:", THREADS)
print("Thresholds:", THRESHOLDS)
print("Repetitions:", REPEATS)
print("New tokens:", NEW_TOKENS)


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
    / "rct_244_results.csv"
)

fieldnames = [
    "workload_id",
    "batch_size",
    "sequence_length",
    "unique_prompts",
    "duplicate_rate_percent",

    "threshold_percent",
    "required_duplicates",
    "expected_decision",
    "actual_decision",
    "decision_correct",
    "correctness",

    "kv_median_ms",
    "combined_median_ms",
    "adaptive_median_ms",

    "adaptive_vs_kv_speedup",
    "adaptive_vs_combined_speedup",
    "adaptive_latency_reduction_percent",

    "paired_adaptive_vs_kv_speedup",
    "paired_adaptive_vs_combined_speedup",

    "duplicate_detection_median_ms",
    "duplicate_detection_fraction_of_kv_percent",

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

    indices = torch.arange(batch) % unique_count
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


def run_adaptive(model, prompts):

    return model.generate(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0
    )


# ==================================================
# TIMING
# ==================================================

def measure(function, prompts, seed):

    torch.manual_seed(seed)

    start = time.perf_counter()

    with torch.inference_mode():
        output = function(prompts)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return elapsed_ms, output


# ==================================================
# DUPLICATE DETECTION TIMING
# ==================================================

def measure_detection(prompts, adaptive_model):

    # Warm-up detection.
    with torch.inference_mode():
        adaptive_model.stable_deduplicate(prompts)

    measurements = []

    for _ in range(DETECTION_REPEATS):

        start = time.perf_counter()

        with torch.inference_mode():
            unique_prompts, inverse_indices = (
                adaptive_model.stable_deduplicate(prompts)
            )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        measurements.append(elapsed_ms)

        del unique_prompts
        del inverse_indices

    return statistics.median(measurements)


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

                if not torch.equal(output[i], output[j]):
                    return False

    return True


# ==================================================
# EXPECTED ADAPTIVE DECISION
# ==================================================

def expected_decision(batch_size, unique_count, threshold):

    duplicate_count = batch_size - unique_count

    required_duplicates = math.ceil(
        threshold * batch_size
    )

    should_deduplicate = (
        duplicate_count >= required_duplicates
        and duplicate_count > 0
    )

    decision = (
        "deduplicated"
        if should_deduplicate
        else "kv_only"
    )

    return decision, required_duplicates


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

for workload_id, workload in enumerate(workloads, start=1):

    batch = workload["batch"]
    sequence = workload["seq"]
    unique_count = workload["unique"]

    duplicate_count = batch - unique_count

    duplicate_rate = (
        duplicate_count / batch
    ) * 100.0

    print("\n" + "=" * 80)
    print(f"WORKLOAD {workload_id}/{len(workloads)}")
    print("=" * 80)

    print(
        f"Batch={batch}, "
        f"Sequence={sequence}, "
        f"Unique={unique_count}, "
        f"Duplicates={duplicate_rate:.1f}%"
    )

    # ----------------------------------------------
    # Generate prompts
    # ----------------------------------------------

    prompts = create_prompts(
        batch,
        sequence,
        unique_count
    )

    # Use one model for stable exact deduplication.
    detector_model = adaptive_models[0.50]

    unique_prompts, inverse_indices = (
        detector_model.stable_deduplicate(prompts)
    )

    detected_unique_count = unique_prompts.size(0)

    if detected_unique_count != unique_count:

        raise RuntimeError(
            "Unexpected number of unique prompts"
        )

    # ----------------------------------------------
    # Measure duplicate detection separately
    # ----------------------------------------------

    detection_ms = measure_detection(
        prompts,
        detector_model
    )

    print(
        f"Duplicate detection median: "
        f"{detection_ms:.4f} ms"
    )

    # ----------------------------------------------
    # Store all methods
    # ----------------------------------------------

    methods = {
        "kv": run_kv,
        "combined": run_combined
    }

    for threshold, adaptive_model in adaptive_models.items():

        methods[f"adaptive_{threshold}"] = (
            lambda p, model=adaptive_model:
                run_adaptive(model, p)
        )

    # ----------------------------------------------
    # Correctness tests for every threshold
    # ----------------------------------------------

    correctness_seed = 70000 + workload_id

    threshold_correctness = {}

    for threshold, adaptive_model in adaptive_models.items():

        expected, required_duplicates = expected_decision(
            batch,
            detected_unique_count,
            threshold
        )

        if expected == "deduplicated":
            reference_prompts = unique_prompts
        else:
            reference_prompts = prompts

        torch.manual_seed(correctness_seed)

        with torch.inference_mode():
            reference_output = run_kv(reference_prompts)

        if expected == "deduplicated":

            expected_output = reference_output.index_select(
                0,
                inverse_indices
            )

        else:
            expected_output = reference_output

        torch.manual_seed(correctness_seed)

        with torch.inference_mode():
            adaptive_output = run_adaptive(
                adaptive_model,
                prompts
            )

        stats = adaptive_model.statistics()

        actual_decision = stats["last_decision"]

        decision_correct = (
            actual_decision == expected
        )

        output_correct = torch.equal(
            adaptive_output,
            expected_output
        )

        shape_correct = validate_shape(
            adaptive_output,
            prompts
        )

        prefix_correct = validate_prompt_preservation(
            adaptive_output,
            prompts
        )

        correctness = (
            decision_correct
            and output_correct
            and shape_correct
            and prefix_correct
        )

        threshold_correctness[threshold] = {
            "correctness": correctness,
            "decision_correct": decision_correct,
            "expected": expected,
            "actual": actual_decision,
            "required_duplicates": required_duplicates
        }

        print(
            f"Threshold {threshold * 100:5.1f}% | "
            f"Expected={expected:11s} | "
            f"Actual={actual_decision:11s} | "
            f"Correct={correctness}"
        )

        del reference_output
        del expected_output
        del adaptive_output

        if not correctness:

            raise RuntimeError(
                f"Correctness failed: workload={workload_id}, "
                f"threshold={threshold}"
            )

    # ----------------------------------------------
    # Combined correctness
    # ----------------------------------------------

    torch.manual_seed(correctness_seed)

    with torch.inference_mode():
        combined_output = run_combined(prompts)

    combined_correctness = (
        validate_shape(combined_output, prompts)
        and validate_prompt_preservation(
            combined_output,
            prompts
        )
        and validate_duplicate_consistency(
            combined_output,
            prompts
        )
    )

    del combined_output

    if not combined_correctness:

        raise RuntimeError(
            f"Combined correctness failed: workload {workload_id}"
        )

    print("Combined correctness:", combined_correctness)

    # ----------------------------------------------
    # Warm-up all methods
    # ----------------------------------------------

    for warmup in range(WARMUP_RUNS):

        warmup_seed = 80000 + workload_id + warmup

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
        name: []
        for name in methods
    }

    paired_ratios = {
        name: []
        for name in methods
        if name != "kv"
    }

    for repeat in range(REPEATS):

        order = list(methods.keys())
        random.shuffle(order)

        pair = {}

        current_seed = (
            SEED
            + workload_id * 10000
            + repeat
        )

        for name in order:

            elapsed, output = measure(
                methods[name],
                prompts,
                current_seed
            )

            timings[name].append(elapsed)
            pair[name] = elapsed

            del output

        for name in paired_ratios:

            paired_ratios[name].append(
                pair["kv"] / pair[name]
            )

    # ----------------------------------------------
    # Calculate medians
    # ----------------------------------------------

    medians = {
        name: statistics.median(values)
        for name, values in timings.items()
    }

    kv_median = medians["kv"]
    combined_median = medians["combined"]

    # ----------------------------------------------
    # Memory estimates
    # ----------------------------------------------

    kv_memory = cache_memory_mib(batch)

    deduplicated_memory = cache_memory_mib(
        detected_unique_count
    )

    # ----------------------------------------------
    # Save one CSV row per threshold
    # ----------------------------------------------

    for threshold in THRESHOLDS:

        method_name = f"adaptive_{threshold}"

        adaptive_median = medians[method_name]

        threshold_info = threshold_correctness[threshold]

        adaptive_vs_kv = (
            kv_median / adaptive_median
        )

        adaptive_vs_combined = (
            combined_median / adaptive_median
        )

        latency_reduction = (
            1.0 - adaptive_median / kv_median
        ) * 100.0

        detection_fraction = (
            detection_ms / kv_median
        ) * 100.0

        result = {
            "workload_id": workload_id,
            "batch_size": batch,
            "sequence_length": sequence,
            "unique_prompts": detected_unique_count,
            "duplicate_rate_percent": duplicate_rate,

            "threshold_percent": threshold * 100,
            "required_duplicates": (
                threshold_info["required_duplicates"]
            ),
            "expected_decision": threshold_info["expected"],
            "actual_decision": threshold_info["actual"],
            "decision_correct": (
                threshold_info["decision_correct"]
            ),
            "correctness": (
                threshold_info["correctness"]
                and combined_correctness
            ),

            "kv_median_ms": kv_median,
            "combined_median_ms": combined_median,
            "adaptive_median_ms": adaptive_median,

            "adaptive_vs_kv_speedup": adaptive_vs_kv,
            "adaptive_vs_combined_speedup": (
                adaptive_vs_combined
            ),
            "adaptive_latency_reduction_percent": (
                latency_reduction
            ),

            "paired_adaptive_vs_kv_speedup": (
                statistics.median(
                    paired_ratios[method_name]
                )
            ),
            "paired_adaptive_vs_combined_speedup": (
                statistics.median([
                    pair["combined"] / pair[method_name]
                    for pair in []
                ]) if False else
                statistics.median([
                    # Calculated from paired measurements below.
                    1.0
                ])
            ),

            "duplicate_detection_median_ms": detection_ms,
            "duplicate_detection_fraction_of_kv_percent": (
                detection_fraction
            ),

            "kv_cache_mib": kv_memory,
            "deduplicated_cache_mib": deduplicated_memory
        }

        results.append(result)

    # ----------------------------------------------
    # Correct paired adaptive vs combined ratios
    # ----------------------------------------------

    # Calculate from the actual paired timings.
    for threshold in THRESHOLDS:

        method_name = f"adaptive_{threshold}"

        paired_values = []

        for repeat in range(REPEATS):

            # The paired ratio is calculated using the
            # paired KV/combined/adaptive observations.
            # Reconstruct from stored per-repeat timings.
            paired_values.append(
                timings["combined"][repeat]
                / timings[method_name][repeat]
            )

        results[-len(THRESHOLDS) + THRESHOLDS.index(threshold)][
            "paired_adaptive_vs_combined_speedup"
        ] = statistics.median(paired_values)

    save_results()

    # ----------------------------------------------
    # Display workload results
    # ----------------------------------------------

    print("\nTiming results:")
    print(f"KV-only:   {kv_median:.2f} ms")
    print(f"Combined:  {combined_median:.2f} ms")

    for threshold in THRESHOLDS:

        adaptive_median = medians[f"adaptive_{threshold}"]

        speedup = kv_median / adaptive_median

        reduction = (
            1.0 - adaptive_median / kv_median
        ) * 100.0

        print(
            f"Adaptive {threshold * 100:5.1f}%: "
            f"{adaptive_median:.2f} ms | "
            f"{speedup:.3f}x | "
            f"{reduction:.2f}% reduction"
        )

    print("Saved completed workload to CSV.")


# ==================================================
# FINAL THRESHOLD SUMMARY
# ==================================================

print("\n" + "=" * 80)
print("RCT 2.4.4 - FINAL THRESHOLD COMPARISON")
print("=" * 80)

print("Total workloads:", len(workloads))
print("Thresholds:", len(THRESHOLDS))
print("Total benchmark rows:", len(results))

print(
    "Correct rows:",
    sum(row["correctness"] for row in results),
    "/",
    len(results)
)

print("\nTHRESHOLD SUMMARY")

summary = []

for threshold in THRESHOLDS:

    threshold_rows = [
        row for row in results
        if row["threshold_percent"] == threshold * 100
    ]

    average_speedup = statistics.mean(
        row["adaptive_vs_kv_speedup"]
        for row in threshold_rows
    )

    average_reduction = statistics.mean(
        row["adaptive_latency_reduction_percent"]
        for row in threshold_rows
    )

    average_latency = statistics.mean(
        row["adaptive_median_ms"]
        for row in threshold_rows
    )

    average_detection = statistics.mean(
        row["duplicate_detection_median_ms"]
        for row in threshold_rows
    )

    summary.append({
        "threshold": threshold,
        "speedup": average_speedup,
        "reduction": average_reduction,
        "latency": average_latency,
        "detection": average_detection
    })

    print(
        f"\nThreshold: {threshold * 100:.1f}%"
    )

    print(
        f"Average speedup: {average_speedup:.3f}x"
    )

    print(
        f"Average latency reduction: {average_reduction:.2f}%"
    )

    print(
        f"Average adaptive latency: {average_latency:.2f} ms"
    )

    print(
        f"Average detection time: {average_detection:.4f} ms"
    )


# Select best threshold by arithmetic mean of per-workload
# end-to-end latency ratios against KV-only.
best = max(summary, key=lambda item: item["speedup"])

print("\n" + "=" * 80)
print("BEST THRESHOLD BY MEAN WORKLOAD SPEEDUP")
print("=" * 80)

print(
    f"Best threshold: {best['threshold'] * 100:.1f}%"
)

print(
    f"Mean speedup: {best['speedup']:.3f}x"
)

print(
    f"Mean latency reduction: {best['reduction']:.2f}%"
)

print("\nResults saved to:")
print(output_path)

print("\nRCT 2.4.4 BENCHMARK COMPLETED")
