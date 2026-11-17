
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
REPEATS = 20
WARMUP_RUNS = 1
NEW_TOKENS = 8

SEEDS = [2441, 2442, 2443]

THRESHOLDS = [
    0.25,
    0.375,
    0.50
]

DETECTION_REPEATS = 30

torch.set_num_threads(THREADS)

print("=" * 80)
print("RCT 2.4.4.1 - REPRODUCIBILITY VALIDATION")
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
print("Seeds:", SEEDS)
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
    / "rct_2441_results.csv"
)

fieldnames = [
    "seed",
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
    "adaptive_correctness",
    "combined_correctness",

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

def measure_detection(prompts, detector):

    with torch.inference_mode():
        detector.stable_deduplicate(prompts)

    measurements = []

    for _ in range(DETECTION_REPEATS):

        start = time.perf_counter()

        with torch.inference_mode():
            unique_prompts, inverse_indices = (
                detector.stable_deduplicate(prompts)
            )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        measurements.append(elapsed_ms)

        del unique_prompts
        del inverse_indices

    return statistics.median(measurements)


# ==================================================
# VALIDATION
# ==================================================

def validate_shape(output, prompts):

    expected_shape = (
        prompts.size(0),
        prompts.size(1) + NEW_TOKENS
    )

    return output.shape == expected_shape


def validate_prefix(output, prompts):

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

def expected_decision(batch, unique_count, threshold):

    duplicate_count = batch - unique_count

    required_duplicates = math.ceil(
        threshold * batch
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
# CACHE MEMORY
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
# BENCHMARK
# ==================================================

for workload_id, workload in enumerate(workloads, start=1):

    batch = workload["batch"]
    sequence = workload["seq"]
    unique_count = workload["unique"]

    duplicate_count = batch - unique_count

    duplicate_rate = (
        duplicate_count / batch
    ) * 100.0

    for seed_index, experiment_seed in enumerate(SEEDS):

        print("\n" + "=" * 80)
        print(
            f"WORKLOAD {workload_id}/{len(workloads)} "
            f"| SEED {experiment_seed}"
        )
        print("=" * 80)

        print(
            f"Batch={batch}, "
            f"Sequence={sequence}, "
            f"Unique={unique_count}, "
            f"Duplicates={duplicate_rate:.1f}%"
        )

        # ------------------------------------------
        # Reproducible prompt generation
        # ------------------------------------------

        torch.manual_seed(
            experiment_seed * 10000 + workload_id
        )

        prompts = create_prompts(
            batch,
            sequence,
            unique_count
        )

        detector = adaptive_models[0.50]

        unique_prompts, inverse_indices = (
            detector.stable_deduplicate(prompts)
        )

        detected_unique_count = unique_prompts.size(0)

        if detected_unique_count != unique_count:
            raise RuntimeError("Unexpected unique prompt count")

        # ------------------------------------------
        # Detection overhead
        # ------------------------------------------

        detection_ms = measure_detection(
            prompts,
            detector
        )

        # ------------------------------------------
        # Method collection
        # ------------------------------------------

        methods = {
            "kv": run_kv,
            "combined": run_combined
        }

        for threshold, adaptive_model in adaptive_models.items():

            methods[f"adaptive_{threshold}"] = (
                lambda p, model=adaptive_model:
                    run_adaptive(model, p)
            )

        # ------------------------------------------
        # Correctness validation
        # ------------------------------------------

        correctness_seed = (
            9000000
            + experiment_seed * 100
            + workload_id
        )

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

            actual = stats["last_decision"]

            decision_correct = actual == expected

            output_correct = torch.equal(
                adaptive_output,
                expected_output
            )

            shape_correct = validate_shape(
                adaptive_output,
                prompts
            )

            prefix_correct = validate_prefix(
                adaptive_output,
                prompts
            )

            adaptive_correctness = (
                decision_correct
                and output_correct
                and shape_correct
                and prefix_correct
            )

            threshold_correctness[threshold] = {
                "expected": expected,
                "actual": actual,
                "required_duplicates": required_duplicates,
                "decision_correct": decision_correct,
                "correctness": adaptive_correctness
            }

            print(
                f"Threshold {threshold * 100:5.1f}% | "
                f"Expected={expected:11s} | "
                f"Actual={actual:11s} | "
                f"Correct={adaptive_correctness}"
            )

            del reference_output
            del expected_output
            del adaptive_output

            if not adaptive_correctness:
                raise RuntimeError(
                    f"Adaptive correctness failed: "
                    f"workload={workload_id}, "
                    f"seed={experiment_seed}, "
                    f"threshold={threshold}"
                )

        # ------------------------------------------
        # Combined correctness
        # ------------------------------------------

        correctness_seed_combined = correctness_seed + 1

        torch.manual_seed(correctness_seed_combined)

        with torch.inference_mode():
            combined_output = run_combined(prompts)

        combined_correctness = (
            validate_shape(combined_output, prompts)
            and validate_prefix(combined_output, prompts)
            and validate_duplicate_consistency(
                combined_output,
                prompts
            )
        )

        del combined_output

        if not combined_correctness:
            raise RuntimeError(
                f"Combined correctness failed: "
                f"workload={workload_id}, "
                f"seed={experiment_seed}"
            )

        # ------------------------------------------
        # Warm-up
        # ------------------------------------------

        for warmup in range(WARMUP_RUNS):

            warmup_seed = (
                experiment_seed * 100000
                + workload_id * 100
                + warmup
            )

            for function in methods.values():

                _, output = measure(
                    function,
                    prompts,
                    warmup_seed
                )

                del output

        # ------------------------------------------
        # Randomized paired measurements
        # ------------------------------------------

        timings = {
            name: []
            for name in methods
        }

        paired_adaptive_kv = {
            threshold: []
            for threshold in THRESHOLDS
        }

        paired_adaptive_combined = {
            threshold: []
            for threshold in THRESHOLDS
        }

        order_rng = random.Random(
            experiment_seed * 1000 + workload_id
        )

        for repeat in range(REPEATS):

            order = list(methods.keys())
            order_rng.shuffle(order)

            pair = {}

            timing_seed = (
                experiment_seed * 1000000
                + workload_id * 1000
                + repeat
            )

            for name in order:

                elapsed, output = measure(
                    methods[name],
                    prompts,
                    timing_seed
                )

                timings[name].append(elapsed)
                pair[name] = elapsed

                del output

            # Correct paired comparisons:
            # All ratios use measurements from this
            # exact repeat and this exact workload.

            for threshold in THRESHOLDS:

                method_name = f"adaptive_{threshold}"

                paired_adaptive_kv[threshold].append(
                    pair["kv"] / pair[method_name]
                )

                paired_adaptive_combined[threshold].append(
                    pair["combined"] / pair[method_name]
                )

        # ------------------------------------------
        # Median latency
        # ------------------------------------------

        medians = {
            name: statistics.median(values)
            for name, values in timings.items()
        }

        kv_median = medians["kv"]
        combined_median = medians["combined"]

        # ------------------------------------------
        # Memory estimates
        # ------------------------------------------

        kv_memory = cache_memory_mib(batch)

        deduplicated_memory = cache_memory_mib(
            detected_unique_count
        )

        # ------------------------------------------
        # CSV rows
        # ------------------------------------------

        for threshold in THRESHOLDS:

            method_name = f"adaptive_{threshold}"

            adaptive_median = medians[method_name]

            info = threshold_correctness[threshold]

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
                "seed": experiment_seed,
                "workload_id": workload_id,
                "batch_size": batch,
                "sequence_length": sequence,
                "unique_prompts": detected_unique_count,
                "duplicate_rate_percent": duplicate_rate,

                "threshold_percent": threshold * 100,
                "required_duplicates": info["required_duplicates"],
                "expected_decision": info["expected"],
                "actual_decision": info["actual"],

                "decision_correct": info["decision_correct"],
                "adaptive_correctness": info["correctness"],
                "combined_correctness": combined_correctness,

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
                        paired_adaptive_kv[threshold]
                    )
                ),

                "paired_adaptive_vs_combined_speedup": (
                    statistics.median(
                        paired_adaptive_combined[threshold]
                    )
                ),

                "duplicate_detection_median_ms": detection_ms,

                "duplicate_detection_fraction_of_kv_percent": (
                    detection_fraction
                ),

                "kv_cache_mib": kv_memory,
                "deduplicated_cache_mib": deduplicated_memory
            }

            results.append(result)

        save_results()

        # ------------------------------------------
        # Display
        # ------------------------------------------

        print("\nMedian timing results:")

        print(f"KV-only:  {kv_median:.2f} ms")
        print(f"Combined: {combined_median:.2f} ms")

        for threshold in THRESHOLDS:

            adaptive_median = medians[
                f"adaptive_{threshold}"
            ]

            print(
                f"Adaptive {threshold * 100:5.1f}%: "
                f"{adaptive_median:.2f} ms | "
                f"{kv_median / adaptive_median:.3f}x"
            )

        print(
            f"Detection overhead: {detection_ms:.4f} ms"
        )

        print("Seed/workload results saved.")


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 80)
print("RCT 2.4.4.1 - FINAL SUMMARY")
print("=" * 80)

print("Seeds:", len(SEEDS))
print("Workloads:", len(workloads))
print("Thresholds:", len(THRESHOLDS))
print("Total result rows:", len(results))

correct_rows = sum(
    row["adaptive_correctness"]
    and row["combined_correctness"]
    and row["decision_correct"]
    for row in results
)

print(
    "Correct rows:",
    correct_rows,
    "/",
    len(results)
)

print("\nTHRESHOLD COMPARISON")

summary = []

for threshold in THRESHOLDS:

    rows = [
        row for row in results
        if row["threshold_percent"] == threshold * 100
    ]

    mean_speedup = statistics.mean(
        row["adaptive_vs_kv_speedup"]
        for row in rows
    )

    median_speedup = statistics.median(
        row["adaptive_vs_kv_speedup"]
        for row in rows
    )

    mean_reduction = statistics.mean(
        row["adaptive_latency_reduction_percent"]
        for row in rows
    )

    mean_adaptive_latency = statistics.mean(
        row["adaptive_median_ms"]
        for row in rows
    )

    mean_kv_latency = statistics.mean(
        row["kv_median_ms"]
        for row in rows
    )

    aggregate_speedup = (
        mean_kv_latency / mean_adaptive_latency
    )

    mean_detection = statistics.mean(
        row["duplicate_detection_median_ms"]
        for row in rows
    )

    summary.append({
        "threshold": threshold,
        "mean_speedup": mean_speedup,
        "median_speedup": median_speedup,
        "mean_reduction": mean_reduction,
        "aggregate_speedup": aggregate_speedup,
        "mean_detection": mean_detection
    })

    print(f"\nThreshold: {threshold * 100:.1f}%")

    print(f"Mean workload speedup: {mean_speedup:.3f}x")
    print(f"Median workload speedup: {median_speedup:.3f}x")

    print(
        f"Mean latency reduction: {mean_reduction:.2f}%"
    )

    print(
        f"Aggregate latency ratio: {aggregate_speedup:.3f}x"
    )

    print(
        f"Mean detection time: {mean_detection:.4f} ms"
    )


# ==================================================
# BEST THRESHOLD
# ==================================================

best = max(
    summary,
    key=lambda item: item["mean_speedup"]
)

print("\n" + "=" * 80)
print("PRELIMINARY BEST THRESHOLD")
print("=" * 80)

print(
    f"Threshold: {best['threshold'] * 100:.1f}%"
)

print(
    f"Mean workload speedup: {best['mean_speedup']:.3f}x"
)

print(
    f"Median workload speedup: {best['median_speedup']:.3f}x"
)

print(
    f"Mean latency reduction: {best['mean_reduction']:.2f}%"
)

print("\nCSV saved to:")
print(output_path)

print("\nRCT 2.4.4.1 COMPLETED")
