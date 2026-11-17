
import sys
import gc
import csv
import json
import time
import random
import statistics
import itertools
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import psutil

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT = (
    PROJECT_DIR / "checkpoints" /
    "rct100m_continued" / "best_model.pt"
)

VALIDATION_DATA = (
    PROJECT_DIR / "data" / "rct30" /
    "validation_expanded_tokens.bin"
)

BATCH_SIZES = [4, 8, 16]
PROMPT_LENGTHS = [8, 16, 24]
DUPLICATION_RATES = [0.0, 0.50, 0.75]

NEW_TOKENS = 8

NUM_CLUSTERS = 20
BENCHMARK_RUNS = 3
WARMUP_RUNS = 1
BOOTSTRAP_ITERATIONS = 20000

CPU_THREADS = 4
MIN_AVAILABLE_RAM_GB = 3.0

SEED = 20261104

METHODS = [
    "standard",
    "kv_cache",
    "combined_no_dedup",
    "combined_dedup",
]

TIMESTAMP = datetime.now(
    timezone.utc
).strftime("%Y%m%dT%H%M%SZ")

OUTPUT_DIR = (
    PROJECT_DIR / "results" /
    "rct100m_generalization" /
    f"run_{TIMESTAMP}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RAW_CSV = OUTPUT_DIR / "raw_timing_records.csv"
PROGRESS_JSON = OUTPUT_DIR / "progress.json"
RESULTS_JSON = OUTPUT_DIR / "experiment_results.json"


# =========================================================
# HELPERS
# =========================================================

def available_ram_gb():
    return psutil.virtual_memory().available / (1024 ** 3)


def check_memory(stage):
    available = available_ram_gb()

    print(
        f"RAM check [{stage}]: "
        f"{available:.2f} GB available"
    )

    if available < MIN_AVAILABLE_RAM_GB:
        raise RuntimeError(
            f"Insufficient available RAM at {stage}: "
            f"{available:.2f} GB. "
            f"Required: {MIN_AVAILABLE_RAM_GB:.2f} GB."
        )


def enable_greedy_sampling():
    original = torch.multinomial

    def greedy_multinomial(
        probabilities,
        num_samples,
        *args,
        **kwargs
    ):
        if num_samples != 1:
            raise ValueError(
                "Expected num_samples=1."
            )

        return probabilities.argmax(
            dim=-1,
            keepdim=True
        )

    torch.multinomial = greedy_multinomial

    return original


def create_workload(base_prompts, unique_count):
    batch_size = len(base_prompts)

    if unique_count == batch_size:
        return base_prompts.clone()

    if unique_count <= 0 or batch_size % unique_count != 0:
        raise ValueError(
            "Unique count must be positive and divide batch size."
        )

    repeats = batch_size // unique_count

    return base_prompts[:unique_count].repeat(
        repeats, 1
    )


def build_functions(
    base_model,
    kv_model,
    combined_model,
    prompts
):
    return {
        "standard": lambda: base_model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0
        ),

        "kv_cache": lambda: kv_model.generate_cached(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0
        ),

        "combined_no_dedup": lambda: combined_model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0,
            deduplicate=False
        ),

        "combined_dedup": lambda: combined_model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0,
            deduplicate=True
        ),
    }


def time_function(function):
    start = time.perf_counter()

    output = function()

    elapsed = time.perf_counter() - start

    return elapsed, output


# =========================================================
# WORKLOAD DESIGN
# =========================================================

def make_configurations():
    configurations = []

    for batch_size, prompt_length, duplication_rate in itertools.product(
        BATCH_SIZES,
        PROMPT_LENGTHS,
        DUPLICATION_RATES
    ):
        unique_count = int(
            round(batch_size * (1.0 - duplication_rate))
        )

        actual_rate = (
            batch_size - unique_count
        ) / batch_size

        name = (
            f"batch{batch_size}_"
            f"prompt{prompt_length}_"
            f"dup{int(round(actual_rate * 100))}"
        )

        configurations.append({
            "name": name,
            "batch_size": batch_size,
            "prompt_length": prompt_length,
            "unique_prompts": unique_count,
            "duplicate_prompts": batch_size - unique_count,
            "duplication_rate": actual_rate,
        })

    return configurations


# =========================================================
# UNIQUE PROMPT SELECTION
# =========================================================

def load_clusters(config, rng):
    """
    Select unique, non-overlapping prompt windows.

    Each batch-size/prompt-length combination receives
    its own disjoint validation-data region.

    Duplicate token sequences are filtered BEFORE
    random sampling.

    The returned clusters can be reused across all
    duplication-rate conditions for the same configuration
    family.
    """

    data = np.memmap(
        VALIDATION_DATA,
        dtype=np.uint16,
        mode="r"
    )

    prompt_length = config["prompt_length"]
    batch_size = config["batch_size"]

    # -----------------------------------------------------
    # Select one of nine disjoint regions.
    # -----------------------------------------------------

    prompt_index = PROMPT_LENGTHS.index(prompt_length)
    batch_index = BATCH_SIZES.index(batch_size)

    region_index = (
        prompt_index * len(BATCH_SIZES) + batch_index
    )

    region_size = len(data) // 9

    region_start = region_index * region_size

    region_end = (
        len(data)
        if region_index == 8
        else (region_index + 1) * region_size
    )

    # -----------------------------------------------------
    # Candidate non-overlapping windows.
    # -----------------------------------------------------

    possible_starts = np.arange(
        region_start,
        region_end - prompt_length,
        prompt_length,
        dtype=np.int64
    )

    required = NUM_CLUSTERS * batch_size

    print(
        "Candidate windows:",
        len(possible_starts)
    )

    # -----------------------------------------------------
    # Remove duplicate token sequences BEFORE sampling.
    # -----------------------------------------------------

    seen = set()
    unique_starts = []

    for start in possible_starts:

        window = np.asarray(
            data[start:start + prompt_length],
            dtype=np.uint16
        )

        fingerprint = window.tobytes()

        if fingerprint not in seen:
            seen.add(fingerprint)
            unique_starts.append(int(start))

    unique_candidate_count = len(unique_starts)

    print(
        "Unique candidate windows:",
        unique_candidate_count
    )

    del seen

    if unique_candidate_count < required:

        del data

        raise RuntimeError(
            f"Not enough unique prompts for {config['name']}. "
            f"Required: {required}; "
            f"available: {unique_candidate_count}."
        )

    # -----------------------------------------------------
    # Randomly select unique windows.
    # -----------------------------------------------------

    selected_indices = rng.choice(
        unique_candidate_count,
        size=required,
        replace=False
    )

    selected = np.asarray(
        unique_starts,
        dtype=np.int64
    )[selected_indices]

    # Sort selected positions for reproducible cluster order.
    selected.sort()

    # -----------------------------------------------------
    # Extract selected token sequences.
    # -----------------------------------------------------

    prompts = np.stack([
        np.array(
            data[start:start + prompt_length],
            dtype=np.int64
        )
        for start in selected
    ])

    del data
    del unique_starts
    del possible_starts
    del selected
    gc.collect()

    # -----------------------------------------------------
    # Final uniqueness verification.
    # -----------------------------------------------------

    actual_unique_count = len(
        np.unique(prompts, axis=0)
    )

    if actual_unique_count != len(prompts):
        raise RuntimeError(
            f"Duplicate base prompts detected in {config['name']}. "
            f"Selected: {len(prompts)}; "
            f"unique: {actual_unique_count}."
        )

    print(
        "Selected unique prompts:",
        actual_unique_count
    )

    prompts = torch.from_numpy(prompts).long()

    clusters = [
        prompts[
            i * batch_size:(i + 1) * batch_size
        ].clone()
        for i in range(NUM_CLUSTERS)
    ]

    del prompts
    gc.collect()

    return clusters


# =========================================================
# STATISTICS
# =========================================================

def bootstrap_ci(values, iterations, rng):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    if len(values) == 0:
        raise ValueError(
            "Cannot calculate confidence interval from empty data."
        )

    indices = rng.integers(
        0,
        len(values),
        size=(iterations, len(values))
    )

    samples = values[indices]

    bootstrap_medians = np.median(
        samples,
        axis=1
    )

    lower, upper = np.percentile(
        bootstrap_medians,
        [2.5, 97.5]
    )

    return {
        "median": float(np.median(values)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
    }


# =========================================================
# SAVE PROGRESS
# =========================================================

def save_progress(completed, total, records):
    data = {
        "completed_configurations": completed,
        "total_configurations": total,
        "raw_timing_records": records,
        "status": (
            "running"
            if completed < total
            else "complete"
        ),
        "updated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with PROGRESS_JSON.open(
        "w",
        encoding="utf-8"
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2
        )


def append_records(records):
    if not records:
        return

    write_header = not RAW_CSV.exists()

    with RAW_CSV.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0].keys())
        )

        if write_header:
            writer.writeheader()

        writer.writerows(records)


# =========================================================
# SINGLE CONFIGURATION
# =========================================================

def run_configuration(
    config,
    clusters,
    base_model,
    kv_model,
    combined_model,
    config_index
):
    name = config["name"]
    batch_size = config["batch_size"]
    unique_count = config["unique_prompts"]

    print("\n" + "=" * 100)
    print("CONFIGURATION:", name)
    print("=" * 100)

    print("Batch size:", batch_size)
    print("Prompt length:", config["prompt_length"])
    print("Generated tokens:", NEW_TOKENS)
    print("Unique prompts:", unique_count)
    print("Duplicate prompts:", config["duplicate_prompts"])

    print(
        "Actual duplication rate:",
        f"{config['duplication_rate'] * 100:.1f}%"
    )

    # -----------------------------------------------------
    # WARM-UP
    # -----------------------------------------------------

    print("\nRunning warm-up...")

    with torch.inference_mode():

        prompts = create_workload(
            clusters[0],
            unique_count
        )

        functions = build_functions(
            base_model,
            kv_model,
            combined_model,
            prompts
        )

        for method in METHODS:

            for _ in range(WARMUP_RUNS):
                functions[method]()

        del prompts
        del functions
        gc.collect()

    # -----------------------------------------------------
    # CORRECTNESS
    # -----------------------------------------------------

    print("Running correctness checks...")

    with torch.inference_mode():

        for cluster_index, base_prompts in enumerate(clusters):

            prompts = create_workload(
                base_prompts,
                unique_count
            )

            functions = build_functions(
                base_model,
                kv_model,
                combined_model,
                prompts
            )

            outputs = {
                method: functions[method]()
                for method in METHODS
            }

            reference = outputs["standard"]

            # All methods must produce exactly identical
            # greedy-decoding outputs.
            for method in METHODS:

                if not torch.equal(
                    reference,
                    outputs[method]
                ):
                    raise AssertionError(
                        f"Output mismatch in {name}, "
                        f"cluster {cluster_index}, "
                        f"method {method}"
                    )

            # Verify duplicate restoration.
            if unique_count < batch_size:

                dedup_output = outputs["combined_dedup"]

                for index in range(unique_count):

                    for repeated_index in range(
                        index + unique_count,
                        batch_size,
                        unique_count
                    ):

                        if not torch.equal(
                            dedup_output[index],
                            dedup_output[repeated_index]
                        ):
                            raise AssertionError(
                                f"Duplicate restoration mismatch: {name}"
                            )

            del outputs
            del functions
            del prompts

            if (cluster_index + 1) % 5 == 0:

                print(
                    f"Correctness: "
                    f"{cluster_index + 1}/{NUM_CLUSTERS}"
                )

            gc.collect()

    print("Correctness PASS.")

    # -----------------------------------------------------
    # TIMING
    # -----------------------------------------------------

    print("Starting randomized timing...")

    order_rng = random.Random(
        SEED + config_index * 1000
    )

    records = []

    cluster_medians = {}

    with torch.inference_mode():

        for cluster_index, base_prompts in enumerate(clusters):

            prompts = create_workload(
                base_prompts,
                unique_count
            )

            functions = build_functions(
                base_model,
                kv_model,
                combined_model,
                prompts
            )

            timings = {
                method: []
                for method in METHODS
            }

            for repetition in range(BENCHMARK_RUNS):

                method_order = METHODS.copy()

                order_rng.shuffle(method_order)

                for method in method_order:

                    elapsed, output = time_function(
                        functions[method]
                    )

                    timings[method].append(elapsed)

                    records.append({
                        "configuration": name,
                        "batch_size": batch_size,
                        "prompt_length": config["prompt_length"],
                        "new_tokens": NEW_TOKENS,
                        "duplication_rate": config["duplication_rate"],
                        "unique_prompts": unique_count,
                        "duplicate_prompts": config["duplicate_prompts"],
                        "cluster": cluster_index,
                        "repetition": repetition,
                        "method": method,
                        "elapsed_seconds": elapsed,
                    })

                    del output

            cluster_medians[cluster_index] = {
                method: statistics.median(
                    timings[method]
                )
                for method in METHODS
            }

            del functions
            del prompts
            gc.collect()

            if (
                (cluster_index + 1) % 5 == 0
                or cluster_index + 1 == NUM_CLUSTERS
            ):

                print(
                    f"Timing: {cluster_index + 1}/{NUM_CLUSTERS}"
                )

    # -----------------------------------------------------
    # PAIRED CLUSTER STATISTICS
    # -----------------------------------------------------

    comparisons = {
        "kv_vs_standard": (
            "standard",
            "kv_cache"
        ),

        "combined_no_dedup_vs_standard": (
            "standard",
            "combined_no_dedup"
        ),

        "combined_dedup_vs_standard": (
            "standard",
            "combined_dedup"
        ),

        "dedup_vs_no_dedup": (
            "combined_no_dedup",
            "combined_dedup"
        ),
    }

    rng = np.random.default_rng(
        SEED + config_index * 10000
    )

    speedup_results = {}

    for label, (baseline, optimized) in comparisons.items():

        ratios = [
            cluster_medians[i][baseline]
            / cluster_medians[i][optimized]
            for i in range(NUM_CLUSTERS)
        ]

        speedup_results[label] = bootstrap_ci(
            ratios,
            BOOTSTRAP_ITERATIONS,
            rng
        )

    # -----------------------------------------------------
    # LATENCY SUMMARY
    # -----------------------------------------------------

    latency_results = {
        method: float(
            statistics.median([
                cluster_medians[i][method]
                for i in range(NUM_CLUSTERS)
            ])
        )
        for method in METHODS
    }

    result = {
        **config,
        "clusters": NUM_CLUSTERS,
        "timed_repetitions": BENCHMARK_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "correctness": "PASS",
        "median_latency_seconds": latency_results,
        "paired_speedups": speedup_results,
    }

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\nResults for", name)

    for metric, values in speedup_results.items():

        print(
            f"{metric:38s} "
            f"{values['median']:.3f}x "
            f"[{values['ci95_lower']:.3f}x, "
            f"{values['ci95_upper']:.3f}x]"
        )

    print("\nMedian latency:")

    for method, value in latency_results.items():

        print(
            f"{method:25s} {value:.6f}s"
        )

    return records, result


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 100)
    print("RCT 100M GENERALIZATION BENCHMARK")
    print("=" * 100)

    print("PyTorch:", torch.__version__)

    torch.set_num_threads(CPU_THREADS)

    print("CPU threads:", torch.get_num_threads())
    print("Checkpoint:", CHECKPOINT)
    print("Validation data:", VALIDATION_DATA)
    print("Output directory:", OUTPUT_DIR)

    check_memory("before model loading")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not VALIDATION_DATA.exists():
        raise FileNotFoundError(VALIDATION_DATA)

    configurations = make_configurations()

    expected_records = (
        len(configurations)
        * NUM_CLUSTERS
        * BENCHMARK_RUNS
        * len(METHODS)
    )

    print("\nConfigurations:", len(configurations))
    print("Batch sizes:", BATCH_SIZES)
    print("Prompt lengths:", PROMPT_LENGTHS)
    print("Duplication rates:", DUPLICATION_RATES)
    print("Clusters per configuration:", NUM_CLUSTERS)
    print("Timed repetitions:", BENCHMARK_RUNS)
    print("Methods:", len(METHODS))
    print("Expected timing records:", expected_records)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # -----------------------------------------------------
    # LOAD MODEL
    # -----------------------------------------------------

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    checkpoint_step = int(checkpoint["step"])

    base_model = RCT100M(
        RCT100MConfig()
    )

    base_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    base_model.eval()

    print("Checkpoint step:", checkpoint_step)
    print("Parameter count:", base_model.parameter_count())

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    combined_model = RCT100MCombined(base_model)
    combined_model.eval()

    del checkpoint
    gc.collect()

    check_memory("after model loading")

    # -----------------------------------------------------
    # GREEDY DECODING
    # -----------------------------------------------------

    original_multinomial = enable_greedy_sampling()

    all_results = []
    total_records = 0

    # Cache base clusters by batch size and prompt length.
    # This ensures the exact same prompts are used for
    # all three duplication-rate conditions.
    cluster_cache = {}

    try:

        rng = np.random.default_rng(SEED)

        for index, config in enumerate(configurations):

            print(
                f"\nConfiguration "
                f"{index + 1}/{len(configurations)}"
            )

            check_memory(
                f"before {config['name']}"
            )

            # -------------------------------------------------
            # REUSE CLUSTERS ACROSS DUPLICATION CONDITIONS
            # -------------------------------------------------

            family_key = (
                config["batch_size"],
                config["prompt_length"]
            )

            if family_key not in cluster_cache:

                print(
                    "Selecting new base prompt clusters for:",
                    family_key
                )

                cluster_cache[family_key] = load_clusters(
                    config,
                    rng
                )

            else:

                print(
                    "Reusing paired base prompt clusters for:",
                    family_key
                )

            clusters = cluster_cache[family_key]

            check_memory(
                f"after prompt selection {config['name']}"
            )

            # -------------------------------------------------
            # RUN CONFIGURATION
            # -------------------------------------------------

            records, result = run_configuration(
                config,
                clusters,
                base_model,
                kv_model,
                combined_model,
                index
            )

            # -------------------------------------------------
            # SAVE RAW RECORDS
            # -------------------------------------------------

            append_records(records)

            total_records += len(records)

            all_results.append(result)

            save_progress(
                index + 1,
                len(configurations),
                total_records
            )

            print(
                f"Saved {len(records)} records. "
                f"Total: {total_records}/{expected_records}"
            )

            del records
            gc.collect()

        # -----------------------------------------------------
        # FINAL EXPERIMENT REPORT
        # -----------------------------------------------------

        experiment = {
            "experiment": "RCT100M generalization benchmark",
            "timestamp_utc": TIMESTAMP,
            "checkpoint_step": checkpoint_step,
            "parameter_count": int(
                base_model.parameter_count()
            ),
            "device": "CPU",
            "cpu_threads": CPU_THREADS,
            "decoding": "greedy_argmax",
            "batch_sizes": BATCH_SIZES,
            "prompt_lengths": PROMPT_LENGTHS,
            "new_tokens": NEW_TOKENS,
            "duplication_rates": DUPLICATION_RATES,
            "clusters_per_configuration": NUM_CLUSTERS,
            "timed_repetitions": BENCHMARK_RUNS,
            "warmup_runs": WARMUP_RUNS,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "methods": METHODS,
            "configuration_results": all_results,
            "raw_timing_records": total_records,

            "notes": [
                "Prompt windows are selected from nine disjoint validation-data regions.",
                "Duplicate token sequences are filtered before random sampling.",
                "Selected prompt windows do not overlap within a region.",
                "The same base prompt clusters are reused across duplication conditions for each batch-size and prompt-length pair.",
                "Paired cluster comparisons use the same base prompt workloads.",
                "Greedy decoding is used for deterministic correctness comparisons.",
                "Identical prompts share the same continuation in deduplicated generation.",
                "Results apply to the tested CPU, checkpoint, batch sizes, prompt lengths and generation length.",
                "This benchmark measures inference behavior and does not establish language-model quality."
            ]
        }

        with RESULTS_JSON.open(
            "w",
            encoding="utf-8"
        ) as handle:

            json.dump(
                experiment,
                handle,
                indent=2
            )

        save_progress(
            len(configurations),
            len(configurations),
            total_records
        )

        print("\n" + "=" * 100)
        print("GENERALIZATION EXPERIMENT COMPLETE")
        print("=" * 100)

        print("Configurations:", len(all_results))
        print("Raw timing records:", total_records)
        print("Expected records:", expected_records)
        print("Raw CSV:", RAW_CSV)
        print("Results JSON:", RESULTS_JSON)
        print("Progress JSON:", PROGRESS_JSON)

    finally:

        torch.multinomial = original_multinomial

        cluster_cache.clear()

        del combined_model
        del kv_model
        del base_model

        gc.collect()


if __name__ == "__main__":
    main()
