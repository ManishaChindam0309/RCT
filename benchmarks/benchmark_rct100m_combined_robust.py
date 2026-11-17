import sys
import gc
import csv
import json
import time
import random
import statistics
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

VALIDATION_DATA = (
    PROJECT_DIR
    / "data"
    / "rct30"
    / "validation_expanded_tokens.bin"
)

BATCH_SIZE = 8
PROMPT_LENGTH = 12
NEW_TOKENS = 8

NUM_CLUSTERS = 30
BENCHMARK_RUNS = 5
WARMUP_RUNS = 1

BOOTSTRAP_ITERATIONS = 20000

SEED = 20261103

METHODS = [
    "standard",
    "kv_cache",
    "combined_no_dedup",
    "combined_dedup"
]

WORKLOADS = {
    "unique_8_of_8": 8,
    "duplicate_4_of_8": 4,
    "duplicate_6_of_8": 2,
    "duplicate_7_of_8": 1
}


# ---------------------------------------------------------
# Deterministic greedy decoding
# ---------------------------------------------------------

def enable_greedy_sampling():

    original_multinomial = torch.multinomial

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

    return original_multinomial


# ---------------------------------------------------------
# Data
# ---------------------------------------------------------

def load_prompt_clusters():

    data = np.memmap(
        VALIDATION_DATA,
        dtype=np.uint16,
        mode="r"
    )

    max_start = len(data) - PROMPT_LENGTH

    rng = np.random.default_rng(SEED)

    # Use non-overlapping windows to reduce dependence
    # between prompts within and across clusters.
    possible_starts = np.arange(
        0,
        max_start,
        PROMPT_LENGTH
    )

    if len(possible_starts) < NUM_CLUSTERS * BATCH_SIZE:
        raise RuntimeError(
            "Not enough non-overlapping validation windows."
        )

    selected_starts = rng.choice(
        possible_starts,
        size=NUM_CLUSTERS * BATCH_SIZE,
        replace=False
    )

    selected_starts.sort()

    prompts = np.stack([
        np.array(
            data[start:start + PROMPT_LENGTH],
            dtype=np.int64
        )
        for start in selected_starts
    ])

    del data

    if len(np.unique(prompts, axis=0)) != len(prompts):
        raise RuntimeError(
            "Duplicate base prompts found. "
            "Regenerate the workload before benchmarking."
        )

    prompts = torch.from_numpy(prompts).long()

    return [
        prompts[
            cluster * BATCH_SIZE:
            (cluster + 1) * BATCH_SIZE
        ].clone()
        for cluster in range(NUM_CLUSTERS)
    ]


def create_workload(base_prompts, unique_count):

    if unique_count == BATCH_SIZE:
        return base_prompts.clone()

    if BATCH_SIZE % unique_count != 0:
        raise ValueError(
            "Unique count must divide batch size."
        )

    repeats = BATCH_SIZE // unique_count

    return base_prompts[:unique_count].repeat(
        repeats,
        1
    )


# ---------------------------------------------------------
# Generation functions
# ---------------------------------------------------------

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
        )
    }


def time_function(function):

    start = time.perf_counter()

    output = function()

    elapsed = time.perf_counter() - start

    return elapsed, output


# ---------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------

def cluster_bootstrap_ci(
    cluster_values,
    iterations,
    seed
):

    values = np.asarray(
        cluster_values,
        dtype=np.float64
    )

    rng = np.random.default_rng(seed)

    bootstrap_medians = np.empty(
        iterations,
        dtype=np.float64
    )

    for index in range(iterations):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_medians[index] = np.median(
            sample
        )

    lower, upper = np.percentile(
        bootstrap_medians,
        [2.5, 97.5]
    )

    return (
        float(np.median(values)),
        float(lower),
        float(upper)
    )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    torch.set_num_threads(4)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 100)
    print("RCT 100M ROBUST COMBINED GENERATION BENCHMARK")
    print("=" * 100)

    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())
    print("Checkpoint:", CHECKPOINT)
    print("Clusters:", NUM_CLUSTERS)
    print("Duplication workloads:", len(WORKLOADS))
    print("Methods:", len(METHODS))
    print("Timed repetitions:", BENCHMARK_RUNS)
    print("Warmups:", WARMUP_RUNS)
    print("Bootstrap iterations:", BOOTSTRAP_ITERATIONS)

    total_timed_runs = (
        NUM_CLUSTERS
        * len(WORKLOADS)
        * len(METHODS)
        * BENCHMARK_RUNS
    )

    print("Expected timed generation calls:", total_timed_runs)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    base_model = RCT100M(RCT100MConfig())

    base_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    base_model.eval()

    print("Checkpoint step:", checkpoint["step"])
    print("Parameters:", base_model.parameter_count())

    # Wrappers share the base model weights.
    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    combined_model = RCT100MCombined(base_model)
    combined_model.eval()

    clusters = load_prompt_clusters()

    print("Independent prompt clusters:", len(clusters))
    print(
        "Base prompt count:",
        len(clusters) * BATCH_SIZE
    )

    original_multinomial = enable_greedy_sampling()

    raw_records = []

    # Correctness results indexed by workload.
    correctness = {}

    # Each cluster has one median timing per method/workload.
    cluster_medians = {}

    # -----------------------------------------------------
    # Warm up all methods
    # -----------------------------------------------------

    print("\nWarming up methods...")

    with torch.inference_mode():

        for workload_name, unique_count in WORKLOADS.items():

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
            gc.collect()

    print("Warmup complete.")

    # -----------------------------------------------------
    # Correctness checks
    # -----------------------------------------------------

    print("\nRunning exact-output correctness checks...")

    with torch.inference_mode():

        for workload_name, unique_count in WORKLOADS.items():

            workload_passes = 0

            for cluster_index in range(NUM_CLUSTERS):

                prompts = create_workload(
                    clusters[cluster_index],
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

                for method in METHODS:

                    if not torch.equal(
                        reference,
                        outputs[method]
                    ):
                        raise AssertionError(
                            f"Output mismatch: "
                            f"{workload_name}, "
                            f"cluster {cluster_index}, "
                            f"method {method}"
                        )

                if unique_count < BATCH_SIZE:

                    dedup_output = outputs[
                        "combined_dedup"
                    ]

                    for index in range(unique_count):

                        for repeated_index in range(
                            index + unique_count,
                            BATCH_SIZE,
                            unique_count
                        ):

                            if not torch.equal(
                                dedup_output[index],
                                dedup_output[repeated_index]
                            ):
                                raise AssertionError(
                                    "Duplicate restoration mismatch."
                                )

                workload_passes += 1

                del outputs
                del prompts
                gc.collect()

            correctness[workload_name] = {
                "clusters_checked": workload_passes,
                "passed": True
            }

            print(
                f"{workload_name}: "
                f"{workload_passes}/{NUM_CLUSTERS} clusters PASS"
            )

    # -----------------------------------------------------
    # Randomized timing experiment
    # -----------------------------------------------------

    print("\nStarting randomized timing experiment...")
    print("Do not run other heavy CPU tasks during this phase.")

    order_rng = random.Random(SEED + 100)

    with torch.inference_mode():

        for workload_name, unique_count in WORKLOADS.items():

            print("\n" + "-" * 90)
            print("WORKLOAD:", workload_name)
            print("Unique prompts:", unique_count)
            print("Duplicate prompts:", BATCH_SIZE - unique_count)
            print("-" * 90)

            cluster_medians[workload_name] = {}

            for cluster_index in range(NUM_CLUSTERS):

                prompts = create_workload(
                    clusters[cluster_index],
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

                        raw_records.append({
                            "workload": workload_name,
                            "unique_prompts": unique_count,
                            "duplicate_prompts": (
                                BATCH_SIZE - unique_count
                            ),
                            "cluster": cluster_index,
                            "repetition": repetition,
                            "method": method,
                            "elapsed_seconds": elapsed
                        })

                        del output

                cluster_medians[workload_name][
                    cluster_index
                ] = {
                    method: statistics.median(
                        timings[method]
                    )
                    for method in METHODS
                }

                del prompts
                gc.collect()

                if (
                    (cluster_index + 1) % 5 == 0
                    or cluster_index + 1 == NUM_CLUSTERS
                ):

                    print(
                        f"Completed cluster "
                        f"{cluster_index + 1}/{NUM_CLUSTERS}"
                    )

    # -----------------------------------------------------
    # Restore random sampling
    # -----------------------------------------------------

    torch.multinomial = original_multinomial

    # -----------------------------------------------------
    # Save raw data
    # -----------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct100m_combined_robust"
        / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_csv = output_dir / "raw_timing_records.csv"

    with raw_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(raw_records[0].keys())
        )

        writer.writeheader()
        writer.writerows(raw_records)

    # -----------------------------------------------------
    # Summary and confidence intervals
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("PAIRED CLUSTER-LEVEL RESULTS")
    print("=" * 100)

    summary = {}

    for workload_index, workload_name in enumerate(WORKLOADS):

        records = cluster_medians[workload_name]

        speedups = {
            "kv_vs_standard": [],
            "combined_no_dedup_vs_standard": [],
            "combined_dedup_vs_standard": [],
            "dedup_vs_no_dedup": []
        }

        method_medians = {
            method: []
            for method in METHODS
        }

        for cluster_index in range(NUM_CLUSTERS):

            row = records[cluster_index]

            for method in METHODS:
                method_medians[method].append(
                    row[method]
                )

            standard = row["standard"]

            speedups["kv_vs_standard"].append(
                standard / row["kv_cache"]
            )

            speedups[
                "combined_no_dedup_vs_standard"
            ].append(
                standard / row["combined_no_dedup"]
            )

            speedups[
                "combined_dedup_vs_standard"
            ].append(
                standard / row["combined_dedup"]
            )

            speedups["dedup_vs_no_dedup"].append(
                row["combined_no_dedup"]
                / row["combined_dedup"]
            )

        summary[workload_name] = {
            "unique_prompts": WORKLOADS[workload_name],
            "duplicate_prompts": (
                BATCH_SIZE - WORKLOADS[workload_name]
            ),
            "median_latency_seconds": {
                method: float(
                    np.median(method_medians[method])
                )
                for method in METHODS
            },
            "paired_speedups": {}
        }

        print(f"\n{workload_name}")

        print(
            f"{'Comparison':38s}"
            f"{'Median':>12s}"
            f"{'95% CI':>27s}"
        )

        for metric_index, (metric, values) in enumerate(
            speedups.items()
        ):

            median, lower, upper = cluster_bootstrap_ci(
                values,
                BOOTSTRAP_ITERATIONS,
                SEED + workload_index * 100 + metric_index
            )

            summary[workload_name][
                "paired_speedups"
            ][metric] = {
                "median": median,
                "ci95_lower": lower,
                "ci95_upper": upper
            }

            print(
                f"{metric:38s}"
                f"{median:11.3f}x"
                f"[{lower:.3f}x, {upper:.3f}x]"
            )

        print("\nMedian latency across clusters:")

        for method in METHODS:

            print(
                f"{method:25s}"
                f"{np.median(method_medians[method]):.6f}s"
            )

    # -----------------------------------------------------
    # Overall paired comparison
    # -----------------------------------------------------

    overall_speedups = {
        "kv_vs_standard": [],
        "combined_dedup_vs_standard": [],
        "dedup_vs_no_dedup": []
    }

    for workload_name in WORKLOADS:

        for cluster_index in range(NUM_CLUSTERS):

            row = cluster_medians[workload_name][
                cluster_index
            ]

            overall_speedups[
                "kv_vs_standard"
            ].append(
                row["standard"] / row["kv_cache"]
            )

            overall_speedups[
                "combined_dedup_vs_standard"
            ].append(
                row["standard"] / row["combined_dedup"]
            )

            overall_speedups[
                "dedup_vs_no_dedup"
            ].append(
                row["combined_no_dedup"]
                / row["combined_dedup"]
            )

    print("\n" + "=" * 100)
    print("OVERALL PAIRED CLUSTER BOOTSTRAP RESULTS")
    print("=" * 100)

    overall_summary = {}

    for metric_index, (metric, values) in enumerate(
        overall_speedups.items()
    ):

        median, lower, upper = cluster_bootstrap_ci(
            values,
            BOOTSTRAP_ITERATIONS,
            SEED + 500 + metric_index
        )

        overall_summary[metric] = {
            "median": median,
            "ci95_lower": lower,
            "ci95_upper": upper
        }

        print(
            f"{metric:35s}"
            f"{median:.3f}x "
            f"[{lower:.3f}x, {upper:.3f}x]"
        )

    # -----------------------------------------------------
    # Save experiment report
    # -----------------------------------------------------

    experiment = {
        "experiment": "RCT100M robust combined generation",
        "timestamp_utc": timestamp,
        "checkpoint_step": int(checkpoint["step"]),
        "parameter_count": int(
            base_model.parameter_count()
        ),
        "device": "CPU",
        "cpu_threads": torch.get_num_threads(),
        "decoding": "greedy_argmax",
        "batch_size": BATCH_SIZE,
        "prompt_length": PROMPT_LENGTH,
        "new_tokens": NEW_TOKENS,
        "clusters": NUM_CLUSTERS,
        "measured_repetitions": BENCHMARK_RUNS,
        "warmup_runs": WARMUP_RUNS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "correctness": correctness,
        "workload_results": summary,
        "overall_results": overall_summary,
        "raw_timing_records": len(raw_records)
    }

    json_path = output_dir / "experiment_results.json"

    with json_path.open(
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            experiment,
            handle,
            indent=2
        )

    print("\n" + "=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print("Output directory:", output_dir)
    print("Raw timings:", raw_csv)
    print("Experiment results:", json_path)
    print("Raw records:", len(raw_records))

    print("\nEXPERIMENT COMPLETE")

    del combined_model
    del kv_model
    del base_model
    del checkpoint

    gc.collect()


if __name__ == "__main__":
    main()
