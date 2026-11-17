
from pathlib import Path
import csv
import json
import random
import statistics
import time
from datetime import datetime, timezone
from collections import defaultdict

import torch


# =========================================================
# 1. CONFIGURATION
# =========================================================

BATCH_SIZES = [4, 8, 16]
PROMPT_LENGTHS = [8, 24, 64]
DUPLICATE_RATES = [0.0, 0.25, 0.50, 0.75, 0.875]

FINGERPRINT_COLUMNS = [2, 4, 8]
THRESHOLDS = [0.10, 0.25, 0.50, 0.75]

CLUSTERS = 10
WARMUP_REPETITIONS = 2
TIMED_REPETITIONS = 5

VOCAB_SIZE = 8192
SEED = 20261104

PROJECT_DIR = Path(__file__).resolve().parent.parent

RESULTS_ROOT = (
    PROJECT_DIR
    / "results"
    / "rct100m_gpu_adaptive_tuning"
)


# =========================================================
# 2. FINGERPRINT DETECTOR
# Matches RCT100MAdaptive.estimate_duplicate_rate()
# =========================================================

@torch.inference_mode()
def estimate_duplicate_rate(prompts, fingerprint_columns):

    batch_size, sequence_length = prompts.shape

    if batch_size < 2:
        return 0.0

    positions = sorted(set(
        round(
            i * (sequence_length - 1)
            / max(fingerprint_columns - 1, 1)
        )
        for i in range(fingerprint_columns)
    ))

    positions = torch.tensor(
        positions,
        dtype=torch.long,
        device=prompts.device
    )

    fingerprints = prompts.index_select(1, positions)

    unique_fingerprints = torch.unique(
        fingerprints,
        dim=0
    )

    return (
        batch_size - unique_fingerprints.size(0)
    ) / batch_size


# =========================================================
# 3. GENERATE NATURAL DUPLICATE WORKLOADS
# =========================================================

def create_natural_prompts(
    batch_size,
    prompt_length,
    duplicate_rate,
    device
):

    unique_count = max(
        1,
        round(batch_size * (1.0 - duplicate_rate))
    )

    unique_prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (unique_count, prompt_length),
        dtype=torch.long,
        device=device
    )

    repeat_indices = (
        torch.arange(batch_size, device=device)
        % unique_count
    )

    prompts = unique_prompts.index_select(
        0,
        repeat_indices
    )

    actual_unique = torch.unique(
        prompts,
        dim=0
    ).size(0)

    actual_rate = (
        batch_size - actual_unique
    ) / batch_size

    return prompts, actual_rate


# =========================================================
# 4. GENERATE DELIBERATE FINGERPRINT COLLISIONS
# All full prompts remain unique.
# Their selected fingerprint positions are identical.
# =========================================================

def create_collision_prompts(
    batch_size,
    prompt_length,
    fingerprint_columns,
    device
):

    selected_positions = sorted(set(
        round(
            i * (prompt_length - 1)
            / max(fingerprint_columns - 1, 1)
        )
        for i in range(fingerprint_columns)
    ))

    unselected_positions = [
        position
        for position in range(prompt_length)
        if position not in selected_positions
    ]

    if not unselected_positions:
        return None

    prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, prompt_length),
        dtype=torch.long,
        device=device
    )

    # Make every fingerprint identical.
    prompts[:, selected_positions] = 1

    # Give every complete prompt a unique identity.
    identity_position = unselected_positions[0]

    prompts[:, identity_position] = (
        100 + torch.arange(
            batch_size,
            device=device
        )
    )

    actual_unique = torch.unique(
        prompts,
        dim=0
    ).size(0)

    if actual_unique != batch_size:
        raise AssertionError(
            "Collision workload contains duplicate full prompts."
        )

    return prompts


# =========================================================
# 5. DETECTOR TIMING
# =========================================================

def measure_detector(
    prompts,
    fingerprint_columns,
    device
):

    torch.cuda.synchronize(device)

    start = time.perf_counter()

    estimate = estimate_duplicate_rate(
        prompts,
        fingerprint_columns
    )

    torch.cuda.synchronize(device)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return estimate, elapsed_ms


# =========================================================
# 6. CSV EXPORT
# =========================================================

def save_csv(path, records):

    if not records:
        raise ValueError(
            f"No records to save: {path}"
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)


# =========================================================
# 7. MAIN
# =========================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. NVIDIA GPU required."
        )

    device = torch.device("cuda")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    run_id = datetime.now(
        timezone.utc
    ).strftime("run_%Y%m%dT%H%M%SZ")

    output_dir = RESULTS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RCT100M ADAPTIVE DETECTOR TUNING")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Batch sizes:", BATCH_SIZES)
    print("Prompt lengths:", PROMPT_LENGTHS)
    print("Duplicate rates:", DUPLICATE_RATES)
    print("Fingerprint columns:", FINGERPRINT_COLUMNS)
    print("Thresholds:", THRESHOLDS)
    print("Clusters:", CLUSTERS)
    print("Output:", output_dir)

    records = []

    # -----------------------------------------------------
    # Natural workloads
    # -----------------------------------------------------

    total_natural = (
        len(BATCH_SIZES)
        * len(PROMPT_LENGTHS)
        * len(DUPLICATE_RATES)
        * CLUSTERS
    )

    completed = 0

    for batch_size in BATCH_SIZES:

        for prompt_length in PROMPT_LENGTHS:

            for requested_rate in DUPLICATE_RATES:

                for cluster_id in range(CLUSTERS):

                    prompts, actual_rate = (
                        create_natural_prompts(
                            batch_size,
                            prompt_length,
                            requested_rate,
                            device
                        )
                    )

                    for columns in FINGERPRINT_COLUMNS:

                        # Warm-up.
                        for _ in range(WARMUP_REPETITIONS):

                            estimate_duplicate_rate(
                                prompts,
                                columns
                            )

                        torch.cuda.synchronize(device)

                        measurements = []

                        for _ in range(TIMED_REPETITIONS):

                            estimate, elapsed_ms = (
                                measure_detector(
                                    prompts,
                                    columns,
                                    device
                                )
                            )

                            measurements.append(elapsed_ms)

                        median_latency = statistics.median(
                            measurements
                        )

                        records.append({
                            "scenario": "natural",
                            "batch_size": batch_size,
                            "prompt_length": prompt_length,
                            "requested_duplicate_rate": requested_rate,
                            "actual_duplicate_rate": actual_rate,
                            "fingerprint_columns": columns,
                            "fingerprint_estimate": estimate,
                            "detector_latency_ms": median_latency,
                            "cluster": cluster_id
                        })

                    completed += 1

                    if completed % 30 == 0:
                        print(
                            f"Natural workloads: "
                            f"{completed}/{total_natural}"
                        )

                    del prompts

    # -----------------------------------------------------
    # Deliberate collision workloads
    # -----------------------------------------------------

    collision_count = 0
    collision_skipped = 0

    for batch_size in BATCH_SIZES:

        for prompt_length in PROMPT_LENGTHS:

            for columns in FINGERPRINT_COLUMNS:

                for cluster_id in range(CLUSTERS):

                    prompts = create_collision_prompts(
                        batch_size,
                        prompt_length,
                        columns,
                        device
                    )

                    if prompts is None:
                        collision_skipped += 1
                        continue

                    actual_rate = 0.0

                    for _ in range(WARMUP_REPETITIONS):

                        estimate_duplicate_rate(
                            prompts,
                            columns
                        )

                    torch.cuda.synchronize(device)

                    measurements = []

                    for _ in range(TIMED_REPETITIONS):

                        estimate, elapsed_ms = (
                            measure_detector(
                                prompts,
                                columns,
                                device
                            )
                        )

                        measurements.append(elapsed_ms)

                    records.append({
                        "scenario": "deliberate_collision",
                        "batch_size": batch_size,
                        "prompt_length": prompt_length,
                        "requested_duplicate_rate": 0.0,
                        "actual_duplicate_rate": actual_rate,
                        "fingerprint_columns": columns,
                        "fingerprint_estimate": estimate,
                        "detector_latency_ms": statistics.median(
                            measurements
                        ),
                        "cluster": cluster_id
                    })

                    collision_count += 1

                    del prompts

    # -----------------------------------------------------
    # Threshold decision analysis
    # -----------------------------------------------------

    summary_groups = defaultdict(list)

    upper_bound_violations = 0

    for record in records:

        actual_rate = record["actual_duplicate_rate"]
        estimate = record["fingerprint_estimate"]

        if estimate + 1e-9 < actual_rate:
            upper_bound_violations += 1

        for threshold in THRESHOLDS:

            triggered = (
                estimate >= threshold
                and estimate > 0.0
            )

            if actual_rate > 0 and not triggered:
                decision = "missed_duplicate_batch"

            elif actual_rate == 0 and triggered:
                decision = "unnecessary_dedup_trigger"

            elif actual_rate > 0 and triggered:
                decision = "useful_dedup_trigger"

            else:
                decision = "correct_kv_only"

            key = (
                record["scenario"],
                record["fingerprint_columns"],
                threshold
            )

            summary_groups[key].append({
                "triggered": triggered,
                "decision": decision,
                "actual_rate": actual_rate,
                "estimate": estimate,
                "latency": record["detector_latency_ms"]
            })

    summaries = []

    for key, group in sorted(summary_groups.items()):

        scenario, columns, threshold = key

        count = len(group)

        triggered_count = sum(
            item["triggered"] for item in group
        )

        false_triggers = sum(
            item["decision"] == "unnecessary_dedup_trigger"
            for item in group
        )

        missed_batches = sum(
            item["decision"] == "missed_duplicate_batch"
            for item in group
        )

        duplicate_batches = sum(
            item["actual_rate"] > 0
            for item in group
        )

        zero_duplicate_batches = sum(
            item["actual_rate"] == 0
            for item in group
        )

        summaries.append({
            "scenario": scenario,
            "fingerprint_columns": columns,
            "threshold": threshold,
            "workload_count": count,
            "triggered_count": triggered_count,
            "trigger_rate": triggered_count / count,
            "false_trigger_count": false_triggers,
            "missed_duplicate_batches": missed_batches,
            "duplicate_batches": duplicate_batches,
            "zero_duplicate_batches": zero_duplicate_batches,
            "mean_actual_duplicate_rate": statistics.mean(
                item["actual_rate"] for item in group
            ),
            "mean_fingerprint_estimate": statistics.mean(
                item["estimate"] for item in group
            ),
            "median_detector_latency_ms": statistics.median(
                item["latency"] for item in group
            )
        })

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    raw_path = output_dir / "detector_records.csv"
    summary_path = output_dir / "decision_summary.csv"
    report_path = output_dir / "report.json"

    save_csv(raw_path, records)
    save_csv(summary_path, summaries)

    report = {
        "run_id": run_id,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": SEED,
        "batch_sizes": BATCH_SIZES,
        "prompt_lengths": PROMPT_LENGTHS,
        "duplicate_rates": DUPLICATE_RATES,
        "fingerprint_columns": FINGERPRINT_COLUMNS,
        "thresholds": THRESHOLDS,
        "clusters": CLUSTERS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "timed_repetitions": TIMED_REPETITIONS,
        "natural_workload_count": total_natural,
        "collision_workload_count": collision_count,
        "collision_configs_skipped": collision_skipped,
        "total_detector_records": len(records),
        "upper_bound_violations": upper_bound_violations,
        "all_fingerprints_respected_duplicate_upper_bound": (
            upper_bound_violations == 0
        ),
        "summaries": summaries,
        "important_limitation": (
            "Detector-only screening. These measurements do not "
            "establish end-to-end inference speedups."
        )
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    # -----------------------------------------------------
    # Print concise report
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("TUNING SUMMARY")
    print("=" * 70)

    print(
        f"{'SCENARIO':<23}"
        f"{'COLS':>5}"
        f"{'THRESH':>9}"
        f"{'TRIGGER':>10}"
        f"{'FALSE':>9}"
        f"{'MISSED':>9}"
        f"{'MEDIAN ms':>12}"
    )

    for item in summaries:

        print(
            f"{item['scenario']:<23}"
            f"{item['fingerprint_columns']:>5}"
            f"{item['threshold']:>9.2f}"
            f"{item['trigger_rate']:>10.3f}"
            f"{item['false_trigger_count']:>9}"
            f"{item['missed_duplicate_batches']:>9}"
            f"{item['median_detector_latency_ms']:>12.4f}"
        )

    print("\nUpper-bound violations:", upper_bound_violations)

    print("\nBENCHMARK COMPLETED")
    print("Output directory:", output_dir)
    print("Raw records:", raw_path)
    print("Decision summary:", summary_path)
    print("JSON report:", report_path)


if __name__ == "__main__":
    main()
