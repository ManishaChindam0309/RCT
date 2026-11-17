
from pathlib import Path
import sys
import csv
import json
import time
import statistics
from datetime import datetime, timezone

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_adaptive import RCT100MAdaptive


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

RESULTS_ROOT = (
    PROJECT_DIR
    / "results"
    / "rct100m_fingerprint_collisions"
)

BATCH_SIZES = [4, 8, 16]
PROMPT_LENGTHS = [8, 24, 64]
DUPLICATE_RATES = [0.0, 0.50, 0.75]
FINGERPRINT_COLUMN_OPTIONS = [2, 4, 8]

WARMUP_REPETITIONS = 3
TIMED_REPETITIONS = 20

THRESHOLD = 0.25
VOCAB_SIZE = 8192
SEED = 20261103


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(device):

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    config = RCT100MConfig(
        vocab_size=8192,
        hidden_size=512,
        num_layers=30,
        num_heads=8,
        intermediate_size=2048,
        max_seq_len=512
    )

    model = RCT100M(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint


# =========================================================
# CREATE RANDOM WORKLOADS
# =========================================================

def create_random_prompts(
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

    indices = (
        torch.arange(batch_size, device=device)
        % unique_count
    )

    prompts = unique_prompts.index_select(
        0,
        indices
    )

    return prompts


# =========================================================
# EXACT DETECTOR POSITION SELECTION
# =========================================================

def get_fingerprint_positions(
    prompt_length,
    fingerprint_columns
):
    """
    This function intentionally matches the exact
    position-selection logic in:

    RCT100MAdaptive.estimate_duplicate_rate()
    """

    positions = sorted(set(
        round(
            i * (prompt_length - 1)
            / max(fingerprint_columns - 1, 1)
        )
        for i in range(fingerprint_columns)
    ))

    return positions


# =========================================================
# CREATE DELIBERATE FINGERPRINT COLLISIONS
# =========================================================

def create_collision_prompts(
    batch_size,
    prompt_length,
    fingerprint_columns,
    device
):
    """
    Creates prompts that:

    1. Have identical selected fingerprint positions.
    2. Are unique as complete prompts.
    3. Therefore produce a false-positive fingerprint
       duplicate estimate.

    The selected positions are identical to the
    adaptive detector's selected positions.
    """

    selected_positions = get_fingerprint_positions(
        prompt_length,
        fingerprint_columns
    )

    selected_set = set(selected_positions)

    unselected_positions = [
        position
        for position in range(prompt_length)
        if position not in selected_set
    ]

    # No collision can be constructed if every token
    # position is included in the fingerprint.
    if not unselected_positions:
        return None

    prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, prompt_length),
        dtype=torch.long,
        device=device
    )

    # Make all fingerprint columns identical.
    prompts[:, selected_positions] = 1

    # Make every complete prompt unique.
    unique_position = unselected_positions[0]

    prompts[:, unique_position] = (
        torch.arange(
            batch_size,
            device=device
        ) + 100
    )

    return prompts


# =========================================================
# MEASURE ACTUAL DUPLICATION
# =========================================================

def actual_duplicate_rate(prompts):

    batch_size = prompts.size(0)

    unique_count = torch.unique(
        prompts,
        dim=0
    ).size(0)

    return (
        batch_size - unique_count
    ) / batch_size


# =========================================================
# MEASURE DETECTOR LATENCY
# =========================================================

def measure_detector(
    adaptive_model,
    prompts,
    device
):

    # Warm-up.
    for _ in range(WARMUP_REPETITIONS):

        adaptive_model.estimate_duplicate_rate(
            prompts
        )

    torch.cuda.synchronize(device)

    measurements = []

    for _ in range(TIMED_REPETITIONS):

        torch.cuda.synchronize(device)

        start = time.perf_counter()

        estimate = (
            adaptive_model.estimate_duplicate_rate(
                prompts
            )
        )

        torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

        if isinstance(estimate, torch.Tensor):
            estimate = estimate.item()

        measurements.append(elapsed)

    return {
        "estimated_duplicate_rate": float(estimate),
        "median_latency_ms": (
            statistics.median(measurements) * 1000
        ),
        "mean_latency_ms": (
            statistics.mean(measurements) * 1000
        )
    }


# =========================================================
# MAIN
# =========================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark."
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )

    output_dir = RESULTS_ROOT / run_id
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("RCT100M FINGERPRINT COLLISION BENCHMARK")
    print("=" * 65)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Threshold:", THRESHOLD)
    print(
        "Fingerprint columns:",
        FINGERPRINT_COLUMN_OPTIONS
    )

    print("\nLoading checkpoint...")

    base_model, checkpoint = load_model(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in base_model.parameters()
    )

    print("Checkpoint step:", checkpoint.get("step"))
    print("Parameters:", f"{parameter_count:,}")

    # Create adaptive detectors.
    detectors = {}

    for columns in FINGERPRINT_COLUMN_OPTIONS:

        detectors[columns] = RCT100MAdaptive(
            base_model,
            duplicate_threshold=THRESHOLD,
            fingerprint_columns=columns
        )

        detectors[columns].eval()

    records = []

    total_tests = 0

    # =====================================================
    # RANDOM WORKLOADS
    # =====================================================

    print("\nStarting random workload tests...")

    for batch_size in BATCH_SIZES:

        for prompt_length in PROMPT_LENGTHS:

            for duplicate_rate in DUPLICATE_RATES:

                prompts = create_random_prompts(
                    batch_size,
                    prompt_length,
                    duplicate_rate,
                    device
                )

                actual_rate = actual_duplicate_rate(
                    prompts
                )

                for columns, detector in detectors.items():

                    total_tests += 1

                    result = measure_detector(
                        detector,
                        prompts,
                        device
                    )

                    estimate = result[
                        "estimated_duplicate_rate"
                    ]

                    false_positive_excess = max(
                        0.0,
                        estimate - actual_rate
                    )

                    triggered = (
                        estimate >= THRESHOLD
                        and estimate > 0
                    )

                    record = {
                        "scenario": "random",
                        "batch_size": batch_size,
                        "prompt_length": prompt_length,
                        "fingerprint_columns": columns,
                        "requested_duplicate_rate": duplicate_rate,
                        "actual_duplicate_rate": actual_rate,
                        "estimated_duplicate_rate": estimate,
                        "false_positive_excess": false_positive_excess,
                        "detector_triggered": triggered,
                        "median_latency_ms": (
                            result["median_latency_ms"]
                        ),
                        "mean_latency_ms": (
                            result["mean_latency_ms"]
                        )
                    }

                    records.append(record)

                    print(
                        f"[{total_tests}] "
                        f"Random B{batch_size} "
                        f"P{prompt_length} "
                        f"D{duplicate_rate:.0%} "
                        f"F{columns}: "
                        f"actual={actual_rate:.3f}, "
                        f"estimated={estimate:.3f}, "
                        f"time={result['median_latency_ms']:.4f} ms"
                    )

                del prompts

    # =====================================================
    # DELIBERATE COLLISION WORKLOADS
    # =====================================================

    print("\nStarting deliberate collision tests...")

    for batch_size in BATCH_SIZES:

        for prompt_length in PROMPT_LENGTHS:

            for columns, detector in detectors.items():

                prompts = create_collision_prompts(
                    batch_size,
                    prompt_length,
                    columns,
                    device
                )

                if prompts is None:

                    print(
                        f"Skipping impossible collision: "
                        f"B{batch_size}, P{prompt_length}, "
                        f"F{columns}"
                    )

                    continue

                actual_rate = actual_duplicate_rate(
                    prompts
                )

                total_tests += 1

                result = measure_detector(
                    detector,
                    prompts,
                    device
                )

                estimate = result[
                    "estimated_duplicate_rate"
                ]

                false_positive_excess = max(
                    0.0,
                    estimate - actual_rate
                )

                triggered = (
                    estimate >= THRESHOLD
                    and estimate > 0
                )

                record = {
                    "scenario": "deliberate_collision",
                    "batch_size": batch_size,
                    "prompt_length": prompt_length,
                    "fingerprint_columns": columns,
                    "requested_duplicate_rate": 0.0,
                    "actual_duplicate_rate": actual_rate,
                    "estimated_duplicate_rate": estimate,
                    "false_positive_excess": false_positive_excess,
                    "detector_triggered": triggered,
                    "median_latency_ms": (
                        result["median_latency_ms"]
                    ),
                    "mean_latency_ms": (
                        result["mean_latency_ms"]
                    )
                }

                records.append(record)

                print(
                    f"[{total_tests}] "
                    f"COLLISION B{batch_size} "
                    f"P{prompt_length} F{columns}: "
                    f"actual={actual_rate:.3f}, "
                    f"estimated={estimate:.3f}, "
                    f"time={result['median_latency_ms']:.4f} ms"
                )

                del prompts

    # =====================================================
    # SAVE CSV
    # =====================================================

    csv_path = output_dir / "fingerprint_results.csv"

    with open(
        csv_path,
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

    # =====================================================
    # SUMMARY
    # =====================================================

    collision_records = [
        row
        for row in records
        if row["scenario"] == "deliberate_collision"
    ]

    random_records = [
        row
        for row in records
        if row["scenario"] == "random"
    ]

    summary = {
        "run_id": run_id,
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_step": checkpoint.get("step"),
        "parameter_count": parameter_count,
        "threshold": THRESHOLD,
        "fingerprint_columns": FINGERPRINT_COLUMN_OPTIONS,
        "total_tests": len(records),
        "random_tests": len(random_records),
        "collision_tests": len(collision_records),
        "collision_false_positive_triggers": sum(
            row["detector_triggered"]
            for row in collision_records
        ),
        "collision_test_count": len(collision_records),
        "records": records
    }

    json_path = output_dir / "report.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # =====================================================
    # PRINT SUMMARY
    # =====================================================

    print("\n" + "=" * 65)
    print("FINGERPRINT COLLISION SUMMARY")
    print("=" * 65)

    print("Total tests:", len(records))
    print("Random tests:", len(random_records))
    print("Collision tests:", len(collision_records))

    print("\nDeliberate collision results:")

    for columns in FINGERPRINT_COLUMN_OPTIONS:

        subset = [
            row
            for row in collision_records
            if row["fingerprint_columns"] == columns
        ]

        if not subset:
            continue

        triggers = sum(
            row["detector_triggered"]
            for row in subset
        )

        mean_estimate = statistics.mean(
            row["estimated_duplicate_rate"]
            for row in subset
        )

        mean_latency = statistics.mean(
            row["median_latency_ms"]
            for row in subset
        )

        print(
            f"F{columns}: "
            f"{triggers}/{len(subset)} "
            f"collision tests triggered; "
            f"mean estimated duplication="
            f"{mean_estimate:.3f}; "
            f"mean detection latency="
            f"{mean_latency:.4f} ms"
        )

    print("\nSaved results:")
    print(csv_path)
    print(json_path)

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
