
from pathlib import Path
import sys
import csv
import json
import random
import statistics
import time
from datetime import datetime, timezone

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
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
    / "rct100m_gpu_matched_collisions"
)

BATCH_SIZES = [4, 8, 16]
PROMPT_LENGTHS = [8, 24, 64]
FINGERPRINT_COLUMNS = [2, 4, 8]

THRESHOLD = 0.25
GENERATED_TOKENS = 8
VOCAB_SIZE = 8192

CLUSTERS = 10
WARMUP_REPETITIONS = 1
TIMED_REPETITIONS = 5
BOOTSTRAP_REPETITIONS = 10000

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
# DETERMINISTIC GREEDY DECODING
# =========================================================

def greedy_multinomial(
    input,
    num_samples,
    replacement=False,
    *,
    generator=None,
    out=None
):

    indices = torch.argmax(
        input,
        dim=-1,
        keepdim=True
    )

    if num_samples > 1:
        indices = indices.expand(
            *indices.shape[:-1],
            num_samples
        )

    if out is not None:
        out.copy_(indices)
        return out

    return indices


# =========================================================
# FINGERPRINT POSITIONS
# Must match the detector implementation exactly.
# =========================================================

def get_fingerprint_positions(
    prompt_length,
    fingerprint_columns
):

    return sorted(set(
        round(
            i * (prompt_length - 1)
            / max(fingerprint_columns - 1, 1)
        )
        for i in range(fingerprint_columns)
    ))


# =========================================================
# CREATE MATCHED PROMPT PAIR
# =========================================================

def create_matched_prompts(
    batch_size,
    prompt_length,
    fingerprint_columns,
    device
):

    selected_positions = get_fingerprint_positions(
        prompt_length,
        fingerprint_columns
    )

    selected_set = set(selected_positions)

    unselected_positions = [
        p
        for p in range(prompt_length)
        if p not in selected_set
    ]

    # We need at least one unselected position to preserve
    # distinct complete prompts after forcing identical fingerprints.
    if not unselected_positions:
        return None, None

    base_prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, prompt_length),
        dtype=torch.long,
        device=device
    )

    # Guarantee unique fingerprints in the control batch.
    first_selected = selected_positions[0]

    base_prompts[:, first_selected] = (
        torch.arange(batch_size, device=device) + 100
    )

    # Guarantee unique full prompts after selected positions
    # are overwritten in the collision variant.
    first_unselected = unselected_positions[0]

    base_prompts[:, first_unselected] = (
        torch.arange(batch_size, device=device) + 200
    )

    # Collision variant starts as an exact copy.
    collision_prompts = base_prompts.clone()

    # Change only fingerprint-selected positions.
    # All prompts now share an identical fingerprint.
    collision_prompts[:, selected_positions] = 1

    return base_prompts, collision_prompts


# =========================================================
# VERIFY MATCHED PAIR
# =========================================================

def verify_matched_pair(
    base_prompts,
    collision_prompts,
    adaptive_model,
    fingerprint_columns
):

    batch_size = base_prompts.size(0)

    selected_positions = get_fingerprint_positions(
        base_prompts.size(1),
        fingerprint_columns
    )

    # Both batches must contain unique complete prompts.
    base_unique = torch.unique(
        base_prompts,
        dim=0
    ).size(0)

    collision_unique = torch.unique(
        collision_prompts,
        dim=0
    ).size(0)

    if base_unique != batch_size:
        raise RuntimeError(
            "Control batch contains duplicate full prompts."
        )

    if collision_unique != batch_size:
        raise RuntimeError(
            "Collision batch contains duplicate full prompts."
        )

    # The only changed positions must be selected positions.
    changed = base_prompts != collision_prompts

    allowed = torch.zeros(
        base_prompts.size(1),
        dtype=torch.bool,
        device=base_prompts.device
    )

    allowed[selected_positions] = True

    if torch.any(changed[:, ~allowed]):
        raise RuntimeError(
            "Unexpected changes outside fingerprint positions."
        )

    if not torch.all(changed[:, allowed]):
        raise RuntimeError(
            "Expected every selected fingerprint position to change."
        )

    control_rate = adaptive_model.estimate_duplicate_rate(
        base_prompts
    )

    collision_rate = adaptive_model.estimate_duplicate_rate(
        collision_prompts
    )

    if isinstance(control_rate, torch.Tensor):
        control_rate = control_rate.item()

    if isinstance(collision_rate, torch.Tensor):
        collision_rate = collision_rate.item()

    if float(control_rate) != 0.0:
        raise RuntimeError(
            f"Control detector rate should be zero, got {control_rate}"
        )

    if float(collision_rate) < THRESHOLD:
        raise RuntimeError(
            "Collision detector did not reach the trigger threshold."
        )

    # Fingerprints should be identical in the collision variant.
    selected = collision_prompts[:, selected_positions]

    if torch.unique(selected, dim=0).size(0) != 1:
        raise RuntimeError(
            "Collision fingerprints are not identical."
        )

    return {
        "control_fingerprint_rate": float(control_rate),
        "collision_fingerprint_rate": float(collision_rate),
        "actual_duplicate_rate": 0.0,
        "changed_token_positions": len(selected_positions),
        "changed_token_cells": int(changed.sum().item())
    }


# =========================================================
# GENERATION AND TIMING
# =========================================================

def run_kv_only(kv_model, prompts):

    return kv_model.generate_cached(
        prompts,
        max_new_tokens=GENERATED_TOKENS,
        temperature=1.0
    )


def run_adaptive(adaptive_model, prompts):

    return adaptive_model.generate(
        prompts,
        max_new_tokens=GENERATED_TOKENS,
        temperature=1.0
    )


def timed_generation(function, prompts, device):

    torch.cuda.synchronize(device)

    start = time.perf_counter()

    output = function(prompts)

    torch.cuda.synchronize(device)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return output, elapsed_ms


# =========================================================
# BENCHMARK ONE SCENARIO
# =========================================================

def benchmark_scenario(
    prompts,
    kv_model,
    adaptive_model,
    device
):

    methods = {
        "kv_only": lambda x: run_kv_only(
            kv_model,
            x
        ),
        "adaptive_rct": lambda x: run_adaptive(
            adaptive_model,
            x
        )
    }

    # Exact output agreement check.
    kv_output = methods["kv_only"](prompts)
    adaptive_output = methods["adaptive_rct"](prompts)

    torch.cuda.synchronize(device)

    if not torch.equal(kv_output, adaptive_output):
        raise RuntimeError(
            "KV-only and adaptive outputs differ."
        )

    del kv_output
    del adaptive_output

    # Warm-up.
    for _ in range(WARMUP_REPETITIONS):

        order = list(methods.keys())
        random.shuffle(order)

        for name in order:
            methods[name](prompts)

        torch.cuda.synchronize(device)

    measurements = {
        "kv_only": [],
        "adaptive_rct": []
    }

    # Randomize order on every timed repetition.
    for _ in range(TIMED_REPETITIONS):

        order = list(methods.keys())
        random.shuffle(order)

        for name in order:

            _, elapsed = timed_generation(
                methods[name],
                prompts,
                device
            )

            measurements[name].append(elapsed)

    kv_median = statistics.median(
        measurements["kv_only"]
    )

    adaptive_median = statistics.median(
        measurements["adaptive_rct"]
    )

    return {
        "kv_only_median_ms": kv_median,
        "adaptive_median_ms": adaptive_median,
        "adaptive_speedup": kv_median / adaptive_median,
        "kv_measurements_ms": measurements["kv_only"],
        "adaptive_measurements_ms": measurements["adaptive_rct"],
        "correctness_passed": True
    }


# =========================================================
# STRATIFIED CLUSTER BOOTSTRAP
# =========================================================

def bootstrap_mean_ci(
    grouped_values,
    repetitions,
    rng
):

    estimates = []

    for _ in range(repetitions):

        sample = []

        for values in grouped_values:

            indices = rng.integers(
                0,
                len(values),
                size=len(values)
            )

            sample.extend(
                values[index]
                for index in indices
            )

        estimates.append(
            float(np.mean(sample))
        )

    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5))
    )


# =========================================================
# MAIN
# =========================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

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

    print("=" * 72)
    print("RCT100M MATCHED-PROMPT COLLISION BENCHMARK")
    print("=" * 72)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Threshold:", THRESHOLD)
    print("Generated tokens:", GENERATED_TOKENS)
    print("Batch sizes:", BATCH_SIZES)
    print("Prompt lengths:", PROMPT_LENGTHS)
    print("Fingerprint columns:", FINGERPRINT_COLUMNS)
    print("Matched clusters:", CLUSTERS)
    print("Warmups:", WARMUP_REPETITIONS)
    print("Timed repetitions:", TIMED_REPETITIONS)

    print("\nLoading checkpoint...")

    base_model, checkpoint = load_model(device)

    parameter_count = sum(
        p.numel()
        for p in base_model.parameters()
    )

    print("Checkpoint step:", checkpoint.get("step"))
    print("Parameters:", f"{parameter_count:,}")

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    records = []
    correctness_checks = 0
    skipped_configs = 0

    original_multinomial = torch.multinomial

    # Greedy decoding ensures exact deterministic comparisons.
    torch.multinomial = greedy_multinomial

    try:

        for batch_size in BATCH_SIZES:

            for prompt_length in PROMPT_LENGTHS:

                for columns in FINGERPRINT_COLUMNS:

                    adaptive_model = RCT100MAdaptive(
                        base_model,
                        duplicate_threshold=THRESHOLD,
                        fingerprint_columns=columns
                    )

                    adaptive_model.eval()

                    # P8/F8 has no unselected token position,
                    # so distinct complete prompts cannot retain
                    # unique identities while all fingerprint tokens
                    # are overwritten.
                    base_check, collision_check = (
                        create_matched_prompts(
                            batch_size,
                            prompt_length,
                            columns,
                            device
                        )
                    )

                    if base_check is None:
                        skipped_configs += 1

                        print(
                            f"\nSkipping impossible configuration: "
                            f"B{batch_size} P{prompt_length} F{columns}"
                        )

                        continue

                    del base_check
                    del collision_check

                    print(
                        f"\nCONFIG B{batch_size} "
                        f"P{prompt_length} F{columns}"
                    )

                    for cluster_id in range(CLUSTERS):

                        base_prompts, collision_prompts = (
                            create_matched_prompts(
                                batch_size,
                                prompt_length,
                                columns,
                                device
                            )
                        )

                        pair_info = verify_matched_pair(
                            base_prompts,
                            collision_prompts,
                            adaptive_model,
                            columns
                        )

                        # Same original prompts, same batch shape,
                        # same non-fingerprint token positions.
                        scenarios = [
                            ("unique_control", base_prompts),
                            ("deliberate_collision", collision_prompts)
                        ]

                        cluster_results = {}

                        for scenario, prompts in scenarios:

                            result = benchmark_scenario(
                                prompts,
                                kv_model,
                                adaptive_model,
                                device
                            )

                            record = {
                                "scenario": scenario,
                                "cluster_id": cluster_id,
                                "batch_size": batch_size,
                                "prompt_length": prompt_length,
                                "fingerprint_columns": columns,
                                "matched_pair": (
                                    f"B{batch_size}_P{prompt_length}_"
                                    f"F{columns}_C{cluster_id}"
                                ),
                                "actual_duplicate_rate": (
                                    pair_info["actual_duplicate_rate"]
                                ),
                                "control_fingerprint_rate": (
                                    pair_info["control_fingerprint_rate"]
                                ),
                                "collision_fingerprint_rate": (
                                    pair_info["collision_fingerprint_rate"]
                                ),
                                "changed_token_positions": (
                                    pair_info["changed_token_positions"]
                                ),
                                "changed_token_cells": (
                                    pair_info["changed_token_cells"]
                                ),
                                "kv_only_median_ms": (
                                    result["kv_only_median_ms"]
                                ),
                                "adaptive_median_ms": (
                                    result["adaptive_median_ms"]
                                ),
                                "adaptive_speedup": (
                                    result["adaptive_speedup"]
                                ),
                                "correctness_passed": (
                                    result["correctness_passed"]
                                ),
                                "kv_measurements_ms": (
                                    result["kv_measurements_ms"]
                                ),
                                "adaptive_measurements_ms": (
                                    result["adaptive_measurements_ms"]
                                )
                            }

                            records.append(record)
                            cluster_results[scenario] = record
                            correctness_checks += 1

                            print(
                                f"  {scenario}: "
                                f"KV={record['kv_only_median_ms']:.3f} ms, "
                                f"Adaptive={record['adaptive_median_ms']:.3f} ms, "
                                f"Speedup={record['adaptive_speedup']:.4f}x"
                            )

                        # Paired cluster effect.
                        control_speedup = (
                            cluster_results["unique_control"][
                                "adaptive_speedup"
                            ]
                        )

                        collision_speedup = (
                            cluster_results["deliberate_collision"][
                                "adaptive_speedup"
                            ]
                        )

                        print(
                            "  Matched collision effect:",
                            f"{collision_speedup - control_speedup:+.4f}x"
                        )

                        del base_prompts
                        del collision_prompts

                    del adaptive_model
                    torch.cuda.empty_cache()

    finally:

        torch.multinomial = original_multinomial

    # =====================================================
    # SAVE CSV
    # =====================================================

    csv_path = output_dir / "raw_results.csv"

    fields = [
        "scenario",
        "cluster_id",
        "batch_size",
        "prompt_length",
        "fingerprint_columns",
        "matched_pair",
        "actual_duplicate_rate",
        "control_fingerprint_rate",
        "collision_fingerprint_rate",
        "changed_token_positions",
        "changed_token_cells",
        "kv_only_median_ms",
        "adaptive_median_ms",
        "adaptive_speedup",
        "correctness_passed"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                key: record[key]
                for key in fields
            })

    # =====================================================
    # STATISTICAL ANALYSIS
    # =====================================================

    configs = sorted({
        (
            row["batch_size"],
            row["prompt_length"],
            row["fingerprint_columns"]
        )
        for row in records
    })

    rng = np.random.default_rng(SEED)

    scenario_summaries = {}

    for scenario in [
        "unique_control",
        "deliberate_collision"
    ]:

        grouped = []
        all_speedups = []
        all_kv = []
        all_adaptive = []

        for config in configs:

            values = []

            for cluster_id in range(CLUSTERS):

                matching = [
                    row for row in records
                    if row["scenario"] == scenario
                    and (
                        row["batch_size"],
                        row["prompt_length"],
                        row["fingerprint_columns"]
                    ) == config
                    and row["cluster_id"] == cluster_id
                ]

                if not matching:
                    continue

                row = matching[0]

                values.append(row["adaptive_speedup"])
                all_speedups.append(row["adaptive_speedup"])
                all_kv.append(row["kv_only_median_ms"])
                all_adaptive.append(row["adaptive_median_ms"])

            if values:
                grouped.append(values)

        ci_low, ci_high = bootstrap_mean_ci(
            grouped,
            BOOTSTRAP_REPETITIONS,
            rng
        )

        scenario_summaries[scenario] = {
            "workload_count": len(all_speedups),
            "mean_speedup": statistics.mean(all_speedups),
            "median_speedup": statistics.median(all_speedups),
            "ci95": [ci_low, ci_high],
            "mean_kv_latency_ms": statistics.mean(all_kv),
            "mean_adaptive_latency_ms": statistics.mean(all_adaptive)
        }

    # Compare collision and control speedups within each
    # identical matched-prompt cluster.
    grouped_differences = []
    all_differences = []

    for config in configs:

        differences = []

        for cluster_id in range(CLUSTERS):

            control = [
                row for row in records
                if row["scenario"] == "unique_control"
                and (
                    row["batch_size"],
                    row["prompt_length"],
                    row["fingerprint_columns"]
                ) == config
                and row["cluster_id"] == cluster_id
            ]

            collision = [
                row for row in records
                if row["scenario"] == "deliberate_collision"
                and (
                    row["batch_size"],
                    row["prompt_length"],
                    row["fingerprint_columns"]
                ) == config
                and row["cluster_id"] == cluster_id
            ]

            if not control or not collision:
                continue

            difference = (
                collision[0]["adaptive_speedup"]
                - control[0]["adaptive_speedup"]
            )

            differences.append(difference)
            all_differences.append(difference)

        if differences:
            grouped_differences.append(differences)

    difference_ci = bootstrap_mean_ci(
        grouped_differences,
        BOOTSTRAP_REPETITIONS,
        rng
    )

    paired_effect = {
        "mean_speedup_difference_collision_minus_control": (
            statistics.mean(all_differences)
        ),
        "median_speedup_difference": (
            statistics.median(all_differences)
        ),
        "ci95": list(difference_ci),
        "matched_cluster_count": len(all_differences)
    }

    report = {
        "run_id": run_id,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint_step": checkpoint.get("step"),
        "parameter_count": parameter_count,
        "threshold": THRESHOLD,
        "generated_tokens": GENERATED_TOKENS,
        "batch_sizes": BATCH_SIZES,
        "prompt_lengths": PROMPT_LENGTHS,
        "fingerprint_columns": FINGERPRINT_COLUMNS,
        "clusters_per_configuration": CLUSTERS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "timed_repetitions": TIMED_REPETITIONS,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "matched_prompt_design": (
            "Collision variant is cloned from the control batch. "
            "Only fingerprint-selected token positions are modified. "
            "Both variants retain unique complete prompts."
        ),
        "total_records": len(records),
        "correctness_checks": correctness_checks,
        "all_correctness_checks_passed": (
            correctness_checks == len(records)
        ),
        "skipped_configurations": skipped_configs,
        "scenario_summaries": scenario_summaries,
        "paired_collision_effect": paired_effect,
        "records": records
    }

    json_path = output_dir / "report.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 72)
    print("MATCHED-PROMPT BENCHMARK SUMMARY")
    print("=" * 72)

    print("Total records:", len(records))
    print("Correctness checks:", correctness_checks)
    print("Skipped configurations:", skipped_configs)

    for scenario, summary in scenario_summaries.items():

        print("\nScenario:", scenario)
        print("Workloads:", summary["workload_count"])

        print(
            "Mean speedup:",
            f"{summary['mean_speedup']:.4f}x"
        )

        print(
            "Median speedup:",
            f"{summary['median_speedup']:.4f}x"
        )

        print(
            "95% bootstrap CI:",
            f"[{summary['ci95'][0]:.4f}, "
            f"{summary['ci95'][1]:.4f}]"
        )

        print(
            "Mean KV latency:",
            f"{summary['mean_kv_latency_ms']:.3f} ms"
        )

        print(
            "Mean adaptive latency:",
            f"{summary['mean_adaptive_latency_ms']:.3f} ms"
        )

    print("\nMatched collision effect:")
    print(
        "Mean speedup difference:",
        f"{paired_effect['mean_speedup_difference_collision_minus_control']:+.4f}x"
    )

    print(
        "95% bootstrap CI:",
        f"[{difference_ci[0]:+.4f}, {difference_ci[1]:+.4f}]"
    )

    print(
        "Matched clusters:",
        paired_effect["matched_cluster_count"]
    )

    print("\nSaved:")
    print(csv_path)
    print(json_path)
    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
