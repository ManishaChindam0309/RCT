
import csv
import random
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference import load_rct_inference


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "rct100m_adaptive_gate_benchmark"
)

PROMPT_LENGTH = 32
NEW_TOKENS = 8

DEDUP_THRESHOLD = 0.3125

GROUPS_PER_WORKLOAD = 20
REPETITIONS_PER_METHOD = 3
WARMUP_REPETITIONS = 1

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 20261106


# Each workload is (batch_size, unique_prompt_count).
WORKLOADS = [
    (4, 4),
    (4, 3),
    (4, 2),
    (4, 1),

    (8, 8),
    (8, 6),
    (8, 5),
    (8, 4),
    (8, 2),
    (8, 1),

    (16, 16),
    (16, 12),
    (16, 11),
    (16, 8),
    (16, 4),
    (16, 2),
]


# ---------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------

def make_prompts(
    batch_size,
    unique_count,
    vocab_size,
    device,
    seed,
):
    """
    Create a batch with exactly unique_count distinct prompts.

    Duplicate prompts are distributed across the batch in
    randomized order.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    while True:

        unique_prompts = torch.randint(
            low=1,
            high=vocab_size,
            size=(unique_count, PROMPT_LENGTH),
            dtype=torch.long,
            generator=generator,
        )

        if torch.unique(
            unique_prompts,
            dim=0,
        ).size(0) == unique_count:
            break

    indices = (
        torch.arange(batch_size) % unique_count
    )

    permutation = torch.randperm(
        batch_size,
        generator=generator,
    )

    indices = indices[permutation]

    prompts = unique_prompts.index_select(
        0,
        indices,
    )

    return prompts.to(device)


# ---------------------------------------------------------
# Timing
# ---------------------------------------------------------

def synchronize():
    torch.cuda.synchronize()


def timed_generation(
    inference,
    prompts,
    mode,
):
    """
    Measure complete generation latency.

    Adaptive timing includes:
      - exact duplicate detection
      - threshold decision
      - KV-cache generation
      - output expansion when deduplication activates
    """

    synchronize()

    start = time.perf_counter()

    output = inference.generate(
        prompts,
        max_new_tokens=NEW_TOKENS,
        mode=mode,
    )

    synchronize()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return output, elapsed_ms


# ---------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------

def paired_cluster_bootstrap(
    kv_group_medians,
    adaptive_group_medians,
    seed,
):
    """
    Paired cluster bootstrap.

    Each independent prompt group is one cluster.
    Resampling preserves the pairing between methods.
    """

    kv_values = np.asarray(
        kv_group_medians,
        dtype=np.float64,
    )

    adaptive_values = np.asarray(
        adaptive_group_medians,
        dtype=np.float64,
    )

    observed = (
        np.median(kv_values)
        / np.median(adaptive_values)
    )

    rng = np.random.default_rng(seed)

    group_count = len(kv_values)

    indices = rng.integers(
        0,
        group_count,
        size=(
            BOOTSTRAP_ITERATIONS,
            group_count,
        ),
    )

    boot_kv = np.median(
        kv_values[indices],
        axis=1,
    )

    boot_adaptive = np.median(
        adaptive_values[indices],
        axis=1,
    )

    ratios = boot_kv / boot_adaptive

    lower, upper = np.percentile(
        ratios,
        [2.5, 97.5],
    )

    return float(observed), float(lower), float(upper)


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark."
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    device = "cuda"

    run_name = datetime.now(
        timezone.utc
    ).strftime("run_%Y%m%dT%H%M%SZ")

    output_dir = RESULTS_ROOT / run_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    measurement_path = (
        output_dir / "benchmark_measurements.csv"
    )

    summary_path = (
        output_dir / "benchmark_summary.csv"
    )

    print("=" * 70)
    print("RCT 100M ADAPTIVE GATE BENCHMARK")
    print("=" * 70)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Checkpoint:",
        CHECKPOINT,
    )

    print(
        "Prompt length:",
        PROMPT_LENGTH,
    )

    print(
        "Generated tokens:",
        NEW_TOKENS,
    )

    print(
        "Dedup threshold:",
        f"{DEDUP_THRESHOLD:.2%}",
    )

    print(
        "Independent groups per workload:",
        GROUPS_PER_WORKLOAD,
    )

    print(
        "Timed repetitions per method:",
        REPETITIONS_PER_METHOD,
    )

    print(
        "Warmups per method:",
        WARMUP_REPETITIONS,
    )

    print(
        "Bootstrap iterations:",
        BOOTSTRAP_ITERATIONS,
    )

    print("\nLoading checkpoint...")

    inference = load_rct_inference(
        CHECKPOINT,
        device=device,
        dedup_threshold=DEDUP_THRESHOLD,
    )

    inference.model.eval()

    vocab_size = inference.model.config.vocab_size

    print(
        "Parameters:",
        f"{inference.model.parameter_count():,}",
    )

    print(
        "Checkpoint step:",
        inference.model.config.num_layers,
        "layers loaded",
    )

    measurement_rows = []
    summary_rows = []

    for workload_index, (
        batch_size,
        unique_count,
    ) in enumerate(WORKLOADS):

        duplicate_count = (
            batch_size - unique_count
        )

        duplicate_rate = (
            duplicate_count / batch_size
        )

        expected_activation = (
            duplicate_rate >= DEDUP_THRESHOLD
        )

        print("\n" + "=" * 70)
        print(
            f"WORKLOAD {workload_index + 1}/{len(WORKLOADS)}"
        )
        print("=" * 70)

        print(f"Batch size: {batch_size}")
        print(f"Unique prompts: {unique_count}")
        print(f"Duplicate prompts: {duplicate_count}")
        print(f"Duplicate rate: {duplicate_rate:.2%}")
        print(
            "Expected gate:",
            "DEDUPLICATE" if expected_activation else "KV ONLY",
        )

        kv_group_medians = []
        adaptive_group_medians = []

        workload_correctness = 0

        torch.cuda.reset_peak_memory_stats()

        for group_index in range(GROUPS_PER_WORKLOAD):

            seed = (
                RANDOM_SEED
                + workload_index * 100000
                + group_index * 100
                + batch_size
            )

            prompts = make_prompts(
                batch_size=batch_size,
                unique_count=unique_count,
                vocab_size=vocab_size,
                device=device,
                seed=seed,
            )

            # -----------------------------------------
            # Correctness validation
            # -----------------------------------------

            reference = inference.generate(
                prompts,
                max_new_tokens=NEW_TOKENS,
                mode="kv",
            )

            adaptive_output = inference.generate(
                prompts,
                max_new_tokens=NEW_TOKENS,
                mode="adaptive",
            )

            synchronize()

            if not torch.equal(
                reference,
                adaptive_output,
            ):
                raise AssertionError(
                    "KV/adaptive output mismatch: "
                    f"batch={batch_size}, "
                    f"unique={unique_count}, "
                    f"group={group_index}"
                )

            workload_correctness += 1

            del reference
            del adaptive_output

            # -----------------------------------------
            # Warmup
            # -----------------------------------------

            for mode in ("kv", "adaptive"):

                for _ in range(WARMUP_REPETITIONS):

                    output, _ = timed_generation(
                        inference,
                        prompts,
                        mode,
                    )

                    del output

            # -----------------------------------------
            # Timed repetitions
            # -----------------------------------------

            group_times = {
                "kv": [],
                "adaptive": [],
            }

            for repetition in range(
                REPETITIONS_PER_METHOD
            ):

                methods = [
                    "kv",
                    "adaptive",
                ]

                random.shuffle(methods)

                for mode in methods:

                    output, elapsed_ms = timed_generation(
                        inference,
                        prompts,
                        mode,
                    )

                    group_times[mode].append(
                        elapsed_ms
                    )

                    measurement_rows.append({
                        "batch_size": batch_size,
                        "unique_prompts": unique_count,
                        "duplicate_prompts": duplicate_count,
                        "duplicate_rate": duplicate_rate,
                        "group": group_index + 1,
                        "repetition": repetition + 1,
                        "method": mode,
                        "latency_ms": elapsed_ms,
                    })

                    del output

            kv_group_medians.append(
                float(np.median(group_times["kv"]))
            )

            adaptive_group_medians.append(
                float(np.median(group_times["adaptive"]))
            )

            del prompts

            if (group_index + 1) % 5 == 0:

                print(
                    f"Completed groups: "
                    f"{group_index + 1}/"
                    f"{GROUPS_PER_WORKLOAD}"
                )

        # ---------------------------------------------
        # Workload summary
        # ---------------------------------------------

        kv_median = float(
            np.median(kv_group_medians)
        )

        adaptive_median = float(
            np.median(adaptive_group_medians)
        )

        speedup, ci_lower, ci_upper = (
            paired_cluster_bootstrap(
                kv_group_medians,
                adaptive_group_medians,
                RANDOM_SEED + workload_index,
            )
        )

        latency_reduction = (
            1.0 - adaptive_median / kv_median
        ) * 100.0

        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        if ci_lower > 1.0:
            result = "ADAPTIVE FASTER"
        elif ci_upper < 1.0:
            result = "ADAPTIVE SLOWER"
        else:
            result = "NO RELIABLE DIFFERENCE"

        summary = {
            "batch_size": batch_size,
            "unique_prompts": unique_count,
            "duplicate_prompts": duplicate_count,
            "duplicate_rate": duplicate_rate,
            "threshold": DEDUP_THRESHOLD,
            "gate_activated": expected_activation,
            "kv_median_ms": kv_median,
            "adaptive_median_ms": adaptive_median,
            "speedup_kv_over_adaptive": speedup,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "latency_reduction_percent": latency_reduction,
            "peak_memory_mb": peak_memory_mb,
            "correctness_groups": workload_correctness,
            "result": result,
        }

        summary_rows.append(summary)

        print("\nRESULT")
        print(f"KV median: {kv_median:.3f} ms")
        print(f"Adaptive median: {adaptive_median:.3f} ms")
        print(f"Speedup: {speedup:.4f}x")
        print(
            f"95% paired bootstrap CI: "
            f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        )
        print(
            f"Latency reduction: "
            f"{latency_reduction:.2f}%"
        )
        print(
            f"Peak allocated GPU memory: "
            f"{peak_memory_mb:.1f} MiB"
        )
        print(
            f"Correctness: "
            f"{workload_correctness}/"
            f"{GROUPS_PER_WORKLOAD}"
        )
        print(f"Conclusion: {result}")

        del kv_group_medians
        del adaptive_group_medians

    # -----------------------------------------------------
    # Save CSV files
    # -----------------------------------------------------

    with open(
        measurement_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(measurement_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(measurement_rows)

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL ADAPTIVE GATE BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"{'Batch':>6} "
        f"{'Dup %':>9} "
        f"{'KV ms':>11} "
        f"{'Adaptive ms':>13} "
        f"{'Speedup':>10} "
        f"{'95% CI':>21}"
    )

    for row in summary_rows:

        ci_text = (
            f"[{row['ci_95_lower']:.3f}, "
            f"{row['ci_95_upper']:.3f}]"
        )

        print(
            f"{row['batch_size']:>6} "
            f"{row['duplicate_rate'] * 100:>8.2f}% "
            f"{row['kv_median_ms']:>11.3f} "
            f"{row['adaptive_median_ms']:>13.3f} "
            f"{row['speedup_kv_over_adaptive']:>9.3f}x "
            f"{ci_text:>21}"
        )

    print("\nAll correctness checks passed.")

    print(
        "Measurement CSV:",
        measurement_path,
    )

    print(
        "Summary CSV:",
        summary_path,
    )

    print("\nBENCHMARK COMPLETED")


if __name__ == "__main__":
    main()
