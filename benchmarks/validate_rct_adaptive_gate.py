
import csv
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference import load_rct_inference


# =========================================================
# Configuration
# =========================================================

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "rct100m_adaptive_gate_confirmation"
)

PROMPT_LENGTH = 32
NEW_TOKENS = 8

DEDUP_THRESHOLD = 0.3125

GROUPS = 30
REPETITIONS = 5
WARMUPS = 2

BOOTSTRAP_ITERATIONS = 10000
SEED = 20261107

# (batch size, unique prompts)
WORKLOADS = [
    (4, 4),
    (4, 3),
    (4, 2),
    (4, 1),

    (8, 6),
    (8, 5),
    (8, 4),
    (8, 2),

    (16, 12),
    (16, 11),
    (16, 8),
    (16, 4),
    (16, 2),
]


# =========================================================
# Prompt creation
# =========================================================

def make_prompts(batch_size, unique_count, vocab_size, device, seed):

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    while True:
        unique = torch.randint(
            1,
            vocab_size,
            (unique_count, PROMPT_LENGTH),
            generator=generator,
            dtype=torch.long,
        )

        if torch.unique(unique, dim=0).size(0) == unique_count:
            break

    indices = torch.arange(batch_size) % unique_count

    permutation = torch.randperm(
        batch_size,
        generator=generator,
    )

    prompts = unique.index_select(
        0,
        indices[permutation],
    )

    return prompts.to(device)


# =========================================================
# Timing
# =========================================================

def synchronize():
    torch.cuda.synchronize()


def timed_generate(inference, prompts, mode):

    synchronize()
    start = time.perf_counter()

    output = inference.generate(
        prompts,
        max_new_tokens=NEW_TOKENS,
        mode=mode,
    )

    synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return output, elapsed_ms


# =========================================================
# Paired cluster bootstrap
# =========================================================

def bootstrap_speedup(kv_values, adaptive_values, seed):

    kv_values = np.asarray(kv_values, dtype=np.float64)
    adaptive_values = np.asarray(adaptive_values, dtype=np.float64)

    observed = (
        np.median(kv_values) / np.median(adaptive_values)
    )

    rng = np.random.default_rng(seed)

    count = len(kv_values)

    indices = rng.integers(
        0,
        count,
        size=(BOOTSTRAP_ITERATIONS, count),
    )

    boot_kv = np.median(kv_values[indices], axis=1)
    boot_adaptive = np.median(adaptive_values[indices], axis=1)

    ratios = boot_kv / boot_adaptive

    lower, upper = np.percentile(
        ratios,
        [2.5, 97.5],
    )

    return float(observed), float(lower), float(upper)


# =========================================================
# Adaptive gate verification
# =========================================================

def verify_gate(inference, batch_size, unique_count, threshold):

    duplicate_count = batch_size - unique_count
    duplicate_rate = duplicate_count / batch_size

    expected_activation = (
        batch_size > 1 and duplicate_rate >= threshold
    )

    # Reset before the current group's adaptive call.
    inference.reset_adaptive_statistics()

    # Use a separate call to trigger the adaptive gate.
    # The caller performs output correctness comparison.
    return expected_activation, duplicate_count


def check_gate_statistics(
    stats,
    expected_activation,
    expected_duplicate_count,
):

    activations = stats["dedup_activations"]
    saved = stats["duplicate_prompts_saved"]

    expected_activations = int(expected_activation)

    if activations != expected_activations:
        raise AssertionError(
            "Incorrect adaptive activation count: "
            f"expected {expected_activations}, got {activations}"
        )

    expected_saved = (
        expected_duplicate_count if expected_activation else 0
    )

    if saved != expected_saved:
        raise AssertionError(
            "Incorrect duplicate-saving count: "
            f"expected {expected_saved}, got {saved}"
        )


# =========================================================
# Main
# =========================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    run_name = datetime.now(
        timezone.utc
    ).strftime("run_%Y%m%dT%H%M%SZ")

    output_dir = RESULTS_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    measurements = []
    summaries = []

    print("=" * 70)
    print("RCT 100M ADAPTIVE GATE CONFIRMATION")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", CHECKPOINT)
    print("Prompt length:", PROMPT_LENGTH)
    print("Generated tokens:", NEW_TOKENS)
    print("Dedup threshold:", f"{DEDUP_THRESHOLD:.2%}")
    print("Independent groups:", GROUPS)
    print("Timed repetitions:", REPETITIONS)
    print("Warmups:", WARMUPS)
    print("Bootstrap iterations:", BOOTSTRAP_ITERATIONS)

    print("\nLoading checkpoint...")

    inference = load_rct_inference(
        CHECKPOINT,
        device="cuda",
        dedup_threshold=DEDUP_THRESHOLD,
    )

    vocab_size = inference.model.config.vocab_size

    print(
        "Parameters:",
        f"{inference.model.parameter_count():,}",
    )

    for workload_index, (batch_size, unique_count) in enumerate(WORKLOADS):

        duplicate_count = batch_size - unique_count
        duplicate_rate = duplicate_count / batch_size

        expected_activation = (
            batch_size > 1
            and duplicate_rate >= DEDUP_THRESHOLD
        )

        print("\n" + "=" * 70)
        print(
            f"WORKLOAD {workload_index + 1}/{len(WORKLOADS)}"
        )
        print("=" * 70)

        print("Batch:", batch_size)
        print("Unique prompts:", unique_count)
        print("Duplicate prompts:", duplicate_count)
        print("Duplicate rate:", f"{duplicate_rate:.2%}")
        print(
            "Expected gate:",
            "DEDUPLICATE" if expected_activation else "KV ONLY",
        )

        kv_group_times = []
        adaptive_group_times = []

        correctness_checks = 0
        gate_checks = 0

        workload_peak_memory = 0.0

        for group in range(GROUPS):

            prompts = make_prompts(
                batch_size,
                unique_count,
                vocab_size,
                "cuda",
                SEED + workload_index * 100000 + group * 100,
            )

            # -----------------------------------------
            # KV reference output
            # -----------------------------------------

            reference = inference.generate(
                prompts,
                max_new_tokens=NEW_TOKENS,
                mode="kv",
            )

            # -----------------------------------------
            # Reset counters before each adaptive call.
            # -----------------------------------------

            inference.reset_adaptive_statistics()

            adaptive_output = inference.generate(
                prompts,
                max_new_tokens=NEW_TOKENS,
                mode="adaptive",
            )

            synchronize()

            if not torch.equal(reference, adaptive_output):
                raise AssertionError(
                    f"Output mismatch at workload {workload_index + 1}, "
                    f"group {group + 1}"
                )

            correctness_checks += 1

            # -----------------------------------------
            # Verify only this group's gate statistics.
            # -----------------------------------------

            stats = inference.adaptive_statistics()

            check_gate_statistics(
                stats,
                expected_activation,
                duplicate_count,
            )

            gate_checks += 1

            del reference, adaptive_output

            # -----------------------------------------
            # Warmups
            # -----------------------------------------

            for mode in ("kv", "adaptive"):

                for _ in range(WARMUPS):

                    output, _ = timed_generate(
                        inference,
                        prompts,
                        mode,
                    )

                    del output

            # -----------------------------------------
            # Timed repetitions
            # -----------------------------------------

            times = {
                "kv": [],
                "adaptive": [],
            }

            for repetition in range(REPETITIONS):

                methods = ["kv", "adaptive"]
                random.shuffle(methods)

                for mode in methods:

                    output, elapsed_ms = timed_generate(
                        inference,
                        prompts,
                        mode,
                    )

                    times[mode].append(elapsed_ms)

                    measurements.append({
                        "batch_size": batch_size,
                        "unique_prompts": unique_count,
                        "duplicate_prompts": duplicate_count,
                        "duplicate_rate": duplicate_rate,
                        "group": group + 1,
                        "repetition": repetition + 1,
                        "method": mode,
                        "latency_ms": elapsed_ms,
                    })

                    del output

            kv_group_times.append(
                float(np.median(times["kv"]))
            )

            adaptive_group_times.append(
                float(np.median(times["adaptive"]))
            )

            workload_peak_memory = max(
                workload_peak_memory,
                torch.cuda.max_memory_allocated() / (1024 ** 2),
            )

            del prompts

            if (group + 1) % 10 == 0:
                print(
                    f"Completed groups: {group + 1}/{GROUPS}"
                )

        # ---------------------------------------------
        # Statistical summary
        # ---------------------------------------------

        kv_median = float(np.median(kv_group_times))
        adaptive_median = float(np.median(adaptive_group_times))

        speedup, ci_lower, ci_upper = bootstrap_speedup(
            kv_group_times,
            adaptive_group_times,
            SEED + workload_index,
        )

        reduction = (
            1.0 - adaptive_median / kv_median
        ) * 100.0

        if ci_lower > 1.0:
            conclusion = "ADAPTIVE FASTER"
        elif ci_upper < 1.0:
            conclusion = "ADAPTIVE SLOWER"
        else:
            conclusion = "INCONCLUSIVE"

        summary = {
            "batch_size": batch_size,
            "unique_prompts": unique_count,
            "duplicate_prompts": duplicate_count,
            "duplicate_rate": duplicate_rate,
            "threshold": DEDUP_THRESHOLD,
            "gate_activated": expected_activation,
            "kv_median_ms": kv_median,
            "adaptive_median_ms": adaptive_median,
            "speedup": speedup,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "latency_reduction_percent": reduction,
            "peak_memory_mb": workload_peak_memory,
            "correctness_checks": correctness_checks,
            "gate_checks": gate_checks,
            "conclusion": conclusion,
        }

        summaries.append(summary)

        print("\nRESULT")
        print(f"KV median: {kv_median:.3f} ms")
        print(f"Adaptive median: {adaptive_median:.3f} ms")
        print(f"Speedup: {speedup:.4f}x")
        print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"Latency reduction: {reduction:.2f}%")
        print(f"Peak memory: {workload_peak_memory:.1f} MiB")
        print(f"Correctness: {correctness_checks}/{GROUPS}")
        print(f"Gate checks: {gate_checks}/{GROUPS}")
        print("Conclusion:", conclusion)

    # =====================================================
    # Save results
    # =====================================================

    measurement_path = (
        output_dir / "confirmation_measurements.csv"
    )

    summary_path = (
        output_dir / "confirmation_summary.csv"
    )

    with open(
        measurement_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(measurements[0].keys()),
        )

        writer.writeheader()
        writer.writerows(measurements)

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summaries)

    # =====================================================
    # Final report
    # =====================================================

    print("\n" + "=" * 70)
    print("CONFIRMATION SUMMARY")
    print("=" * 70)

    print(
        f"{'Batch':>6} "
        f"{'Dup %':>9} "
        f"{'KV ms':>11} "
        f"{'Adaptive ms':>13} "
        f"{'Speedup':>10} "
        f"{'95% CI':>21}"
    )

    for row in summaries:

        ci = (
            f"[{row['ci_95_lower']:.3f}, "
            f"{row['ci_95_upper']:.3f}]"
        )

        print(
            f"{row['batch_size']:>6} "
            f"{row['duplicate_rate'] * 100:>8.2f}% "
            f"{row['kv_median_ms']:>11.3f} "
            f"{row['adaptive_median_ms']:>13.3f} "
            f"{row['speedup']:>9.3f}x "
            f"{ci:>21}"
        )

    print("\nAll correctness and gate checks passed.")
    print("Measurements:", measurement_path)
    print("Summary:", summary_path)
    print("\nCONFIRMATION COMPLETED")


if __name__ == "__main__":
    main()
