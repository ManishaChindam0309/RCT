
from pathlib import Path
import sys
import time
import random
import csv
import json
from datetime import datetime, timezone
from statistics import median

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_prefix import RCT100MPrefixCache


CONFIGURATIONS = [
    (4, 24),
    (8, 16),
    (8, 24),
    (16, 16),
    (16, 24),
]

TOTAL_SEQUENCE_LENGTH = 32
NUM_CLUSTERS = 30
WARMUP_REPETITIONS = 2
TIMED_REPETITIONS = 5
SEED = 20261106


def timed_call(function, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    result = function()

    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms, result


def bootstrap_median_ci(
    values,
    seed=42,
    samples=10000,
):
    rng = random.Random(seed)
    estimates = []

    for _ in range(samples):
        sample = [
            rng.choice(values)
            for _ in range(len(values))
        ]
        estimates.append(median(sample))

    estimates.sort()

    lower = estimates[int(0.025 * samples)]
    upper = estimates[int(0.975 * samples)]

    return lower, upper


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")

    checkpoint_path = (
        PROJECT_DIR
        / "checkpoints"
        / "rct100m_continued"
        / "best_model.pt"
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = RCT100MConfig(**checkpoint["model_config"])

    model = RCT100M(config)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print("Training step:", checkpoint["step"])
    print("Parameters:", f"{model.parameter_count():,}")
    print("Precision: FP32")
    print("Independent clusters per configuration:", NUM_CLUSTERS)

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct100m_prefix_validation"
        / datetime.now(timezone.utc).strftime(
            "run_%Y%m%dT%H%M%SZ"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = []
    summary_records = []

    rng = random.Random(SEED)

    total_correctness_checks = 0

    for batch_size, prefix_length in CONFIGURATIONS:

        suffix_length = TOTAL_SEQUENCE_LENGTH - prefix_length

        print("\n" + "=" * 72)
        print(
            f"BATCH={batch_size}, "
            f"PREFIX={prefix_length}, "
            f"SUFFIX={suffix_length}"
        )

        cluster_standard_medians = []
        cluster_warm_medians = []
        cluster_speedups = []
        cluster_cold_times = []

        cluster_standard_memory = []
        cluster_cached_memory = []

        for cluster_id in range(NUM_CLUSTERS):

            cluster_seed = (
                SEED
                + batch_size * 100000
                + prefix_length * 1000
                + cluster_id
            )

            torch.manual_seed(cluster_seed)
            torch.cuda.manual_seed_all(cluster_seed)

            # Generate an independent prefix for this cluster.
            prefix = torch.randint(
                0,
                config.vocab_size,
                (1, prefix_length),
                device=device,
                dtype=torch.long,
            )

            suffixes = torch.randint(
                0,
                config.vocab_size,
                (batch_size, suffix_length),
                device=device,
                dtype=torch.long,
            )

            prompts = torch.cat(
                [
                    prefix.expand(batch_size, -1),
                    suffixes,
                ],
                dim=1,
            )

            prefix_model = RCT100MPrefixCache(
                model,
                cache_capacity=2,
            )
            prefix_model.eval()

            standard_function = lambda: model(prompts)

            cached_function = lambda: (
                prefix_model.forward_cached(
                    prompts,
                    prefix_length,
                )
            )

            with torch.inference_mode():

                # Standard reference.
                standard_logits = standard_function()

                # Measure cold cache: first request for this prefix.
                torch.cuda.reset_peak_memory_stats(device)

                cold_ms, cold_logits = timed_call(
                    cached_function,
                    device,
                )

                cold_peak_mb = (
                    torch.cuda.max_memory_allocated(device)
                    / 1024**2
                )

                # Correctness check for cold cache.
                if not torch.allclose(
                    standard_logits,
                    cold_logits,
                    rtol=1e-4,
                    atol=1e-4,
                ):
                    raise RuntimeError(
                        f"Cold correctness failed: "
                        f"batch={batch_size}, "
                        f"prefix={prefix_length}, "
                        f"cluster={cluster_id}"
                    )

                total_correctness_checks += 1

                # Verify that the prefix was cached.
                stats = prefix_model.cache_statistics()

                if stats["misses"] != 1:
                    raise RuntimeError(
                        "Expected one cold-cache miss."
                    )

                # Warm up both methods.
                for _ in range(WARMUP_REPETITIONS):
                    standard_function()
                    cached_function()

                # Warm-cache correctness check.
                warm_logits = cached_function()

                if not torch.allclose(
                    standard_logits,
                    warm_logits,
                    rtol=1e-4,
                    atol=1e-4,
                ):
                    raise RuntimeError(
                        f"Warm correctness failed: "
                        f"batch={batch_size}, "
                        f"prefix={prefix_length}, "
                        f"cluster={cluster_id}"
                    )

                total_correctness_checks += 1

                # Measure standard peak memory.
                torch.cuda.reset_peak_memory_stats(device)
                standard_function()
                torch.cuda.synchronize(device)

                standard_peak_mb = (
                    torch.cuda.max_memory_allocated(device)
                    / 1024**2
                )

                # Measure warm-cache peak memory.
                torch.cuda.reset_peak_memory_stats(device)
                cached_function()
                torch.cuda.synchronize(device)

                cached_peak_mb = (
                    torch.cuda.max_memory_allocated(device)
                    / 1024**2
                )

                standard_times = []
                cached_times = []

                # Randomize execution order within each paired repeat.
                for repetition in range(TIMED_REPETITIONS):

                    methods = [
                        "standard",
                        "prefix_cached",
                    ]
                    rng.shuffle(methods)

                    measurements = {}

                    for method in methods:

                        if method == "standard":
                            elapsed, _ = timed_call(
                                standard_function,
                                device,
                            )
                        else:
                            elapsed, _ = timed_call(
                                cached_function,
                                device,
                            )

                        measurements[method] = elapsed

                    standard_ms = measurements["standard"]
                    cached_ms = measurements["prefix_cached"]

                    standard_times.append(standard_ms)
                    cached_times.append(cached_ms)

                    raw_records.append({
                        "batch_size": batch_size,
                        "prefix_length": prefix_length,
                        "suffix_length": suffix_length,
                        "cluster_id": cluster_id,
                        "repetition": repetition,
                        "standard_ms": standard_ms,
                        "prefix_cached_ms": cached_ms,
                        "paired_speedup": (
                            standard_ms / cached_ms
                        ),
                        "cold_cache_ms": cold_ms,
                        "correctness_passed": True,
                    })

            standard_cluster_median = median(standard_times)
            cached_cluster_median = median(cached_times)

            # One paired result per independent prompt cluster.
            cluster_speedup = (
                standard_cluster_median
                / cached_cluster_median
            )

            cluster_standard_medians.append(
                standard_cluster_median
            )
            cluster_warm_medians.append(
                cached_cluster_median
            )
            cluster_speedups.append(cluster_speedup)
            cluster_cold_times.append(cold_ms)

            cluster_standard_memory.append(standard_peak_mb)
            cluster_cached_memory.append(cached_peak_mb)

            if (cluster_id + 1) % 5 == 0:
                print(
                    f"Completed clusters: "
                    f"{cluster_id + 1}/{NUM_CLUSTERS}"
                )

            del prefix_model
            del prompts
            del standard_logits
            del cold_logits
            del warm_logits

        # Confidence interval resamples independent clusters,
        # not individual timing repetitions.
        ci_lower, ci_upper = bootstrap_median_ci(
            cluster_speedups,
            seed=SEED + batch_size + prefix_length,
        )

        median_standard = median(cluster_standard_medians)
        median_warm = median(cluster_warm_medians)
        median_speedup = median(cluster_speedups)
        median_cold = median(cluster_cold_times)

        standard_memory = median(cluster_standard_memory)
        cached_memory = median(cluster_cached_memory)

        summary = {
            "batch_size": batch_size,
            "prefix_length": prefix_length,
            "suffix_length": suffix_length,
            "independent_clusters": NUM_CLUSTERS,
            "timed_repetitions_per_cluster": TIMED_REPETITIONS,
            "standard_median_ms": median_standard,
            "cold_cache_median_ms": median_cold,
            "warm_cache_median_ms": median_warm,
            "median_cluster_speedup": median_speedup,
            "speedup_ci_lower": ci_lower,
            "speedup_ci_upper": ci_upper,
            "standard_peak_memory_mb": standard_memory,
            "warm_cache_peak_memory_mb": cached_memory,
            "memory_change_mb": cached_memory - standard_memory,
            "correctness_checks": NUM_CLUSTERS * 2,
            "correctness_passed": True,
        }

        summary_records.append(summary)

        print("\nConfiguration summary:")
        print(f"Standard median: {median_standard:.3f} ms")
        print(f"Cold cache median: {median_cold:.3f} ms")
        print(f"Warm cache median: {median_warm:.3f} ms")
        print(f"Paired cluster speedup: {median_speedup:.3f}x")
        print(
            f"95% cluster-bootstrap CI: "
            f"[{ci_lower:.3f}, {ci_upper:.3f}]"
        )
        print(
            f"Standard peak memory: {standard_memory:.2f} MB"
        )
        print(
            f"Warm cache peak memory: {cached_memory:.2f} MB"
        )
        print("Correctness: PASS")

    raw_path = output_dir / "raw_timings.csv"
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "report.json"

    with open(
        raw_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(raw_records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(raw_records)

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_records)

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_step": checkpoint["step"],
        "parameters": model.parameter_count(),
        "precision": "FP32",
        "total_sequence_length": TOTAL_SEQUENCE_LENGTH,
        "independent_clusters_per_configuration": NUM_CLUSTERS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "timed_repetitions_per_cluster": TIMED_REPETITIONS,
        "total_correctness_checks": total_correctness_checks,
        "all_correctness_passed": all(
            row["correctness_passed"]
            for row in summary_records
        ),
        "configurations": summary_records,
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 72)
    print("PREFIX CACHE VALIDATION COMPLETED")
    print("Configurations:", len(summary_records))
    print("Independent clusters:", NUM_CLUSTERS)
    print("Raw timing records:", len(raw_records))
    print("Correctness checks:", total_correctness_checks)
    print("Correctness: PASS")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
