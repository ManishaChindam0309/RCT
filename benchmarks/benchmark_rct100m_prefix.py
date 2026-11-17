
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


BATCH_SIZES = [4, 8, 16]
PREFIX_LENGTHS = [8, 16, 24]
TOTAL_SEQUENCE_LENGTH = 32

WARMUP_REPETITIONS = 5
TIMED_REPETITIONS = 20
SEED = 20261106


def timed_call(function, device):
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    result = function()

    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000

    return elapsed, result


def bootstrap_median_ci(values, seed=42, samples=5000):
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

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct100m_prefix_benchmark"
        / datetime.now(timezone.utc).strftime(
            "run_%Y%m%dT%H%M%SZ"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = []
    summary_records = []

    rng = random.Random(SEED)

    for batch_size in BATCH_SIZES:

        for prefix_length in PREFIX_LENGTHS:

            suffix_length = (
                TOTAL_SEQUENCE_LENGTH - prefix_length
            )

            print("\n" + "=" * 65)
            print(
                f"BATCH={batch_size}, "
                f"PREFIX={prefix_length}, "
                f"SUFFIX={suffix_length}"
            )

            torch.manual_seed(
                SEED + batch_size * 100 + prefix_length
            )

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

            shared_prefix = prefix.expand(batch_size, -1)

            prompts = torch.cat(
                [shared_prefix, suffixes],
                dim=1,
            )

            prefix_model = RCT100MPrefixCache(
                model,
                cache_capacity=4,
            )

            prefix_model.eval()

            standard_function = lambda: model(prompts)

            cached_function = lambda: (
                prefix_model.forward_cached(
                    prompts,
                    prefix_length,
                )
            )

            # Standard reference.
            with torch.inference_mode():
                standard_logits = standard_function()

            # First prefix call: cold cache.
            torch.cuda.reset_peak_memory_stats(device)

            with torch.inference_mode():
                cold_ms, cold_logits = timed_call(
                    cached_function,
                    device,
                )

            cold_peak_mb = (
                torch.cuda.max_memory_allocated(device)
                / 1024**2
            )

            if not torch.allclose(
                standard_logits,
                cold_logits,
                rtol=1e-4,
                atol=1e-4,
            ):
                raise RuntimeError(
                    "Cold prefix-cache correctness failed."
                )

            # Second call should be a cache hit.
            with torch.inference_mode():
                warm_ms, warm_logits = timed_call(
                    cached_function,
                    device,
                )

            if not torch.allclose(
                standard_logits,
                warm_logits,
                rtol=1e-4,
                atol=1e-4,
            ):
                raise RuntimeError(
                    "Warm prefix-cache correctness failed."
                )

            # Warm up both paths.
            with torch.inference_mode():

                for _ in range(WARMUP_REPETITIONS):
                    standard_function()
                    cached_function()

            torch.cuda.synchronize(device)

            # Measure standard memory peak.
            torch.cuda.reset_peak_memory_stats(device)

            with torch.inference_mode():
                standard_function()

            torch.cuda.synchronize(device)

            standard_peak_mb = (
                torch.cuda.max_memory_allocated(device)
                / 1024**2
            )

            # Measure prefix-cache memory peak.
            torch.cuda.reset_peak_memory_stats(device)

            with torch.inference_mode():
                cached_function()

            torch.cuda.synchronize(device)

            cached_peak_mb = (
                torch.cuda.max_memory_allocated(device)
                / 1024**2
            )

            standard_times = []
            cached_times = []
            paired_speedups = []

            for repetition in range(TIMED_REPETITIONS):

                methods = [
                    "standard",
                    "prefix_cached",
                ]

                rng.shuffle(methods)

                measurements = {}

                with torch.inference_mode():

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

                paired_speedup = (
                    standard_ms / cached_ms
                )

                paired_speedups.append(paired_speedup)

                raw_records.append({
                    "batch_size": batch_size,
                    "prefix_length": prefix_length,
                    "suffix_length": suffix_length,
                    "repetition": repetition,
                    "standard_ms": standard_ms,
                    "prefix_cached_ms": cached_ms,
                    "paired_speedup": paired_speedup,
                })

            median_standard = median(standard_times)
            median_cached = median(cached_times)
            median_speedup = median(paired_speedups)

            ci_lower, ci_upper = bootstrap_median_ci(
                paired_speedups,
                seed=SEED + batch_size + prefix_length,
            )

            cache_stats = prefix_model.cache_statistics()

            summary = {
                "batch_size": batch_size,
                "prefix_length": prefix_length,
                "suffix_length": suffix_length,
                "standard_median_ms": median_standard,
                "prefix_cold_ms": cold_ms,
                "prefix_warm_median_ms": median_cached,
                "median_paired_speedup": median_speedup,
                "speedup_ci_lower": ci_lower,
                "speedup_ci_upper": ci_upper,
                "standard_peak_memory_mb": standard_peak_mb,
                "prefix_cold_peak_memory_mb": cold_peak_mb,
                "prefix_warm_peak_memory_mb": cached_peak_mb,
                "cache_hits": cache_stats["hits"],
                "cache_misses": cache_stats["misses"],
                "correctness_passed": True,
            }

            summary_records.append(summary)

            print(
                f"Standard median: {median_standard:.3f} ms"
            )
            print(
                f"Prefix cold: {cold_ms:.3f} ms"
            )
            print(
                f"Prefix warm median: {median_cached:.3f} ms"
            )
            print(
                f"Warm speedup: {median_speedup:.3f}x"
            )
            print(
                f"95% bootstrap CI: "
                f"[{ci_lower:.3f}, {ci_upper:.3f}]"
            )
            print(
                f"Standard peak memory: {standard_peak_mb:.2f} MB"
            )
            print(
                f"Prefix warm peak memory: {cached_peak_mb:.2f} MB"
            )
            print("Correctness: PASS")
            print("Cache:", cache_stats)

            del prefix_model
            torch.cuda.empty_cache()

    raw_path = output_dir / "raw_timings.csv"
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "report.json"

    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(raw_records[0].keys()),
        )
        writer.writeheader()
        writer.writerows(raw_records)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
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
        "total_sequence_length": TOTAL_SEQUENCE_LENGTH,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "timed_repetitions": TIMED_REPETITIONS,
        "all_correctness_passed": all(
            row["correctness_passed"]
            for row in summary_records
        ),
        "configurations": summary_records,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 65)
    print("PREFIX CACHE BENCHMARK COMPLETED")
    print("Configurations:", len(summary_records))
    print("Raw measurements:", len(raw_records))
    print("Correctness: PASS")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
