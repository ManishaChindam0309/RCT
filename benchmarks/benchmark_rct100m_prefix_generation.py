
import sys
import time
import random
import statistics
import csv
import gc
from pathlib import Path
from datetime import datetime, timezone

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from inference import load_rct_inference


CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

BATCH_SIZES = [2, 4, 8]
PREFIX_LENGTHS = [8, 16, 24]

PROMPT_LEN = 32
MAX_NEW_TOKENS = 8

WARMUP_RUNS = 3
BENCHMARK_RUNS = 30
BOOTSTRAP_SAMPLES = 10000

PREFIX_CACHE_CAPACITY = 16
DEDUP_THRESHOLD = 0.75

SEED = 20261106


def synchronize(device):
    torch.cuda.synchronize(device)


def create_prompt(inference, batch_size, prefix_len, device):

    vocab_size = inference.model.config.vocab_size

    shared_prefix = torch.randint(
        0,
        vocab_size,
        (1, prefix_len),
        device=device,
    )

    suffixes = torch.randint(
        0,
        vocab_size,
        (batch_size, PROMPT_LEN - prefix_len),
        device=device,
    )

    return torch.cat(
        [
            shared_prefix.expand(batch_size, -1),
            suffixes,
        ],
        dim=1,
    )


def build_methods(inference, prompt, prefix_len):

    return {
        "standard": lambda: inference.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            mode="standard",
        ),

        "kv": lambda: inference.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            mode="kv",
        ),

        "prefix_manual": lambda: inference.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            mode="prefix",
            prefix_len=prefix_len,
            fallback=False,
        ),

        "prefix_auto": lambda: inference.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            mode="prefix",
            fallback=False,
        ),
    }


def measure_latency(function, device):

    synchronize(device)

    baseline_allocated = torch.cuda.memory_allocated(device)

    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    output = function()

    synchronize(device)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    additional_peak_mb = max(
        0,
        peak_allocated - baseline_allocated,
    ) / (1024 ** 2)

    return {
        "latency_ms": elapsed_ms,
        "additional_peak_memory_mb": additional_peak_mb,
        "peak_allocated_memory_mb": (
            peak_allocated / (1024 ** 2)
        ),
        "peak_reserved_memory_mb": (
            peak_reserved / (1024 ** 2)
        ),
        "output": output,
    }


def measure_cold_prefix(
    method,
    inference,
    device,
    expected_output,
):

    # Remove previous entries so each method starts cold.
    inference.prefix_model.prefix_cache.clear_cache()

    gc.collect()
    synchronize(device)
    torch.cuda.empty_cache()

    baseline_allocated = torch.cuda.memory_allocated(device)

    result = measure_latency(
        method,
        device,
    )

    if not torch.equal(
        expected_output,
        result["output"],
    ):
        raise AssertionError(
            "Cold prefix generation output mismatch."
        )

    cold_latency = result["latency_ms"]
    cold_peak = result["additional_peak_memory_mb"]

    del result["output"]
    del result

    gc.collect()
    synchronize(device)

    # Measure memory that remains allocated by the persistent
    # prefix cache after the generated output has been released.
    persistent_allocated = torch.cuda.memory_allocated(device)

    persistent_cache_mb = max(
        0,
        persistent_allocated - baseline_allocated,
    ) / (1024 ** 2)

    return {
        "cold_latency_ms": cold_latency,
        "cold_additional_peak_memory_mb": cold_peak,
        "persistent_cache_memory_mb": persistent_cache_mb,
    }


def bootstrap_speedup_ci(
    kv_timings,
    prefix_timings,
    samples=BOOTSTRAP_SAMPLES,
    seed=SEED,
):

    if len(kv_timings) != len(prefix_timings):
        raise ValueError(
            "Paired timing arrays must have equal lengths."
        )

    ratios = [
        kv / prefix
        for kv, prefix in zip(
            kv_timings,
            prefix_timings,
        )
    ]

    rng = random.Random(seed)

    bootstrap_medians = []

    for _ in range(samples):

        selected = [
            rng.choice(ratios)
            for _ in range(len(ratios))
        ]

        bootstrap_medians.append(
            statistics.median(selected)
        )

    bootstrap_medians.sort()

    lower_index = int(0.025 * samples)
    upper_index = int(0.975 * samples)

    return (
        bootstrap_medians[lower_index],
        bootstrap_medians[min(upper_index, samples - 1)],
    )


def validate_outputs(
    methods,
    inference,
    prompt,
    expected_prefix,
):

    detected = inference.detect_shared_prefix(prompt)

    if detected != expected_prefix:
        raise AssertionError(
            f"Expected common prefix {expected_prefix}, "
            f"detected {detected}."
        )

    reference = methods["standard"]()

    for name in (
        "kv",
        "prefix_manual",
        "prefix_auto",
    ):

        output = methods[name]()

        if not torch.equal(reference, output):
            raise AssertionError(
                f"Exact output mismatch: {name}"
            )

        del output

    return reference


def benchmark_workload(
    inference,
    batch_size,
    prefix_len,
    device,
    workload_index,
    total_workloads,
):

    prompt = create_prompt(
        inference,
        batch_size,
        prefix_len,
        device,
    )

    methods = build_methods(
        inference,
        prompt,
        prefix_len,
    )

    print("\n" + "=" * 78)
    print(
        f"WORKLOAD {workload_index}/{total_workloads} "
        f"| Batch={batch_size} "
        f"| Prefix={prefix_len}"
    )
    print("=" * 78)

    # --------------------------------------------------
    # Correctness validation
    # --------------------------------------------------

    print("Validating exact output agreement...")

    reference = validate_outputs(
        methods,
        inference,
        prompt,
        prefix_len,
    )

    print("All four methods: exact match PASS")

    # --------------------------------------------------
    # Independent cold-cache measurements
    # --------------------------------------------------

    print("\nIndependent cold-cache measurements:")

    cold_results = {}

    for name in (
        "prefix_manual",
        "prefix_auto",
    ):

        cold = measure_cold_prefix(
            methods[name],
            inference,
            device,
            reference,
        )

        cold_results[name] = cold

        print(
            f"{name}: "
            f"{cold['cold_latency_ms']:.3f} ms "
            f"| Persistent cache: "
            f"{cold['persistent_cache_memory_mb']:.2f} MB "
            f"| Additional peak: "
            f"{cold['cold_additional_peak_memory_mb']:.2f} MB"
        )

    # --------------------------------------------------
    # Warm-up
    # --------------------------------------------------

    print("\nRunning warm-up iterations...")

    for _ in range(WARMUP_RUNS):

        for name in methods:

            output = methods[name]()

            if not torch.equal(reference, output):
                raise AssertionError(
                    f"Warm-up output mismatch: {name}"
                )

            del output

    # --------------------------------------------------
    # Paired randomized timing
    # --------------------------------------------------

    timings = {
        name: []
        for name in methods
    }

    memory_results = {
        name: []
        for name in methods
    }

    peak_allocated_results = {
        name: []
        for name in methods
    }

    peak_reserved_results = {
        name: []
        for name in methods
    }

    print("\nRunning paired randomized benchmark...")

    for run_index in range(BENCHMARK_RUNS):

        order = list(methods.keys())
        random.shuffle(order)

        for name in order:

            result = measure_latency(
                methods[name],
                device,
            )

            if not torch.equal(
                reference,
                result["output"],
            ):
                raise AssertionError(
                    f"Output mismatch in {name}, "
                    f"run {run_index + 1}."
                )

            timings[name].append(
                result["latency_ms"]
            )

            memory_results[name].append(
                result["additional_peak_memory_mb"]
            )

            peak_allocated_results[name].append(
                result["peak_allocated_memory_mb"]
            )

            peak_reserved_results[name].append(
                result["peak_reserved_memory_mb"]
            )

            del result

        print(
            f"Run {run_index + 1:02d}/{BENCHMARK_RUNS} complete",
            flush=True,
        )

    # --------------------------------------------------
    # Summary statistics
    # --------------------------------------------------

    medians = {
        name: statistics.median(values)
        for name, values in timings.items()
    }

    memory_medians = {
        name: statistics.median(values)
        for name, values in memory_results.items()
    }

    print("\n" + "-" * 78)
    print("WARM-CACHE RESULTS")
    print("-" * 78)

    print(
        f"{'Method':<18}"
        f"{'Median ms':>14}"
        f"{'P10 ms':>12}"
        f"{'P90 ms':>12}"
        f"{'Vs KV':>12}"
        f"{'Extra MB':>12}"
    )

    for name in methods:

        sorted_times = sorted(timings[name])

        p10 = sorted_times[
            int(0.10 * (len(sorted_times) - 1))
        ]

        p90 = sorted_times[
            int(0.90 * (len(sorted_times) - 1))
        ]

        speedup_vs_kv = (
            medians["kv"] / medians[name]
        )

        print(
            f"{name:<18}"
            f"{medians[name]:>14.3f}"
            f"{p10:>12.3f}"
            f"{p90:>12.3f}"
            f"{speedup_vs_kv:>11.3f}x"
            f"{memory_medians[name]:>12.2f}"
        )

    # --------------------------------------------------
    # Paired speedup confidence intervals
    # --------------------------------------------------

    print("\nPaired speedup confidence intervals:")

    confidence_intervals = {}

    for name in (
        "prefix_manual",
        "prefix_auto",
    ):

        ci_low, ci_high = bootstrap_speedup_ci(
            timings["kv"],
            timings[name],
        )

        speedup = (
            medians["kv"] / medians[name]
        )

        confidence_intervals[name] = (
            ci_low,
            ci_high,
        )

        print(
            f"{name}: {speedup:.3f}x "
            f"| 95% CI [{ci_low:.3f}, {ci_high:.3f}]"
        )

    # --------------------------------------------------
    # CSV rows
    # --------------------------------------------------

    rows = []

    for name in methods:

        ci = confidence_intervals.get(
            name,
            ("", ""),
        )

        rows.append({
            "batch_size": batch_size,
            "prompt_length": PROMPT_LEN,
            "prefix_length": prefix_len,
            "new_tokens": MAX_NEW_TOKENS,
            "method": name,
            "timed_runs": BENCHMARK_RUNS,
            "median_latency_ms": medians[name],
            "minimum_latency_ms": min(timings[name]),
            "maximum_latency_ms": max(timings[name]),
            "mean_latency_ms": statistics.mean(timings[name]),
            "stdev_latency_ms": (
                statistics.stdev(timings[name])
                if len(timings[name]) > 1
                else 0.0
            ),
            "speedup_vs_kv": (
                medians["kv"] / medians[name]
            ),
            "paired_speedup_ci95_lower": ci[0],
            "paired_speedup_ci95_upper": ci[1],
            "median_additional_peak_memory_mb": (
                memory_medians[name]
            ),
            "median_peak_allocated_memory_mb": (
                statistics.median(
                    peak_allocated_results[name]
                )
            ),
            "median_peak_reserved_memory_mb": (
                statistics.median(
                    peak_reserved_results[name]
                )
            ),
            "cold_latency_ms": (
                cold_results[name]["cold_latency_ms"]
                if name in cold_results
                else ""
            ),
            "cold_persistent_cache_memory_mb": (
                cold_results[name]["persistent_cache_memory_mb"]
                if name in cold_results
                else ""
            ),
            "cold_additional_peak_memory_mb": (
                cold_results[name]["cold_additional_peak_memory_mb"]
                if name in cold_results
                else ""
            ),
        })

    return rows


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    random.seed(SEED)

    print("=" * 78)
    print("RCT100M PREFIX GENERATION CONFIRMATION BENCHMARK")
    print("=" * 78)

    print("Checkpoint:", CHECKPOINT)
    print("GPU:", torch.cuda.get_device_name(device))
    print("CUDA:", torch.version.cuda)
    print("PyTorch:", torch.__version__)

    print("Batch sizes:", BATCH_SIZES)
    print("Prefix lengths:", PREFIX_LENGTHS)
    print("Prompt length:", PROMPT_LEN)
    print("New tokens:", MAX_NEW_TOKENS)
    print("Warm-up runs:", WARMUP_RUNS)
    print("Timed runs:", BENCHMARK_RUNS)
    print("Bootstrap samples:", BOOTSTRAP_SAMPLES)

    inference = load_rct_inference(
        CHECKPOINT,
        device=device,
        prefix_cache_capacity=PREFIX_CACHE_CAPACITY,
        dedup_threshold=DEDUP_THRESHOLD,
    )

    inference.model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in inference.model.parameters()
    )

    print("Model parameters:", parameter_count)

    total_workloads = (
        len(BATCH_SIZES) * len(PREFIX_LENGTHS)
    )

    all_rows = []

    workload_index = 0

    for batch_size in BATCH_SIZES:

        for prefix_len in PREFIX_LENGTHS:

            workload_index += 1

            rows = benchmark_workload(
                inference=inference,
                batch_size=batch_size,
                prefix_len=prefix_len,
                device=device,
                workload_index=workload_index,
                total_workloads=total_workloads,
            )

            all_rows.extend(rows)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct100m_prefix_confirmation"
        / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "confirmation_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(all_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 78)
    print("CONFIRMATION BENCHMARK COMPLETE")
    print("=" * 78)

    print("Workloads:", total_workloads)
    print("Methods per workload: 4")
    print("Timed repetitions per method:", BENCHMARK_RUNS)
    print("Exact output validation: PASS")
    print("Independent cold-cache measurements: COMPLETE")
    print("Paired speedup confidence intervals: COMPLETE")

    print("\nSaved results:")
    print(csv_path)

    print(
        "\nFinal prefix cache statistics:",
        inference.prefix_model.prefix_cache.cache_statistics(),
    )

    print("\nRCT100M PREFIX CONFIRMATION: COMPLETE")


if __name__ == "__main__":
    main()
