
import csv
import math
import random
import statistics
import time

from datetime import datetime, timezone
from pathlib import Path

import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model.transformer import RCTModel

# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------

BATCH_SIZES = [1, 2, 4, 8, 16]

SEQ_LEN = 64
HIDDEN_SIZE = 128
LAYERS = 4
HEADS = 4
VOCAB_SIZE = 1000

WARMUP_RUNS = 5
TIMED_RUNS = 20

SEED = 42


# --------------------------------------------------
# Device utilities
# --------------------------------------------------

def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------
# Create controlled duplicate workloads
# --------------------------------------------------

def create_prompts(batch_size, unique_count, device):

    base_prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (unique_count, SEQ_LEN),
        device=device,
        dtype=torch.long
    )

    # Repeat unique prompts to construct the batch.
    repeats = math.ceil(batch_size / unique_count)

    prompts = base_prompts.repeat(
        repeats,
        1
    )[:batch_size]

    return prompts


# --------------------------------------------------
# Measure one forward pass
# --------------------------------------------------

def measure_latency(model, prompts, mode, device):

    synchronize(device)

    start = time.perf_counter()

    with torch.inference_mode():

        output = model(
            prompts,
            ffn_mode=mode
        )

    synchronize(device)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return elapsed_ms, output


# --------------------------------------------------
# Benchmark one workload
# --------------------------------------------------

def benchmark_workload(
    model,
    batch_size,
    unique_count,
    device
):

    prompts = create_prompts(
        batch_size,
        unique_count,
        device
    )

    # Check numerical correctness.
    with torch.inference_mode():

        standard_output = model(
            prompts,
            ffn_mode="standard"
        )

        model.reset_dedup_statistics()

        dedup_output = model(
            prompts,
            ffn_mode="deduplicated"
        )

    numerical_match = torch.allclose(
        standard_output,
        dedup_output,
        atol=1e-5,
        rtol=1e-5
    )

    if not numerical_match:

        raise RuntimeError(
            "Numerical correctness check failed: "
            f"batch={batch_size}, unique={unique_count}"
        )

    # Collect actual FFN reuse statistics.
    stats = model.dedup_statistics()

    total_inputs = sum(
        item["total_inputs"]
        for item in stats.values()
    )

    unique_inputs = sum(
        item["unique_inputs"]
        for item in stats.values()
    )

    reuse_ratio = (
        1 - unique_inputs / total_inputs
        if total_inputs > 0
        else 0.0
    )

    # Warm up both execution paths.
    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):

            model(
                prompts,
                ffn_mode="standard"
            )

            model(
                prompts,
                ffn_mode="deduplicated"
            )

    standard_times = []
    dedup_times = []

    # Randomize measurement order to reduce order bias.
    methods = [
        "standard",
        "deduplicated"
    ]

    rng = random.Random(
        SEED + batch_size * 100 + unique_count
    )

    for _ in range(TIMED_RUNS):

        order = methods.copy()
        rng.shuffle(order)

        for method in order:

            elapsed, _ = measure_latency(
                model,
                prompts,
                method,
                device
            )

            if method == "standard":
                standard_times.append(elapsed)

            else:
                dedup_times.append(elapsed)

    standard_median = statistics.median(
        standard_times
    )

    dedup_median = statistics.median(
        dedup_times
    )

    speedup = (
        standard_median / dedup_median
        if dedup_median > 0
        else 0.0
    )

    standard_p95 = sorted(standard_times)[
        math.ceil(0.95 * len(standard_times)) - 1
    ]

    dedup_p95 = sorted(dedup_times)[
        math.ceil(0.95 * len(dedup_times)) - 1
    ]

    return {
        "batch_size": batch_size,
        "unique_prompts": unique_count,
        "duplicate_prompt_fraction": (
            1 - unique_count / batch_size
        ),
        "reuse_ratio": reuse_ratio,
        "standard_median_ms": standard_median,
        "dedup_median_ms": dedup_median,
        "speedup": speedup,
        "standard_p95_ms": standard_p95,
        "dedup_p95_ms": dedup_p95,
        "correctness": numerical_match
    }


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main():

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n======================================")
    print("RCT FFN DEDUPLICATION BENCHMARK")
    print("======================================")

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device)
        )

    print("\nModel configuration:")
    print("Vocabulary:", VOCAB_SIZE)
    print("Hidden size:", HIDDEN_SIZE)
    print("Layers:", LAYERS)
    print("Heads:", HEADS)
    print("Sequence length:", SEQ_LEN)

    print("\nBenchmark configuration:")
    print("Warmup runs:", WARMUP_RUNS)
    print("Timed runs per method:", TIMED_RUNS)

    model = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        layers=LAYERS,
        heads=HEADS,
        max_seq_len=SEQ_LEN
    ).to(device)

    model.eval()

    results = []

    total_configs = 0

    for batch_size in BATCH_SIZES:

        unique_counts = sorted(
            set([
                batch_size,
                max(1, batch_size // 2),
                1
            ]),
            reverse=True
        )

        for unique_count in unique_counts:

            total_configs += 1

            print("\n--------------------------------------")

            print(
                f"Batch={batch_size}, "
                f"Unique prompts={unique_count}"
            )

            result = benchmark_workload(
                model,
                batch_size,
                unique_count,
                device
            )

            results.append(result)

            print(
                "Prompt duplication:",
                f"{result['duplicate_prompt_fraction']:.1%}"
            )

            print(
                "Actual FFN reuse:",
                f"{result['reuse_ratio']:.2%}"
            )

            print(
                "Standard median:",
                f"{result['standard_median_ms']:.4f} ms"
            )

            print(
                "Deduplicated median:",
                f"{result['dedup_median_ms']:.4f} ms"
            )

            print(
                "Speedup:",
                f"{result['speedup']:.3f}x"
            )

            print(
                "Numerical correctness:",
                "PASS" if result["correctness"] else "FAIL"
            )

    # Save results.
    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        Path("results")
        / "rct_ffn_dedup_benchmark"
        / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = output_dir / "benchmark_results.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    # Final summary.
    print("\n======================================")
    print("BENCHMARK SUMMARY")
    print("======================================")

    print(
        f"{'Batch':>6} "
        f"{'Unique':>7} "
        f"{'Duplication':>12} "
        f"{'FFN Reuse':>11} "
        f"{'Standard':>12} "
        f"{'Dedup':>12} "
        f"{'Speedup':>9}"
    )

    for result in results:

        print(
            f"{result['batch_size']:>6} "
            f"{result['unique_prompts']:>7} "
            f"{result['duplicate_prompt_fraction']:>11.1%} "
            f"{result['reuse_ratio']:>10.1%} "
            f"{result['standard_median_ms']:>10.3f}ms "
            f"{result['dedup_median_ms']:>10.3f}ms "
            f"{result['speedup']:>8.3f}x"
        )

    print("\nConfigurations tested:", total_configs)

    print(
        "Correctness checks:",
        sum(r["correctness"] for r in results),
        "/",
        len(results)
    )

    print("\nResults saved to:")
    print(csv_path)

    print("\n===== BENCHMARK COMPLETE =====")


if __name__ == "__main__":
    main()
