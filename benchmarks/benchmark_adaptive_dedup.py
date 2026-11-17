
import sys
import csv
import math
import random
import statistics
import time

from datetime import datetime, timezone
from pathlib import Path

# Add project root to Python import path.
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

METHODS = [
    "standard",
    "ffn_deduplicated",
    "adaptive_prompt"
]


# --------------------------------------------------
# Device utilities
# --------------------------------------------------

def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------
# Create controlled workloads
# --------------------------------------------------

def create_prompts(batch_size, unique_count, device):

    base_prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (unique_count, SEQ_LEN),
        device=device,
        dtype=torch.long
    )

    repeats = math.ceil(
        batch_size / unique_count
    )

    prompts = base_prompts.repeat(
        repeats,
        1
    )[:batch_size]

    return prompts


# --------------------------------------------------
# Execute a selected method
# --------------------------------------------------

def execute_method(model, prompts, method):

    if method == "standard":

        return model(
            prompts,
            ffn_mode="standard",
            prompt_mode="standard"
        )

    if method == "ffn_deduplicated":

        return model(
            prompts,
            ffn_mode="deduplicated",
            prompt_mode="standard"
        )

    if method == "adaptive_prompt":

        return model(
            prompts,
            ffn_mode="standard",
            prompt_mode="adaptive"
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# --------------------------------------------------
# Measure execution latency
# --------------------------------------------------

def measure_latency(
    model,
    prompts,
    method,
    device
):

    synchronize(device)

    start = time.perf_counter()

    with torch.inference_mode():

        output = execute_method(
            model,
            prompts,
            method
        )

    synchronize(device)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return elapsed_ms, output


# --------------------------------------------------
# Percentile utility
# --------------------------------------------------

def percentile(values, fraction):

    ordered = sorted(values)

    index = math.ceil(
        fraction * len(ordered)
    ) - 1

    return ordered[max(0, index)]


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

    # ----------------------------------------------
    # Correctness checks
    # ----------------------------------------------

    with torch.inference_mode():

        standard_output = execute_method(
            model,
            prompts,
            "standard"
        )

        ffn_output = execute_method(
            model,
            prompts,
            "ffn_deduplicated"
        )

        adaptive_output = execute_method(
            model,
            prompts,
            "adaptive_prompt"
        )

    ffn_correct = torch.allclose(
        standard_output,
        ffn_output,
        atol=1e-5,
        rtol=1e-5
    )

    adaptive_correct = torch.allclose(
        standard_output,
        adaptive_output,
        atol=1e-5,
        rtol=1e-5
    )

    if not ffn_correct:
        raise RuntimeError(
            "FFN deduplication correctness failed."
        )

    if not adaptive_correct:
        raise RuntimeError(
            "Adaptive prompt correctness failed."
        )

    # ----------------------------------------------
    # Collect FFN deduplication statistics
    # ----------------------------------------------

    model.reset_dedup_statistics()

    with torch.inference_mode():

        execute_method(
            model,
            prompts,
            "ffn_deduplicated"
        )

    ffn_stats = model.dedup_statistics()

    total_inputs = sum(
        item["total_inputs"]
        for item in ffn_stats.values()
    )

    unique_inputs = sum(
        item["unique_inputs"]
        for item in ffn_stats.values()
    )

    ffn_reuse_ratio = (
        1 - unique_inputs / total_inputs
        if total_inputs > 0
        else 0.0
    )

    # ----------------------------------------------
    # Collect adaptive prompt statistics
    # ----------------------------------------------

    model.reset_prompt_dedup_statistics()

    with torch.inference_mode():

        execute_method(
            model,
            prompts,
            "adaptive_prompt"
        )

    prompt_stats = model.prompt_dedup_statistics()

    # ----------------------------------------------
    # Warm up all methods
    # ----------------------------------------------

    rng = random.Random(
        SEED + batch_size * 100 + unique_count
    )

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):

            order = METHODS.copy()
            rng.shuffle(order)

            for method in order:

                execute_method(
                    model,
                    prompts,
                    method
                )

    # ----------------------------------------------
    # Timed measurements
    # ----------------------------------------------

    measurements = {
        method: []
        for method in METHODS
    }

    for _ in range(TIMED_RUNS):

        order = METHODS.copy()
        rng.shuffle(order)

        for method in order:

            elapsed, _ = measure_latency(
                model,
                prompts,
                method,
                device
            )

            measurements[method].append(
                elapsed
            )

    medians = {
        method: statistics.median(
            measurements[method]
        )
        for method in METHODS
    }

    standard_median = medians["standard"]

    ffn_median = medians["ffn_deduplicated"]

    adaptive_median = medians["adaptive_prompt"]

    return {
        "batch_size": batch_size,
        "unique_prompts": unique_count,

        "duplicate_prompt_fraction": (
            1 - unique_count / batch_size
        ),

        "ffn_reuse_ratio": ffn_reuse_ratio,

        "adaptive_duplicate_ratio": (
            prompt_stats["duplicate_ratio"]
        ),

        "adaptive_activations": (
            prompt_stats["dedup_activations"]
        ),

        "adaptive_bypasses": (
            prompt_stats["bypasses"]
        ),

        "standard_median_ms": standard_median,

        "ffn_median_ms": ffn_median,

        "adaptive_median_ms": adaptive_median,

        "ffn_speedup": (
            standard_median / ffn_median
        ),

        "adaptive_speedup": (
            standard_median / adaptive_median
        ),

        "adaptive_vs_ffn_speedup": (
            ffn_median / adaptive_median
        ),

        "standard_p95_ms": percentile(
            measurements["standard"],
            0.95
        ),

        "ffn_p95_ms": percentile(
            measurements["ffn_deduplicated"],
            0.95
        ),

        "adaptive_p95_ms": percentile(
            measurements["adaptive_prompt"],
            0.95
        ),

        "ffn_correctness": ffn_correct,

        "adaptive_correctness": adaptive_correct
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

    print("\n==========================================")
    print("RCT ADAPTIVE DEDUPLICATION BENCHMARK")
    print("==========================================")

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

    # ----------------------------------------------
    # Run all workloads
    # ----------------------------------------------

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

            print("\n------------------------------------------")

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
                "FFN reuse:",
                f"{result['ffn_reuse_ratio']:.2%}"
            )

            print(
                "Adaptive duplicate ratio:",
                f"{result['adaptive_duplicate_ratio']:.2%}"
            )

            print(
                "Standard:",
                f"{result['standard_median_ms']:.4f} ms"
            )

            print(
                "FFN dedup:",
                f"{result['ffn_median_ms']:.4f} ms"
            )

            print(
                "Adaptive prompt:",
                f"{result['adaptive_median_ms']:.4f} ms"
            )

            print(
                "FFN speedup:",
                f"{result['ffn_speedup']:.3f}x"
            )

            print(
                "Adaptive speedup:",
                f"{result['adaptive_speedup']:.3f}x"
            )

            print(
                "Adaptive vs FFN:",
                f"{result['adaptive_vs_ffn_speedup']:.3f}x"
            )

            print(
                "Correctness:",
                "PASS"
                if (
                    result["ffn_correctness"]
                    and result["adaptive_correctness"]
                )
                else "FAIL"
            )

    # ----------------------------------------------
    # Save results
    # ----------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        Path("results")
        / "rct_adaptive_dedup_benchmark"
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

    # ----------------------------------------------
    # Summary
    # ----------------------------------------------

    print("\n==========================================")
    print("THREE-WAY BENCHMARK SUMMARY")
    print("==========================================")

    print(
        f"{'Batch':>5} "
        f"{'Unique':>6} "
        f"{'Dup %':>7} "
        f"{'FFN reuse':>10} "
        f"{'Standard':>11} "
        f"{'FFN':>11} "
        f"{'Adaptive':>11} "
        f"{'Adapt speed':>11}"
    )

    for result in results:

        print(
            f"{result['batch_size']:>5} "
            f"{result['unique_prompts']:>6} "
            f"{result['duplicate_prompt_fraction']:>6.1%} "
            f"{result['ffn_reuse_ratio']:>9.1%} "
            f"{result['standard_median_ms']:>9.3f}ms "
            f"{result['ffn_median_ms']:>9.3f}ms "
            f"{result['adaptive_median_ms']:>9.3f}ms "
            f"{result['adaptive_speedup']:>10.3f}x"
        )

    print("\nConfigurations tested:", total_configs)

    print(
        "FFN correctness checks:",
        sum(
            result["ffn_correctness"]
            for result in results
        ),
        "/",
        len(results)
    )

    print(
        "Adaptive correctness checks:",
        sum(
            result["adaptive_correctness"]
            for result in results
        ),
        "/",
        len(results)
    )

    print("\nResults saved to:")
    print(csv_path)

    print("\n===== BENCHMARK COMPLETE =====")


if __name__ == "__main__":
    main()
