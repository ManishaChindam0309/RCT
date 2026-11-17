
import csv
import gc
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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

PROMPT_LENGTH = 32
NEW_TOKENS = 8

BATCH_SIZES = [2, 4, 8]
PREFIX_LENGTHS = [8, 16, 24]

WARMUP_RUNS = 3
TIMED_RUNS = 30
BOOTSTRAP_SAMPLES = 10000

SEED = 20261106


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_prompts(
    batch_size,
    prefix_len,
    prompt_length,
    vocab_size,
    device,
):
    shared_prefix = torch.randint(
        0,
        vocab_size,
        (1, prefix_len),
        device=device,
    )

    suffix = torch.randint(
        0,
        vocab_size,
        (
            batch_size,
            prompt_length - prefix_len,
        ),
        device=device,
    )

    return torch.cat(
        [
            shared_prefix.expand(batch_size, -1),
            suffix,
        ],
        dim=1,
    )


def timed_generate(
    inference,
    prompt,
    mode,
):
    synchronize()
    start = time.perf_counter()

    output = inference.generate(
        prompt,
        max_new_tokens=NEW_TOKENS,
        mode=mode,
    )

    synchronize()
    elapsed = time.perf_counter() - start

    return output, elapsed * 1000.0


def bootstrap_median_ci(
    values,
    samples=BOOTSTRAP_SAMPLES,
):
    if not values:
        raise ValueError("Cannot bootstrap an empty dataset.")

    rng = random.Random(SEED)
    estimates = []

    count = len(values)

    for _ in range(samples):
        selected = [
            values[rng.randrange(count)]
            for _ in range(count)
        ]

        estimates.append(
            statistics.median(selected)
        )

    estimates.sort()

    lower_index = int(0.025 * samples)
    upper_index = int(0.975 * samples)

    return (
        estimates[lower_index],
        estimates[min(upper_index, samples - 1)],
    )


def run_workload(
    inference,
    batch_size,
    prefix_len,
    device,
    raw_writer,
):
    print()
    print("-" * 78)
    print(
        f"WORKLOAD: batch={batch_size}, "
        f"prefix={prefix_len}, "
        f"prompt={PROMPT_LENGTH}, "
        f"new_tokens={NEW_TOKENS}"
    )
    print("-" * 78)

    prompt = create_prompts(
        batch_size=batch_size,
        prefix_len=prefix_len,
        prompt_length=PROMPT_LENGTH,
        vocab_size=inference.model.config.vocab_size,
        device=device,
    )

    detected = inference.detect_shared_prefix(prompt)

    if detected != prefix_len:
        raise AssertionError(
            f"Expected prefix {prefix_len}, detected {detected}."
        )

    # Confirm correctness before collecting timing data.
    expected, _ = timed_generate(
        inference,
        prompt,
        "kv",
    )

    actual, _ = timed_generate(
        inference,
        prompt,
        "auto",
    )

    if not torch.equal(expected, actual):
        raise AssertionError(
            f"Output mismatch: batch={batch_size}, "
            f"prefix={prefix_len}"
        )

    print("Correctness check: PASS")

    # Warm up both methods.
    for warmup in range(WARMUP_RUNS):
        methods = ["kv", "auto"]
        random.shuffle(methods)

        for method in methods:
            timed_generate(
                inference,
                prompt,
                method,
            )

    synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    paired_speedups = []
    kv_times = []
    auto_times = []

    for repetition in range(TIMED_RUNS):
        methods = ["kv", "auto"]
        random.shuffle(methods)

        results = {}

        for method in methods:
            _, elapsed_ms = timed_generate(
                inference,
                prompt,
                method,
            )

            results[method] = elapsed_ms

        kv_ms = results["kv"]
        auto_ms = results["auto"]

        speedup = kv_ms / auto_ms

        kv_times.append(kv_ms)
        auto_times.append(auto_ms)
        paired_speedups.append(speedup)

        raw_writer.writerow(
            {
                "batch_size": batch_size,
                "prefix_length": prefix_len,
                "prompt_length": PROMPT_LENGTH,
                "new_tokens": NEW_TOKENS,
                "repetition": repetition + 1,
                "kv_ms": f"{kv_ms:.6f}",
                "auto_ms": f"{auto_ms:.6f}",
                "paired_speedup": f"{speedup:.6f}",
            }
        )

    kv_median = statistics.median(kv_times)
    auto_median = statistics.median(auto_times)

    median_speedup = statistics.median(
        paired_speedups
    )

    ci_lower, ci_upper = bootstrap_median_ci(
        paired_speedups
    )

    latency_reduction = (
        (kv_median - auto_median)
        / kv_median
        * 100.0
    )

    if ci_lower > 1.0:
        conclusion = "AUTO FASTER"
    elif ci_upper < 1.0:
        conclusion = "KV FASTER"
    else:
        conclusion = "INCONCLUSIVE"

    result = {
        "batch_size": batch_size,
        "prefix_length": prefix_len,
        "prompt_length": PROMPT_LENGTH,
        "new_tokens": NEW_TOKENS,
        "timed_runs": TIMED_RUNS,
        "kv_median_ms": kv_median,
        "auto_median_ms": auto_median,
        "paired_median_speedup": median_speedup,
        "speedup_ci95_lower": ci_lower,
        "speedup_ci95_upper": ci_upper,
        "latency_reduction_percent": latency_reduction,
        "conclusion": conclusion,
    }

    print(f"KV median:             {kv_median:.3f} ms")
    print(f"Auto median:           {auto_median:.3f} ms")
    print(f"Paired median speedup:  {median_speedup:.4f}x")
    print(
        f"95% CI:                "
        f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    )
    print(f"Latency reduction:     {latency_reduction:.2f}%")
    print(f"Conclusion:            {conclusion}")

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark."
        )

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")

    print("=" * 78)
    print("RCT ADAPTIVE PREFIX GATE CONFIRMATION BENCHMARK")
    print("=" * 78)

    print("Checkpoint:", CHECKPOINT)
    print("GPU:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Prompt length:", PROMPT_LENGTH)
    print("New tokens:", NEW_TOKENS)
    print("Warmup runs:", WARMUP_RUNS)
    print("Timed runs:", TIMED_RUNS)
    print("Bootstrap samples:", BOOTSTRAP_SAMPLES)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    print("\nLoading RCT100M...")

    inference = load_rct_inference(
        CHECKPOINT,
        device=device,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in inference.model.parameters()
    )

    print("Model parameters:", parameter_count)
    print(
        "Adaptive dedup threshold:",
        inference.dedup_threshold,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct_adaptive_prefix_gate_confirmation"
        / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = output_dir / "raw_measurements.csv"
    summary_path = output_dir / "summary.csv"

    raw_fields = [
        "batch_size",
        "prefix_length",
        "prompt_length",
        "new_tokens",
        "repetition",
        "kv_ms",
        "auto_ms",
        "paired_speedup",
    ]

    summary_fields = [
        "batch_size",
        "prefix_length",
        "prompt_length",
        "new_tokens",
        "timed_runs",
        "kv_median_ms",
        "auto_median_ms",
        "paired_median_speedup",
        "speedup_ci95_lower",
        "speedup_ci95_upper",
        "latency_reduction_percent",
        "conclusion",
    ]

    results = []

    total_workloads = (
        len(BATCH_SIZES) * len(PREFIX_LENGTHS)
    )

    workload_number = 0

    with raw_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as raw_file:

        raw_writer = csv.DictWriter(
            raw_file,
            fieldnames=raw_fields,
        )

        raw_writer.writeheader()

        for batch_size in BATCH_SIZES:
            for prefix_len in PREFIX_LENGTHS:

                workload_number += 1

                print(
                    f"\nWorkload {workload_number}/"
                    f"{total_workloads}"
                )

                result = run_workload(
                    inference=inference,
                    batch_size=batch_size,
                    prefix_len=prefix_len,
                    device=device,
                    raw_writer=raw_writer,
                )

                results.append(result)

                raw_file.flush()

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as summary_file:

        summary_writer = csv.DictWriter(
            summary_file,
            fieldnames=summary_fields,
        )

        summary_writer.writeheader()

        for result in results:
            summary_writer.writerow(result)

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    print(
        f"{'Batch':>6} "
        f"{'Prefix':>7} "
        f"{'KV ms':>11} "
        f"{'Auto ms':>11} "
        f"{'Speedup':>10} "
        f"{'95% CI':>23} "
        f"{'Conclusion':>14}"
    )

    for result in results:
        ci_text = (
            f"[{result['speedup_ci95_lower']:.3f}, "
            f"{result['speedup_ci95_upper']:.3f}]"
        )

        print(
            f"{result['batch_size']:6d} "
            f"{result['prefix_length']:7d} "
            f"{result['kv_median_ms']:11.3f} "
            f"{result['auto_median_ms']:11.3f} "
            f"{result['paired_median_speedup']:9.3f}x "
            f"{ci_text:>23} "
            f"{result['conclusion']:>14}"
        )

    auto_faster = sum(
        result["conclusion"] == "AUTO FASTER"
        for result in results
    )

    kv_faster = sum(
        result["conclusion"] == "KV FASTER"
        for result in results
    )

    inconclusive = sum(
        result["conclusion"] == "INCONCLUSIVE"
        for result in results
    )

    print("\nWorkload conclusions:")
    print("Auto reliably faster:", auto_faster)
    print("KV reliably faster:", kv_faster)
    print("Inconclusive:", inconclusive)

    print("\nRaw measurements:")
    print(raw_path)

    print("\nSummary:")
    print(summary_path)

    print("\n" + "=" * 78)
    print("RCT ADAPTIVE PREFIX GATE BENCHMARK: COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
