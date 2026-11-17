
import sys
import csv
import random
import statistics
import time

from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model.transformer import RCTModel


# --------------------------------------------------
# Experiment configuration
# --------------------------------------------------

CONFIGS = [
    (8, 1),    # 87.5% duplicate prompts
    (16, 8),   # 50% duplicate prompts
    (16, 1),   # 93.75% duplicate prompts
]

SEQ_LEN = 64
HIDDEN_SIZE = 128
LAYERS = 4
HEADS = 4
VOCAB_SIZE = 1000

INDEPENDENT_GROUPS = 30
WARMUP_RUNS = 2
REPETITIONS = 5
BOOTSTRAP_SAMPLES = 10000

SEED = 20261106


# --------------------------------------------------
# GPU utilities
# --------------------------------------------------

def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------
# Prompt generation
# --------------------------------------------------

def create_prompts(batch_size, unique_count, device):

    unique_prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (unique_count, SEQ_LEN),
        device=device,
        dtype=torch.long
    )

    repeats = (
        batch_size + unique_count - 1
    ) // unique_count

    prompts = unique_prompts.repeat(
        repeats,
        1
    )[:batch_size]

    # Randomize positions while preserving duplicate counts.
    permutation = torch.randperm(
        batch_size,
        device=device
    )

    return prompts[permutation].contiguous()


# --------------------------------------------------
# Model execution
# --------------------------------------------------

def execute(model, prompts, method):

    if method == "standard":

        return model(
            prompts,
            ffn_mode="standard",
            prompt_mode="standard"
        )

    if method == "adaptive":

        return model(
            prompts,
            ffn_mode="standard",
            prompt_mode="adaptive"
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# --------------------------------------------------
# Latency measurement
# --------------------------------------------------

def measure(model, prompts, method, device):

    synchronize(device)

    start = time.perf_counter()

    with torch.inference_mode():

        output = execute(
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
# Bootstrap confidence interval
# --------------------------------------------------

def bootstrap_ci(
    cluster_speedups,
    samples,
    seed
):

    rng = random.Random(seed)

    count = len(cluster_speedups)

    bootstrap_medians = []

    for _ in range(samples):

        sampled = [
            cluster_speedups[
                rng.randrange(count)
            ]
            for _ in range(count)
        ]

        bootstrap_medians.append(
            statistics.median(sampled)
        )

    bootstrap_medians.sort()

    lower_index = int(
        0.025 * samples
    )

    upper_index = int(
        0.975 * samples
    )

    return (
        bootstrap_medians[lower_index],
        bootstrap_medians[upper_index]
    )


# --------------------------------------------------
# One independent prompt group
# --------------------------------------------------

def run_cluster(
    model,
    batch_size,
    unique_count,
    device,
    cluster_id,
    rng
):

    prompts = create_prompts(
        batch_size,
        unique_count,
        device
    )

    # Correctness validation.
    with torch.inference_mode():

        standard_output = execute(
            model,
            prompts,
            "standard"
        )

        model.reset_prompt_dedup_statistics()

        adaptive_output = execute(
            model,
            prompts,
            "adaptive"
        )

    max_difference = (
        standard_output - adaptive_output
    ).abs().max().item()

    correctness = torch.allclose(
        standard_output,
        adaptive_output,
        atol=1e-5,
        rtol=1e-5
    )

    if not correctness:

        raise RuntimeError(
            f"Correctness failure in cluster {cluster_id}"
        )

    # Validate adaptive deduplication statistics.
    stats = model.prompt_dedup_statistics()

    if stats["original_prompts"] != batch_size:
        raise RuntimeError(
            "Unexpected original prompt count."
        )

    if stats["unique_prompts"] != unique_count:
        raise RuntimeError(
            "Unexpected unique prompt count."
        )

    # Warm up both methods.
    with torch.inference_mode():

        for _ in range(WARMUP_RUNS):

            order = [
                "standard",
                "adaptive"
            ]

            rng.shuffle(order)

            for method in order:

                execute(
                    model,
                    prompts,
                    method
                )

    # Timed measurements.
    timings = {
        "standard": [],
        "adaptive": []
    }

    for _ in range(REPETITIONS):

        order = [
            "standard",
            "adaptive"
        ]

        rng.shuffle(order)

        for method in order:

            elapsed, _ = measure(
                model,
                prompts,
                method,
                device
            )

            timings[method].append(
                elapsed
            )

    standard_median = statistics.median(
        timings["standard"]
    )

    adaptive_median = statistics.median(
        timings["adaptive"]
    )

    paired_speedup = (
        standard_median / adaptive_median
    )

    return {
        "cluster_id": cluster_id,
        "batch_size": batch_size,
        "unique_prompts": unique_count,
        "duplicate_fraction": (
            1 - unique_count / batch_size
        ),
        "standard_median_ms": standard_median,
        "adaptive_median_ms": adaptive_median,
        "paired_speedup": paired_speedup,
        "max_abs_difference": max_difference,
        "correctness": correctness
    }


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n==========================================")
    print("RCT ADAPTIVE DEDUPLICATION CONFIRMATION")
    print("==========================================")

    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(device)
        )

    print("\nModel configuration:")
    print("Hidden size:", HIDDEN_SIZE)
    print("Layers:", LAYERS)
    print("Heads:", HEADS)
    print("Sequence length:", SEQ_LEN)

    print("\nExperiment configuration:")
    print("Independent groups:", INDEPENDENT_GROUPS)
    print("Warmups per method:", WARMUP_RUNS)
    print("Timed repetitions:", REPETITIONS)
    print("Bootstrap samples:", BOOTSTRAP_SAMPLES)

    model = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        layers=LAYERS,
        heads=HEADS,
        max_seq_len=SEQ_LEN
    ).to(device)

    model.eval()

    all_cluster_results = []
    summary_results = []

    for config_index, (batch_size, unique_count) in enumerate(CONFIGS):

        print("\n==========================================")
        print(
            f"BATCH {batch_size} | "
            f"UNIQUE PROMPTS {unique_count}"
        )
        print("==========================================")

        rng = random.Random(
            SEED + config_index * 1000
        )

        cluster_results = []

        for cluster_id in range(
            INDEPENDENT_GROUPS
        ):

            result = run_cluster(
                model=model,
                batch_size=batch_size,
                unique_count=unique_count,
                device=device,
                cluster_id=cluster_id,
                rng=rng
            )

            cluster_results.append(result)
            all_cluster_results.append(result)

            if (cluster_id + 1) % 5 == 0:

                print(
                    f"Completed "
                    f"{cluster_id + 1}/"
                    f"{INDEPENDENT_GROUPS} groups"
                )

        # Cluster-level paired measurements.
        speedups = [
            result["paired_speedup"]
            for result in cluster_results
        ]

        standard_times = [
            result["standard_median_ms"]
            for result in cluster_results
        ]

        adaptive_times = [
            result["adaptive_median_ms"]
            for result in cluster_results
        ]

        observed_speedup = statistics.median(
            speedups
        )

        ci_lower, ci_upper = bootstrap_ci(
            speedups,
            BOOTSTRAP_SAMPLES,
            SEED + config_index
        )

        standard_median = statistics.median(
            standard_times
        )

        adaptive_median = statistics.median(
            adaptive_times
        )

        latency_reduction = (
            1 - adaptive_median / standard_median
        ) * 100

        summary = {
            "batch_size": batch_size,
            "unique_prompts": unique_count,
            "duplicate_fraction": (
                1 - unique_count / batch_size
            ),
            "independent_groups": INDEPENDENT_GROUPS,
            "standard_median_ms": standard_median,
            "adaptive_median_ms": adaptive_median,
            "paired_speedup": observed_speedup,
            "ci95_lower": ci_lower,
            "ci95_upper": ci_upper,
            "latency_reduction_percent": latency_reduction,
            "correctness_checks": sum(
                result["correctness"]
                for result in cluster_results
            )
        }

        summary_results.append(summary)

        print("\nConfiguration results:")

        print(
            "Standard median:",
            f"{standard_median:.4f} ms"
        )

        print(
            "Adaptive median:",
            f"{adaptive_median:.4f} ms"
        )

        print(
            "Paired cluster speedup:",
            f"{observed_speedup:.4f}x"
        )

        print(
            "95% bootstrap CI:",
            f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        )

        print(
            "Latency reduction:",
            f"{latency_reduction:.2f}%"
        )

        print(
            "Correctness:",
            summary["correctness_checks"],
            "/",
            INDEPENDENT_GROUPS
        )

        if ci_lower > 1.0:

            print(
                "Interpretation: CI excludes 1.0 "
                "(adaptive advantage supported)."
            )

        elif ci_upper < 1.0:

            print(
                "Interpretation: CI excludes 1.0 "
                "(standard advantage supported)."
            )

        else:

            print(
                "Interpretation: CI includes 1.0 "
                "(inconclusive)."
            )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        Path("results")
        / "rct_adaptive_dedup_confirmation"
        / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cluster_path = (
        output_dir / "cluster_measurements.csv"
    )

    summary_path = (
        output_dir / "confirmation_summary.csv"
    )

    with open(
        cluster_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                all_cluster_results[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(all_cluster_results)

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_results[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(summary_results)

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n==========================================")
    print("CONFIRMATION SUMMARY")
    print("==========================================")

    print(
        f"{'Batch':>6} "
        f"{'Unique':>7} "
        f"{'Standard':>12} "
        f"{'Adaptive':>12} "
        f"{'Speedup':>10} "
        f"{'95% CI':>23}"
    )

    for result in summary_results:

        ci_text = (
            f"[{result['ci95_lower']:.3f}, "
            f"{result['ci95_upper']:.3f}]"
        )

        print(
            f"{result['batch_size']:>6} "
            f"{result['unique_prompts']:>7} "
            f"{result['standard_median_ms']:>10.3f}ms "
            f"{result['adaptive_median_ms']:>10.3f}ms "
            f"{result['paired_speedup']:>9.3f}x "
            f"{ci_text:>23}"
        )

    print("\nTotal independent groups:")
    print(len(all_cluster_results))

    print("\nTotal correctness checks:")
    print(
        sum(
            result["correctness"]
            for result in all_cluster_results
        )
    )

    print("\nResults saved to:")
    print(summary_path)

    print(cluster_path)

    print("\n===== CONFIRMATION COMPLETE =====")


if __name__ == "__main__":
    main()
