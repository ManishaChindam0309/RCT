
import sys
import csv
import time
import random
import statistics
from pathlib import Path
from datetime import datetime, timezone

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_prefix_generation import RCT100MPrefixGeneration


CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

CONFIGURATIONS = [
    {"batch_size": 16, "prefix_len": 16},
    {"batch_size": 16, "prefix_len": 24},
]

PROMPT_LEN = 32
MAX_NEW_TOKENS = 8

NUM_CLUSTERS = 30
REPETITIONS = 5
WARMUP_RUNS = 2
BOOTSTRAP_SAMPLES = 10000

SEED = 20261108


def load_model(device):

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    config = RCT100MConfig(**checkpoint["model_config"])

    model = RCT100M(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def standard_greedy(model, prompt):

    generated = prompt.clone()

    for _ in range(MAX_NEW_TOKENS):

        logits = model(generated)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated


@torch.inference_mode()
def kv_greedy(kv_model, prompt):

    cache = kv_model.create_cache(prompt.size(0))

    logits = kv_model.decode_step(prompt, cache)

    generated = prompt.clone()

    for step in range(MAX_NEW_TOKENS):

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        if step + 1 < MAX_NEW_TOKENS:
            logits = kv_model.decode_step(
                next_token,
                cache,
            )

    return generated


def synchronize(device):
    torch.cuda.synchronize(device)


def measure(function, device):

    synchronize(device)

    start = time.perf_counter()

    output = function()

    synchronize(device)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return elapsed_ms, output


def paired_bootstrap_ratio(
    kv_values,
    prefix_values,
    seed,
):

    rng = random.Random(seed)

    count = len(kv_values)

    observed = (
        statistics.median(kv_values)
        / statistics.median(prefix_values)
    )

    samples = []

    for _ in range(BOOTSTRAP_SAMPLES):

        indices = [
            rng.randrange(count)
            for _ in range(count)
        ]

        sampled_kv = [
            kv_values[i] for i in indices
        ]

        sampled_prefix = [
            prefix_values[i] for i in indices
        ]

        denominator = statistics.median(
            sampled_prefix
        )

        if denominator > 0:

            samples.append(
                statistics.median(sampled_kv)
                / denominator
            )

    samples.sort()

    # Bonferroni-adjusted 97.5% interval:
    # two configurations, family-wise error target 5%.

    lower_index = int(0.0125 * len(samples))
    upper_index = int(0.9875 * len(samples)) - 1

    lower = samples[lower_index]
    upper = samples[upper_index]

    # Also report the ordinary 95% interval.

    ordinary_lower = samples[
        int(0.025 * len(samples))
    ]

    ordinary_upper = samples[
        int(0.975 * len(samples)) - 1
    ]

    return (
        observed,
        lower,
        upper,
        ordinary_lower,
        ordinary_upper,
    )


def run_configuration(
    model,
    kv_model,
    prefix_model,
    device,
    batch_size,
    prefix_len,
    config_index,
):

    print("\n" + "=" * 72)
    print(
        f"CONFIRMATION: batch={batch_size}, "
        f"prefix={prefix_len}, prompt={PROMPT_LEN}"
    )
    print("=" * 72)

    rng = random.Random(
        SEED + config_index * 1000
    )

    cluster_results = []
    cold_latencies = []

    total_correctness_checks = 0

    for cluster_index in range(NUM_CLUSTERS):

        shared_prefix = torch.randint(
            0,
            model.config.vocab_size,
            (1, prefix_len),
            device=device,
        )

        suffixes = torch.randint(
            0,
            model.config.vocab_size,
            (batch_size, PROMPT_LEN - prefix_len),
            device=device,
        )

        prompt = torch.cat(
            [
                shared_prefix.expand(batch_size, -1),
                suffixes,
            ],
            dim=1,
        )

        reference = standard_greedy(
            model,
            prompt,
        )

        methods = {
            "kv": lambda: kv_greedy(
                kv_model,
                prompt,
            ),
            "prefix": lambda: prefix_model.generate_greedy(
                prompt,
                prefix_len=prefix_len,
                max_new_tokens=MAX_NEW_TOKENS,
            ),
        }

        # Verify ordinary KV output.

        kv_output = methods["kv"]()

        if not torch.equal(reference, kv_output):
            raise AssertionError(
                f"KV mismatch in cluster {cluster_index + 1}"
            )

        # Measure one cold prefix-cache execution.

        prefix_model.prefix_cache.clear_cache()

        cold_ms, cold_output = measure(
            methods["prefix"],
            device,
        )

        if not torch.equal(reference, cold_output):
            raise AssertionError(
                f"Cold prefix mismatch in cluster "
                f"{cluster_index + 1}"
            )

        cold_latencies.append(cold_ms)

        # Warm up both methods.

        for _ in range(WARMUP_RUNS):

            for name in ("kv", "prefix"):

                output = methods[name]()

                if not torch.equal(reference, output):
                    raise AssertionError(
                        f"Warm-up mismatch: {name}"
                    )

        times = {
            "kv": [],
            "prefix": [],
        }

        # Randomize execution order for each repetition.

        for _ in range(REPETITIONS):

            order = ["kv", "prefix"]
            rng.shuffle(order)

            for name in order:

                elapsed_ms, output = measure(
                    methods[name],
                    device,
                )

                if not torch.equal(reference, output):
                    raise AssertionError(
                        f"Timed output mismatch: {name}, "
                        f"cluster {cluster_index + 1}"
                    )

                times[name].append(elapsed_ms)
                total_correctness_checks += 1

        cluster_result = {
            "cluster": cluster_index + 1,
            "kv_ms": statistics.median(times["kv"]),
            "prefix_ms": statistics.median(times["prefix"]),
            "cold_prefix_ms": cold_ms,
        }

        cluster_results.append(cluster_result)

        print(
            f"Cluster {cluster_index + 1:02d}/{NUM_CLUSTERS}: "
            f"KV={cluster_result['kv_ms']:.3f} ms, "
            f"Prefix={cluster_result['prefix_ms']:.3f} ms, "
            f"Cold={cold_ms:.3f} ms",
            flush=True,
        )

    kv_values = [
        item["kv_ms"] for item in cluster_results
    ]

    prefix_values = [
        item["prefix_ms"] for item in cluster_results
    ]

    kv_median = statistics.median(kv_values)
    prefix_median = statistics.median(prefix_values)

    (
        speedup,
        adjusted_low,
        adjusted_high,
        ordinary_low,
        ordinary_high,
    ) = paired_bootstrap_ratio(
        kv_values,
        prefix_values,
        SEED + config_index,
    )

    cold_median = statistics.median(cold_latencies)

    latency_reduction = (
        (kv_median - prefix_median)
        / kv_median
    ) * 100

    print("\nConfiguration result:")
    print(f"  KV median: {kv_median:.3f} ms")
    print(f"  Prefix median: {prefix_median:.3f} ms")
    print(f"  Speedup: {speedup:.4f}x")
    print(f"  Latency reduction: {latency_reduction:.2f}%")

    print(
        f"  Ordinary 95% CI: "
        f"[{ordinary_low:.4f}, {ordinary_high:.4f}]"
    )

    print(
        f"  Bonferroni-adjusted 97.5% CI: "
        f"[{adjusted_low:.4f}, {adjusted_high:.4f}]"
    )

    print(f"  Cold prefix median: {cold_median:.3f} ms")
    print(
        f"  Correctness checks: "
        f"{total_correctness_checks}"
    )

    result = {
        "batch_size": batch_size,
        "prefix_len": prefix_len,
        "prompt_len": PROMPT_LEN,
        "new_tokens": MAX_NEW_TOKENS,
        "clusters": NUM_CLUSTERS,
        "repetitions": REPETITIONS,
        "kv_median_ms": kv_median,
        "prefix_median_ms": prefix_median,
        "speedup_vs_kv": speedup,
        "latency_reduction_percent": latency_reduction,
        "ordinary_95_ci_lower": ordinary_low,
        "ordinary_95_ci_upper": ordinary_high,
        "adjusted_97_5_ci_lower": adjusted_low,
        "adjusted_97_5_ci_upper": adjusted_high,
        "cold_prefix_median_ms": cold_median,
        "correctness_checks": total_correctness_checks,
        "correctness": "PASS",
    }

    return result, cluster_results


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("=" * 72)
    print("RCT100M PREFIX GENERATION CONFIRMATION")
    print("=" * 72)

    model, checkpoint = load_model(device)

    kv_model = RCT100MKV(model)
    kv_model.eval()

    prefix_model = RCT100MPrefixGeneration(
        model,
        cache_capacity=16,
    )
    prefix_model.eval()

    print(f"Checkpoint step: {checkpoint['step']}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Precision: {next(model.parameters()).dtype}")
    print(f"Configurations: {CONFIGURATIONS}")
    print(f"Prompt length: {PROMPT_LEN}")
    print(f"Generated tokens: {MAX_NEW_TOKENS}")
    print(f"Independent clusters: {NUM_CLUSTERS}")
    print(f"Timed repetitions: {REPETITIONS}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Bootstrap samples: {BOOTSTRAP_SAMPLES}")

    results = []
    all_cluster_rows = []

    for index, configuration in enumerate(
        CONFIGURATIONS,
        start=1,
    ):

        result, cluster_rows = run_configuration(
            model,
            kv_model,
            prefix_model,
            device,
            configuration["batch_size"],
            configuration["prefix_len"],
            index,
        )

        results.append(result)

        for row in cluster_rows:

            all_cluster_rows.append({
                "batch_size": configuration["batch_size"],
                "prefix_len": configuration["prefix_len"],
                **row,
            })

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

    summary_path = output_dir / "confirmation_summary.csv"
    clusters_path = output_dir / "cluster_measurements.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    with clusters_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(all_cluster_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_cluster_rows)

    print("\n" + "=" * 72)
    print("FINAL CONFIRMATION SUMMARY")
    print("=" * 72)

    print(
        f"{'Batch':>6} {'Prefix':>8} "
        f"{'KV ms':>12} {'Prefix ms':>12} "
        f"{'Speedup':>10} {'Adjusted 97.5% CI':>25}"
    )

    for result in results:

        interval = (
            f"[{result['adjusted_97_5_ci_lower']:.3f}, "
            f"{result['adjusted_97_5_ci_upper']:.3f}]"
        )

        print(
            f"{result['batch_size']:>6} "
            f"{result['prefix_len']:>8} "
            f"{result['kv_median_ms']:>12.3f} "
            f"{result['prefix_median_ms']:>12.3f} "
            f"{result['speedup_vs_kv']:>9.3f}x "
            f"{interval:>25}"
        )

    print(f"\nSummary CSV: {summary_path}")
    print(f"Cluster CSV: {clusters_path}")
    print("All configurations completed.")
    print("RCT100M PREFIX CONFIRMATION: COMPLETE")


if __name__ == "__main__":
    main()
