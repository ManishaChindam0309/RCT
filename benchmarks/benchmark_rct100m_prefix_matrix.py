
import sys
import time
import random
import csv
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

BATCH_SIZES = [4, 8, 16]
PREFIX_LENGTHS = [8, 16, 24]
PROMPT_LEN = 32
MAX_NEW_TOKENS = 8

NUM_CLUSTERS = 20
REPETITIONS = 3
WARMUP_RUNS = 2
BOOTSTRAP_SAMPLES = 10000

SEED = 20261107


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

    elapsed_ms = (time.perf_counter() - start) * 1000

    return elapsed_ms, output


def bootstrap_speedup(kv_values, prefix_values, seed):

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
            kv_values[index] for index in indices
        ]

        sampled_prefix = [
            prefix_values[index] for index in indices
        ]

        denominator = statistics.median(sampled_prefix)

        if denominator > 0:

            samples.append(
                statistics.median(sampled_kv) / denominator
            )

    samples.sort()

    lower = samples[int(0.025 * len(samples))]
    upper = samples[int(0.975 * len(samples))]

    return observed, lower, upper


def run_configuration(
    model,
    kv_model,
    prefix_model,
    device,
    batch_size,
    prefix_len,
    config_index,
):

    print(
        f"\nConfiguration: batch={batch_size}, "
        f"prefix={prefix_len}, prompt={PROMPT_LEN}",
        flush=True,
    )

    cluster_results = []

    cold_latencies = []

    total_checks = 0

    rng = random.Random(
        SEED + config_index * 1000
    )

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

        methods = {
            "standard": lambda: standard_greedy(
                model,
                prompt,
            ),
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

        reference = methods["standard"]()

        kv_output = methods["kv"]()

        if not torch.equal(reference, kv_output):
            raise AssertionError(
                f"KV mismatch: batch={batch_size}, "
                f"prefix={prefix_len}, cluster={cluster_index}"
            )

        prefix_model.prefix_cache.clear_cache()

        cold_ms, cold_output = measure(
            methods["prefix"],
            device,
        )

        if not torch.equal(reference, cold_output):
            raise AssertionError(
                f"Cold prefix mismatch: batch={batch_size}, "
                f"prefix={prefix_len}, cluster={cluster_index}"
            )

        cold_latencies.append(cold_ms)

        for _ in range(WARMUP_RUNS):

            for name in ("standard", "kv", "prefix"):

                output = methods[name]()

                if not torch.equal(reference, output):
                    raise AssertionError(
                        f"Warm-up mismatch: {name}"
                    )

        times = {
            "standard": [],
            "kv": [],
            "prefix": [],
        }

        for _ in range(REPETITIONS):

            order = ["standard", "kv", "prefix"]
            rng.shuffle(order)

            for name in order:

                elapsed_ms, output = measure(
                    methods[name],
                    device,
                )

                if not torch.equal(reference, output):
                    raise AssertionError(
                        f"Timed mismatch: {name}"
                    )

                times[name].append(elapsed_ms)
                total_checks += 1

        cluster_results.append({
            name: statistics.median(values)
            for name, values in times.items()
        })

        print(
            f"  Cluster {cluster_index + 1:02d}/{NUM_CLUSTERS}: "
            f"standard={statistics.median(times['standard']):.2f} ms, "
            f"KV={statistics.median(times['kv']):.2f} ms, "
            f"prefix={statistics.median(times['prefix']):.2f} ms",
            flush=True,
        )

    standard_clusters = [
        item["standard"] for item in cluster_results
    ]

    kv_clusters = [
        item["kv"] for item in cluster_results
    ]

    prefix_clusters = [
        item["prefix"] for item in cluster_results
    ]

    standard_median = statistics.median(standard_clusters)
    kv_median = statistics.median(kv_clusters)
    prefix_median = statistics.median(prefix_clusters)

    speedup_kv, ci_kv_low, ci_kv_high = bootstrap_speedup(
        kv_clusters,
        prefix_clusters,
        SEED + config_index,
    )

    speedup_standard = (
        statistics.median(standard_clusters)
        / prefix_median
    )

    cold_median = statistics.median(cold_latencies)

    result = {
        "batch_size": batch_size,
        "prefix_len": prefix_len,
        "prompt_len": PROMPT_LEN,
        "new_tokens": MAX_NEW_TOKENS,
        "clusters": NUM_CLUSTERS,
        "repetitions": REPETITIONS,
        "standard_ms": standard_median,
        "kv_ms": kv_median,
        "prefix_ms": prefix_median,
        "prefix_speedup_vs_standard": speedup_standard,
        "prefix_speedup_vs_kv": speedup_kv,
        "ci95_lower": ci_kv_low,
        "ci95_upper": ci_kv_high,
        "cold_prefix_ms": cold_median,
        "correctness_checks": total_checks,
        "correctness": "PASS",
    }

    print("\nConfiguration summary:")
    print(f"  Standard median: {standard_median:.3f} ms")
    print(f"  KV median: {kv_median:.3f} ms")
    print(f"  Prefix median: {prefix_median:.3f} ms")
    print(f"  Prefix vs standard: {speedup_standard:.3f}x")
    print(f"  Prefix vs KV: {speedup_kv:.3f}x")
    print(
        f"  Paired cluster-bootstrap 95% CI: "
        f"[{ci_kv_low:.3f}, {ci_kv_high:.3f}]"
    )
    print(f"  Cold prefix median: {cold_median:.3f} ms")
    print(f"  Correctness checks: {total_checks}")

    return result


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("=" * 78)
    print("RCT100M PREFIX GENERATION WORKLOAD MATRIX")
    print("=" * 78)

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
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Prefix lengths: {PREFIX_LENGTHS}")
    print(f"Prompt length: {PROMPT_LEN}")
    print(f"Generated tokens: {MAX_NEW_TOKENS}")
    print(f"Clusters/configuration: {NUM_CLUSTERS}")
    print(f"Timed repetitions: {REPETITIONS}")
    print(f"Warm-ups: {WARMUP_RUNS}")

    results = []

    config_index = 0

    for batch_size in BATCH_SIZES:

        for prefix_len in PREFIX_LENGTHS:

            config_index += 1

            result = run_configuration(
                model,
                kv_model,
                prefix_model,
                device,
                batch_size,
                prefix_len,
                config_index,
            )

            results.append(result)

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct100m_prefix_generation_matrix"
        / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / "matrix_results.csv"

    with csv_path.open(
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

    print("\n" + "=" * 78)
    print("FINAL MATRIX SUMMARY")
    print("=" * 78)

    print(
        f"{'Batch':>6} {'Prefix':>8} "
        f"{'KV ms':>12} {'Prefix ms':>12} "
        f"{'Speedup':>10} {'95% CI':>23}"
    )

    for result in results:

        interval = (
            f"[{result['ci95_lower']:.3f}, "
            f"{result['ci95_upper']:.3f}]"
        )

        print(
            f"{result['batch_size']:>6} "
            f"{result['prefix_len']:>8} "
            f"{result['kv_ms']:>12.3f} "
            f"{result['prefix_ms']:>12.3f} "
            f"{result['prefix_speedup_vs_kv']:>9.3f}x "
            f"{interval:>23}"
        )

    print(f"\nResults saved to: {csv_path}")
    print("All nine configurations completed.")
    print("RCT100M PREFIX GENERATION MATRIX: COMPLETE")


if __name__ == "__main__":
    main()
