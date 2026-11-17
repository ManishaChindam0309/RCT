
import sys
import time
import random
import statistics
from pathlib import Path

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

BATCH_SIZE = 8
PREFIX_LEN = 24
PROMPT_LEN = 32
MAX_NEW_TOKENS = 8

NUM_CLUSTERS = 30
REPETITIONS = 5
WARMUP_RUNS = 2
BOOTSTRAP_SAMPLES = 10000

SEED = 20261106


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


def bootstrap_speedup(kv_clusters, prefix_clusters):

    rng = random.Random(SEED + 100)

    cluster_count = len(kv_clusters)

    observed_speedup = (
        statistics.median(kv_clusters)
        / statistics.median(prefix_clusters)
    )

    bootstrap_values = []

    for _ in range(BOOTSTRAP_SAMPLES):

        indices = [
            rng.randrange(cluster_count)
            for _ in range(cluster_count)
        ]

        sampled_kv = [
            kv_clusters[index]
            for index in indices
        ]

        sampled_prefix = [
            prefix_clusters[index]
            for index in indices
        ]

        denominator = statistics.median(sampled_prefix)

        if denominator > 0:

            bootstrap_values.append(
                statistics.median(sampled_kv)
                / denominator
            )

    bootstrap_values.sort()

    lower_index = int(0.025 * len(bootstrap_values))
    upper_index = int(0.975 * len(bootstrap_values))

    return (
        observed_speedup,
        bootstrap_values[lower_index],
        bootstrap_values[upper_index],
    )


def summarize(values):

    return {
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    rng = random.Random(SEED)

    print("=" * 78)
    print("RCT100M PREFIX GENERATION VALIDATION")
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
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Prefix length: {PREFIX_LEN}")
    print(f"Prompt length: {PROMPT_LEN}")
    print(f"Generated tokens: {MAX_NEW_TOKENS}")
    print(f"Independent prompt groups: {NUM_CLUSTERS}")
    print(f"Timed repetitions per method/group: {REPETITIONS}")
    print(f"Warm-up repetitions per method/group: {WARMUP_RUNS}")
    print(f"Bootstrap samples: {BOOTSTRAP_SAMPLES}")

    cluster_results = []

    all_standard = []
    all_kv = []
    all_prefix = []
    all_cold = []

    total_correctness_checks = 0

    for cluster_index in range(NUM_CLUSTERS):

        # Create a new shared prefix for each independent group.
        shared_prefix = torch.randint(
            0,
            model.config.vocab_size,
            (1, PREFIX_LEN),
            device=device,
        )

        suffixes = torch.randint(
            0,
            model.config.vocab_size,
            (BATCH_SIZE, PROMPT_LEN - PREFIX_LEN),
            device=device,
        )

        prompt = torch.cat(
            [
                shared_prefix.expand(BATCH_SIZE, -1),
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
                prefix_len=PREFIX_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            ),
        }

        # Establish the standard reference output.
        reference = methods["standard"]()

        # Validate ordinary KV generation.
        kv_output = methods["kv"]()

        if not torch.equal(reference, kv_output):
            raise AssertionError(
                f"KV mismatch in cluster {cluster_index + 1}"
            )

        # Start with a genuinely cold prefix cache.
        prefix_model.prefix_cache.clear_cache()

        cold_ms, cold_output = measure(
            methods["prefix"],
            device,
        )

        if not torch.equal(reference, cold_output):
            raise AssertionError(
                f"Cold prefix mismatch in cluster {cluster_index + 1}"
            )

        all_cold.append(cold_ms)

        # The cold call populated the cache.
        # Now perform warm-up runs.
        for _ in range(WARMUP_RUNS):

            for name in ("standard", "kv", "prefix"):

                output = methods[name]()

                if not torch.equal(reference, output):
                    raise AssertionError(
                        f"Warm-up mismatch: {name}, "
                        f"cluster {cluster_index + 1}"
                    )

        cluster_times = {
            "standard": [],
            "kv": [],
            "prefix": [],
        }

        # Randomize execution order for every repetition.
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
                        f"Timed mismatch: {name}, "
                        f"cluster {cluster_index + 1}"
                    )

                cluster_times[name].append(elapsed_ms)

                total_correctness_checks += 1

        for name in ("standard", "kv", "prefix"):

            all_values = {
                "standard": all_standard,
                "kv": all_kv,
                "prefix": all_prefix,
            }

            all_values[name].extend(
                cluster_times[name]
            )

        cluster_results.append({
            name: statistics.median(values)
            for name, values in cluster_times.items()
        })

        print(
            f"Cluster {cluster_index + 1:02d}/{NUM_CLUSTERS} "
            f"complete | "
            f"standard={statistics.median(cluster_times['standard']):.2f} ms | "
            f"KV={statistics.median(cluster_times['kv']):.2f} ms | "
            f"prefix={statistics.median(cluster_times['prefix']):.2f} ms",
            flush=True,
        )

    # Per-cluster paired observations.
    standard_clusters = [
        item["standard"] for item in cluster_results
    ]

    kv_clusters = [
        item["kv"] for item in cluster_results
    ]

    prefix_clusters = [
        item["prefix"] for item in cluster_results
    ]

    # Cluster-level bootstrap confidence interval.
    speedup, ci_lower, ci_upper = bootstrap_speedup(
        kv_clusters,
        prefix_clusters,
    )

    standard_summary = summarize(all_standard)
    kv_summary = summarize(all_kv)
    prefix_summary = summarize(all_prefix)
    cold_summary = summarize(all_cold)

    print("\n" + "=" * 78)
    print("VALIDATION RESULTS")
    print("=" * 78)

    print(
        f"{'Method':<20}"
        f"{'Median (ms)':>15}"
        f"{'Mean (ms)':>15}"
        f"{'Min (ms)':>13}"
        f"{'Max (ms)':>13}"
    )

    for name, result in (
        ("Standard", standard_summary),
        ("KV cache", kv_summary),
        ("Prefix + KV", prefix_summary),
    ):

        print(
            f"{name:<20}"
            f"{result['median']:>15.3f}"
            f"{result['mean']:>15.3f}"
            f"{result['minimum']:>13.3f}"
            f"{result['maximum']:>13.3f}"
        )

    print("\nSpeedups based on pooled median latency:")

    print(
        "KV vs standard:",
        f"{standard_summary['median'] / kv_summary['median']:.3f}x",
    )

    print(
        "Prefix + KV vs standard:",
        f"{standard_summary['median'] / prefix_summary['median']:.3f}x",
    )

    print(
        "Prefix + KV vs KV:",
        f"{kv_summary['median'] / prefix_summary['median']:.3f}x",
    )

    print("\nPaired cluster-bootstrap result:")

    print(f"Observed speedup: {speedup:.3f}x")
    print(
        f"95% confidence interval: "
        f"[{ci_lower:.3f}, {ci_upper:.3f}]"
    )

    print("\nCold-cache prefix latency:")

    print(f"Median: {cold_summary['median']:.3f} ms")
    print(f"Mean: {cold_summary['mean']:.3f} ms")
    print(f"Minimum: {cold_summary['minimum']:.3f} ms")
    print(f"Maximum: {cold_summary['maximum']:.3f} ms")

    print("\nCorrectness:")

    print(f"Timed correctness checks: {total_correctness_checks}")
    print("All outputs matched standard greedy generation: PASS")

    print("\nCache statistics:")

    print(prefix_model.prefix_cache.cache_statistics())

    print("\nExperiment configuration:")
    print(f"Independent clusters: {NUM_CLUSTERS}")
    print(f"Timed measurements: {len(all_standard) * 3}")
    print(f"Cold-cache measurements: {len(all_cold)}")
    print(f"Precision: {next(model.parameters()).dtype}")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    print("\nRCT100M PREFIX GENERATION VALIDATION: COMPLETE")


if __name__ == "__main__":
    main()
