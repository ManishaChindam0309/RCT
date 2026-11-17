
from pathlib import Path
from datetime import datetime
import sys
import csv
import json
import time
import random
import statistics

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined


CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

BATCH_SIZES = [4, 8, 16]
PROMPT_LENGTHS = [8, 24, 64]

DUPLICATE_RATES = {
    "0%": 1.00,
    "50%": 0.50,
    "75%": 0.25,
}

NEW_TOKENS = 8

CLUSTERS = 20
REPEATS = 5
WARMUPS = 2
BOOTSTRAP_SAMPLES = 10000

METHODS = [
    "standard_no_cache",
    "kv_cache",
    "rct_dedup_kv",
]


def synchronize():
    torch.cuda.synchronize()


def theoretical_cache_mib(config, batch_size):
    elements = (
        batch_size
        * config.num_layers
        * 2
        * config.num_heads
        * config.max_seq_len
        * (config.hidden_size // config.num_heads)
    )

    return elements * 4 / (1024 ** 2)


@torch.inference_mode()
def greedy_standard(model, prompts):
    generated = prompts

    for _ in range(NEW_TOKENS):
        logits = model(generated)

        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

    return generated


@torch.inference_mode()
def greedy_kv(model, prompts):
    cache = model.create_cache(prompts.size(0))

    logits = model.decode_step(prompts, cache)
    generated = prompts

    for step in range(NEW_TOKENS):
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

        if step + 1 < NEW_TOKENS:
            logits = model.decode_step(
                next_token,
                cache
            )

    return generated


@torch.inference_mode()
def greedy_rct_dedup(model, prompts):
    unique_prompts, inverse_indices = (
        model._stable_deduplicate(prompts)
    )

    unique_outputs = greedy_kv(
        model.kv_model,
        unique_prompts
    )

    return unique_outputs.index_select(
        0,
        inverse_indices
    )


def create_workload(
    batch_size,
    prompt_length,
    unique_count,
    seed
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    while True:
        unique_prompts = torch.randint(
            low=0,
            high=8192,
            size=(unique_count, prompt_length),
            generator=generator,
            dtype=torch.long
        )

        if torch.unique(
            unique_prompts,
            dim=0
        ).size(0) == unique_count:
            break

    indices = (
        torch.arange(batch_size) % unique_count
    )

    prompts = unique_prompts.index_select(
        0,
        indices
    )

    return prompts.cuda()


def run_timed(function, model, prompts):
    synchronize()

    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    output = function(model, prompts)

    synchronize()

    elapsed = time.perf_counter() - start

    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**2
    )

    del output

    return elapsed, peak_memory


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(BOOTSTRAP_SAMPLES, len(values)),
        replace=True
    )

    medians = np.median(samples, axis=1)

    return [
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5))
    ]


def speedup_summary(cluster_values, seed):
    ratios = np.asarray(
        [
            cluster_values[c]["kv_cache"]
            / cluster_values[c]["rct_dedup_kv"]
            for c in sorted(cluster_values)
        ],
        dtype=np.float64
    )

    return {
        "median_speedup": float(np.median(ratios)),
        "mean_speedup": float(np.mean(ratios)),
        "ci95_median": bootstrap_ci(ratios, seed),
        "cluster_speedups": ratios.tolist(),
    }


def save_csv(path, records):
    if not records:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    torch.manual_seed(2026)
    random.seed(2026)

    device = torch.cuda.get_device_name(0)

    print("GPU:", device)
    print("PyTorch:", torch.__version__)
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    config = RCT100MConfig(
        **checkpoint["model_config"]
    )

    base_model = RCT100M(config)

    base_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    del checkpoint

    base_model = base_model.cuda()
    base_model.eval()

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    combined = RCT100MCombined(base_model)
    combined.eval()

    functions = {
        "standard_no_cache": greedy_standard,
        "kv_cache": greedy_kv,
        "rct_dedup_kv": greedy_rct_dedup,
    }

    models = {
        "standard_no_cache": base_model,
        "kv_cache": kv_model,
        "rct_dedup_kv": combined,
    }

    print("Parameters:", base_model.parameter_count())
    print("Checkpoint: step 1500")
    print("Precision: FP32")
    print("Generated tokens:", NEW_TOKENS)

    print("\nTheoretical full KV-cache allocation:")

    for batch in BATCH_SIZES:
        print(
            f"Batch {batch}: "
            f"{theoretical_cache_mib(config, batch):.2f} MiB"
        )

    raw_records = []
    memory_records = []
    correctness_records = []
    all_summaries = {}

    configuration_index = 0

    total_configurations = (
        len(BATCH_SIZES)
        * len(PROMPT_LENGTHS)
        * len(DUPLICATE_RATES)
    )

    for batch_size in BATCH_SIZES:

        for prompt_length in PROMPT_LENGTHS:

            for duplicate_rate, unique_fraction in (
                DUPLICATE_RATES.items()
            ):

                configuration_index += 1

                unique_count = max(
                    1,
                    int(batch_size * unique_fraction)
                )

                config_key = (
                    f"B{batch_size}"
                    f"_P{prompt_length}"
                    f"_D{duplicate_rate}"
                )

                print("\n" + "=" * 75)
                print(
                    f"Configuration "
                    f"{configuration_index}/{total_configurations}"
                )
                print(
                    f"Batch={batch_size}, "
                    f"Prompt={prompt_length}, "
                    f"Duplicates={duplicate_rate}, "
                    f"Unique={unique_count}"
                )
                print("=" * 75)

                # Warm up all methods for this configuration.
                warmup_prompts = create_workload(
                    batch_size,
                    prompt_length,
                    unique_count,
                    seed=500000 + configuration_index
                )

                try:
                    for method in METHODS:
                        for _ in range(WARMUPS):
                            output = functions[method](
                                models[method],
                                warmup_prompts
                            )
                            del output

                    synchronize()

                except torch.cuda.OutOfMemoryError:
                    print(
                        "GPU OUT OF MEMORY during warm-up. "
                        "Stopping safely."
                    )
                    raise

                del warmup_prompts

                cluster_values = {}

                for cluster in range(CLUSTERS):

                    prompts = create_workload(
                        batch_size,
                        prompt_length,
                        unique_count,
                        seed=(
                            600000
                            + configuration_index * 1000
                            + cluster
                        )
                    )

                    actual_unique = torch.unique(
                        prompts,
                        dim=0
                    ).size(0)

                    if actual_unique != unique_count:
                        raise RuntimeError(
                            "Incorrect duplicate workload."
                        )

                    # Exact greedy-output correctness.
                    try:
                        standard_output = greedy_standard(
                            base_model,
                            prompts
                        )

                        kv_output = greedy_kv(
                            kv_model,
                            prompts
                        )

                        rct_output = greedy_rct_dedup(
                            combined,
                            prompts
                        )

                    except torch.cuda.OutOfMemoryError:
                        print(
                            "GPU OUT OF MEMORY during correctness. "
                            "Stopping safely."
                        )
                        raise

                    standard_kv_exact = torch.equal(
                        standard_output,
                        kv_output
                    )

                    kv_rct_exact = torch.equal(
                        kv_output,
                        rct_output
                    )

                    if not standard_kv_exact or not kv_rct_exact:
                        raise RuntimeError(
                            f"Correctness failure: {config_key}, "
                            f"cluster {cluster}"
                        )

                    correctness_records.append({
                        "configuration": config_key,
                        "batch_size": batch_size,
                        "prompt_length": prompt_length,
                        "duplicate_rate": duplicate_rate,
                        "cluster": cluster,
                        "unique_prompts": unique_count,
                        "standard_vs_kv_exact": standard_kv_exact,
                        "kv_vs_rct_exact": kv_rct_exact,
                    })

                    del standard_output
                    del kv_output
                    del rct_output

                    cluster_values[cluster] = {
                        method: [] for method in METHODS
                    }

                    # Randomized method order on every repetition.
                    for repeat in range(REPEATS):

                        order = METHODS.copy()
                        random.shuffle(order)

                        for method in order:

                            try:
                                elapsed, peak_memory = run_timed(
                                    functions[method],
                                    models[method],
                                    prompts
                                )

                            except torch.cuda.OutOfMemoryError:
                                print(
                                    "GPU OUT OF MEMORY during timing. "
                                    "Stopping safely."
                                )
                                raise

                            cluster_values[cluster][method].append(
                                elapsed
                            )

                            raw_records.append({
                                "configuration": config_key,
                                "batch_size": batch_size,
                                "prompt_length": prompt_length,
                                "duplicate_rate": duplicate_rate,
                                "unique_prompts": unique_count,
                                "cluster": cluster,
                                "repeat": repeat + 1,
                                "method": method,
                                "latency_seconds": elapsed,
                                "output_tokens_per_second": (
                                    batch_size * NEW_TOKENS / elapsed
                                ),
                            })

                            memory_records.append({
                                "configuration": config_key,
                                "batch_size": batch_size,
                                "prompt_length": prompt_length,
                                "duplicate_rate": duplicate_rate,
                                "cluster": cluster,
                                "repeat": repeat + 1,
                                "method": method,
                                "peak_allocated_mb": peak_memory,
                            })

                    del prompts

                    print(
                        f"Cluster {cluster + 1}/{CLUSTERS}: "
                        "correctness and timing passed"
                    )

                # One median per method per independent workload cluster.
                cluster_medians = {}

                for cluster in range(CLUSTERS):
                    cluster_medians[cluster] = {}

                    for method in METHODS:
                        cluster_medians[cluster][method] = (
                            statistics.median(
                                cluster_values[cluster][method]
                            )
                        )

                result = {
                    "batch_size": batch_size,
                    "prompt_length": prompt_length,
                    "duplicate_rate": duplicate_rate,
                    "unique_prompts": unique_count,
                    "theoretical_full_cache_mib": (
                        theoretical_cache_mib(
                            config,
                            batch_size
                        )
                    ),
                    "methods": {},
                    "rct_vs_kv": speedup_summary(
                        cluster_medians,
                        seed=70000 + configuration_index
                    ),
                }

                for method_index, method in enumerate(METHODS):

                    values = [
                        cluster_medians[c][method]
                        for c in range(CLUSTERS)
                    ]

                    result["methods"][method] = {
                        "median_latency_seconds": float(
                            np.median(values)
                        ),
                        "mean_latency_seconds": float(
                            np.mean(values)
                        ),
                        "ci95_median_latency": bootstrap_ci(
                            values,
                            seed=80000 + configuration_index * 10
                            + method_index
                        ),
                    }

                all_summaries[config_key] = result

                print(
                    "Median RCT speedup vs KV:",
                    f"{result['rct_vs_kv']['median_speedup']:.3f}x",
                    "| 95% CI:",
                    result["rct_vs_kv"]["ci95_median"]
                )

                del cluster_values
                del cluster_medians

                torch.cuda.synchronize()

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct100m_gpu_generalization"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_csv(
        output_dir / "raw_timings.csv",
        raw_records
    )

    save_csv(
        output_dir / "memory_records.csv",
        memory_records
    )

    save_csv(
        output_dir / "correctness.csv",
        correctness_records
    )

    report = {
        "experiment": "RCT-100M GPU workload generalization",
        "gpu": device,
        "pytorch": torch.__version__,
        "checkpoint_step": 1500,
        "precision": "FP32",
        "batch_sizes": BATCH_SIZES,
        "prompt_lengths": PROMPT_LENGTHS,
        "duplicate_rates": list(DUPLICATE_RATES.keys()),
        "generated_tokens": NEW_TOKENS,
        "clusters_per_configuration": CLUSTERS,
        "repetitions_per_cluster": REPEATS,
        "warmups_per_method_configuration": WARMUPS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "configuration_count": total_configurations,
        "correctness_checks": len(correctness_records),
        "timing_records": len(raw_records),
        "memory_records": len(memory_records),
        "summaries": all_summaries,
    }

    report_path = output_dir / "report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 75)
    print("FINAL GENERALIZATION SUMMARY")
    print("=" * 75)

    for key, result in all_summaries.items():
        speedup = result["rct_vs_kv"]

        print(
            f"{key:20s} "
            f"speedup={speedup['median_speedup']:.3f}x "
            f"CI={speedup['ci95_median']}"
        )

    print("\nConfigurations:", total_configurations)
    print("Correctness checks:", len(correctness_records))
    print("Timing records:", len(raw_records))
    print("Memory records:", len(memory_records))
    print("Output directory:", output_dir)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
