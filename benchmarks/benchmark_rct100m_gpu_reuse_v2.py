
from pathlib import Path
from datetime import datetime
import sys
import csv
import json
import gc
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

BATCH_SIZE = 8
PROMPT_LENGTH = 24
NEW_TOKENS = 8

CLUSTERS = 30
REPEATS = 10
WARMUPS = 2
BOOTSTRAP_SAMPLES = 10000

DUPLICATE_RATES = {
    "0%": 8,
    "50%": 4,
    "75%": 2,
    "87.5%": 1,
}

METHODS = [
    "standard_no_cache",
    "kv_cache",
    "rct_dedup_kv",
]


def synchronize():
    torch.cuda.synchronize()


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


def create_workload(unique_count, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    while True:
        unique_prompts = torch.randint(
            low=0,
            high=8192,
            size=(unique_count, PROMPT_LENGTH),
            generator=generator,
            dtype=torch.long
        )

        if torch.unique(
            unique_prompts,
            dim=0
        ).size(0) == unique_count:
            break

    indices = (
        torch.arange(BATCH_SIZE) % unique_count
    )

    return unique_prompts.index_select(
        0,
        indices
    ).cuda()


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

    bootstrap_medians = np.median(samples, axis=1)

    return [
        float(np.percentile(bootstrap_medians, 2.5)),
        float(np.percentile(bootstrap_medians, 97.5))
    ]


def paired_speedup_summary(cluster_latencies, numerator, denominator, seed):
    ratios = []

    for cluster in sorted(cluster_latencies):
        num = cluster_latencies[cluster][numerator]
        den = cluster_latencies[cluster][denominator]

        ratios.append(den / num)

    ratios = np.asarray(ratios, dtype=np.float64)

    return {
        "median_speedup": float(np.median(ratios)),
        "mean_speedup": float(np.mean(ratios)),
        "ci95_median": bootstrap_ci(ratios, seed),
        "cluster_ratios": ratios.tolist(),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    torch.manual_seed(2026)
    random.seed(2026)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3, 2
    ), "GB")

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    config = RCT100MConfig(**checkpoint["model_config"])

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
    print("Batch:", BATCH_SIZE)
    print("Prompt length:", PROMPT_LENGTH)
    print("Generated tokens:", NEW_TOKENS)

    raw_records = []
    correctness_records = []
    cluster_latencies = {}
    memory_records = []

    for rate_index, (rate, unique_count) in enumerate(
        DUPLICATE_RATES.items()
    ):

        print("\n" + "=" * 75)
        print(
            f"Duplicate rate {rate}: "
            f"{unique_count}/{BATCH_SIZE} unique prompts"
        )
        print("=" * 75)

        cluster_latencies[rate] = {}

        for cluster in range(CLUSTERS):

            prompts = create_workload(
                unique_count,
                seed=70000 + rate_index * 10000 + cluster
            )

            actual_unique = torch.unique(
                prompts,
                dim=0
            ).size(0)

            if actual_unique != unique_count:
                raise RuntimeError(
                    "Incorrect duplicate workload."
                )

            # Verify all three methods produce identical output.
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
                    f"Correctness failure: {rate}, cluster {cluster}"
                )

            correctness_records.append({
                "duplicate_rate": rate,
                "cluster": cluster,
                "unique_prompts": unique_count,
                "standard_vs_kv_exact": standard_kv_exact,
                "kv_vs_rct_exact": kv_rct_exact,
            })

            del standard_output, kv_output, rct_output

            # Warm up each method.
            for method in METHODS:
                for _ in range(WARMUPS):
                    output = functions[method](
                        models[method],
                        prompts
                    )
                    del output

            # Randomized method order reduces systematic timing bias.
            cluster_latencies[rate][cluster] = {
                method: [] for method in METHODS
            }

            for repeat in range(REPEATS):

                order = METHODS.copy()
                random.shuffle(order)

                for method in order:

                    elapsed, peak_memory = run_timed(
                        functions[method],
                        models[method],
                        prompts
                    )

                    cluster_latencies[rate][cluster][method].append(
                        elapsed
                    )

                    memory_records.append({
                        "duplicate_rate": rate,
                        "cluster": cluster,
                        "repeat": repeat + 1,
                        "method": method,
                        "peak_allocated_mb": peak_memory,
                    })

                    raw_records.append({
                        "duplicate_rate": rate,
                        "cluster": cluster,
                        "repeat": repeat + 1,
                        "method": method,
                        "latency_seconds": elapsed,
                        "output_tokens_per_second": (
                            BATCH_SIZE * NEW_TOKENS / elapsed
                        ),
                        "unique_prompts": unique_count,
                    })

            print(
                f"Cluster {cluster + 1}/{CLUSTERS}: "
                f"correctness passed; timing completed"
            )

            del prompts
            gc.collect()

    # Calculate median latency for every method and cluster.
    paired_clusters = {}

    for rate in DUPLICATE_RATES:
        paired_clusters[rate] = {}

        for cluster in range(CLUSTERS):
            paired_clusters[rate][cluster] = {}

            for method in METHODS:
                paired_clusters[rate][cluster][method] = (
                    statistics.median(
                        cluster_latencies[rate][cluster][method]
                    )
                )

    # Build statistical summaries.
    summary = {}

    for rate_index, rate in enumerate(DUPLICATE_RATES):

        summary[rate] = {
            "methods": {},
            "rct_vs_kv": paired_speedup_summary(
                paired_clusters[rate],
                "rct_dedup_kv",
                "kv_cache",
                seed=9000 + rate_index
            ),
            "rct_vs_standard": paired_speedup_summary(
                paired_clusters[rate],
                "rct_dedup_kv",
                "standard_no_cache",
                seed=10000 + rate_index
            ),
        }

        for method in METHODS:
            values = [
                paired_clusters[rate][cluster][method]
                for cluster in range(CLUSTERS)
            ]

            summary[rate]["methods"][method] = {
                "median_latency_seconds": float(
                    np.median(values)
                ),
                "mean_latency_seconds": float(
                    np.mean(values)
                ),
                "ci95_median_latency": bootstrap_ci(
                    values,
                    seed=11000 + rate_index + METHODS.index(method)
                ),
            }

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct100m_gpu_reuse_v2"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(filename, records):
        path = output_dir / filename

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

        return path

    timing_path = save_csv(
        "raw_timings.csv",
        raw_records
    )

    memory_path = save_csv(
        "memory_records.csv",
        memory_records
    )

    report = {
        "experiment": "RCT-100M GPU statistical reuse benchmark",
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "checkpoint_step": 1500,
        "precision": "FP32",
        "batch_size": BATCH_SIZE,
        "prompt_length": PROMPT_LENGTH,
        "generated_tokens": NEW_TOKENS,
        "clusters_per_rate": CLUSTERS,
        "repetitions_per_cluster": REPEATS,
        "warmups_per_method": WARMUPS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "correctness_passed": len(correctness_records),
        "correctness": correctness_records,
        "summary": summary,
    }

    report_path = output_dir / "report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 75)
    print("STATISTICAL GPU REUSE RESULTS")
    print("=" * 75)

    for rate in DUPLICATE_RATES:

        print(f"\nDuplicate rate: {rate}")

        for method in METHODS:
            result = summary[rate]["methods"][method]

            print(
                f"{method:22s} "
                f"median={result['median_latency_seconds']:.5f}s "
                f"95% CI="
                f"{result['ci95_median_latency']}"
            )

        result = summary[rate]["rct_vs_kv"]

        print(
            "RCT speedup vs KV:",
            f"{result['median_speedup']:.3f}x",
            "| 95% CI:",
            result["ci95_median"]
        )

    print("\nCorrectness checks:", len(correctness_records))
    print("Raw timing records:", len(raw_records))
    print("Timing CSV:", timing_path)
    print("Memory CSV:", memory_path)
    print("Report JSON:", report_path)


if __name__ == "__main__":
    main()
