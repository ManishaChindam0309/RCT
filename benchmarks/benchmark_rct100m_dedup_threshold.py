
import csv
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "rct100m_dedup_threshold"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROMPT_LENGTH = 32
NEW_TOKENS = 8

INDEPENDENT_GROUPS = 30
WARMUP_REPETITIONS = 1
TIMED_REPETITIONS = 5

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20261106

# Batch size remains 16.
# Unique prompt counts give the following duplicate rates:
# 16 =   0%
# 12 =  25%
#  8 =  50%
#  4 =  75%
#  2 =  87.5%
#  1 =  93.75%

WORKLOADS = [
    {"batch_size": 16, "unique_prompts": 16},
    {"batch_size": 16, "unique_prompts": 12},
    {"batch_size": 16, "unique_prompts": 8},
    {"batch_size": 16, "unique_prompts": 4},
    {"batch_size": 16, "unique_prompts": 2},
    {"batch_size": 16, "unique_prompts": 1},
]

METHODS = [
    "standard",
    "kv",
    "dedup_kv",
]


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    config = RCT100MConfig(
        **checkpoint["model_config"]
    )

    model = RCT100M(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    kv_model = RCT100MKV(model)
    kv_model.eval()

    return model, kv_model, checkpoint, config


# ============================================================
# PROMPT GENERATION
# ============================================================

def make_prompt_batch(
    batch_size,
    unique_count,
    prompt_length,
    vocab_size,
    seed,
):
    """
    Create a batch containing exactly unique_count distinct
    complete prompts.

    Prompts are distributed as evenly as possible. This also
    supports workloads where batch_size is not divisible by
    unique_count, such as batch 16 with 12 unique prompts.
    """

    if unique_count < 1 or unique_count > batch_size:
        raise ValueError(
            "unique_count must be between 1 and batch_size."
        )

    if prompt_length < 1:
        raise ValueError(
            "prompt_length must be positive."
        )

    if vocab_size <= unique_count:
        raise ValueError(
            "Vocabulary must be larger than unique_count."
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    unique_prompts = torch.randint(
        low=0,
        high=vocab_size,
        size=(unique_count, prompt_length),
        generator=generator,
        dtype=torch.long,
    )

    # Guarantee that every complete prompt is distinct.
    unique_prompts[:, 0] = (
        torch.arange(
            unique_count,
            dtype=torch.long,
        ) + 1
    )

    # Distribute batch rows across unique prompts.
    # Counts differ by at most one when not evenly divisible.
    assignments = (
        torch.arange(batch_size) % unique_count
    )

    permutation = torch.randperm(
        batch_size,
        generator=generator,
    )

    assignments = assignments[permutation]

    prompts = unique_prompts.index_select(
        0,
        assignments,
    )

    return prompts.to(DEVICE)


# ============================================================
# GENERATION METHODS
# ============================================================

@torch.inference_mode()
def standard_greedy(
    model,
    prompts,
    max_new_tokens,
):
    """
    Full-context greedy autoregressive generation.
    """

    generated = prompts.clone()
    prompt_length = prompts.size(1)

    for _ in range(max_new_tokens):

        context = generated[
            :, -model.config.max_seq_len:
        ]

        logits = model(context)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated[:, prompt_length:]


@torch.inference_mode()
def kv_greedy(
    kv_model,
    prompts,
    max_new_tokens,
):
    """
    Incremental greedy generation using the existing KV cache.

    RCT100MKV.create_cache() accepts batch_size only.
    """

    batch_size = prompts.size(0)

    cache = kv_model.create_cache(batch_size)

    logits = kv_model.decode_step(
        prompts,
        cache,
    )

    generated_tokens = []

    for _ in range(max_new_tokens):

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated_tokens.append(next_token)

        logits = kv_model.decode_step(
            next_token,
            cache,
        )

    return torch.cat(
        generated_tokens,
        dim=1,
    )


@torch.inference_mode()
def dedup_kv_greedy(
    kv_model,
    prompts,
    max_new_tokens,
):
    """
    Deduplicate exact complete prompts, generate once per
    unique prompt using KV caching, and restore original order.
    """

    unique_prompts, inverse = torch.unique(
        prompts,
        dim=0,
        sorted=True,
        return_inverse=True,
    )

    unique_outputs = kv_greedy(
        kv_model,
        unique_prompts,
        max_new_tokens,
    )

    return unique_outputs.index_select(
        0,
        inverse,
    )


# ============================================================
# METHOD DISPATCH
# ============================================================

@torch.inference_mode()
def run_method(
    method,
    model,
    kv_model,
    prompts,
):

    if method == "standard":
        return standard_greedy(
            model,
            prompts,
            NEW_TOKENS,
        )

    if method == "kv":
        return kv_greedy(
            kv_model,
            prompts,
            NEW_TOKENS,
        )

    if method == "dedup_kv":
        return dedup_kv_greedy(
            kv_model,
            prompts,
            NEW_TOKENS,
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# ============================================================
# TIMING
# ============================================================

def timed_run(
    method,
    model,
    kv_model,
    prompts,
):

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    output = run_method(
        method,
        model,
        kv_model,
        prompts,
    )

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000.0

    return output, elapsed_ms


# ============================================================
# STATISTICS
# ============================================================

def bootstrap_median_ci(
    values,
    samples=BOOTSTRAP_SAMPLES,
    seed=BOOTSTRAP_SEED,
):
    """
    Bootstrap confidence interval for the median.

    Each value represents one independent prompt group.
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) == 0:
        raise ValueError(
            "Cannot bootstrap an empty array."
        )

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        low=0,
        high=len(values),
        size=(samples, len(values)),
    )

    resampled = values[indices]

    bootstrap_medians = np.median(
        resampled,
        axis=1,
    )

    lower, upper = np.percentile(
        bootstrap_medians,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


def summarize_method_times(
    cluster_results,
    method,
):

    values = [
        row[method]
        for row in cluster_results
    ]

    return float(
        np.median(values)
    )


def summarize_speedup(
    cluster_results,
    numerator,
    denominator,
    seed,
):
    """
    Paired speedup:

        numerator latency / denominator latency

    Values above 1.0 mean the denominator is faster.
    """

    ratios = np.asarray(
        [
            row[numerator] / row[denominator]
            for row in cluster_results
        ],
        dtype=np.float64,
    )

    median_ratio = float(
        np.median(ratios)
    )

    lower, upper = bootstrap_median_ci(
        ratios,
        seed=seed,
    )

    return median_ratio, lower, upper


# ============================================================
# CSV OUTPUT
# ============================================================

def write_csv(path, rows):

    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 72)
    print("RCT100M DEDUPLICATION THRESHOLD BENCHMARK")
    print("=" * 72)

    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(DEVICE)}"
        )

    print(f"Checkpoint: {CHECKPOINT_PATH}")

    print()
    print("Experiment configuration:")
    print(f"Prompt length: {PROMPT_LENGTH}")
    print(f"New tokens: {NEW_TOKENS}")
    print(f"Independent groups: {INDEPENDENT_GROUPS}")
    print(f"Warmups per method/config: {WARMUP_REPETITIONS}")
    print(f"Timed repetitions per group: {TIMED_REPETITIONS}")
    print(f"Bootstrap samples: {BOOTSTRAP_SAMPLES}")

    print()
    print("Methods:")
    print("1. Standard full-context greedy")
    print("2. KV-cached greedy")
    print("3. Exact deduplication + KV-cached greedy")

    print()
    print("Duplication levels:")

    for workload in WORKLOADS:

        batch_size = workload["batch_size"]
        unique_count = workload["unique_prompts"]

        duplicate_rate = (
            1.0 - unique_count / batch_size
        ) * 100.0

        print(
            f"Batch {batch_size}, "
            f"unique {unique_count}, "
            f"duplicates {duplicate_rate:.2f}%"
        )

    # --------------------------------------------------------
    # Load checkpoint and models
    # --------------------------------------------------------

    model, kv_model, checkpoint, config = load_model()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"Checkpoint step: "
        f"{checkpoint.get('step', 'unknown')}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint.get('val_loss', 'unknown')}"
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        RESULTS_ROOT / f"run_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summary_rows = []
    all_cluster_rows = []

    # ========================================================
    # WORKLOAD LOOP
    # ========================================================

    for workload_index, workload in enumerate(WORKLOADS):

        batch_size = workload["batch_size"]
        unique_count = workload["unique_prompts"]

        duplicate_count = batch_size - unique_count

        duplicate_rate = (
            duplicate_count / batch_size
        ) * 100.0

        print()
        print("=" * 72)

        print(
            f"BATCH={batch_size} | "
            f"UNIQUE={unique_count} | "
            f"DUPLICATE PROMPTS={duplicate_count} | "
            f"DUPLICATES={duplicate_rate:.2f}%"
        )

        print("=" * 72)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEVICE)

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        warmup_prompts = make_prompt_batch(
            batch_size,
            unique_count,
            PROMPT_LENGTH,
            config.vocab_size,
            seed=900000 + workload_index,
        )

        warmup_outputs = {}

        warmup_order = METHODS.copy()

        random.Random(
            1000 + workload_index
        ).shuffle(warmup_order)

        for method in warmup_order:

            output, _ = timed_run(
                method,
                model,
                kv_model,
                warmup_prompts,
            )

            warmup_outputs[method] = output

        if not (
            torch.equal(
                warmup_outputs["standard"],
                warmup_outputs["kv"],
            )
            and torch.equal(
                warmup_outputs["standard"],
                warmup_outputs["dedup_kv"],
            )
        ):
            raise AssertionError(
                "Warmup correctness mismatch."
            )

        # ----------------------------------------------------
        # Independent prompt groups
        # ----------------------------------------------------

        cluster_results = []

        for group_index in range(INDEPENDENT_GROUPS):

            prompts = make_prompt_batch(
                batch_size,
                unique_count,
                PROMPT_LENGTH,
                config.vocab_size,
                seed=(
                    20261106
                    + workload_index * 100000
                    + group_index
                ),
            )

            # Independent standard reference.
            # It is calculated outside the timed methods.
            reference_output = standard_greedy(
                model,
                prompts,
                NEW_TOKENS,
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            group_times = {
                method: []
                for method in METHODS
            }

            # ------------------------------------------------
            # Timed repetitions
            # ------------------------------------------------

            for repetition in range(TIMED_REPETITIONS):

                method_order = METHODS.copy()

                random.Random(
                    500000
                    + workload_index * 10000
                    + group_index * 100
                    + repetition
                ).shuffle(method_order)

                for method in method_order:

                    output, elapsed_ms = timed_run(
                        method,
                        model,
                        kv_model,
                        prompts,
                    )

                    # Compare every timed output against
                    # the same standard reference.
                    if not torch.equal(
                        reference_output,
                        output,
                    ):
                        raise AssertionError(
                            f"Correctness mismatch: "
                            f"workload={workload_index}, "
                            f"group={group_index}, "
                            f"repetition={repetition}, "
                            f"method={method}"
                        )

                    group_times[method].append(
                        elapsed_ms
                    )

            # ------------------------------------------------
            # Per-group medians
            # ------------------------------------------------

            group_medians = {
                method: float(
                    np.median(group_times[method])
                )
                for method in METHODS
            }

            cluster_results.append(
                group_medians
            )

            all_cluster_rows.append({
                "batch_size": batch_size,
                "unique_prompts": unique_count,
                "duplicate_prompts": duplicate_count,
                "duplicate_rate_percent": duplicate_rate,
                "group": group_index + 1,
                "standard_ms": group_medians["standard"],
                "kv_ms": group_medians["kv"],
                "dedup_kv_ms": group_medians["dedup_kv"],
                "standard_over_kv": (
                    group_medians["standard"]
                    / group_medians["kv"]
                ),
                "kv_over_dedup": (
                    group_medians["kv"]
                    / group_medians["dedup_kv"]
                ),
                "standard_over_dedup": (
                    group_medians["standard"]
                    / group_medians["dedup_kv"]
                ),
            })

            if (group_index + 1) % 5 == 0:

                print(
                    f"Completed "
                    f"{group_index + 1}/"
                    f"{INDEPENDENT_GROUPS} groups"
                )

        # ====================================================
        # AGGREGATE TIMING RESULTS
        # ====================================================

        standard_ms = summarize_method_times(
            cluster_results,
            "standard",
        )

        kv_ms = summarize_method_times(
            cluster_results,
            "kv",
        )

        dedup_ms = summarize_method_times(
            cluster_results,
            "dedup_kv",
        )

        standard_over_kv = summarize_speedup(
            cluster_results,
            "standard",
            "kv",
            BOOTSTRAP_SEED + workload_index * 10 + 1,
        )

        kv_over_dedup = summarize_speedup(
            cluster_results,
            "kv",
            "dedup_kv",
            BOOTSTRAP_SEED + workload_index * 10 + 2,
        )

        standard_over_dedup = summarize_speedup(
            cluster_results,
            "standard",
            "dedup_kv",
            BOOTSTRAP_SEED + workload_index * 10 + 3,
        )

        # Positive percentage means dedup+KV is faster.
        dedup_latency_reduction = (
            1.0 - dedup_ms / kv_ms
        ) * 100.0

        if DEVICE.type == "cuda":

            peak_memory_mib = (
                torch.cuda.max_memory_allocated(DEVICE)
                / (1024 ** 2)
            )

        else:
            peak_memory_mib = None

        # ----------------------------------------------------
        # Display workload results
        # ----------------------------------------------------

        print()
        print("Results:")

        print(
            f"Standard: {standard_ms:.3f} ms"
        )

        print(
            f"KV only: {kv_ms:.3f} ms"
        )

        print(
            f"Dedup + KV: {dedup_ms:.3f} ms"
        )

        print(
            f"Standard / KV: "
            f"{standard_over_kv[0]:.3f}x "
            f"[{standard_over_kv[1]:.3f}, "
            f"{standard_over_kv[2]:.3f}]"
        )

        print(
            f"KV / Dedup+KV: "
            f"{kv_over_dedup[0]:.3f}x "
            f"[{kv_over_dedup[1]:.3f}, "
            f"{kv_over_dedup[2]:.3f}]"
        )

        print(
            f"Standard / Dedup+KV: "
            f"{standard_over_dedup[0]:.3f}x "
            f"[{standard_over_dedup[1]:.3f}, "
            f"{standard_over_dedup[2]:.3f}]"
        )

        print(
            f"Dedup latency reduction vs KV: "
            f"{dedup_latency_reduction:.2f}%"
        )

        if peak_memory_mib is not None:

            print(
                f"Peak allocated GPU memory: "
                f"{peak_memory_mib:.1f} MiB"
            )

        print(
            f"Correctness: "
            f"{INDEPENDENT_GROUPS}/"
            f"{INDEPENDENT_GROUPS} groups"
        )

        # ----------------------------------------------------
        # Save summary row
        # ----------------------------------------------------

        all_summary_rows.append({

            "batch_size": batch_size,

            "unique_prompts": unique_count,

            "duplicate_prompts": duplicate_count,

            "duplicate_rate_percent": duplicate_rate,

            "independent_groups": INDEPENDENT_GROUPS,

            "timed_repetitions": TIMED_REPETITIONS,

            "standard_median_ms": standard_ms,

            "kv_median_ms": kv_ms,

            "dedup_kv_median_ms": dedup_ms,

            "standard_over_kv": standard_over_kv[0],

            "standard_over_kv_ci95_low": standard_over_kv[1],

            "standard_over_kv_ci95_high": standard_over_kv[2],

            "kv_over_dedup": kv_over_dedup[0],

            "kv_over_dedup_ci95_low": kv_over_dedup[1],

            "kv_over_dedup_ci95_high": kv_over_dedup[2],

            "standard_over_dedup": standard_over_dedup[0],

            "standard_over_dedup_ci95_low": standard_over_dedup[1],

            "standard_over_dedup_ci95_high": standard_over_dedup[2],

            "dedup_latency_reduction_percent": dedup_latency_reduction,

            "peak_allocated_gpu_memory_mib": peak_memory_mib,

            "correctness_groups": INDEPENDENT_GROUPS,
        })

    # ========================================================
    # SAVE CSV FILES
    # ========================================================

    summary_path = (
        output_dir / "threshold_summary.csv"
    )

    clusters_path = (
        output_dir / "cluster_measurements.csv"
    )

    write_csv(
        summary_path,
        all_summary_rows,
    )

    write_csv(
        clusters_path,
        all_cluster_rows,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 72)
    print("FINAL RCT100M DEDUPLICATION THRESHOLD SUMMARY")
    print("=" * 72)

    print(
        f"{'Batch':>5} "
        f"{'Unique':>7} "
        f"{'Dup %':>8} "
        f"{'Standard':>12} "
        f"{'KV':>12} "
        f"{'Dedup+KV':>12} "
        f"{'KV/Dedup':>10}"
    )

    for row in all_summary_rows:

        print(
            f"{row['batch_size']:>5} "
            f"{row['unique_prompts']:>7} "
            f"{row['duplicate_rate_percent']:>7.2f}% "
            f"{row['standard_median_ms']:>10.2f}ms "
            f"{row['kv_median_ms']:>10.2f}ms "
            f"{row['dedup_kv_median_ms']:>10.2f}ms "
            f"{row['kv_over_dedup']:>9.3f}x"
        )

    # --------------------------------------------------------
    # Confidence intervals
    # --------------------------------------------------------

    print()
    print(
        "Paired cluster-bootstrap 95% confidence intervals"
    )

    print(
        "KV / Dedup+KV isolates the additional "
        "deduplication effect."
    )

    for row in all_summary_rows:

        print(
            f"Duplicates {row['duplicate_rate_percent']:.2f}%: "
            f"{row['kv_over_dedup']:.3f}x "
            f"[{row['kv_over_dedup_ci95_low']:.3f}, "
            f"{row['kv_over_dedup_ci95_high']:.3f}]"
        )

    # --------------------------------------------------------
    # Correctness totals
    # --------------------------------------------------------

    total_correctness_checks = (
        len(WORKLOADS)
        * INDEPENDENT_GROUPS
        * TIMED_REPETITIONS
        * len(METHODS)
    )

    total_independent_groups = (
        len(WORKLOADS) * INDEPENDENT_GROUPS
    )

    print()
    print("Correctness totals:")

    print(
        f"{total_correctness_checks} timed method checks "
        f"against standard reference outputs"
    )

    print(
        f"{total_independent_groups} "
        f"independent prompt groups"
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print()
    print("Saved files:")

    print(summary_path.resolve())
    print(clusters_path.resolve())

    print()
    print("===== RCT100M DEDUPLICATION THRESHOLD COMPLETE =====")


if __name__ == "__main__":
    main()
