
import os
import sys

# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Imports
# ============================================================

import csv
import math
import random
import statistics
import time

import psutil
import sentencepiece as spm
import torch

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined
from model.rct_100m_adaptive import RCT100MAdaptive


# ============================================================
# Configuration
# ============================================================

BATCH_SIZES = [4, 8, 16]

DUPLICATE_RATES = [
    0.0,
    0.25,
    0.50,
    0.75,
    0.875,
]

THRESHOLDS = [
    0.25,
    0.375,
    0.50,
]

PROMPT_LENGTH = 32
NEW_TOKENS = 8

REPEATS = 5
CPU_THREADS = 4

SEEDS = [2461, 2462, 2463]

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "realistic",
)

TOKENIZER_PATH = os.path.join(
    DATA_DIR,
    "rct_expanded.model",
)

CORPUS_PATH = os.path.join(
    DATA_DIR,
    "prompts_expanded.txt",
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "benchmarks",
    "rct_247_results.csv",
)


# ============================================================
# Environment
# ============================================================

torch.set_num_threads(CPU_THREADS)
torch.manual_seed(2470)
random.seed(2470)

process = psutil.Process(os.getpid())


def get_rss_mib():
    return process.memory_info().rss / (1024 ** 2)


# ============================================================
# Load realistic prompt corpus
# ============================================================

def load_prompts():

    with open(
        CORPUS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        prompts = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(prompts) != 200:
        raise ValueError(
            f"Expected 200 prompts, found {len(prompts)}"
        )

    if len(set(prompts)) != 200:
        raise ValueError(
            "Expanded corpus must contain 200 unique prompts"
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=TOKENIZER_PATH
    )

    encoded = []

    for prompt in prompts:

        ids = tokenizer.encode(
            prompt,
            out_type=int,
        )

        ids = ids[:PROMPT_LENGTH]

        if len(ids) < PROMPT_LENGTH:
            ids += [0] * (
                PROMPT_LENGTH - len(ids)
            )

        encoded.append(ids)

    tokens = torch.tensor(
        encoded,
        dtype=torch.long,
    )

    if tokens.max().item() >= 8192:
        raise ValueError(
            "Tokenizer produced an ID outside model vocabulary"
        )

    print(f"Loaded prompts: {len(prompts)}")
    print(f"Encoded tensor: {tuple(tokens.shape)}")
    print(
        f"Tokenizer vocabulary: {tokenizer.vocab_size()}"
    )

    return tokens


# ============================================================
# Stable deduplication
# ============================================================

def stable_deduplicate(prompts):

    unique_sorted, inverse_sorted = torch.unique(
        prompts,
        dim=0,
        sorted=True,
        return_inverse=True,
    )

    unique_count = unique_sorted.size(0)

    first_positions = torch.full(
        (unique_count,),
        prompts.size(0),
        dtype=torch.long,
        device=prompts.device,
    )

    positions = torch.arange(
        prompts.size(0),
        device=prompts.device,
    )

    first_positions.scatter_reduce_(
        0,
        inverse_sorted,
        positions,
        reduce="amin",
        include_self=True,
    )

    order = torch.argsort(first_positions)

    unique_prompts = unique_sorted.index_select(
        0,
        order,
    )

    sorted_to_stable = torch.empty_like(order)

    sorted_to_stable[order] = torch.arange(
        unique_count,
        device=prompts.device,
    )

    inverse = sorted_to_stable.index_select(
        0,
        inverse_sorted,
    )

    return unique_prompts, inverse


# ============================================================
# Create benchmark workloads
# ============================================================

def make_workload(
    corpus,
    batch_size,
    duplicate_rate,
    seed,
):

    generator = torch.Generator()
    generator.manual_seed(seed)

    duplicate_count = round(
        batch_size * duplicate_rate
    )

    unique_count = (
        batch_size - duplicate_count
    )

    if unique_count < 1:
        raise ValueError(
            "At least one unique prompt is required"
        )

    indices = torch.randperm(
        corpus.size(0),
        generator=generator,
    )[:unique_count]

    unique_prompts = corpus.index_select(
        0,
        indices,
    )

    prompts = [
        row for row in unique_prompts
    ]

    for i in range(duplicate_count):

        source_index = i % unique_count

        prompts.append(
            unique_prompts[source_index]
        )

    prompts = torch.stack(prompts)

    order = torch.randperm(
        batch_size,
        generator=generator,
    )

    prompts = prompts.index_select(
        0,
        order,
    )

    actual_unique, inverse = stable_deduplicate(
        prompts
    )

    actual_unique_count = actual_unique.size(0)

    actual_duplicate_rate = (
        batch_size - actual_unique_count
    ) / batch_size

    return (
        prompts,
        actual_unique,
        inverse,
        actual_unique_count,
        actual_duplicate_rate,
    )


# ============================================================
# Generation functions
# ============================================================

def run_kv(kv_model, prompts):

    return kv_model.generate_cached(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0,
    )


def run_combined(combined_model, prompts):

    return combined_model.generate(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0,
        deduplicate=True,
    )


def run_adaptive(adaptive_model, prompts):

    return adaptive_model.generate(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0,
    )


# ============================================================
# Timed execution
# ============================================================

def timed_call(function, seed):

    torch.manual_seed(seed)

    with torch.inference_mode():

        start = time.perf_counter()

        output = function()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

    return (
        output,
        elapsed_ms,
        get_rss_mib(),
    )


# ============================================================
# Correctness validation
# ============================================================

def check_correctness(
    kv_model,
    combined_model,
    adaptive_model,
    prompts,
    unique_prompts,
    inverse,
    seed,
    batch_size,
    unique_count,
    threshold,
):

    duplicate_count = (
        batch_size - unique_count
    )

    expected_dedup = (
        duplicate_count > 0
        and duplicate_count >= math.ceil(
            threshold * batch_size
        )
    )

    expected_decision = (
        "deduplicated"
        if expected_dedup
        else "kv_only"
    )

    # --------------------------------------------------------
    # Unique-prompt KV reference
    # --------------------------------------------------------

    reference_unique, _, _ = timed_call(
        lambda: run_kv(
            kv_model,
            unique_prompts,
        ),
        seed,
    )

    reference_deduplicated = (
        reference_unique.index_select(
            0,
            inverse,
        )
    )

    # --------------------------------------------------------
    # Combined model correctness
    # --------------------------------------------------------

    combined_output, _, _ = timed_call(
        lambda: run_combined(
            combined_model,
            prompts,
        ),
        seed,
    )

    combined_correct = torch.equal(
        reference_deduplicated,
        combined_output,
    )

    # --------------------------------------------------------
    # Adaptive model correctness
    #
    # Deduplicated decision:
    # Compare with unique prompts restored to original order.
    #
    # KV-only decision:
    # Compare with the original full prompt batch.
    # --------------------------------------------------------

    if expected_dedup:

        adaptive_reference = (
            reference_deduplicated
        )

    else:

        adaptive_reference, _, _ = timed_call(
            lambda: run_kv(
                kv_model,
                prompts,
            ),
            seed,
        )

    adaptive_output, _, _ = timed_call(
        lambda: run_adaptive(
            adaptive_model,
            prompts,
        ),
        seed,
    )

    adaptive_correct = torch.equal(
        adaptive_reference,
        adaptive_output,
    )

    # --------------------------------------------------------
    # KV reproducibility
    # --------------------------------------------------------

    kv_first, _, _ = timed_call(
        lambda: run_kv(
            kv_model,
            prompts,
        ),
        seed,
    )

    kv_second, _, _ = timed_call(
        lambda: run_kv(
            kv_model,
            prompts,
        ),
        seed,
    )

    kv_reproducible = torch.equal(
        kv_first,
        kv_second,
    )

    # --------------------------------------------------------
    # Adaptive decision correctness
    # --------------------------------------------------------

    actual_decision = (
        adaptive_model.statistics()[
            "last_decision"
        ]
    )

    decision_correct = (
        actual_decision == expected_decision
    )

    return {
        "combined_correct": combined_correct,
        "adaptive_correct": adaptive_correct,
        "kv_reproducible": kv_reproducible,
        "decision_correct": decision_correct,
        "expected_decision": expected_decision,
        "actual_decision": actual_decision,
    }


# ============================================================
# Warmup
# ============================================================

def warmup(
    kv_model,
    combined_model,
    adaptive_models,
    prompts,
):

    with torch.inference_mode():

        run_kv(
            kv_model,
            prompts,
        )

        run_combined(
            combined_model,
            prompts,
        )

        for model in adaptive_models.values():

            run_adaptive(
                model,
                prompts,
            )


# ============================================================
# Benchmark one workload
# ============================================================

def benchmark_workload(
    kv_model,
    combined_model,
    adaptive_model,
    prompts,
    unique_prompts,
    inverse,
    seed,
    batch_size,
    unique_count,
    duplicate_rate,
    threshold,
):

    correctness = check_correctness(
        kv_model=kv_model,
        combined_model=combined_model,
        adaptive_model=adaptive_model,
        prompts=prompts,
        unique_prompts=unique_prompts,
        inverse=inverse,
        seed=seed,
        batch_size=batch_size,
        unique_count=unique_count,
        threshold=threshold,
    )

    correctness_keys = [
        "combined_correct",
        "adaptive_correct",
        "kv_reproducible",
        "decision_correct",
    ]

    if not all(
        correctness[key]
        for key in correctness_keys
    ):

        raise AssertionError(
            f"Correctness validation failed: {correctness}"
        )

    # Warmup before timing.

    with torch.inference_mode():

        run_kv(
            kv_model,
            prompts,
        )

        run_combined(
            combined_model,
            prompts,
        )

        run_adaptive(
            adaptive_model,
            prompts,
        )

    kv_times = []
    combined_times = []
    adaptive_times = []

    rss_samples = [
        get_rss_mib()
    ]

    methods = [
        "kv",
        "combined",
        "adaptive",
    ]

    # --------------------------------------------------------
    # Randomized benchmark execution
    # --------------------------------------------------------

    for repeat in range(REPEATS):

        order = methods.copy()

        random.Random(
            seed * 1000
            + repeat
            + int(threshold * 1000)
        ).shuffle(order)

        for method in order:

            call_seed = seed + repeat + 1

            if method == "kv":

                _, elapsed, rss = timed_call(
                    lambda: run_kv(
                        kv_model,
                        prompts,
                    ),
                    call_seed,
                )

                kv_times.append(elapsed)

            elif method == "combined":

                _, elapsed, rss = timed_call(
                    lambda: run_combined(
                        combined_model,
                        prompts,
                    ),
                    call_seed,
                )

                combined_times.append(elapsed)

            else:

                _, elapsed, rss = timed_call(
                    lambda: run_adaptive(
                        adaptive_model,
                        prompts,
                    ),
                    call_seed,
                )

                adaptive_times.append(elapsed)

            rss_samples.append(rss)

    # --------------------------------------------------------
    # Median latency
    # --------------------------------------------------------

    kv_median = statistics.median(
        kv_times
    )

    combined_median = statistics.median(
        combined_times
    )

    adaptive_median = statistics.median(
        adaptive_times
    )

    speedup_vs_kv = (
        kv_median / adaptive_median
    )

    speedup_vs_combined = (
        combined_median / adaptive_median
    )

    latency_reduction = (
        (kv_median - adaptive_median)
        / kv_median
    ) * 100

    # --------------------------------------------------------
    # Paired comparisons
    # --------------------------------------------------------

    paired_speedups = []

    paired_reductions = []

    for kv_time, adaptive_time in zip(
        kv_times,
        adaptive_times,
    ):

        paired_speedups.append(
            kv_time / adaptive_time
        )

        paired_reductions.append(
            (
                (kv_time - adaptive_time)
                / kv_time
            ) * 100
        )

    median_paired_speedup = (
        statistics.median(
            paired_speedups
        )
    )

    median_paired_reduction = (
        statistics.median(
            paired_reductions
        )
    )

    rss_after = get_rss_mib()

    # --------------------------------------------------------
    # Memory estimates
    # --------------------------------------------------------

    kv_cache_memory_mib = (
        60.0 * batch_size
    )

    unique_cache_memory_mib = (
        60.0 * unique_count
    )

    return {

        "seed": seed,

        "batch_size": batch_size,

        "unique_prompts": unique_count,

        "duplicate_rate": duplicate_rate,

        "threshold": threshold,

        "expected_decision": (
            correctness["expected_decision"]
        ),

        "actual_decision": (
            correctness["actual_decision"]
        ),

        "combined_correct": (
            correctness["combined_correct"]
        ),

        "adaptive_correct": (
            correctness["adaptive_correct"]
        ),

        "kv_reproducible": (
            correctness["kv_reproducible"]
        ),

        "decision_correct": (
            correctness["decision_correct"]
        ),

        "kv_only_ms": kv_median,

        "combined_ms": combined_median,

        "adaptive_ms": adaptive_median,

        "adaptive_speedup_vs_kv": (
            speedup_vs_kv
        ),

        "adaptive_speedup_vs_combined": (
            speedup_vs_combined
        ),

        "median_paired_speedup": (
            median_paired_speedup
        ),

        "median_paired_reduction_percent": (
            median_paired_reduction
        ),

        "latency_reduction_percent": (
            latency_reduction
        ),

        "rss_before_mib": rss_samples[0],

        "rss_after_mib": rss_after,

        "rss_change_mib": (
            rss_after - rss_samples[0]
        ),

        "max_sampled_rss_mib": max(
            rss_samples
        ),

        "kv_cache_memory_mib": (
            kv_cache_memory_mib
        ),

        "unique_cache_memory_mib": (
            unique_cache_memory_mib
        ),
    }


# ============================================================
# Main benchmark
# ============================================================

def main():

    print("=" * 75)
    print("RCT 2.4.7 - Adaptive Threshold Validation")
    print("=" * 75)

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"CPU threads: {CPU_THREADS}"
    )

    print(
        f"Repeats: {REPEATS}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Thresholds: {THRESHOLDS}"
    )

    # --------------------------------------------------------
    # Load prompts
    # --------------------------------------------------------

    corpus = load_prompts()

    # --------------------------------------------------------
    # Create base model
    # --------------------------------------------------------

    print("\nCreating base 100M model...")

    base_model = RCT100M()
    base_model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in base_model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: {parameter_count:,}"
    )

    # --------------------------------------------------------
    # Create separate KV model
    # --------------------------------------------------------

    kv_model = RCT100MKV(
        base_model
    )

    kv_model.eval()

    # --------------------------------------------------------
    # IMPORTANT FIX:
    #
    # RCT100MCombined receives the original RCT100M model.
    # Do NOT pass kv_model here.
    #
    # The combined wrapper manages its KV implementation.
    # --------------------------------------------------------

    combined_model = RCT100MCombined(
        base_model
    )

    combined_model.eval()

    # --------------------------------------------------------
    # Create adaptive models
    # --------------------------------------------------------

    adaptive_models = {}

    for threshold in THRESHOLDS:

        adaptive_models[threshold] = (
            RCT100MAdaptive(
                base_model,
                duplicate_threshold=threshold,
            )
        )

        adaptive_models[threshold].eval()

    print(
        f"Initial RSS: {get_rss_mib():.2f} MiB"
    )

    # --------------------------------------------------------
    # Generate valid workload configurations
    # --------------------------------------------------------

    workloads = []

    for batch_size in BATCH_SIZES:

        for duplicate_rate in DUPLICATE_RATES:

            duplicate_count = round(
                batch_size * duplicate_rate
            )

            unique_count = (
                batch_size - duplicate_count
            )

            if unique_count < 1:
                continue

            workloads.append(
                (
                    batch_size,
                    duplicate_rate,
                )
            )

    expected_records = (
        len(workloads)
        * len(SEEDS)
        * len(THRESHOLDS)
    )

    print(
        f"\nValid workloads: {len(workloads)}"
    )

    print(
        f"Expected records: {expected_records}"
    )

    print("\nStarting benchmark...\n")

    rows = []

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    for seed in SEEDS:

        for batch_size, duplicate_rate in workloads:

            (
                prompts,
                unique_prompts,
                inverse,
                unique_count,
                actual_duplicate_rate,
            ) = make_workload(
                corpus,
                batch_size,
                duplicate_rate,
                seed + batch_size * 100,
            )

            print(
                f"\nSeed {seed} | "
                f"Batch {batch_size} | "
                f"Unique {unique_count} | "
                f"Duplicates {actual_duplicate_rate:.1%}"
            )

            warmup(
                kv_model,
                combined_model,
                adaptive_models,
                prompts,
            )

            for threshold in THRESHOLDS:

                adaptive_model = (
                    adaptive_models[threshold]
                )

                adaptive_model.reset_statistics()

                print(
                    f"  Threshold {threshold:.3f}...",
                    end=" ",
                    flush=True,
                )

                row = benchmark_workload(
                    kv_model=kv_model,
                    combined_model=combined_model,
                    adaptive_model=adaptive_model,
                    prompts=prompts,
                    unique_prompts=unique_prompts,
                    inverse=inverse,
                    seed=seed,
                    batch_size=batch_size,
                    unique_count=unique_count,
                    duplicate_rate=actual_duplicate_rate,
                    threshold=threshold,
                )

                rows.append(row)

                print(
                    f"Decision={row['actual_decision']} | "
                    f"KV={row['kv_only_ms']:.2f}ms | "
                    f"Adaptive={row['adaptive_ms']:.2f}ms | "
                    f"Speedup={row['adaptive_speedup_vs_kv']:.3f}x | "
                    f"Correct=True"
                )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True,
    )

    with open(
        OUTPUT_CSV,
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

    # --------------------------------------------------------
    # Final correctness summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("RCT 2.4.7 BENCHMARK SUMMARY")
    print("=" * 75)

    correctness_fields = [
        "combined_correct",
        "adaptive_correct",
        "kv_reproducible",
        "decision_correct",
    ]

    all_passed = True

    for field in correctness_fields:

        passed = sum(
            bool(row[field])
            for row in rows
        )

        print(
            f"{field}: {passed}/{len(rows)}"
        )

        if passed != len(rows):
            all_passed = False

    if all_passed:

        print(
            "\nAll correctness checks passed."
        )

    else:

        print(
            "\nWARNING: Correctness validation failed. "
            "Do not select a threshold yet."
        )

    # --------------------------------------------------------
    # Threshold comparison
    # --------------------------------------------------------

    print("\nThreshold comparison:")

    for threshold in THRESHOLDS:

        threshold_rows = [
            row
            for row in rows
            if row["threshold"] == threshold
        ]

        mean_speedup = statistics.mean(
            row["adaptive_speedup_vs_kv"]
            for row in threshold_rows
        )

        median_speedup = statistics.median(
            row["adaptive_speedup_vs_kv"]
            for row in threshold_rows
        )

        mean_reduction = statistics.mean(
            row["latency_reduction_percent"]
            for row in threshold_rows
        )

        mean_paired_speedup = statistics.mean(
            row["median_paired_speedup"]
            for row in threshold_rows
        )

        dedup_count = sum(
            row["actual_decision"] == "deduplicated"
            for row in threshold_rows
        )

        kv_only_count = sum(
            row["actual_decision"] == "kv_only"
            for row in threshold_rows
        )

        print(
            f"\nThreshold: {threshold:.3f}"
        )

        print(
            f"  Mean speedup: {mean_speedup:.3f}x"
        )

        print(
            f"  Median speedup: {median_speedup:.3f}x"
        )

        print(
            f"  Mean latency reduction: "
            f"{mean_reduction:.2f}%"
        )

        print(
            f"  Mean paired speedup: "
            f"{mean_paired_speedup:.3f}x"
        )

        print(
            f"  Deduplicated decisions: {dedup_count}"
        )

        print(
            f"  KV-only decisions: {kv_only_count}"
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)

    print(
        f"CSV saved: {OUTPUT_CSV}"
    )

    print(
        f"Final RSS: {get_rss_mib():.2f} MiB"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
