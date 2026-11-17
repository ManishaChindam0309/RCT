
import csv
import math
import random
import statistics
import sys
import time
from pathlib import Path

import sentencepiece as spm
import torch

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined
from model.rct_100m_adaptive import RCT100MAdaptive


CORPUS_PATH = (
    PROJECT_ROOT / "data" / "realistic"
    / "prompts_holdout_400.txt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT / "data" / "realistic"
    / "rct_expanded.model"
)

OUTPUT_CSV = (
    PROJECT_ROOT / "benchmarks"
    / "rct_249_holdout_results.csv"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT / "benchmarks"
    / "rct_249_holdout_summary.txt"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

BATCH_SIZES = [4, 8, 16]

DUPLICATE_RATES = [
    0.0,
    0.25,
    0.50,
    0.75,
    0.875,
]

THRESHOLDS = [0.25, 0.375, 0.50]

SEEDS = [2491, 2492, 2493]

PROMPT_LENGTH = 32
NEW_TOKENS = 8
REPEATS = 5
CPU_THREADS = 4
TEMPERATURE = 1.0

THEORETICAL_KV_MB_PER_SEQUENCE = 60.0


# ============================================================
# STABLE DEDUPLICATION
# ============================================================

def stable_deduplicate(prompts):
    """
    Deduplicate prompts while preserving the order
    of their first occurrence.

    Returns:
        unique_stable:
            Unique prompts in first-occurrence order.

        inverse_stable:
            Maps each original batch row to its
            corresponding stable unique-prompt row.
    """

    unique, inverse = torch.unique(
        prompts,
        dim=0,
        return_inverse=True,
    )

    first_positions = torch.full(
        (unique.shape[0],),
        prompts.shape[0],
        dtype=torch.long,
        device=prompts.device,
    )

    positions = torch.arange(
        prompts.shape[0],
        device=prompts.device,
    )

    first_positions.scatter_reduce_(
        0,
        inverse,
        positions,
        reduce="amin",
        include_self=True,
    )

    order = torch.argsort(first_positions)

    # CRITICAL FIX:
    # Reorder the unique tensor, not the original prompts.
    unique_stable = unique.index_select(
        0,
        order,
    )

    inverse_order = torch.empty_like(order)

    inverse_order[order] = torch.arange(
        order.shape[0],
        device=order.device,
    )

    inverse_stable = inverse_order[inverse]

    return unique_stable, inverse_stable


# ============================================================
# LOAD HOLDOUT CORPUS
# ============================================================

def load_prompt_tokens():

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(CORPUS_PATH)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(TOKENIZER_PATH)

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        texts = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(texts) != 400:
        raise ValueError(
            f"Expected 400 prompts, found {len(texts)}"
        )

    if len(set(texts)) != 400:
        raise ValueError(
            "Holdout corpus contains duplicate text."
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_PATH)
    )

    encoded = []

    for text in texts:

        ids = tokenizer.encode(
            text,
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

    unique_count = torch.unique(
        tokens,
        dim=0,
    ).shape[0]

    if unique_count != 400:
        raise ValueError(
            "Some prompts have identical token sequences "
            "after truncation and padding."
        )

    return texts, tokens


# ============================================================
# WORKLOAD GENERATION
# ============================================================

def make_workload(
    prompt_pool,
    batch_size,
    duplicate_rate,
    seed,
):

    rng = random.Random(seed)

    duplicate_count = round(
        batch_size * duplicate_rate
    )

    unique_count = batch_size - duplicate_count

    if unique_count < 1:
        raise ValueError(
            "At least one unique prompt is required."
        )

    selected_indices = rng.sample(
        range(prompt_pool.shape[0]),
        unique_count,
    )

    rows = [
        prompt_pool[index]
        for index in selected_indices
    ]

    for index in range(duplicate_count):

        rows.append(
            prompt_pool[
                selected_indices[index % unique_count]
            ]
        )

    rng.shuffle(rows)

    prompts = torch.stack(rows)

    unique_prompts, inverse = stable_deduplicate(
        prompts
    )

    if unique_prompts.shape[0] != unique_count:
        raise AssertionError(
            "Unexpected number of unique prompts."
        )

    actual_duplicate_rate = (
        batch_size - unique_prompts.shape[0]
    ) / batch_size

    # Verify the stable inverse mapping.
    if not torch.equal(
        unique_prompts.index_select(0, inverse),
        prompts,
    ):
        raise AssertionError(
            "Stable deduplication inverse mapping failed."
        )

    return (
        prompts,
        unique_prompts,
        inverse,
        actual_duplicate_rate,
    )


# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def generate_kv(model, prompts, seed):

    torch.manual_seed(seed)

    with torch.inference_mode():

        return model.generate_cached(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=TEMPERATURE,
        )


def generate_combined(model, prompts, seed):

    torch.manual_seed(seed)

    with torch.inference_mode():

        return model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=TEMPERATURE,
            deduplicate=True,
        )


def generate_adaptive(model, prompts, seed):

    torch.manual_seed(seed)

    with torch.inference_mode():

        return model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=TEMPERATURE,
        )


# ============================================================
# TIMING
# ============================================================

def timed_call(function, seed):

    torch.manual_seed(seed)

    start = time.perf_counter()

    with torch.inference_mode():
        function()

    elapsed = time.perf_counter() - start

    return elapsed * 1000.0


# ============================================================
# EXPECTED ADAPTIVE DECISION
# ============================================================

def should_deduplicate(
    batch_size,
    duplicate_count,
    threshold,
):

    required_duplicates = math.ceil(
        threshold * batch_size
    )

    return (
        duplicate_count >= required_duplicates
        and duplicate_count > 0
    )


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 75)
    print("RCT 2.4.9 - INDEPENDENT HOLDOUT VALIDATION")
    print("=" * 75)

    torch.set_num_threads(CPU_THREADS)

    print("\nLoading holdout prompts and tokenizer...")

    texts, prompt_pool = load_prompt_tokens()

    print(f"Text prompts: {len(texts)}")

    print(
        "Unique token sequences:",
        prompt_pool.shape[0],
    )

    print("Prompt length:", PROMPT_LENGTH)

    # --------------------------------------------------------
    # Create valid workload configurations
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

            actual_rate = (
                duplicate_count / batch_size
            )

            workloads.append((
                batch_size,
                unique_count,
                actual_rate,
            ))

    expected_records = (
        len(workloads)
        * len(SEEDS)
        * len(THRESHOLDS)
    )

    print(
        "\nValid workload configurations:",
        len(workloads),
    )

    print("Seeds:", SEEDS)
    print("Thresholds:", THRESHOLDS)
    print("Timing repetitions:", REPEATS)
    print("Expected CSV records:", expected_records)

    # --------------------------------------------------------
    # Initialize models
    # --------------------------------------------------------

    print("\nInitializing RCT 100M...")

    base_model = RCT100M()
    base_model.eval()

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    combined_model = RCT100MCombined(base_model)
    combined_model.eval()

    adaptive_models = {}

    for threshold in THRESHOLDS:

        adaptive_models[threshold] = RCT100MAdaptive(
            base_model,
            duplicate_threshold=threshold,
        )

        adaptive_models[threshold].eval()

    print(
        "Model parameters:",
        f"{base_model.parameter_count():,}",
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    results = []

    correctness = {
        "combined_correct": 0,
        "adaptive_correct": 0,
        "kv_reproducible": 0,
        "decision_correct": 0,
        "total_workloads": 0,
        "total_adaptive": 0,
    }

    workload_number = 0

    # ========================================================
    # WORKLOAD LOOP
    # ========================================================

    for batch_size, unique_count, duplicate_rate in workloads:

        workload_number += 1

        for seed in SEEDS:

            workload_seed = (
                seed
                + batch_size * 100
                + int(duplicate_rate * 1000)
            )

            (
                prompts,
                unique_prompts,
                inverse,
                actual_rate,
            ) = make_workload(
                prompt_pool,
                batch_size,
                duplicate_rate,
                workload_seed,
            )

            duplicate_count = (
                batch_size - unique_prompts.shape[0]
            )

            # ------------------------------------------------
            # KV reference generation
            # ------------------------------------------------

            kv_full = generate_kv(
                kv_model,
                prompts,
                seed,
            )

            kv_full_repeat = generate_kv(
                kv_model,
                prompts,
                seed,
            )

            kv_unique = generate_kv(
                kv_model,
                unique_prompts,
                seed,
            )

            expected_deduplicated = (
                kv_unique.index_select(
                    0,
                    inverse,
                )
            )

            kv_reproducible = torch.equal(
                kv_full,
                kv_full_repeat,
            )

            correctness["total_workloads"] += 1

            correctness["kv_reproducible"] += int(
                kv_reproducible
            )

            # ------------------------------------------------
            # Combined correctness
            # ------------------------------------------------

            combined_output = generate_combined(
                combined_model,
                prompts,
                seed,
            )

            combined_correct = torch.equal(
                combined_output,
                expected_deduplicated,
            )

            correctness["combined_correct"] += int(
                combined_correct
            )

            # ------------------------------------------------
            # Warmup
            # ------------------------------------------------

            generate_kv(
                kv_model,
                prompts,
                seed + 10000,
            )

            generate_combined(
                combined_model,
                prompts,
                seed + 10000,
            )

            for model in adaptive_models.values():

                generate_adaptive(
                    model,
                    prompts,
                    seed + 10000,
                )

            # ------------------------------------------------
            # Timing functions
            # ------------------------------------------------

            methods = {
                "kv": lambda: kv_model.generate_cached(
                    prompts,
                    max_new_tokens=NEW_TOKENS,
                    temperature=TEMPERATURE,
                ),

                "combined": lambda: combined_model.generate(
                    prompts,
                    max_new_tokens=NEW_TOKENS,
                    temperature=TEMPERATURE,
                    deduplicate=True,
                ),
            }

            for threshold, model in adaptive_models.items():

                methods[f"adaptive_{threshold}"] = (
                    lambda model=model: model.generate(
                        prompts,
                        max_new_tokens=NEW_TOKENS,
                        temperature=TEMPERATURE,
                    )
                )

            measurements = {
                name: []
                for name in methods
            }

            # ------------------------------------------------
            # Randomized timing repetitions
            # ------------------------------------------------

            for repetition in range(REPEATS):

                order = list(methods.keys())

                random.Random(
                    workload_seed + repetition
                ).shuffle(order)

                for name in order:

                    elapsed = timed_call(
                        methods[name],
                        seed + repetition,
                    )

                    measurements[name].append(
                        elapsed
                    )

            kv_ms = statistics.median(
                measurements["kv"]
            )

            combined_ms = statistics.median(
                measurements["combined"]
            )

            # ------------------------------------------------
            # Adaptive validation and measurements
            # ------------------------------------------------

            for threshold in THRESHOLDS:

                adaptive_model = adaptive_models[
                    threshold
                ]

                expected_dedup = should_deduplicate(
                    batch_size,
                    duplicate_count,
                    threshold,
                )

                adaptive_output = generate_adaptive(
                    adaptive_model,
                    prompts,
                    seed,
                )

                if expected_dedup:

                    expected_output = (
                        expected_deduplicated
                    )

                    expected_label = "deduplicated"

                else:

                    expected_output = kv_full

                    expected_label = "kv_only"

                adaptive_correct = torch.equal(
                    adaptive_output,
                    expected_output,
                )

                adaptive_stats = (
                    adaptive_model.statistics()
                )

                actual_decision = adaptive_stats[
                    "last_decision"
                ]

                decision_correct = (
                    actual_decision == expected_label
                )

                adaptive_ms = statistics.median(
                    measurements[
                        f"adaptive_{threshold}"
                    ]
                )

                correctness["total_adaptive"] += 1

                correctness["adaptive_correct"] += int(
                    adaptive_correct
                )

                correctness["decision_correct"] += int(
                    decision_correct
                )

                # ------------------------------------------------
                # CSV record
                # ------------------------------------------------

                row = {
                    "seed": seed,
                    "batch_size": batch_size,
                    "unique_prompts": unique_count,
                    "duplicate_count": duplicate_count,
                    "duplicate_rate": actual_rate,
                    "threshold": threshold,
                    "expected_decision": expected_label,
                    "actual_decision": actual_decision,
                    "kv_only_ms": kv_ms,
                    "combined_ms": combined_ms,
                    "adaptive_ms": adaptive_ms,
                    "combined_speedup_vs_kv": (
                        kv_ms / combined_ms
                    ),
                    "adaptive_speedup_vs_kv": (
                        kv_ms / adaptive_ms
                    ),
                    "adaptive_latency_reduction_pct": (
                        1.0 - adaptive_ms / kv_ms
                    ) * 100.0,
                    "theoretical_unique_kv_mb": (
                        unique_count
                        * THEORETICAL_KV_MB_PER_SEQUENCE
                    ),
                    "kv_reproducible": kv_reproducible,
                    "combined_correct": combined_correct,
                    "adaptive_correct": adaptive_correct,
                    "decision_correct": decision_correct,
                }

                results.append(row)

            print(
                f"[{workload_number}/{len(workloads)}] "
                f"Batch={batch_size}, "
                f"duplicates={actual_rate:.1%}, "
                f"seed={seed}, "
                f"KV={kv_ms:.2f} ms, "
                f"Combined={combined_ms:.2f} ms"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
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

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    summary_lines = [
        "RCT 2.4.9 - INDEPENDENT HOLDOUT VALIDATION",
        "",
        f"Holdout prompts: {len(texts)}",
        f"Workload configurations: {len(workloads)}",
        f"Seeds: {SEEDS}",
        f"Thresholds: {THRESHOLDS}",
        f"Timing repetitions: {REPEATS}",
        f"CPU threads: {CPU_THREADS}",
        f"Generated tokens: {NEW_TOKENS}",
        "",
        "THRESHOLD SUMMARY",
        "",
    ]

    print("\n" + "=" * 75)
    print("THRESHOLD SUMMARY")
    print("=" * 75)

    threshold_summaries = []

    for threshold in THRESHOLDS:

        rows = [
            row
            for row in results
            if row["threshold"] == threshold
        ]

        speedups = [
            row["adaptive_speedup_vs_kv"]
            for row in rows
        ]

        reductions = [
            row["adaptive_latency_reduction_pct"]
            for row in rows
        ]

        item = {
            "threshold": threshold,
            "records": len(rows),
            "mean_speedup": statistics.mean(speedups),
            "median_speedup": statistics.median(speedups),
            "mean_reduction": statistics.mean(reductions),
        }

        threshold_summaries.append(item)

        print(
            f"\nThreshold: {threshold:.3f}"
        )

        print(
            f"Mean speedup: "
            f"{item['mean_speedup']:.3f}x"
        )

        print(
            f"Median speedup: "
            f"{item['median_speedup']:.3f}x"
        )

        print(
            f"Mean latency reduction: "
            f"{item['mean_reduction']:.2f}%"
        )

        summary_lines.extend([
            f"Threshold: {threshold:.3f}",
            f"Records: {item['records']}",
            f"Mean speedup: {item['mean_speedup']:.4f}x",
            f"Median speedup: {item['median_speedup']:.4f}x",
            (
                "Mean latency reduction: "
                f"{item['mean_reduction']:.2f}%"
            ),
            "",
        ])

    # ========================================================
    # CORRECTNESS SUMMARY
    # ========================================================

    print("\nCORRECTNESS")
    print("-" * 75)

    print(
        f"Combined: "
        f"{correctness['combined_correct']}/"
        f"{correctness['total_workloads']}"
    )

    print(
        f"Adaptive: "
        f"{correctness['adaptive_correct']}/"
        f"{correctness['total_adaptive']}"
    )

    print(
        f"KV reproducibility: "
        f"{correctness['kv_reproducible']}/"
        f"{correctness['total_workloads']}"
    )

    print(
        f"Decision correctness: "
        f"{correctness['decision_correct']}/"
        f"{correctness['total_adaptive']}"
    )

    summary_lines.extend([
        "CORRECTNESS",
        "",
        (
            f"Combined: {correctness['combined_correct']}/"
            f"{correctness['total_workloads']}"
        ),
        (
            f"Adaptive: {correctness['adaptive_correct']}/"
            f"{correctness['total_adaptive']}"
        ),
        (
            f"KV reproducibility: "
            f"{correctness['kv_reproducible']}/"
            f"{correctness['total_workloads']}"
        ),
        (
            f"Decision correctness: "
            f"{correctness['decision_correct']}/"
            f"{correctness['total_adaptive']}"
        ),
        "",
        "LIMITATIONS",
        "",
        "The model is randomly initialized and untrained.",
        "The holdout corpus uses five shared prompt templates.",
        "This is controlled workload validation, not production traffic.",
        "KV memory values are theoretical full-capacity estimates.",
        "One CPU machine and four CPU threads were used.",
        "",
    ])

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    OUTPUT_SUMMARY.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("RCT 2.4.9 BENCHMARK COMPLETE")
    print("=" * 75)

    print("Records:", len(results))
    print("CSV:", OUTPUT_CSV)
    print("Summary:", OUTPUT_SUMMARY)

    # Do not silently accept failed correctness checks.
    all_correct = (
        correctness["combined_correct"]
        == correctness["total_workloads"]
        and correctness["adaptive_correct"]
        == correctness["total_adaptive"]
        and correctness["kv_reproducible"]
        == correctness["total_workloads"]
        and correctness["decision_correct"]
        == correctness["total_adaptive"]
    )

    if not all_correct:
        print(
            "\nWARNING: Correctness validation failed. "
            "Do not accept speedup results as validated."
        )
    else:
        print(
            "\nSUCCESS: All correctness checks passed."
        )


if __name__ == "__main__":
    main()
