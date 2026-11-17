
import csv
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_adaptive import RCT100MAdaptive


# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = ROOT / "data" / "realistic"

TOKENIZER_FILE = DATA_DIR / "rct_expanded.model"
PROMPT_FILE = DATA_DIR / "prompts_expanded.txt"

OUTPUT_FILE = ROOT / "benchmarks" / "rct_246_results.csv"

BATCH_SIZES = [4, 8, 16]
DUPLICATE_RATES = [0.0, 0.25, 0.50, 0.75, 0.875]

SEEDS = [2461, 2462, 2463]

PROMPT_LENGTH = 32
NEW_TOKENS = 8
REPEATS = 5
CPU_THREADS = 4

THRESHOLD = 0.25


# ==================================================
# STABLE DEDUPLICATION
# ==================================================

def stable_deduplicate(prompts):
    unique_prompts, inverse = torch.unique(
        prompts,
        dim=0,
        return_inverse=True,
    )

    unique_count = unique_prompts.size(0)
    batch_size = prompts.size(0)

    first_positions = torch.full(
        (unique_count,),
        batch_size,
        dtype=torch.long,
        device=prompts.device,
    )

    first_positions.scatter_reduce_(
        0,
        inverse,
        torch.arange(
            batch_size,
            device=prompts.device,
        ),
        reduce="amin",
        include_self=True,
    )

    order = torch.argsort(first_positions)

    unique_prompts = unique_prompts.index_select(
        0,
        order,
    )

    old_to_new = torch.empty_like(order)

    old_to_new[order] = torch.arange(
        unique_count,
        device=prompts.device,
    )

    inverse = old_to_new[inverse]

    return unique_prompts, inverse


# ==================================================
# LOAD EXPANDED TEXT CORPUS
# ==================================================

def load_prompts():
    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(TOKENIZER_FILE)

    if not PROMPT_FILE.exists():
        raise FileNotFoundError(PROMPT_FILE)

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_FILE)
    )

    texts = [
        line.strip()
        for line in PROMPT_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(texts) != 200:
        raise ValueError(
            f"Expected 200 prompts, found {len(texts)}"
        )

    if len(set(texts)) != 200:
        raise ValueError(
            "Expanded corpus contains duplicate text prompts."
        )

    prompts = []

    for text in texts:
        token_ids = tokenizer.encode(
            text,
            out_type=int,
        )

        token_ids = token_ids[:PROMPT_LENGTH]

        token_ids += [
            0
        ] * (PROMPT_LENGTH - len(token_ids))

        prompts.append(
            torch.tensor(
                token_ids,
                dtype=torch.long,
            )
        )

    result = torch.stack(prompts)

    if result.min().item() < 0:
        raise ValueError("Negative token ID found.")

    if result.max().item() >= 8192:
        raise ValueError("Token ID exceeds model vocabulary.")

    print("\nEXPANDED CORPUS")
    print("-" * 55)
    print("Text prompts:", len(texts))
    print("Unique text prompts:", len(set(texts)))
    print("Tokenizer vocabulary:", tokenizer.get_piece_size())
    print("Prompt tensor shape:", tuple(result.shape))
    print("Prompt length:", PROMPT_LENGTH)

    return result


# ==================================================
# BUILD CONTROLLED WORKLOAD
# ==================================================

def make_workload(all_prompts, batch_size, requested_rate):
    duplicate_count = round(
        batch_size * requested_rate
    )

    unique_count = batch_size - duplicate_count

    if unique_count < 1:
        raise ValueError(
            "At least one unique prompt is required."
        )

    selected = all_prompts[:unique_count]

    rows = list(selected)

    for index in range(duplicate_count):
        rows.append(
            selected[index % unique_count]
        )

    prompts = torch.stack(rows)

    actual_rate = duplicate_count / batch_size

    return prompts, unique_count, actual_rate


# ==================================================
# GENERATION FUNCTIONS
# ==================================================

def run_kv(kv_model, prompts):
    return kv_model.generate_cached(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0,
    )


def run_combined(kv_model, prompts):
    unique, inverse = stable_deduplicate(prompts)

    unique_output = run_kv(
        kv_model,
        unique,
    )

    return unique_output.index_select(
        0,
        inverse,
    )


def run_adaptive(adaptive_model, prompts):
    return adaptive_model.generate(
        prompts,
        max_new_tokens=NEW_TOKENS,
        temperature=1.0,
    )


# ==================================================
# TIMING
# ==================================================

def timed_call(function, seed):
    torch.manual_seed(seed)

    start = time.perf_counter()

    with torch.inference_mode():
        output = function()

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return output, elapsed


def median(values):
    return statistics.median(values)


# ==================================================
# CORRECTNESS
# ==================================================

def check_correctness(
    kv_model,
    adaptive_model,
    prompts,
    unique_prompts,
    inverse,
    seed,
    batch_size,
    unique_count,
):
    reference_unique, _ = timed_call(
        lambda: run_kv(
            kv_model,
            unique_prompts,
        ),
        seed,
    )

    reference = reference_unique.index_select(
        0,
        inverse,
    )

    combined_output, _ = timed_call(
        lambda: run_combined(
            kv_model,
            prompts,
        ),
        seed,
    )

    combined_correct = torch.equal(
        reference,
        combined_output,
    )

    adaptive_output, _ = timed_call(
        lambda: run_adaptive(
            adaptive_model,
            prompts,
        ),
        seed,
    )

    adaptive_correct = torch.equal(
        reference,
        adaptive_output,
    )

    kv_first, _ = timed_call(
        lambda: run_kv(
            kv_model,
            prompts,
        ),
        seed,
    )

    kv_second, _ = timed_call(
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

    duplicate_count = batch_size - unique_count

    expected_dedup = (
        duplicate_count > 0
        and duplicate_count >= math.ceil(
            THRESHOLD * batch_size
        )
    )

    expected_decision = (
        "deduplicated"
        if expected_dedup
        else "kv_only"
    )

    actual_decision = (
        adaptive_model.statistics()["last_decision"]
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


# ==================================================
# ONE WORKLOAD / ONE SEED
# ==================================================

def benchmark_workload(
    kv_model,
    adaptive_model,
    prompts,
    batch_size,
    unique_count,
    duplicate_rate,
    seed,
    workload_number,
    total_workloads,
):
    unique_prompts, inverse = stable_deduplicate(
        prompts
    )

    print("\n" + "=" * 70)
    print(
        f"Workload {workload_number}/{total_workloads} "
        f"| Batch={batch_size} "
        f"| Unique={unique_count} "
        f"| Duplicates={duplicate_rate:.1%} "
        f"| Seed={seed}"
    )

    checks = check_correctness(
        kv_model=kv_model,
        adaptive_model=adaptive_model,
        prompts=prompts,
        unique_prompts=unique_prompts,
        inverse=inverse,
        seed=seed,
        batch_size=batch_size,
        unique_count=unique_count,
    )

    print("Combined correctness:", checks["combined_correct"])
    print("Adaptive correctness:", checks["adaptive_correct"])
    print("KV reproducibility:", checks["kv_reproducible"])
    print("Expected decision:", checks["expected_decision"])
    print("Actual decision:", checks["actual_decision"])
    print("Decision correctness:", checks["decision_correct"])

    methods = {
        "kv_only": lambda: run_kv(
            kv_model,
            prompts,
        ),
        "combined": lambda: run_combined(
            kv_model,
            prompts,
        ),
        "adaptive": lambda: run_adaptive(
            adaptive_model,
            prompts,
        ),
    }

    # Warm-up.
    for function in methods.values():
        timed_call(function, seed)

    timings = {
        name: []
        for name in methods
    }

    rng = random.Random(seed)

    for repeat in range(REPEATS):
        order = list(methods.keys())
        rng.shuffle(order)

        for name in order:
            _, elapsed = timed_call(
                methods[name],
                seed + repeat + 1,
            )

            timings[name].append(elapsed)

    kv_ms = median(timings["kv_only"])
    combined_ms = median(timings["combined"])
    adaptive_ms = median(timings["adaptive"])

    speedup_vs_kv = kv_ms / adaptive_ms
    speedup_vs_combined = combined_ms / adaptive_ms

    reduction = (
        (kv_ms - adaptive_ms) / kv_ms
    ) * 100

    print("\nTIMING")
    print("-" * 50)
    print(f"KV-only:       {kv_ms:.2f} ms")
    print(f"Combined:      {combined_ms:.2f} ms")
    print(f"Adaptive:      {adaptive_ms:.2f} ms")
    print(f"Adaptive/KV:   {speedup_vs_kv:.3f}x")
    print(f"Adaptive/combined: {speedup_vs_combined:.3f}x")
    print(f"Latency saved: {reduction:.2f}%")

    # Theoretical full-capacity FP32 KV cache.
    kv_memory_mib = 60.0 * batch_size
    unique_memory_mib = 60.0 * unique_count

    return {
        "seed": seed,
        "batch_size": batch_size,
        "unique_prompts": unique_count,
        "duplicate_rate": duplicate_rate,
        "expected_decision": checks["expected_decision"],
        "actual_decision": checks["actual_decision"],
        "combined_correct": checks["combined_correct"],
        "adaptive_correct": checks["adaptive_correct"],
        "kv_reproducible": checks["kv_reproducible"],
        "decision_correct": checks["decision_correct"],
        "kv_only_ms": kv_ms,
        "combined_ms": combined_ms,
        "adaptive_ms": adaptive_ms,
        "adaptive_speedup_vs_kv": speedup_vs_kv,
        "adaptive_speedup_vs_combined": speedup_vs_combined,
        "latency_reduction_percent": reduction,
        "kv_cache_memory_mib": kv_memory_mib,
        "unique_cache_memory_mib": unique_memory_mib,
    }


# ==================================================
# SUMMARY HELPERS
# ==================================================

def print_group_summary(results, key):
    groups = sorted(
        set(row[key] for row in results)
    )

    for value in groups:
        group = [
            row for row in results
            if row[key] == value
        ]

        speedup = statistics.mean(
            row["adaptive_speedup_vs_kv"]
            for row in group
        )

        reduction = statistics.mean(
            row["latency_reduction_percent"]
            for row in group
        )

        print(
            f"{str(value):>8} | "
            f"{speedup:>8.3f}x | "
            f"{reduction:>8.2f}% | "
            f"{len(group):>3}"
        )


# ==================================================
# MAIN
# ==================================================

def main():
    torch.set_num_threads(CPU_THREADS)

    print("RCT 2.4.6 EXPANDED REALISTIC BENCHMARK")
    print("=" * 70)
    print("CPU threads:", CPU_THREADS)
    print("Generation tokens:", NEW_TOKENS)
    print("Repeats:", REPEATS)
    print("Seeds:", SEEDS)
    print("Adaptive threshold:", THRESHOLD)

    all_prompts = load_prompts()

    print("\nInitializing 100M model...")

    base_model = RCT100M()
    base_model.eval()

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    adaptive_model = RCT100MAdaptive(
        base_model,
        duplicate_threshold=THRESHOLD,
    )
    adaptive_model.eval()

    print(
        "Parameters:",
        f"{base_model.parameter_count():,}",
    )

    # ----------------------------------------------
    # Build only feasible workloads.
    # ----------------------------------------------

    workloads = []

    for batch_size in BATCH_SIZES:
        for requested_rate in DUPLICATE_RATES:

            maximum_rate = (
                batch_size - 1
            ) / batch_size

            if requested_rate > maximum_rate:
                print(
                    f"Skipping impossible workload: "
                    f"batch={batch_size}, "
                    f"requested_duplicates={requested_rate:.1%}"
                )
                continue

            prompts, unique_count, actual_rate = make_workload(
                all_prompts,
                batch_size,
                requested_rate,
            )

            workloads.append(
                (
                    batch_size,
                    prompts,
                    unique_count,
                    actual_rate,
                )
            )

    expected_records = len(workloads) * len(SEEDS)

    print("\nVALIDATION PLAN")
    print("-" * 50)
    print("Valid workload configurations:", len(workloads))
    print("Random seeds:", len(SEEDS))
    print("Expected records:", expected_records)

    results = []
    workload_number = 0

    for seed in SEEDS:
        print("\n\n")
        print("#" * 70)
        print("RANDOM SEED:", seed)
        print("#" * 70)

        for (
            batch_size,
            prompts,
            unique_count,
            actual_rate,
        ) in workloads:

            workload_number += 1

            row = benchmark_workload(
                kv_model=kv_model,
                adaptive_model=adaptive_model,
                prompts=prompts,
                batch_size=batch_size,
                unique_count=unique_count,
                duplicate_rate=actual_rate,
                seed=seed,
                workload_number=workload_number,
                total_workloads=expected_records,
            )

            results.append(row)

    # ----------------------------------------------
    # Save CSV.
    # ----------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
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

    # ----------------------------------------------
    # Final summary.
    # ----------------------------------------------

    print("\n\n")
    print("=" * 70)
    print("RCT 2.4.6 FINAL SUMMARY")
    print("=" * 70)

    total = len(results)

    print("Total records:", total)
    print("Expected records:", expected_records)

    checks = [
        "combined_correct",
        "adaptive_correct",
        "kv_reproducible",
        "decision_correct",
    ]

    for check in checks:
        passed = sum(
            bool(row[check])
            for row in results
        )

        print(f"{check}: {passed}/{total}")

    print(
        "Mean adaptive speedup:",
        f"{statistics.mean(r['adaptive_speedup_vs_kv'] for r in results):.3f}x",
    )

    print(
        "Median adaptive speedup:",
        f"{statistics.median(r['adaptive_speedup_vs_kv'] for r in results):.3f}x",
    )

    print(
        "Mean latency reduction:",
        f"{statistics.mean(r['latency_reduction_percent'] for r in results):.2f}%",
    )

    print("\nRESULTS BY DUPLICATE RATE")
    print("Rate     | Speedup  | Saved    | Runs")
    print("-" * 50)

    print_group_summary(
        results,
        "duplicate_rate",
    )

    print("\nRESULTS BY BATCH SIZE")
    print("Batch    | Speedup  | Saved    | Runs")
    print("-" * 50)

    print_group_summary(
        results,
        "batch_size",
    )

    print("\nCSV:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
