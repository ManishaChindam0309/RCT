
import csv
import math
import random
import statistics
import sys
import time
from pathlib import Path

# --------------------------------------------------
# Project import path
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sentencepiece as spm
import torch

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_adaptive import RCT100MAdaptive


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = ROOT / "data" / "realistic"
OUTPUT_FILE = ROOT / "benchmarks" / "rct_245_results.csv"

BATCH_SIZES = [4, 8]
DUPLICATE_RATES = [0.0, 0.25, 0.50, 0.75]

PROMPT_LENGTH = 32
NEW_TOKENS = 8
REPEATS = 5
CPU_THREADS = 4
THRESHOLD = 0.25

SEED = 2450


# --------------------------------------------------
# Stable exact deduplication
# --------------------------------------------------

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


# --------------------------------------------------
# Load realistic text prompts
# --------------------------------------------------

def load_real_prompts():
    model_file = DATA_DIR / "rct_realistic.model"
    text_file = DATA_DIR / "prompts.txt"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Tokenizer model not found: {model_file}"
        )

    if not text_file.exists():
        raise FileNotFoundError(
            f"Prompt corpus not found: {text_file}"
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(model_file)
    )

    text_prompts = [
        line.strip()
        for line in text_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    unique_text = list(dict.fromkeys(text_prompts))

    encoded = [
        tokenizer.encode(text, out_type=int)
        for text in unique_text
    ]

    prepared = []

    for tokens in encoded:
        tokens = tokens[:PROMPT_LENGTH]

        tokens = tokens + [
            0
        ] * (PROMPT_LENGTH - len(tokens))

        prepared.append(
            torch.tensor(tokens, dtype=torch.long)
        )

    print("\nREAL TEXT DATA")
    print("-" * 50)
    print("Total text prompts:", len(text_prompts))
    print("Unique text prompts:", len(unique_text))
    print("Tokenizer vocabulary:", tokenizer.get_piece_size())

    return prepared


# --------------------------------------------------
# Controlled duplicate workloads
# --------------------------------------------------

def make_workload(unique_prompts, batch_size, duplicate_rate):
    duplicate_count = round(
        batch_size * duplicate_rate
    )

    unique_count = batch_size - duplicate_count

    if unique_count < 1:
        raise ValueError(
            "Every workload needs at least one unique prompt."
        )

    if unique_count > len(unique_prompts):
        raise ValueError(
            "Not enough unique prompts for this workload."
        )

    selected = unique_prompts[:unique_count]

    rows = list(selected)

    for index in range(duplicate_count):
        rows.append(
            selected[index % unique_count]
        )

    return torch.stack(rows), unique_count


# --------------------------------------------------
# Generation methods
# --------------------------------------------------

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


# --------------------------------------------------
# Timing
# --------------------------------------------------

def timed_call(function, seed):
    torch.manual_seed(seed)

    start = time.perf_counter()

    with torch.inference_mode():
        output = function()

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return output, elapsed_ms


def median(values):
    return statistics.median(values)


# --------------------------------------------------
# Correctness verification
# --------------------------------------------------

def verify_correctness(
    kv_model,
    adaptive_model,
    prompts,
    reference_unique,
    inverse,
    workload_index,
    batch_size,
    unique_count,
):
    correctness_seed = SEED + workload_index

    # Generate reference output from stable unique prompts.
    reference_output, _ = timed_call(
        lambda: run_kv(
            kv_model,
            reference_unique,
        ),
        correctness_seed,
    )

    reference = reference_output.index_select(
        0,
        inverse,
    )

    # Verify combined implementation.
    combined_check, _ = timed_call(
        lambda: run_combined(
            kv_model,
            prompts,
        ),
        correctness_seed,
    )

    combined_correct = torch.equal(
        combined_check,
        reference,
    )

    # Verify adaptive implementation.
    adaptive_check, _ = timed_call(
        lambda: run_adaptive(
            adaptive_model,
            prompts,
        ),
        correctness_seed,
    )

    adaptive_correct = torch.equal(
        adaptive_check,
        reference,
    )

    # Verify KV-only reproducibility.
    kv_check_1, _ = timed_call(
        lambda: run_kv(
            kv_model,
            prompts,
        ),
        correctness_seed,
    )

    kv_check_2, _ = timed_call(
        lambda: run_kv(
            kv_model,
            prompts,
        ),
        correctness_seed,
    )

    kv_reproducible = torch.equal(
        kv_check_1,
        kv_check_2,
    )

    # Verify adaptive decision.
    duplicate_count = batch_size - unique_count

    expected_dedup = (
        duplicate_count
        >= math.ceil(THRESHOLD * batch_size)
        and duplicate_count > 0
    )

    adaptive_stats = adaptive_model.statistics()

    actual_decision = adaptive_stats["last_decision"]

    expected_decision = (
        "deduplicated"
        if expected_dedup
        else "kv_only"
    )

    decision_correct = (
        actual_decision == expected_decision
    )

    return {
        "reference": reference,
        "combined_correct": combined_correct,
        "adaptive_correct": adaptive_correct,
        "kv_reproducible": kv_reproducible,
        "expected_dedup": expected_dedup,
        "actual_decision": actual_decision,
        "decision_correct": decision_correct,
    }


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main():
    torch.set_num_threads(CPU_THREADS)

    torch.manual_seed(SEED)
    random.seed(SEED)

    print("\nLoading realistic workloads...")

    unique_prompts = load_real_prompts()

    print("\nInitializing 100M model...")

    base_model = RCT100M()
    base_model.eval()

    # KV-only and combined implementation.
    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    # Adaptive model receives the original RCT100M.
    # It manages its own KV wrapper internally.
    adaptive_model = RCT100MAdaptive(
        base_model,
        duplicate_threshold=THRESHOLD,
    )
    adaptive_model.eval()

    print("Parameters:", f"{base_model.parameter_count():,}")
    print("CPU threads:", CPU_THREADS)
    print("Adaptive threshold:", THRESHOLD)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    workloads = [
        (batch_size, rate)
        for batch_size in BATCH_SIZES
        for rate in DUPLICATE_RATES
    ]

    for workload_index, (
        batch_size,
        duplicate_rate,
    ) in enumerate(workloads):

        prompts, unique_count = make_workload(
            unique_prompts,
            batch_size,
            duplicate_rate,
        )

        actual_duplicate_rate = (
            batch_size - unique_count
        ) / batch_size

        print("\n" + "=" * 65)
        print(
            f"Workload {workload_index + 1}/{len(workloads)} | "
            f"Batch={batch_size} | "
            f"Unique={unique_count} | "
            f"Duplicates={actual_duplicate_rate:.1%}"
        )

        reference_unique, inverse = stable_deduplicate(
            prompts
        )

        # --------------------------------------------------
        # Correctness checks are separate from timing.
        # --------------------------------------------------

        checks = verify_correctness(
            kv_model=kv_model,
            adaptive_model=adaptive_model,
            prompts=prompts,
            reference_unique=reference_unique,
            inverse=inverse,
            workload_index=workload_index,
            batch_size=batch_size,
            unique_count=unique_count,
        )

        print(
            "Combined correctness:",
            checks["combined_correct"],
        )

        print(
            "Adaptive correctness:",
            checks["adaptive_correct"],
        )

        print(
            "KV reproducibility:",
            checks["kv_reproducible"],
        )

        print(
            "Expected decision:",
            "deduplicated"
            if checks["expected_dedup"]
            else "kv_only",
        )

        print(
            "Actual decision:",
            checks["actual_decision"],
        )

        print(
            "Decision correctness:",
            checks["decision_correct"],
        )

        # --------------------------------------------------
        # Timed measurements
        # --------------------------------------------------

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

        timings = {
            name: [] for name in methods
        }

        # Warm-up.
        for name, function in methods.items():
            timed_call(
                function,
                SEED + workload_index,
            )

        # Randomized repeated measurements.
        for repeat in range(REPEATS):
            order = list(methods.keys())
            random.shuffle(order)

            for name in order:
                _, elapsed = timed_call(
                    methods[name],
                    SEED + workload_index + repeat,
                )

                timings[name].append(elapsed)

        kv_ms = median(timings["kv_only"])
        combined_ms = median(timings["combined"])
        adaptive_ms = median(timings["adaptive"])

        kv_speedup = kv_ms / adaptive_ms
        combined_speedup = combined_ms / adaptive_ms

        latency_reduction = (
            (kv_ms - adaptive_ms) / kv_ms * 100
        )

        # Theoretical FP32 KV-cache allocation.
        # 60 MiB per sequence for this architecture.
        kv_memory_mib = 60.0 * batch_size
        unique_memory_mib = 60.0 * unique_count

        row = {
            "batch_size": batch_size,
            "unique_prompts": unique_count,
            "duplicate_rate": actual_duplicate_rate,
            "expected_dedup": checks["expected_dedup"],
            "actual_decision": checks["actual_decision"],
            "decision_correct": checks["decision_correct"],
            "kv_only_ms": kv_ms,
            "combined_ms": combined_ms,
            "adaptive_ms": adaptive_ms,
            "adaptive_speedup_vs_kv": kv_speedup,
            "adaptive_speedup_vs_combined": combined_speedup,
            "latency_reduction_percent": latency_reduction,
            "combined_correct": checks["combined_correct"],
            "adaptive_correct": checks["adaptive_correct"],
            "kv_reproducible": checks["kv_reproducible"],
            "kv_cache_memory_mib": kv_memory_mib,
            "unique_cache_memory_mib": unique_memory_mib,
        }

        results.append(row)

        print("\nTIMING RESULTS")
        print("-" * 50)
        print(f"KV-only:       {kv_ms:.2f} ms")
        print(f"Combined:      {combined_ms:.2f} ms")
        print(f"Adaptive:      {adaptive_ms:.2f} ms")
        print(f"Adaptive/KV:   {kv_speedup:.3f}x")
        print(f"Latency saved: {latency_reduction:.2f}%")

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("RCT 2.4.5 SUMMARY")
    print("=" * 65)

    print("Workloads:", len(results))

    print(
        "Combined correctness:",
        sum(r["combined_correct"] for r in results),
        "/",
        len(results),
    )

    print(
        "Adaptive correctness:",
        sum(r["adaptive_correct"] for r in results),
        "/",
        len(results),
    )

    print(
        "KV reproducibility:",
        sum(r["kv_reproducible"] for r in results),
        "/",
        len(results),
    )

    print(
        "Decision correctness:",
        sum(r["decision_correct"] for r in results),
        "/",
        len(results),
    )

    print(
        "Mean adaptive speedup:",
        f"{statistics.mean(r['adaptive_speedup_vs_kv'] for r in results):.3f}x",
    )

    print(
        "Mean latency reduction:",
        f"{statistics.mean(r['latency_reduction_percent'] for r in results):.2f}%",
    )

    print("CSV:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
