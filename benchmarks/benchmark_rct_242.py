
import csv
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined


# ==================================================
# CONFIGURATION
# ==================================================

THREADS = 4
REPEATS = 15
WARMUP_RUNS = 1
NEW_TOKENS = 8
SEED = 242

torch.set_num_threads(THREADS)
torch.manual_seed(SEED)
random.seed(SEED)

process = psutil.Process()

print("=" * 75)
print("RCT 2.4.2 - EXPANDED WORKLOAD VALIDATION")
print("=" * 75)


# ==================================================
# LOAD MODELS
# ==================================================

print("\nLoading RCT 100M...")

base_model = RCT100M()
base_model.eval()

kv_model = RCT100MKV(base_model)
kv_model.eval()

combined_model = RCT100MCombined(base_model)
combined_model.eval()

print("Parameters:", f"{base_model.parameter_count():,}")
print("CPU threads:", THREADS)
print("Repetitions:", REPEATS)


# ==================================================
# WORKLOAD MATRIX
# ==================================================

workloads = [
    # Batch 1
    {"batch": 1, "seq": 16, "unique": 1},

    # Batch 2
    {"batch": 2, "seq": 16, "unique": 2},
    {"batch": 2, "seq": 16, "unique": 1},

    # Batch 4
    {"batch": 4, "seq": 16, "unique": 4},
    {"batch": 4, "seq": 16, "unique": 3},
    {"batch": 4, "seq": 16, "unique": 2},
    {"batch": 4, "seq": 16, "unique": 1},

    # Batch 8
    {"batch": 8, "seq": 64, "unique": 8},
    {"batch": 8, "seq": 64, "unique": 4},
    {"batch": 8, "seq": 64, "unique": 2},
    {"batch": 8, "seq": 64, "unique": 1},

    # Batch 16
    {"batch": 16, "seq": 128, "unique": 16},
    {"batch": 16, "seq": 128, "unique": 8},
    {"batch": 16, "seq": 128, "unique": 4},
    {"batch": 16, "seq": 128, "unique": 2},
]


# ==================================================
# PROMPT GENERATION
# ==================================================

def create_prompts(batch, sequence, unique_count):

    unique_prompts = torch.randint(
        0,
        base_model.config.vocab_size,
        (unique_count, sequence),
        dtype=torch.long
    )

    indices = torch.arange(batch) % unique_count

    indices = indices[torch.randperm(batch)]

    prompts = unique_prompts[indices].contiguous()

    return prompts


# ==================================================
# GENERATION METHODS
# ==================================================

def run_standard(prompts):

    with torch.inference_mode():

        return base_model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0
        )


def run_kv(prompts):

    with torch.inference_mode():

        return kv_model.generate_cached(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0
        )


def run_combined(prompts):

    with torch.inference_mode():

        return combined_model.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            temperature=1.0,
            deduplicate=True
        )


# ==================================================
# TIMING
# ==================================================

def measure(function, prompts, seed):

    torch.manual_seed(seed)

    start = time.perf_counter()

    output = function(prompts)

    elapsed_ms = (
        time.perf_counter() - start
    ) * 1000

    return elapsed_ms, output


# ==================================================
# VALIDATION
# ==================================================

def validate_shape(output, prompts):

    expected = (
        prompts.size(0),
        prompts.size(1) + NEW_TOKENS
    )

    return output.shape == expected


def validate_prompt_preservation(output, prompts):

    return torch.equal(
        output[:, :prompts.size(1)],
        prompts
    )


def validate_duplicate_outputs(output, prompts):

    for i in range(prompts.size(0)):

        for j in range(i + 1, prompts.size(0)):

            if torch.equal(prompts[i], prompts[j]):

                if not torch.equal(output[i], output[j]):
                    return False

    return True


# ==================================================
# MEMORY HELPERS
# ==================================================

def rss_mib():

    return process.memory_info().rss / (1024 ** 2)


# ==================================================
# BENCHMARK
# ==================================================

methods = {
    "standard": run_standard,
    "kv": run_kv,
    "combined": run_combined
}

results = []

for workload_id, workload in enumerate(workloads, start=1):

    batch = workload["batch"]
    sequence = workload["seq"]
    unique_count = workload["unique"]

    duplicate_rate = (
        1 - unique_count / batch
    ) * 100

    print("\n" + "-" * 75)

    print(
        f"Workload {workload_id}/{len(workloads)}"
    )

    print(
        f"Batch={batch}, Sequence={sequence}, "
        f"Unique={unique_count}, "
        f"Duplicates={duplicate_rate:.1f}%"
    )

    prompts = create_prompts(
        batch,
        sequence,
        unique_count
    )

    # Stable unique prompt ordering.
    unique_prompts, inverse_indices = (
        combined_model._stable_deduplicate(prompts)
    )

    # ----------------------------------------------
    # Correctness validation
    # ----------------------------------------------

    torch.manual_seed(777)

    standard_output = run_standard(prompts)

    torch.manual_seed(777)

    kv_output = run_kv(prompts)

    # Standard and KV should match with the same
    # batch order and random seed.
    if not torch.equal(standard_output, kv_output):

        raise RuntimeError(
            "Standard/KV mismatch in workload "
            f"{workload_id}"
        )

    torch.manual_seed(777)

    unique_kv_output = run_kv(unique_prompts)

    torch.manual_seed(777)

    combined_output = run_combined(prompts)

    expected_combined = unique_kv_output.index_select(
        0,
        inverse_indices
    )

    combined_correct = torch.equal(
        combined_output,
        expected_combined
    )

    if not combined_correct:

        raise RuntimeError(
            "Combined/KV unique-prompt mismatch in workload "
            f"{workload_id}"
        )

    correctness = (
        validate_shape(standard_output, prompts)
        and validate_prompt_preservation(standard_output, prompts)
        and validate_duplicate_outputs(combined_output, prompts)
        and combined_correct
    )

    if not correctness:

        raise RuntimeError(
            "Output validation failed in workload "
            f"{workload_id}"
        )

    print("Correctness: True")

    del standard_output
    del kv_output
    del unique_kv_output
    del combined_output
    del expected_combined

    # ----------------------------------------------
    # Warm-up
    # ----------------------------------------------

    for warmup in range(WARMUP_RUNS):

        warmup_seed = 10000 + workload_id + warmup

        for function in methods.values():

            measure(
                function,
                prompts,
                warmup_seed
            )

    # ----------------------------------------------
    # Memory observation
    # ----------------------------------------------

    rss_before = rss_mib()

    # ----------------------------------------------
    # Randomized paired timing
    # ----------------------------------------------

    timings = {
        "standard": [],
        "kv": [],
        "combined": []
    }

    paired_kv = []
    paired_combined = []

    for repeat in range(REPEATS):

        order = list(methods.keys())
        random.shuffle(order)

        pair = {}

        current_seed = (
            SEED + workload_id * 1000 + repeat
        )

        for name in order:

            elapsed, _ = measure(
                methods[name],
                prompts,
                current_seed
            )

            timings[name].append(elapsed)
            pair[name] = elapsed

        paired_kv.append(
            pair["standard"] / pair["kv"]
        )

        paired_combined.append(
            pair["standard"] / pair["combined"]
        )

    rss_after = rss_mib()

    # ----------------------------------------------
    # Calculate results
    # ----------------------------------------------

    medians = {
        name: statistics.median(values)
        for name, values in timings.items()
    }

    kv_speedup = (
        medians["standard"] / medians["kv"]
    )

    combined_speedup = (
        medians["standard"] / medians["combined"]
    )

    combined_vs_kv = (
        medians["kv"] / medians["combined"]
    )

    kv_reduction = (
        1 - medians["kv"] / medians["standard"]
    ) * 100

    combined_reduction = (
        1 - medians["combined"] / medians["standard"]
    ) * 100

    # Theoretical FP32 KV allocation for all unique prompts.
    cache_bytes = (
        2
        * base_model.config.num_layers
        * unique_count
        * base_model.config.max_seq_len
        * base_model.config.hidden_size
        * 4
    )

    cache_mib = cache_bytes / (1024 ** 2)

    result = {

        "batch_size": batch,
        "sequence_length": sequence,
        "unique_prompts": unique_count,
        "duplicate_rate_percent": duplicate_rate,

        "generated_tokens": NEW_TOKENS,
        "correctness": correctness,

        "standard_median_ms": medians["standard"],
        "kv_median_ms": medians["kv"],
        "combined_median_ms": medians["combined"],

        "kv_speedup": kv_speedup,
        "combined_speedup": combined_speedup,
        "combined_vs_kv_speedup": combined_vs_kv,

        "kv_latency_reduction_percent": kv_reduction,
        "combined_latency_reduction_percent": combined_reduction,

        "paired_kv_speedup": statistics.median(paired_kv),
        "paired_combined_speedup": statistics.median(paired_combined),

        "kv_cache_theoretical_mib": cache_mib,

        "process_rss_before_mib": rss_before,
        "process_rss_after_mib": rss_after,
        "process_rss_change_mib": rss_after - rss_before
    }

    results.append(result)

    print(
        f"Standard: {medians['standard']:.2f} ms"
    )

    print(
        f"KV: {medians['kv']:.2f} ms"
    )

    print(
        f"Combined: {medians['combined']:.2f} ms"
    )

    print(
        f"KV speedup: {kv_speedup:.3f}x"
    )

    print(
        f"Combined speedup: {combined_speedup:.3f}x"
    )

    print(
        f"Combined vs KV: {combined_vs_kv:.3f}x"
    )

    print(
        f"Theoretical KV memory: {cache_mib:.2f} MiB"
    )


# ==================================================
# SAVE RESULTS
# ==================================================

output_path = (
    PROJECT_ROOT
    / "benchmarks"
    / "rct_242_results.csv"
)

with output_path.open(
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 75)
print("RCT 2.4.2 FINAL SUMMARY")
print("=" * 75)

print("Workloads:", len(results))

print(
    "Correctness:",
    sum(r["correctness"] for r in results),
    "/",
    len(results)
)

print(
    "Average KV speedup:",
    f"{statistics.mean(r['kv_speedup'] for r in results):.3f}x"
)

print(
    "Average combined speedup:",
    f"{statistics.mean(r['combined_speedup'] for r in results):.3f}x"
)

print(
    "Average combined vs KV:",
    f"{statistics.mean(r['combined_vs_kv_speedup'] for r in results):.3f}x"
)

print(
    "Average combined latency reduction:",
    f"{statistics.mean(r['combined_latency_reduction_percent'] for r in results):.2f}%"
)

print(
    "\nResults saved:",
    output_path
)

print("\nRCT 2.4.2 BENCHMARK COMPLETED")
