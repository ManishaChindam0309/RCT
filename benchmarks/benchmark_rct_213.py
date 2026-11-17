
import os
import sys
import time
import random
import statistics
import argparse
import gc
import csv
import math

import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from model.transformer import RCTModel
from model.prompt_deduplication import PromptDeduplicator


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 2000
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
MAX_SEQ_LEN = 256

BATCH_SIZES = [4, 8, 16]
SEQUENCE_LENGTHS = [64, 128, 256]

DUPLICATE_RATES = [
    0.0,
    0.25,
    0.50,
    0.75,
    0.875
]

PROMPT_LENGTH_FRACTIONS = [
    0.25,
    0.50,
    0.75,
    1.00
]

WARMUP_RUNS = 3
DEFAULT_RUNS = 10
DEFAULT_SEEDS = 2

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "benchmarks",
    "rct_213_results.csv"
)


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=MAX_SEQ_LEN
    )

    model.eval()

    return model


# ============================================================
# CREATE REALISTIC MIXED BATCH
# ============================================================

def create_batch(
    batch_size,
    sequence_length,
    duplicate_rate,
    seed
):

    if not 0.0 <= duplicate_rate < 1.0:
        raise ValueError(
            "duplicate_rate must be between 0 and 1."
        )

    rng = random.Random(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if duplicate_rate == 0.0:

        unique_count = batch_size

    else:

        unique_count = max(
            1,
            round(batch_size * (1.0 - duplicate_rate))
        )

    unique_prompts = []
    unique_signatures = set()
    active_lengths = []

    possible_lengths = sorted(set(
        max(
            8,
            min(
                sequence_length,
                int(sequence_length * fraction)
            )
        )
        for fraction in PROMPT_LENGTH_FRACTIONS
    ))

    for _ in range(unique_count):

        prompt_length = rng.choice(
            possible_lengths
        )

        while True:

            active_tokens = torch.randint(
                1,
                VOCAB_SIZE,
                (prompt_length,),
                generator=generator,
                dtype=torch.long
            )

            signature = tuple(
                active_tokens.tolist()
            )

            if signature not in unique_signatures:

                unique_signatures.add(signature)

                break

        prompt = torch.zeros(
            sequence_length,
            dtype=torch.long
        )

        prompt[:prompt_length] = active_tokens

        unique_prompts.append(prompt)

        active_lengths.append(prompt_length)

    unique_prompts = torch.stack(
        unique_prompts
    )

    indices = list(range(unique_count))

    while len(indices) < batch_size:

        indices.append(
            rng.randrange(unique_count)
        )

    rng.shuffle(indices)

    tokens = unique_prompts[
        torch.tensor(indices, dtype=torch.long)
    ]

    actual_unique = torch.unique(
        tokens,
        dim=0
    ).size(0)

    actual_duplicate_rate = (
        batch_size - actual_unique
    ) / batch_size

    actual_lengths = [
        active_lengths[index]
        for index in indices
    ]

    return {
        "tokens": tokens,
        "actual_duplicate_rate": actual_duplicate_rate,
        "actual_unique": actual_unique,
        "active_lengths": actual_lengths,
        "unique_lengths": active_lengths
    }


# ============================================================
# CACHE CONTROL
# ============================================================

def clear_caches(model):

    for block in model.blocks:
        block.ffn.invalidate_cache()


# ============================================================
# STANDARD INFERENCE
# ============================================================

@torch.inference_mode()
def standard_inference(model, tokens):

    clear_caches(model)

    start = time.perf_counter()

    output = model(
        tokens,
        prompt_mode="standard",
        ffn_mode="standard"
    )

    elapsed = time.perf_counter() - start

    return output, elapsed


# ============================================================
# STANDALONE PROMPT DEDUPLICATION
# ============================================================

@torch.inference_mode()
def standalone_dedup(
    model,
    tokens,
    deduplicator
):

    clear_caches(model)

    start = time.perf_counter()

    unique_tokens, inverse_indices = (
        deduplicator.deduplicate(tokens)
    )

    unique_outputs = model(
        unique_tokens,
        ffn_mode="standard"
    )

    output = deduplicator.restore(
        unique_outputs,
        inverse_indices
    )

    elapsed = time.perf_counter() - start

    return output, elapsed


# ============================================================
# INTEGRATED ADAPTIVE DEDUPLICATION
# ============================================================

@torch.inference_mode()
def integrated_adaptive(model, tokens):

    clear_caches(model)

    start = time.perf_counter()

    output = model(
        tokens,
        prompt_mode="adaptive",
        ffn_mode="standard"
    )

    elapsed = time.perf_counter() - start

    return output, elapsed


# ============================================================
# CORRECTNESS
# ============================================================

def check_correctness(
    model,
    tokens,
    deduplicator
):

    reference, _ = standard_inference(
        model,
        tokens
    )

    standalone, _ = standalone_dedup(
        model,
        tokens,
        deduplicator
    )

    integrated, _ = integrated_adaptive(
        model,
        tokens
    )

    standalone_ok = torch.allclose(
        reference,
        standalone,
        rtol=1e-4,
        atol=1e-5
    )

    integrated_ok = torch.allclose(
        reference,
        integrated,
        rtol=1e-4,
        atol=1e-5
    )

    max_standalone_diff = (
        reference - standalone
    ).abs().max().item()

    max_integrated_diff = (
        reference - integrated
    ).abs().max().item()

    return {
        "standalone": standalone_ok,
        "integrated": integrated_ok,
        "max_standalone_diff": max_standalone_diff,
        "max_integrated_diff": max_integrated_diff
    }


# ============================================================
# MEASUREMENT
# ============================================================

def measure(
    model,
    tokens,
    mode,
    deduplicator
):

    if mode == "standard":

        _, elapsed = standard_inference(
            model,
            tokens
        )

    elif mode == "standalone":

        _, elapsed = standalone_dedup(
            model,
            tokens,
            deduplicator
        )

    elif mode == "integrated":

        _, elapsed = integrated_adaptive(
            model,
            tokens
        )

    else:

        raise ValueError(
            "Unknown benchmark mode."
        )

    return elapsed * 1000


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, percentile_value):

    sorted_values = sorted(values)

    index = max(
        0,
        math.ceil(
            percentile_value * len(sorted_values)
        ) - 1
    )

    index = min(
        index,
        len(sorted_values) - 1
    )

    return sorted_values[index]


# ============================================================
# BENCHMARK ONE WORKLOAD
# ============================================================

def benchmark_case(
    model,
    tokens,
    actual_duplicate_rate,
    runs,
    deduplicator
):

    correctness = check_correctness(
        model,
        tokens,
        deduplicator
    )

    if not correctness["standalone"]:
        raise RuntimeError(
            "Standalone deduplication correctness failure."
        )

    if not correctness["integrated"]:
        raise RuntimeError(
            "Integrated deduplication correctness failure."
        )

    modes = [
        "standard",
        "standalone",
        "integrated"
    ]

    measurements = {
        mode: [] for mode in modes
    }

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    for _ in range(WARMUP_RUNS):

        shuffled = modes.copy()

        random.shuffle(shuffled)

        for mode in shuffled:

            measure(
                model,
                tokens,
                mode,
                deduplicator
            )

    # --------------------------------------------------------
    # RANDOMIZED PAIRED MEASUREMENTS
    # --------------------------------------------------------

    for _ in range(runs):

        shuffled = modes.copy()

        random.shuffle(shuffled)

        round_times = {}

        for mode in shuffled:

            elapsed = measure(
                model,
                tokens,
                mode,
                deduplicator
            )

            measurements[mode].append(elapsed)

            round_times[mode] = elapsed

    # --------------------------------------------------------
    # MEDIAN LATENCY
    # --------------------------------------------------------

    medians = {
        mode: statistics.median(values)
        for mode, values in measurements.items()
    }

    # --------------------------------------------------------
    # P95 LATENCY
    # --------------------------------------------------------

    p95 = {
        mode: percentile(values, 0.95)
        for mode, values in measurements.items()
    }

    # --------------------------------------------------------
    # PAIRED SPEEDUPS
    # --------------------------------------------------------

    paired_speedups = [
        standard_time / integrated_time
        for standard_time, integrated_time in zip(
            measurements["standard"],
            measurements["integrated"]
        )
    ]

    paired_standalone_speedups = [
        standard_time / standalone_time
        for standard_time, standalone_time in zip(
            measurements["standard"],
            measurements["standalone"]
        )
    ]

    paired_integrated_speedup = statistics.median(
        paired_speedups
    )

    paired_standalone_speedup = statistics.median(
        paired_standalone_speedups
    )

    # --------------------------------------------------------
    # RATIO OF MEDIANS
    # --------------------------------------------------------

    integrated_speedup = (
        medians["standard"] /
        medians["integrated"]
    )

    standalone_speedup = (
        medians["standard"] /
        medians["standalone"]
    )

    # --------------------------------------------------------
    # LATENCY REDUCTION
    # --------------------------------------------------------

    integrated_reduction = (
        1.0 -
        medians["integrated"] /
        medians["standard"]
    ) * 100

    standalone_reduction = (
        1.0 -
        medians["standalone"] /
        medians["standard"]
    ) * 100

    # --------------------------------------------------------
    # ADAPTIVE STATISTICS
    # --------------------------------------------------------

    model.reset_prompt_dedup_statistics()

    integrated_adaptive(
        model,
        tokens
    )

    stats = model.prompt_dedup_statistics()

    # --------------------------------------------------------
    # TOKEN STORAGE ESTIMATE
    # --------------------------------------------------------

    input_memory = (
        tokens.numel() *
        tokens.element_size()
    )

    unique_tokens = torch.unique(
        tokens,
        dim=0
    )

    unique_memory = (
        unique_tokens.numel() *
        unique_tokens.element_size()
    )

    return {
        "correctness": correctness,
        "duplicate_rate": actual_duplicate_rate,
        "medians": medians,
        "p95": p95,
        "speedups": {
            "standalone": standalone_speedup,
            "integrated": integrated_speedup
        },
        "paired_speedups": {
            "standalone": paired_standalone_speedup,
            "integrated": paired_integrated_speedup
        },
        "reductions": {
            "standalone": standalone_reduction,
            "integrated": integrated_reduction
        },
        "paired_integrated_speedups": paired_speedups,
        "stats": stats,
        "input_memory": input_memory,
        "unique_memory": unique_memory
    }


# ============================================================
# CSV EXPORT
# ============================================================

def save_csv(results):

    os.makedirs(
        os.path.dirname(CSV_PATH),
        exist_ok=True
    )

    fieldnames = [
        "seed",
        "batch",
        "sequence",
        "requested_duplicate_rate",
        "actual_duplicate_rate",
        "unique_prompts",
        "minimum_active_length",
        "maximum_active_length",
        "average_active_length",
        "correctness",
        "max_integrated_difference",
        "standard_median_ms",
        "standalone_median_ms",
        "integrated_median_ms",
        "standard_p95_ms",
        "standalone_p95_ms",
        "integrated_p95_ms",
        "integrated_speedup_ratio_of_medians",
        "integrated_paired_speedup",
        "integrated_latency_reduction_percent",
        "input_token_memory_bytes",
        "unique_token_memory_bytes"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            writer.writerow({
                "seed": result["seed"],
                "batch": result["batch"],
                "sequence": result["sequence"],
                "requested_duplicate_rate": (
                    result["requested_duplicate_rate"]
                ),
                "actual_duplicate_rate": (
                    result["duplicate_rate"]
                ),
                "unique_prompts": result["unique_prompts"],
                "minimum_active_length": (
                    result["minimum_active_length"]
                ),
                "maximum_active_length": (
                    result["maximum_active_length"]
                ),
                "average_active_length": (
                    result["average_active_length"]
                ),
                "correctness": result["correctness"],
                "max_integrated_difference": (
                    result["max_integrated_difference"]
                ),
                "standard_median_ms": (
                    result["medians"]["standard"]
                ),
                "standalone_median_ms": (
                    result["medians"]["standalone"]
                ),
                "integrated_median_ms": (
                    result["medians"]["integrated"]
                ),
                "standard_p95_ms": (
                    result["p95"]["standard"]
                ),
                "standalone_p95_ms": (
                    result["p95"]["standalone"]
                ),
                "integrated_p95_ms": (
                    result["p95"]["integrated"]
                ),
                "integrated_speedup_ratio_of_medians": (
                    result["speedups"]["integrated"]
                ),
                "integrated_paired_speedup": (
                    result["paired_speedups"]["integrated"]
                ),
                "integrated_latency_reduction_percent": (
                    result["reductions"]["integrated"]
                ),
                "input_token_memory_bytes": (
                    result["input_memory"]
                ),
                "unique_token_memory_bytes": (
                    result["unique_memory"]
                )
            })


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_summary(results):

    print("\n" + "=" * 125)

    print("RCT 2.1.3 FINAL RESULTS")

    print("=" * 125)

    print(
        f"{'Seed':>5} "
        f"{'Batch':>5} "
        f"{'Seq':>5} "
        f"{'Dup':>8} "
        f"{'Std ms':>10} "
        f"{'Int ms':>10} "
        f"{'Ratio':>9} "
        f"{'Paired':>9} "
        f"{'Reduction':>11}"
    )

    for result in results:

        print(
            f"{result['seed']:>5} "
            f"{result['batch']:>5} "
            f"{result['sequence']:>5} "
            f"{result['duplicate_rate']:>7.1%} "
            f"{result['medians']['standard']:>10.3f} "
            f"{result['medians']['integrated']:>10.3f} "
            f"{result['speedups']['integrated']:>8.3f}x "
            f"{result['paired_speedups']['integrated']:>8.3f}x "
            f"{result['reductions']['integrated']:>9.2f}%"
        )

    # --------------------------------------------------------
    # GROUP BY ACTUAL DUPLICATE RATE
    # --------------------------------------------------------

    print("\nAverage performance by actual duplicate rate:")

    actual_rates = sorted(set(
        round(result["duplicate_rate"], 4)
        for result in results
    ))

    for rate in actual_rates:

        matching = [
            result
            for result in results
            if round(result["duplicate_rate"], 4) == rate
        ]

        ratio_average = statistics.mean(
            result["speedups"]["integrated"]
            for result in matching
        )

        paired_average = statistics.mean(
            result["paired_speedups"]["integrated"]
            for result in matching
        )

        reduction_average = statistics.mean(
            result["reductions"]["integrated"]
            for result in matching
        )

        print(
            f"{rate:.1%}: "
            f"ratio-of-medians={ratio_average:.3f}x, "
            f"paired={paired_average:.3f}x, "
            f"reduction={reduction_average:.2f}%, "
            f"experiments={len(matching)}"
        )

    # --------------------------------------------------------
    # CORRECTNESS SUMMARY
    # --------------------------------------------------------

    correct_count = sum(
        result["correctness"]
        for result in results
    )

    print(
        "\nCorrect experiments:",
        correct_count,
        "/",
        len(results)
    )

    # --------------------------------------------------------
    # OVERALL SPEEDUPS
    # --------------------------------------------------------

    print(
        "Average ratio-of-medians speedup:",
        f"{statistics.mean(result['speedups']['integrated'] for result in results):.3f}x"
    )

    print(
        "Average paired speedup:",
        f"{statistics.mean(result['paired_speedups']['integrated'] for result in results):.3f}x"
    )

    print(
        "Average latency reduction:",
        f"{statistics.mean(result['reductions']['integrated'] for result in results):.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "RCT 2.1.3 realistic mixed-workload benchmark"
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS
    )

    parser.add_argument(
        "--quick",
        action="store_true"
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    if args.seeds < 1:
        parser.error("--seeds must be positive")

    runs = 3 if args.quick else args.runs

    seeds = 1 if args.quick else args.seeds

    # --------------------------------------------------------
    # CPU SETUP
    # --------------------------------------------------------

    torch.set_num_threads(4)

    try:

        torch.set_num_interop_threads(1)

    except RuntimeError:

        pass

    print("=" * 115)

    print("RCT 2.1.3 REALISTIC MIXED-WORKLOAD VALIDATION")

    print("=" * 115)

    print("Device: CPU")
    print("CPU threads:", torch.get_num_threads())
    print("Measurement runs:", runs)
    print("Random seeds:", seeds)
    print("Batch sizes:", BATCH_SIZES)
    print("Sequence lengths:", SEQUENCE_LENGTHS)
    print("Duplicate rates:", DUPLICATE_RATES)
    print("Prompt length fractions:", PROMPT_LENGTH_FRACTIONS)

    total_per_seed = (
        len(BATCH_SIZES)
        * len(SEQUENCE_LENGTHS)
        * len(DUPLICATE_RATES)
    )

    total = total_per_seed * seeds

    all_results = []

    experiment = 0

    # --------------------------------------------------------
    # MULTIPLE RANDOM SEEDS
    # --------------------------------------------------------

    for seed_index in range(seeds):

        seed = 213 + seed_index

        random.seed(seed)
        torch.manual_seed(seed)

        print("\n" + "#" * 115)

        print(
            f"SEED {seed_index + 1}/{seeds}: {seed}"
        )

        print("#" * 115)

        model = create_model()

        deduplicator = PromptDeduplicator()

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print(
            "Model parameters:",
            f"{parameter_count:,}"
        )

        # ----------------------------------------------------
        # WORKLOAD MATRIX
        # ----------------------------------------------------

        for sequence_length in SEQUENCE_LENGTHS:

            for batch_size in BATCH_SIZES:

                for duplicate_rate in DUPLICATE_RATES:

                    experiment += 1

                    workload_seed = (
                        seed * 100000
                        + batch_size * 1000
                        + sequence_length * 10
                        + int(duplicate_rate * 1000)
                    )

                    batch_data = create_batch(
                        batch_size,
                        sequence_length,
                        duplicate_rate,
                        workload_seed
                    )

                    tokens = batch_data["tokens"]

                    actual_rate = (
                        batch_data["actual_duplicate_rate"]
                    )

                    print("\n" + "-" * 115)

                    print(
                        f"Experiment {experiment}/{total}: "
                        f"seed={seed}, "
                        f"batch={batch_size}, "
                        f"sequence={sequence_length}, "
                        f"requested_duplicates={duplicate_rate:.1%}"
                    )

                    print(
                        "Actual duplicate rate:",
                        f"{actual_rate:.2%}"
                    )

                    print(
                        "Unique prompts:",
                        batch_data["actual_unique"]
                    )

                    print(
                        "Active prompt lengths:",
                        batch_data["active_lengths"]
                    )

                    result = benchmark_case(
                        model,
                        tokens,
                        actual_rate,
                        runs,
                        deduplicator
                    )

                    active_lengths = (
                        batch_data["active_lengths"]
                    )

                    # ------------------------------------------------
                    # REPORT RESULTS
                    # ------------------------------------------------

                    print(
                        "Correctness:",
                        result["correctness"]["standalone"],
                        result["correctness"]["integrated"]
                    )

                    print(
                        "Maximum integrated difference:",
                        f"{result['correctness']['max_integrated_diff']:.8f}"
                    )

                    print(
                        "Standard median:",
                        f"{result['medians']['standard']:.4f} ms"
                    )

                    print(
                        "Standalone median:",
                        f"{result['medians']['standalone']:.4f} ms"
                    )

                    print(
                        "Integrated median:",
                        f"{result['medians']['integrated']:.4f} ms"
                    )

                    print(
                        "Integrated P95:",
                        f"{result['p95']['integrated']:.4f} ms"
                    )

                    print(
                        "Ratio-of-medians speedup:",
                        f"{result['speedups']['integrated']:.3f}x"
                    )

                    print(
                        "Paired median speedup:",
                        f"{result['paired_speedups']['integrated']:.3f}x"
                    )

                    print(
                        "Latency reduction:",
                        f"{result['reductions']['integrated']:.2f}%"
                    )

                    print(
                        "Input token storage:",
                        f"{result['input_memory'] / 1024:.2f} KiB"
                    )

                    print(
                        "Unique token storage:",
                        f"{result['unique_memory'] / 1024:.2f} KiB"
                    )

                    print(
                        "Adaptive statistics:",
                        result["stats"]
                    )

                    # ------------------------------------------------
                    # SAVE RESULT
                    # ------------------------------------------------

                    all_results.append({
                        "seed": seed,
                        "batch": batch_size,
                        "sequence": sequence_length,
                        "requested_duplicate_rate": duplicate_rate,
                        "duplicate_rate": actual_rate,
                        "unique_prompts": batch_data["actual_unique"],
                        "minimum_active_length": min(active_lengths),
                        "maximum_active_length": max(active_lengths),
                        "average_active_length": statistics.mean(active_lengths),
                        **result,
                        "correctness": (
                            result["correctness"]["standalone"]
                            and result["correctness"]["integrated"]
                        ),
                        "max_integrated_difference": (
                            result["correctness"]["max_integrated_diff"]
                        )
                    })

                    gc.collect()

        del model

        gc.collect()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_summary(all_results)

    # --------------------------------------------------------
    # CSV EXPORT
    # --------------------------------------------------------

    save_csv(all_results)

    print("\nCSV saved to:")

    print(CSV_PATH)

    print("\nRCT 2.1.3 benchmark completed.")


if __name__ == "__main__":
    main()
