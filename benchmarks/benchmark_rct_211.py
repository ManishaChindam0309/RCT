
import os
import sys
import time
import random
import statistics
import argparse
import gc

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

BATCH_SIZES = [2, 4, 8, 16]
SEQUENCE_LENGTHS = [64, 128, 256]

DUPLICATE_RATES = [
    0.0,
    0.25,
    0.50,
    0.75,
    0.875
]

WARMUP_RUNS = 5
DEFAULT_RUNS = 30


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
# CREATE RANDOMIZED MIXED BATCH
# ============================================================

def create_batch(
    batch_size,
    sequence_length,
    duplicate_rate
):

    if not 0.0 <= duplicate_rate < 1.0:
        raise ValueError(
            "duplicate_rate must be between 0 and 1."
        )

    if duplicate_rate == 0.0:

        unique_count = batch_size

    else:

        unique_count = max(
            1,
            round(batch_size * (1.0 - duplicate_rate))
        )

    unique_prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (unique_count, sequence_length)
    )

    # Make sure the requested number of unique prompts
    # is actually represented in the final batch.
    indices = list(range(unique_count))

    while len(indices) < batch_size:

        indices.append(
            random.randrange(unique_count)
        )

    random.shuffle(indices)

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

    return tokens, actual_duplicate_rate


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
# INTEGRATED RCT ADAPTIVE DEDUPLICATION
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

        raise ValueError("Unknown benchmark mode.")

    return elapsed * 1000


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

        for mode in shuffled:

            elapsed = measure(
                model,
                tokens,
                mode,
                deduplicator
            )

            measurements[mode].append(elapsed)

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

    p95 = {}

    for mode, values in measurements.items():

        sorted_values = sorted(values)

        index = min(
            len(sorted_values) - 1,
            int(0.95 * len(sorted_values))
        )

        p95[mode] = sorted_values[index]

    # --------------------------------------------------------
    # SPEEDUPS
    # --------------------------------------------------------

    speedups = {
        mode: medians["standard"] / medians[mode]
        for mode in ("standalone", "integrated")
    }

    # --------------------------------------------------------
    # LATENCY REDUCTION
    # --------------------------------------------------------

    reductions = {
        mode: (
            1.0 - medians[mode] / medians["standard"]
        ) * 100
        for mode in ("standalone", "integrated")
    }

    # --------------------------------------------------------
    # ADAPTIVE STATISTICS
    # --------------------------------------------------------

    model.reset_prompt_dedup_statistics()

    _, _ = integrated_adaptive(
        model,
        tokens
    )

    stats = model.prompt_dedup_statistics()

    return {
        "correctness": correctness,
        "duplicate_rate": actual_duplicate_rate,
        "medians": medians,
        "p95": p95,
        "speedups": speedups,
        "reductions": reductions,
        "stats": stats
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS
    )

    parser.add_argument(
        "--quick",
        action="store_true"
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    runs = 3 if args.quick else args.runs

    # --------------------------------------------------------
    # CPU SETUP
    # --------------------------------------------------------

    random.seed(211)

    torch.manual_seed(211)

    torch.set_num_threads(4)

    try:

        torch.set_num_interop_threads(1)

    except RuntimeError:

        pass

    print("=" * 110)

    print(
        "RCT 2.1.1 INTEGRATED ADAPTIVE PROMPT DEDUPLICATION"
    )

    print("=" * 110)

    print("Device: CPU")
    print("CPU threads:", torch.get_num_threads())
    print("Runs:", runs)

    model = create_model()

    deduplicator = PromptDeduplicator()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Parameters:", f"{parameter_count:,}")

    print("Batch sizes:", BATCH_SIZES)
    print("Sequence lengths:", SEQUENCE_LENGTHS)
    print("Requested duplicate rates:", DUPLICATE_RATES)

    # --------------------------------------------------------
    # EXPERIMENTS
    # --------------------------------------------------------

    total = (
        len(BATCH_SIZES)
        * len(SEQUENCE_LENGTHS)
        * len(DUPLICATE_RATES)
    )

    all_results = []

    experiment = 0

    for sequence_length in SEQUENCE_LENGTHS:

        for batch_size in BATCH_SIZES:

            for duplicate_rate in DUPLICATE_RATES:

                experiment += 1

                print("\n" + "=" * 110)

                print(
                    f"Experiment {experiment}/{total}: "
                    f"batch={batch_size}, "
                    f"sequence={sequence_length}, "
                    f"requested_duplicates={duplicate_rate:.1%}"
                )

                tokens, actual_rate = create_batch(
                    batch_size,
                    sequence_length,
                    duplicate_rate
                )

                result = benchmark_case(
                    model,
                    tokens,
                    actual_rate,
                    runs,
                    deduplicator
                )

                print(
                    "Actual duplicate rate:",
                    f"{actual_rate:.2%}"
                )

                print(
                    "Unique prompts:",
                    result["stats"]["unique_prompts"]
                )

                print(
                    "Adaptive mode:",
                    (
                        "deduplicated"
                        if result["stats"]["dedup_activations"] > 0
                        else "bypassed"
                    )
                )

                print(
                    "Correctness:",
                    result["correctness"]["standalone"],
                    result["correctness"]["integrated"]
                )

                print(
                    "Maximum numerical difference:",
                    f"{result['correctness']['max_integrated_diff']:.8f}"
                )

                print(
                    "Standard:",
                    f"{result['medians']['standard']:.4f} ms"
                )

                print(
                    "Standalone:",
                    f"{result['medians']['standalone']:.4f} ms, "
                    f"{result['speedups']['standalone']:.3f}x"
                )

                print(
                    "Integrated:",
                    f"{result['medians']['integrated']:.4f} ms, "
                    f"{result['speedups']['integrated']:.3f}x"
                )

                print(
                    "Integrated P95:",
                    f"{result['p95']['integrated']:.4f} ms"
                )

                all_results.append({
                    "batch": batch_size,
                    "sequence": sequence_length,
                    "requested_duplicate_rate": duplicate_rate,
                    **result
                })

                gc.collect()

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 120)

    print("RCT 2.1.1 FINAL RESULTS")

    print("=" * 120)

    print(
        f"{'Batch':>5} "
        f"{'Seq':>5} "
        f"{'Dup %':>8} "
        f"{'Standard':>12} "
        f"{'Standalone':>12} "
        f"{'Integrated':>12} "
        f"{'Speedup':>10} "
        f"{'Reduction':>11}"
    )

    for result in all_results:

        print(
            f"{result['batch']:>5} "
            f"{result['sequence']:>5} "
            f"{result['duplicate_rate']:>7.1%} "
            f"{result['medians']['standard']:>11.3f} "
            f"{result['medians']['standalone']:>11.3f} "
            f"{result['medians']['integrated']:>11.3f} "
            f"{result['speedups']['integrated']:>9.3f}x "
            f"{result['reductions']['integrated']:>9.2f}%"
        )

    # --------------------------------------------------------
    # AVERAGE SPEEDUPS BY DUPLICATE RATE
    # --------------------------------------------------------

    print("\nAverage integrated speedup by actual duplicate rate:")

    for duplicate_rate in DUPLICATE_RATES:

        matching = [
            result
            for result in all_results
            if result["requested_duplicate_rate"] == duplicate_rate
        ]

        if not matching:
            continue

        average_speedup = statistics.mean(
            result["speedups"]["integrated"]
            for result in matching
        )

        print(
            f"{duplicate_rate:.1%}: "
            f"{average_speedup:.3f}x"
        )

    # --------------------------------------------------------
    # CORRECTNESS SUMMARY
    # --------------------------------------------------------

    correct_count = sum(
        result["correctness"]["standalone"]
        and result["correctness"]["integrated"]
        for result in all_results
    )

    print("\nCorrect experiments:", correct_count, "/", total)

    print(
        "Average integrated speedup:",
        f"{statistics.mean(result['speedups']['integrated'] for result in all_results):.3f}x"
    )

    print("\nRCT 2.1.1 benchmark completed.")


if __name__ == "__main__":
    main()
