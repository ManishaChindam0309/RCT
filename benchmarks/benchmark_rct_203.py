
import os
import sys
import time
import random
import math
import statistics
import argparse

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

SEED = 20203

VOCAB_SIZE = 2000
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
MAX_SEQ_LEN = 256

CPU_THREADS = 4

WARMUP_RUNS = 5
BENCHMARK_RUNS = 30

BATCH_SIZES = [1, 2, 4, 8]
SEQUENCE_LENGTHS = [64, 128]

PATTERNS = [
    "identical",
    "paired",
    "unique"
]


# ============================================================
# SETUP
# ============================================================

def setup():

    random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(CPU_THREADS)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print("=" * 75)
    print("RCT 2.0.3 INPUT-LEVEL PROMPT DEDUPLICATION")
    print("=" * 75)

    print("PyTorch:", torch.__version__)
    print("Device: CPU")
    print("Threads:", torch.get_num_threads())


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
# CACHE CONTROL
# ============================================================

def clear_ffn_caches(model):

    for block in model.blocks:
        block.ffn.invalidate_cache()


# ============================================================
# PROMPT GENERATION
# ============================================================

def create_batch(batch_size, sequence_length, pattern):

    if pattern == "identical":

        prompt = torch.randint(
            0,
            VOCAB_SIZE,
            (1, sequence_length)
        )

        return prompt.repeat(batch_size, 1)

    if pattern == "paired":

        unique_count = math.ceil(batch_size / 2)

        prompts = torch.randint(
            0,
            VOCAB_SIZE,
            (unique_count, sequence_length)
        )

        return torch.stack([
            prompts[index // 2]
            for index in range(batch_size)
        ])

    if pattern == "unique":

        return torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, sequence_length)
        )

    raise ValueError("Unknown workload pattern")


# ============================================================
# STANDARD TRANSFORMER
# ============================================================

@torch.inference_mode()
def standard_inference(model, tokens):

    clear_ffn_caches(model)

    start = time.perf_counter()

    output = model(
        tokens,
        ffn_mode="standard"
    )

    elapsed = time.perf_counter() - start

    return output, elapsed


# ============================================================
# FFN DEDUPLICATION
# ============================================================

@torch.inference_mode()
def ffn_dedup_inference(model, tokens):

    clear_ffn_caches(model)

    start = time.perf_counter()

    output = model(
        tokens,
        ffn_mode="deduplicated"
    )

    elapsed = time.perf_counter() - start

    return output, elapsed


# ============================================================
# INPUT-LEVEL PROMPT DEDUPLICATION
# ============================================================

@torch.inference_mode()
def prompt_dedup_inference(
    model,
    tokens,
    deduplicator
):

    clear_ffn_caches(model)

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
# CORRECTNESS
# ============================================================

def check_correctness(
    model,
    tokens,
    deduplicator
):

    standard, _ = standard_inference(
        model,
        tokens
    )

    ffn_dedup, _ = ffn_dedup_inference(
        model,
        tokens
    )

    prompt_dedup, _ = prompt_dedup_inference(
        model,
        tokens,
        deduplicator
    )

    ffn_difference = (
        standard - ffn_dedup
    ).abs().max().item()

    prompt_difference = (
        standard - prompt_dedup
    ).abs().max().item()

    ffn_correct = torch.allclose(
        standard,
        ffn_dedup,
        rtol=1e-4,
        atol=1e-5
    )

    prompt_correct = torch.allclose(
        standard,
        prompt_dedup,
        rtol=1e-4,
        atol=1e-5
    )

    return {
        "ffn_correct": ffn_correct,
        "prompt_correct": prompt_correct,
        "ffn_difference": ffn_difference,
        "prompt_difference": prompt_difference
    }


# ============================================================
# BENCHMARK ONE MODE
# ============================================================

def measure_mode(
    model,
    tokens,
    mode,
    deduplicator,
    runs
):

    if mode == "standard":

        inference = lambda: standard_inference(
            model,
            tokens
        )

    elif mode == "ffn":

        inference = lambda: ffn_dedup_inference(
            model,
            tokens
        )

    elif mode == "prompt":

        inference = lambda: prompt_dedup_inference(
            model,
            tokens,
            deduplicator
        )

    else:

        raise ValueError("Unknown benchmark mode")

    for _ in range(WARMUP_RUNS):
        inference()

    measurements = []

    for _ in range(runs):

        _, elapsed = inference()

        measurements.append(
            elapsed * 1000
        )

    return measurements


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_experiment(
    model,
    batch_size,
    sequence_length,
    pattern,
    runs,
    deduplicator
):

    tokens = create_batch(
        batch_size,
        sequence_length,
        pattern
    )

    # --------------------------------------------------------
    # CORRECTNESS
    # --------------------------------------------------------

    correctness = check_correctness(
        model,
        tokens,
        deduplicator
    )

    if not (
        correctness["ffn_correct"]
        and correctness["prompt_correct"]
    ):

        raise RuntimeError(
            "Numerical correctness failed"
        )

    # --------------------------------------------------------
    # MEASUREMENTS
    # --------------------------------------------------------

    results = {
        "standard": [],
        "ffn": [],
        "prompt": []
    }

    modes = [
        "standard",
        "ffn",
        "prompt"
    ]

    for _ in range(runs):

        random.shuffle(modes)

        for mode in modes:

            if mode == "standard":

                _, elapsed = standard_inference(
                    model,
                    tokens
                )

            elif mode == "ffn":

                _, elapsed = ffn_dedup_inference(
                    model,
                    tokens
                )

            else:

                _, elapsed = prompt_dedup_inference(
                    model,
                    tokens,
                    deduplicator
                )

            results[mode].append(
                elapsed * 1000
            )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    medians = {
        mode: statistics.median(values)
        for mode, values in results.items()
    }

    speedups = {
        mode: medians["standard"] / medians[mode]
        for mode in ("ffn", "prompt")
    }

    reductions = {
        mode: (
            1 - medians[mode] / medians["standard"]
        ) * 100
        for mode in ("ffn", "prompt")
    }

    # --------------------------------------------------------
    # DUPLICATE STATISTICS
    # --------------------------------------------------------

    unique_tokens, inverse_indices = (
        deduplicator.deduplicate(tokens)
    )

    prompt_statistics = deduplicator.statistics()

    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "pattern": pattern,
        "correctness": correctness,
        "medians": medians,
        "speedups": speedups,
        "reductions": reductions,
        "unique_prompts": unique_tokens.shape[0],
        "prompt_statistics": prompt_statistics
    }


# ============================================================
# DISPLAY
# ============================================================

def display_result(result):

    print(
        f"\nBatch={result['batch_size']} "
        f"Sequence={result['sequence_length']} "
        f"Pattern={result['pattern']}"
    )

    print(
        "FFN correctness:",
        result["correctness"]["ffn_correct"]
    )

    print(
        "Prompt correctness:",
        result["correctness"]["prompt_correct"]
    )

    print(
        "Unique prompts:",
        result["unique_prompts"]
    )

    print(
        "Duplicate ratio:",
        f"{result['prompt_statistics']['duplicate_ratio']:.2%}"
    )

    print(
        f"Standard: {result['medians']['standard']:.4f} ms"
    )

    print(
        f"FFN dedup: {result['medians']['ffn']:.4f} ms"
    )

    print(
        f"Prompt dedup: {result['medians']['prompt']:.4f} ms"
    )

    print(
        f"FFN speedup: {result['speedups']['ffn']:.3f}x"
    )

    print(
        f"Prompt speedup: {result['speedups']['prompt']:.3f}x"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=BENCHMARK_RUNS
    )

    parser.add_argument(
        "--quick",
        action="store_true"
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    runs = 5 if args.quick else args.runs

    setup()

    model = create_model()

    deduplicator = PromptDeduplicator()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("\nParameters:", f"{parameter_count:,}")
    print("Benchmark runs:", runs)

    all_results = []

    total = (
        len(BATCH_SIZES)
        * len(SEQUENCE_LENGTHS)
        * len(PATTERNS)
    )

    experiment = 0

    for sequence_length in SEQUENCE_LENGTHS:

        for batch_size in BATCH_SIZES:

            for pattern in PATTERNS:

                experiment += 1

                print("\n" + "=" * 75)

                print(
                    f"Experiment {experiment}/{total}"
                )

                result = run_experiment(
                    model,
                    batch_size,
                    sequence_length,
                    pattern,
                    runs,
                    deduplicator
                )

                display_result(result)

                all_results.append(result)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 95)
    print("RCT 2.0.3 FINAL SUMMARY")
    print("=" * 95)

    print(
        f"{'Batch':>6} "
        f"{'Seq':>6} "
        f"{'Pattern':>12} "
        f"{'Standard':>12} "
        f"{'FFN':>10} "
        f"{'Prompt':>10} "
        f"{'FFN Spd':>10} "
        f"{'Prompt Spd':>12}"
    )

    for result in all_results:

        print(
            f"{result['batch_size']:>6} "
            f"{result['sequence_length']:>6} "
            f"{result['pattern']:>12} "
            f"{result['medians']['standard']:>11.4f} "
            f"{result['medians']['ffn']:>9.4f} "
            f"{result['medians']['prompt']:>9.4f} "
            f"{result['speedups']['ffn']:>9.3f}x "
            f"{result['speedups']['prompt']:>10.3f}x"
        )

    # --------------------------------------------------------
    # PATTERN AVERAGES
    # --------------------------------------------------------

    print("\nAverage speedups by pattern:")

    for pattern in PATTERNS:

        matching = [
            result
            for result in all_results
            if result["pattern"] == pattern
        ]

        ffn_average = statistics.mean(
            result["speedups"]["ffn"]
            for result in matching
        )

        prompt_average = statistics.mean(
            result["speedups"]["prompt"]
            for result in matching
        )

        print(
            f"{pattern}: "
            f"FFN={ffn_average:.3f}x, "
            f"Prompt={prompt_average:.3f}x"
        )

    print(
        "\nCorrect experiments:",
        len(all_results),
        "/",
        total
    )

    print("\nRCT 2.0.3 benchmark completed.")


if __name__ == "__main__":
    main()
