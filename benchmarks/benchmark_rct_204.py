
import os
import sys
import time
import random
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

VOCAB_SIZE = 2000
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
MAX_SEQ_LEN = 256

BATCH_SIZES = [1, 2, 4, 8]
SEQUENCE_LENGTHS = [64, 128]

PATTERNS = [
    "identical",
    "paired",
    "unique"
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
# INPUT GENERATION
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

        unique_count = (batch_size + 1) // 2

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

    raise ValueError("Unknown pattern")


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
def standard(model, tokens):

    clear_caches(model)

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
def ffn_dedup(model, tokens):

    clear_caches(model)

    start = time.perf_counter()

    output = model(
        tokens,
        ffn_mode="deduplicated"
    )

    elapsed = time.perf_counter() - start

    return output, elapsed


# ============================================================
# RCT 2.0.3 PROMPT DEDUPLICATION
# ============================================================

@torch.inference_mode()
def prompt_dedup(model, tokens, deduplicator):

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
# RCT 2.0.4 ADAPTIVE PROMPT DEDUPLICATION
# ============================================================

@torch.inference_mode()
def adaptive_prompt(model, tokens, deduplicator):

    clear_caches(model)

    start = time.perf_counter()

    output, stats = deduplicator.forward(
        model,
        tokens
    )

    elapsed = time.perf_counter() - start

    return output, elapsed, stats


# ============================================================
# CORRECTNESS VALIDATION
# ============================================================

def verify(model, tokens, deduplicator):

    reference, _ = standard(
        model,
        tokens
    )

    ffn_output, _ = ffn_dedup(
        model,
        tokens
    )

    prompt_output, _ = prompt_dedup(
        model,
        tokens,
        deduplicator
    )

    adaptive_output, _, _ = adaptive_prompt(
        model,
        tokens,
        deduplicator
    )

    return {
        "ffn": torch.allclose(
            reference,
            ffn_output,
            rtol=1e-4,
            atol=1e-5
        ),

        "prompt": torch.allclose(
            reference,
            prompt_output,
            rtol=1e-4,
            atol=1e-5
        ),

        "adaptive": torch.allclose(
            reference,
            adaptive_output,
            rtol=1e-4,
            atol=1e-5
        )
    }


# ============================================================
# SINGLE MODE MEASUREMENT
# ============================================================

def measure(model, tokens, mode, deduplicator):

    if mode == "standard":

        _, elapsed = standard(
            model,
            tokens
        )

    elif mode == "ffn":

        _, elapsed = ffn_dedup(
            model,
            tokens
        )

    elif mode == "prompt":

        _, elapsed = prompt_dedup(
            model,
            tokens,
            deduplicator
        )

    elif mode == "adaptive":

        _, elapsed, _ = adaptive_prompt(
            model,
            tokens,
            deduplicator
        )

    else:

        raise ValueError("Unknown benchmark mode")

    return elapsed * 1000


# ============================================================
# BENCHMARK ONE WORKLOAD
# ============================================================

def benchmark_case(
    model,
    tokens,
    runs,
    deduplicator
):

    correctness = verify(
        model,
        tokens,
        deduplicator
    )

    if not all(correctness.values()):

        raise RuntimeError(
            f"Correctness failure: {correctness}"
        )

    modes = [
        "standard",
        "ffn",
        "prompt",
        "adaptive"
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
    # RANDOMIZED MEASUREMENT
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
    # MEDIANS
    # --------------------------------------------------------

    medians = {
        mode: statistics.median(values)
        for mode, values in measurements.items()
    }

    # --------------------------------------------------------
    # SPEEDUPS
    # --------------------------------------------------------

    speedups = {
        mode: medians["standard"] / medians[mode]
        for mode in ("ffn", "prompt", "adaptive")
    }

    # --------------------------------------------------------
    # LATENCY REDUCTION
    # --------------------------------------------------------

    reductions = {
        mode: (
            1 - medians[mode] / medians["standard"]
        ) * 100

        for mode in ("ffn", "prompt", "adaptive")
    }

    # --------------------------------------------------------
    # ADAPTIVE STATISTICS
    # FIXED: adaptive_prompt RETURNS THREE VALUES
    # --------------------------------------------------------

    _, _, stats = adaptive_prompt(
        model,
        tokens,
        deduplicator
    )

    return {
        "correctness": correctness,
        "medians": medians,
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

    runs = 5 if args.quick else args.runs

    # --------------------------------------------------------
    # CPU SETUP
    # --------------------------------------------------------

    random.seed(204)

    torch.manual_seed(204)

    torch.set_num_threads(4)

    try:

        torch.set_num_interop_threads(1)

    except RuntimeError:

        pass

    print("=" * 85)

    print(
        "RCT 2.0.4 ADAPTIVE PROMPT DEDUPLICATION BENCHMARK"
    )

    print("=" * 85)

    model = create_model()

    deduplicator = PromptDeduplicator()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", parameter_count)

    print("CPU threads:", torch.get_num_threads())

    print("Runs:", runs)

    # --------------------------------------------------------
    # EXPERIMENTS
    # --------------------------------------------------------

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

                print("\n" + "=" * 85)

                print(
                    f"Experiment {experiment}/{total}: "
                    f"batch={batch_size}, "
                    f"seq={sequence_length}, "
                    f"pattern={pattern}"
                )

                tokens = create_batch(
                    batch_size,
                    sequence_length,
                    pattern
                )

                result = benchmark_case(
                    model,
                    tokens,
                    runs,
                    deduplicator
                )

                print(
                    "Correctness:",
                    result["correctness"]
                )

                print(
                    "Adaptive mode:",
                    result["stats"]["mode"]
                )

                print(
                    "Unique prompts:",
                    result["stats"]["unique_prompts"]
                )

                print(
                    "Duplicate ratio:",
                    f"{result['stats']['duplicate_ratio']:.2%}"
                )

                print(
                    "Standard:",
                    f"{result['medians']['standard']:.4f} ms"
                )

                for mode in (
                    "ffn",
                    "prompt",
                    "adaptive"
                ):

                    print(
                        f"{mode}: "
                        f"{result['medians'][mode]:.4f} ms, "
                        f"{result['speedups'][mode]:.3f}x, "
                        f"{result['reductions'][mode]:.2f}%"
                    )

                all_results.append({
                    "batch": batch_size,
                    "sequence": sequence_length,
                    "pattern": pattern,
                    **result
                })

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 110)

    print("RCT 2.0.4 FINAL SUMMARY")

    print("=" * 110)

    print(
        f"{'Batch':>5} "
        f"{'Seq':>5} "
        f"{'Pattern':>11} "
        f"{'Standard':>11} "
        f"{'FFN':>10} "
        f"{'Prompt':>10} "
        f"{'Adaptive':>10} "
        f"{'Adapt Spd':>11} "
        f"{'Mode':>22}"
    )

    for result in all_results:

        print(
            f"{result['batch']:>5} "
            f"{result['sequence']:>5} "
            f"{result['pattern']:>11} "
            f"{result['medians']['standard']:>10.3f} "
            f"{result['medians']['ffn']:>9.3f} "
            f"{result['medians']['prompt']:>9.3f} "
            f"{result['medians']['adaptive']:>9.3f} "
            f"{result['speedups']['adaptive']:>10.3f}x "
            f"{result['stats']['mode']:>22}"
        )

    # --------------------------------------------------------
    # AVERAGE SPEEDUPS
    # --------------------------------------------------------

    print("\nAverage speedups by workload:")

    for pattern in PATTERNS:

        matching = [
            item
            for item in all_results
            if item["pattern"] == pattern
        ]

        average_ffn = statistics.mean(
            item["speedups"]["ffn"]
            for item in matching
        )

        average_prompt = statistics.mean(
            item["speedups"]["prompt"]
            for item in matching
        )

        average_adaptive = statistics.mean(
            item["speedups"]["adaptive"]
            for item in matching
        )

        print(
            f"{pattern}: "
            f"FFN={average_ffn:.3f}x, "
            f"Prompt={average_prompt:.3f}x, "
            f"Adaptive={average_adaptive:.3f}x"
        )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    correct_count = sum(
        all(result["correctness"].values())
        for result in all_results
    )

    print(
        "\nCorrect experiments:",
        correct_count,
        "/",
        total
    )

    print("\nRCT 2.0.4 benchmark completed.")


if __name__ == "__main__":

    main()
