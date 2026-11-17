
import sys
import time
import statistics
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import RCTModel


# -------------------------------
# Benchmark configuration
# -------------------------------

VOCAB_SIZE = 8000
HIDDEN_SIZE = 256
LAYERS = 4
HEADS = 4

MAX_SEQ_LEN = 512
CACHE_CAPACITY = 8

PREFIX_LENGTHS = [32, 64, 96, 128, 192]
SUFFIX_LENGTHS = [16, 32, 64]

WARMUP = 3
ITERATIONS = 20
COLD_ITERATIONS = 3


def measure(function, warmup=WARMUP, iterations=ITERATIONS):

    for _ in range(warmup):
        function()

    timings = []

    for _ in range(iterations):

        start = time.perf_counter()
        function()
        end = time.perf_counter()

        timings.append((end - start) * 1000)

    return statistics.median(timings)


def create_models():

    baseline = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        layers=LAYERS,
        heads=HEADS,
        max_seq_len=MAX_SEQ_LEN
    )

    rct = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        layers=LAYERS,
        heads=HEADS,
        max_seq_len=MAX_SEQ_LEN,
        prefix_cache_capacity=CACHE_CAPACITY
    )

    rct.load_state_dict(baseline.state_dict())

    baseline.eval()
    rct.eval()

    # Disable FFN caching in both models.
    # This isolates prefix/KV cache performance.
    for block in baseline.blocks:
        block.ffn.train()

    for block in rct.blocks:
        block.ffn.train()

    return baseline, rct


def create_prompt(prefix_length, suffix_length):

    prefix = torch.randint(
        0,
        VOCAB_SIZE,
        (1, prefix_length)
    )

    suffix_a = torch.randint(
        0,
        VOCAB_SIZE,
        (1, suffix_length)
    )

    suffix_b = torch.randint(
        0,
        VOCAB_SIZE,
        (1, suffix_length)
    )

    prompt_a = torch.cat(
        [prefix, suffix_a],
        dim=1
    )

    prompt_b = torch.cat(
        [prefix, suffix_b],
        dim=1
    )

    return prompt_a, prompt_b


def benchmark_case(baseline, rct, prefix_length, suffix_length):

    prompt_a, prompt_b = create_prompt(
        prefix_length,
        suffix_length
    )

    rct.invalidate_caches()

    with torch.inference_mode():

        reference = baseline(prompt_b)

        # Standard inference.
        standard_time = measure(
            lambda: baseline(prompt_b)
        )

        # First request: cache miss and full inference.
        start = time.perf_counter()

        first_output = rct.forward_cached(
            prompt_a,
            prefix_length
        )

        first_request_time = (
            time.perf_counter() - start
        ) * 1000

        # Second request: same prefix, different suffix.
        cached_output = rct.forward_cached(
            prompt_b,
            prefix_length
        )

        # Verify numerical correctness.
        max_difference = (
            cached_output - reference
        ).abs().max().item()

        equivalent = torch.allclose(
            cached_output,
            reference,
            rtol=1e-4,
            atol=1e-4
        )

        # Cached inference latency.
        cached_time = measure(
            lambda: rct.forward_cached(
                prompt_b,
                prefix_length
            )
        )

        # Cold request median.
        def cold_request():

            rct.invalidate_caches()

            rct.forward_cached(
                prompt_a,
                prefix_length
            )

        cold_time = measure(
            cold_request,
            warmup=0,
            iterations=COLD_ITERATIONS
        )

        speedup = standard_time / cached_time

        reduction = (
            (standard_time - cached_time)
            / standard_time
        ) * 100

        extra_cost = max(
            0,
            first_request_time - standard_time
        )

        saving = standard_time - cached_time

        break_even = (
            extra_cost / saving
            if saving > 0
            else float("inf")
        )

        return {
            "prefix": prefix_length,
            "suffix": suffix_length,
            "total": prefix_length + suffix_length,
            "standard": standard_time,
            "cold": cold_time,
            "first": first_request_time,
            "cached": cached_time,
            "speedup": speedup,
            "reduction": reduction,
            "break_even": break_even,
            "difference": max_difference,
            "correct": equivalent
        }


def test_cache_capacity():

    print("\n===== MULTI-PREFIX CAPACITY TEST =====")

    _, rct = create_models()

    rct.eval()

    prefix_length = 32
    suffix_length = 16

    prompts = []

    for i in range(10):

        prefix = torch.randint(
            0,
            VOCAB_SIZE,
            (1, prefix_length)
        )

        suffix = torch.randint(
            0,
            VOCAB_SIZE,
            (1, suffix_length)
        )

        prompts.append(
            torch.cat([prefix, suffix], dim=1)
        )

    with torch.inference_mode():

        # Insert ten distinct prefixes into capacity-eight cache.
        for prompt in prompts:

            rct.forward_cached(
                prompt,
                prefix_length
            )

        print(
            "After inserting 10 prefixes:",
            rct.prefix_cache_statistics()
        )

        # Access newest entry.
        rct.forward_cached(
            prompts[9],
            prefix_length
        )

        # Access oldest entry, which should have been evicted.
        rct.forward_cached(
            prompts[0],
            prefix_length
        )

        print(
            "After reaccessing newest and oldest:",
            rct.prefix_cache_statistics()
        )

        stats = rct.prefix_cache_statistics()

        assert stats["entries"] == CACHE_CAPACITY
        assert stats["capacity"] == CACHE_CAPACITY
        assert stats["evictions"] >= 2

        print("Cache capacity test: PASSED")


def main():

    torch.manual_seed(42)
    torch.set_num_threads(4)

    print("===== RCT 1.0 WORKLOAD MATRIX =====")

    print("Device: CPU")
    print("Threads:", torch.get_num_threads())
    print("Layers:", LAYERS)
    print("Hidden size:", HIDDEN_SIZE)
    print("Cache capacity:", CACHE_CAPACITY)

    print(
        "FFN caching: disabled in both models"
    )

    baseline, rct = create_models()

    results = []

    with torch.inference_mode():

        for prefix_length in PREFIX_LENGTHS:

            for suffix_length in SUFFIX_LENGTHS:

                total_length = (
                    prefix_length + suffix_length
                )

                if total_length > MAX_SEQ_LEN:
                    continue

                print(
                    f"\nTesting prefix={prefix_length}, "
                    f"suffix={suffix_length}"
                )

                result = benchmark_case(
                    baseline,
                    rct,
                    prefix_length,
                    suffix_length
                )

                results.append(result)

                print(
                    f"Standard: {result['standard']:.3f} ms | "
                    f"Cached: {result['cached']:.3f} ms | "
                    f"Speedup: {result['speedup']:.2f}x | "
                    f"Correct: {result['correct']}"
                )

    print("\n===== COMPLETE WORKLOAD RESULTS =====")

    print(
        "Prefix | Suffix | Total | Standard | Cached | "
        "Speedup | Reduction | Break-even | Correct"
    )

    for r in results:

        print(
            f"{r['prefix']:>6} | "
            f"{r['suffix']:>6} | "
            f"{r['total']:>5} | "
            f"{r['standard']:>8.3f} | "
            f"{r['cached']:>6.3f} | "
            f"{r['speedup']:>7.2f}x | "
            f"{r['reduction']:>8.2f}% | "
            f"{r['break_even']:>9.1f} | "
            f"{r['correct']}"
        )

    # Aggregate measurements.
    successful = [
        r for r in results
        if r["correct"]
    ]

    faster = [
        r for r in successful
        if r["cached"] < r["standard"]
    ]

    print("\n===== SUMMARY =====")

    print("Total workload cases:", len(results))
    print("Numerically correct cases:", len(successful))
    print("Cases faster with caching:", len(faster))

    if successful:

        average_speedup = statistics.mean(
            r["speedup"] for r in successful
        )

        average_reduction = statistics.mean(
            r["reduction"] for r in successful
        )

        print(
            f"Average speedup: {average_speedup:.3f}x"
        )

        print(
            f"Average latency reduction: "
            f"{average_reduction:.2f}%"
        )

    if len(successful) != len(results):
        raise AssertionError(
            "At least one workload failed numerical correctness"
        )

    test_cache_capacity()

    print("\n===== RCT 1.0 BENCHMARK COMPLETE =====")


if __name__ == "__main__":
    main()
