
import sys
import time
import statistics
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import RCTModel


def measure(function, warmup=10, iterations=50):

    for _ in range(warmup):
        function()

    timings = []

    for _ in range(iterations):
        start = time.perf_counter()
        function()
        end = time.perf_counter()

        timings.append((end - start) * 1000)

    return statistics.median(timings)


def main():

    torch.manual_seed(42)
    torch.set_num_threads(4)

    vocab_size = 8000
    hidden_size = 256
    layers = 4
    heads = 4

    sequence_length = 128
    prefix_length = 96
    suffix_length = sequence_length - prefix_length

    print("===== RCT 0.9.1 BENCHMARK =====")

    baseline = RCTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        max_seq_len=512
    )

    rct = RCTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        max_seq_len=512,
        prefix_cache_capacity=8
    )

    rct.load_state_dict(baseline.state_dict())

    baseline.eval()
    rct.eval()

    # Isolate prefix/KV cache performance.
    for block in baseline.blocks:
        block.ffn.train()

    for block in rct.blocks:
        block.ffn.train()

    shared_prefix = torch.randint(
        0,
        vocab_size,
        (1, prefix_length)
    )

    suffix_a = torch.randint(
        0,
        vocab_size,
        (1, suffix_length)
    )

    suffix_b = torch.randint(
        0,
        vocab_size,
        (1, suffix_length)
    )

    prompt_a = torch.cat(
        [shared_prefix, suffix_a],
        dim=1
    )

    prompt_b = torch.cat(
        [shared_prefix, suffix_b],
        dim=1
    )

    with torch.inference_mode():

        reference = baseline(prompt_b)

        # Standard inference benchmark.
        standard_time = measure(
            lambda: baseline(prompt_b)
        )

        # Measure cold request latency.
        def cold_request():

            rct.invalidate_caches()

            return rct.forward_cached(
                prompt_a,
                prefix_length
            )

        cold_time = measure(
            cold_request,
            warmup=3,
            iterations=10
        )

        # Reset and build one prefix entry.
        rct.invalidate_caches()

        start = time.perf_counter()

        first_output = rct.forward_cached(
            prompt_a,
            prefix_length
        )

        first_request_time = (
            time.perf_counter() - start
        ) * 1000

        # Reuse the same prefix with a different suffix.
        cached_output = rct.forward_cached(
            prompt_b,
            prefix_length
        )

        cached_time = measure(
            lambda: rct.forward_cached(
                prompt_b,
                prefix_length
            )
        )

        # Correctness check.
        max_difference = (
            cached_output - reference
        ).abs().max().item()

        equivalent = torch.allclose(
            cached_output,
            reference,
            rtol=1e-4,
            atol=1e-4
        )

        speedup = standard_time / cached_time

        latency_reduction = (
            (standard_time - cached_time)
            / standard_time
        ) * 100

        saved_per_request = standard_time - cached_time

        extra_initial_cost = max(
            0,
            first_request_time - standard_time
        )

        break_even = (
            extra_initial_cost / saved_per_request
            if saved_per_request > 0
            else float("inf")
        )

        print("\n===== CONFIGURATION =====")

        print("Device: CPU")
        print("Sequence length:", sequence_length)
        print("Prefix length:", prefix_length)
        print("Suffix length:", suffix_length)
        print("Layers:", layers)
        print("Hidden size:", hidden_size)

        print("\n===== CORRECTNESS =====")

        print("Maximum difference:", max_difference)
        print("Numerically equivalent:", equivalent)

        print("\n===== LATENCY =====")

        print(f"Standard inference: {standard_time:.4f} ms")
        print(f"Cold request median: {cold_time:.4f} ms")
        print(f"First request: {first_request_time:.4f} ms")
        print(f"Cached inference: {cached_time:.4f} ms")

        print(f"Speedup: {speedup:.3f}x")
        print(f"Latency reduction: {latency_reduction:.2f}%")

        print("\n===== BREAK-EVEN =====")

        print(
            f"Extra initial cost: {extra_initial_cost:.4f} ms"
        )

        print(
            f"Saving per cached request: {saved_per_request:.4f} ms"
        )

        print(
            f"Estimated break-even cached requests: {break_even:.1f}"
        )

        print("\n===== CACHE STATISTICS =====")

        print(rct.prefix_cache_statistics())

        # Calculate memory used by cached tensors.
        cache_memory = 0

        for entry in rct._prefix_cache.values():

            for record in entry["records"]:

                for name in ("prefix_key", "prefix_value"):

                    tensor = record[name]

                    cache_memory += (
                        tensor.numel() * tensor.element_size()
                    )

            tensor = entry["final_prefix"]

            cache_memory += (
                tensor.numel() * tensor.element_size()
            )

        print("\n===== MEMORY =====")

        print(
            f"Cached tensor memory: "
            f"{cache_memory / (1024 ** 2):.3f} MB"
        )

        if not equivalent:
            raise AssertionError(
                "Cached output differs beyond tolerance"
            )

        print("\n===== BENCHMARK COMPLETE =====")


if __name__ == "__main__":
    main()
