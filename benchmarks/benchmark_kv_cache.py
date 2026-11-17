
import sys
import time
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model.transformer import RCTModel


def median_latency(function, repeats=50):

    measurements = []

    for _ in range(repeats):

        start = time.perf_counter()
        function()
        elapsed = time.perf_counter() - start

        measurements.append(elapsed * 1000)

    return statistics.median(measurements)


def main():

    torch.manual_seed(42)
    torch.set_num_threads(4)

    print("\n===== RCT 0.7 KV CACHE BENCHMARK =====")

    vocab_size = 8000
    hidden_size = 256
    layers = 4
    heads = 4
    sequence_length = 128
    prefix_length = 96

    model = RCTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        max_seq_len=512
    )

    baseline = RCTModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        layers=layers,
        heads=heads,
        max_seq_len=512
    )

    baseline.load_state_dict(model.state_dict())

    model.eval()
    baseline.eval()

    # Disable FFN caching in BOTH models.
    # This isolates the KV-cache experiment.
    for block in model.blocks:
        block.ffn.train()

    for block in baseline.blocks:
        block.ffn.train()

    prompt_a = torch.randint(
        0,
        vocab_size,
        (1, sequence_length)
    )

    prompt_b = prompt_a.clone()

    prompt_b[:, prefix_length:] = torch.randint(
        0,
        vocab_size,
        (1, sequence_length - prefix_length)
    )

    print("Sequence length:", sequence_length)
    print("Shared prefix length:", prefix_length)
    print("Suffix length:", sequence_length - prefix_length)
    print("Layers:", layers)
    print("Hidden size:", hidden_size)
    print("FFN caching: disabled in both models")

    with torch.inference_mode():

        # Build prefix cache from prompt A.
        start = time.perf_counter()

        model.forward_cached(
            prompt_a,
            prefix_length
        )

        cache_build_time = (
            time.perf_counter() - start
        ) * 1000

        # Warm up both paths.
        for _ in range(10):

            baseline(prompt_b)

            model.forward_cached(
                prompt_b,
                prefix_length
            )

        # Compare outputs.
        reference_output = baseline(prompt_b)

        cached_output = model.forward_cached(
            prompt_b,
            prefix_length
        )

        absolute_difference = (
            cached_output - reference_output
        ).abs()

        max_difference = absolute_difference.max().item()

        mean_difference = absolute_difference.mean().item()

        # Explicit numerical tolerance.
        rtol = 1e-4
        atol = 1e-4

        is_correct = torch.allclose(
            cached_output,
            reference_output,
            rtol=rtol,
            atol=atol
        )

        # Measure standard inference.
        standard_latency = median_latency(
            lambda: baseline(prompt_b)
        )

        # Measure KV-cached inference.
        cached_latency = median_latency(
            lambda: model.forward_cached(
                prompt_b,
                prefix_length
            )
        )

        speedup = standard_latency / cached_latency

        reduction = (
            (standard_latency - cached_latency)
            / standard_latency
        ) * 100

        print("\n===== CORRECTNESS =====")

        print("Maximum absolute difference:", max_difference)
        print("Mean absolute difference:", mean_difference)
        print("Relative tolerance:", rtol)
        print("Absolute tolerance:", atol)
        print("Numerically equivalent:", is_correct)

        print("\n===== LATENCY RESULTS =====")

        print(
            f"Standard inference: {standard_latency:.4f} ms"
        )

        print(
            f"Cache construction: {cache_build_time:.4f} ms"
        )

        print(
            f"Cached inference: {cached_latency:.4f} ms"
        )

        print("\n===== PERFORMANCE ANALYSIS =====")

        print(f"Speedup: {speedup:.3f}x")
        print(f"Latency reduction: {reduction:.2f}%")

        print("\n===== PREFIX CACHE STATISTICS =====")

        print(model.prefix_cache_statistics())

        print("\n===== BENCHMARK COMPLETE =====")


if __name__ == "__main__":
    main()
