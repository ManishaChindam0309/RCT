
import os
import sys
import time
import random
import statistics

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from model.adaptive_reusable_ffn import AdaptiveReusableFFN


def measure(function, model, x):
    start = time.perf_counter()
    function(model, x)
    return (time.perf_counter() - start) * 1000


def standard(model, x):
    return model(x, force_mode="standard")


def deduplicated(model, x):
    return model(x, force_mode="deduplicated")


def adaptive(model, x):
    return model(x)


def main():

    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)

    hidden_size = 256
    sequence_length = 32

    unique_counts = [4, 8, 10, 12, 16, 24, 32]

    trials = 100

    model = AdaptiveReusableFFN(
        hidden_size=hidden_size,
        expansion=4
    )

    model.eval()

    modes = {
        "Standard": standard,
        "Deduplicated": deduplicated,
        "Adaptive": adaptive
    }

    print("=" * 75)
    print("RCT 1.8.6 ADAPTIVE FFN PERFORMANCE BENCHMARK")
    print("=" * 75)

    print("Sequence length:", sequence_length)
    print("Hidden size:", hidden_size)
    print("Trials:", trials)
    print("CPU threads:", torch.get_num_threads())

    with torch.inference_mode():

        for unique_count in unique_counts:

            unique_tokens = torch.randn(
                1,
                unique_count,
                hidden_size
            )

            indices = (
                torch.arange(sequence_length)
                % unique_count
            )

            x = unique_tokens.index_select(
                1,
                indices
            )

            print("\n" + "-" * 75)
            print(
                f"UNIQUE TOKENS: {unique_count}/{sequence_length}"
            )
            print("-" * 75)

            # Verify numerical correctness.
            reference = model.network(x)

            for name, function in modes.items():

                output = function(model, x)

                if not torch.allclose(
                    reference,
                    output,
                    rtol=1e-5,
                    atol=1e-6
                ):
                    raise AssertionError(
                        f"{name} numerical mismatch"
                    )

            # Warmup.
            for _ in range(20):
                for function in modes.values():
                    function(model, x)

            timings = {
                name: []
                for name in modes
            }

            # Randomized order reduces ordering bias.
            for _ in range(trials):

                order = list(modes.items())
                random.shuffle(order)

                for name, function in order:

                    elapsed = measure(
                        function,
                        model,
                        x
                    )

                    timings[name].append(elapsed)

            medians = {
                name: statistics.median(values)
                for name, values in timings.items()
            }

            standard_time = medians["Standard"]

            for name in modes:

                speedup = (
                    standard_time / medians[name]
                )

                reduction = (
                    1 - medians[name] / standard_time
                ) * 100

                print(
                    f"{name:15s}: "
                    f"{medians[name]:.6f} ms | "
                    f"Speedup {speedup:.3f}x | "
                    f"Latency reduction {reduction:.2f}%"
                )

            print(
                "Numerical correctness: PASSED"
            )

            print(
                "Adaptive statistics:",
                model.reuse_statistics()
            )

            model.reset_statistics()

    print("\n" + "=" * 75)
    print("RCT 1.8.6 BENCHMARK COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
