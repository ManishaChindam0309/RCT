
import os
import sys

# Add the project root to Python's import path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from model.reusable_block import ReusableFFN


def test_repeated_identical_inputs():
    torch.manual_seed(42)

    hidden_size = 256
    repetitions = 8

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=32
    )

    ffn.eval()

    # Create one token and reuse the exact same input.
    token = torch.randn(1, 1, hidden_size)

    outputs = []

    with torch.inference_mode():
        for _ in range(repetitions):
            outputs.append(ffn(token))

    # All outputs must be exactly equal.
    for output in outputs[1:]:
        assert torch.equal(outputs[0], output), (
            "Repeated outputs are not identical."
        )

    stats = ffn.cache_statistics()

    print("\n" + "=" * 55)
    print("REPEATED IDENTICAL INPUT TEST")
    print("=" * 55)

    print("Expected network executions:", 1)
    print("Actual network executions:", stats["network_executions"])
    print("Cache hits:", stats["cache_hits"])
    print("Cache misses:", stats["cache_misses"])
    print("Output equality: PASSED")

    assert stats["network_executions"] == 1
    assert stats["cache_hits"] == repetitions - 1
    assert stats["cache_misses"] == 1

    print("Repeated input reuse: PASSED")


def test_unique_inputs():
    torch.manual_seed(123)

    hidden_size = 256
    repetitions = 8

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4,
        cache_capacity=32
    )

    ffn.eval()

    # Generate eight different token inputs.
    inputs = [
        torch.randn(1, 1, hidden_size)
        for _ in range(repetitions)
    ]

    with torch.inference_mode():
        for token in inputs:
            ffn(token)

    stats = ffn.cache_statistics()

    print("\n" + "=" * 55)
    print("UNIQUE INPUT TEST")
    print("=" * 55)

    print("Expected network executions:", repetitions)
    print("Actual network executions:", stats["network_executions"])
    print("Cache hits:", stats["cache_hits"])
    print("Cache misses:", stats["cache_misses"])

    assert stats["network_executions"] == repetitions
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == repetitions

    print("Unique input handling: PASSED")


if __name__ == "__main__":
    test_repeated_identical_inputs()
    test_unique_inputs()

    print("\n" + "=" * 55)
    print("RCT 1.7 EXECUTION REUSE TESTS PASSED")
    print("=" * 55)
