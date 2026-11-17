import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import torch

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_adaptive_ffn import (
    RCT100MAdaptiveFFNExecutor
)


def main():

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    print("=" * 65)
    print("RCT 100M ADAPTIVE FFN CORRECTNESS TEST")
    print("=" * 65)

    config = RCT100MConfig()

    model = RCT100M(config)

    checkpoint_path = (
        PROJECT_DIR
        / "checkpoints"
        / "rct100m_continued"
        / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print("Checkpoint step:", checkpoint["step"])
    print("Parameters:", model.parameter_count())

    ffn = model.blocks[0].ffn

    executor = RCT100MAdaptiveFFNExecutor(
        ffn,
        duplicate_threshold=0.25,
        sample_size=32
    )

    batch_size = 1
    sequence_length = 32
    hidden_size = config.hidden_size

    # ------------------------------------------------
    # Test 1: Random activations (mostly unique)
    # ------------------------------------------------

    print("\nTEST 1: Random activations")

    x_random = torch.randn(
        batch_size,
        sequence_length,
        hidden_size
    )

    with torch.inference_mode():

        expected = ffn(x_random)

        actual = executor.execute(
            x_random,
            mode="exact_dedup"
        )

    max_difference = (
        expected - actual
    ).abs().max().item()

    mean_difference = (
        expected - actual
    ).abs().mean().item()

    print("Max absolute difference:", max_difference)
    print("Mean absolute difference:", mean_difference)

    assert torch.allclose(
        expected,
        actual,
        atol=1e-5,
        rtol=1e-4
    ), "Random activation equivalence failed"

    print("PASS: Random activation equivalence")

    # ------------------------------------------------
    # Test 2: Repeated activations
    # ------------------------------------------------

    print("\nTEST 2: Repeated activations")

    unique_vectors = torch.randn(
        8,
        hidden_size
    )

    indices = torch.tensor(
        [
            0, 1, 2, 3,
            0, 1, 2, 3,
            4, 5, 6, 7,
            4, 5, 6, 7,
            0, 1, 2, 3,
            4, 5, 6, 7,
            0, 1, 2, 3,
            4, 5, 6, 7,
        ],
        dtype=torch.long
    )

    x_repeated = unique_vectors[
        indices
    ].unsqueeze(0)

    with torch.inference_mode():

        expected = ffn(x_repeated)

        actual = executor.execute(
            x_repeated,
            mode="exact_dedup"
        )

    max_difference = (
        expected - actual
    ).abs().max().item()

    mean_difference = (
        expected - actual
    ).abs().mean().item()

    print("Input vectors:", sequence_length)
    print("Expected unique vectors:", 8)
    print("Max absolute difference:", max_difference)
    print("Mean absolute difference:", mean_difference)

    assert torch.allclose(
        expected,
        actual,
        atol=1e-5,
        rtol=1e-4
    ), "Repeated activation equivalence failed"

    print("PASS: Repeated activation equivalence")

    # ------------------------------------------------
    # Test 3: Adaptive bypass
    # ------------------------------------------------

    print("\nTEST 3: Adaptive bypass")

    executor.reset_statistics()

    with torch.inference_mode():

        expected = ffn(x_random)

        actual = executor.execute(
            x_random,
            mode="adaptive"
        )

    assert torch.allclose(
        expected,
        actual,
        atol=1e-5,
        rtol=1e-4
    ), "Adaptive bypass equivalence failed"

    stats = executor.statistics()

    print("Adaptive bypasses:", stats["adaptive_bypasses"])
    print("Adaptive acceptances:", stats["adaptive_acceptances"])

    assert stats["adaptive_bypasses"] == 1

    print("PASS: Adaptive bypass")

    # ------------------------------------------------
    # Test 4: Adaptive acceptance
    # ------------------------------------------------

    print("\nTEST 4: Adaptive acceptance")

    executor.reset_statistics()

    with torch.inference_mode():

        expected = ffn(x_repeated)

        actual = executor.execute(
            x_repeated,
            mode="adaptive"
        )

    assert torch.allclose(
        expected,
        actual,
        atol=1e-5,
        rtol=1e-4
    ), "Adaptive acceptance equivalence failed"

    stats = executor.statistics()

    print("Adaptive bypasses:", stats["adaptive_bypasses"])
    print("Adaptive acceptances:", stats["adaptive_acceptances"])
    print("Actual reuse ratio:", stats["reuse_ratio"])

    assert stats["adaptive_acceptances"] == 1
    assert stats["adaptive_bypasses"] == 0

    print("PASS: Adaptive acceptance")

    # ------------------------------------------------
    # Final report
    # ------------------------------------------------

    print("\n" + "=" * 65)
    print("ALL ADAPTIVE FFN CORRECTNESS TESTS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
