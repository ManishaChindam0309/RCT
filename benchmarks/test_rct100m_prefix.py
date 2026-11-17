
from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_prefix import RCT100MPrefixCache


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")

    checkpoint_path = (
        PROJECT_DIR
        / "checkpoints"
        / "rct100m_continued"
        / "best_model.pt"
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", checkpoint_path)

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = RCT100MConfig(**checkpoint["model_config"])

    model = RCT100M(config)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print("Training step:", checkpoint["step"])
    print("Validation loss:", checkpoint["val_loss"])
    print("Parameters:", f"{model.parameter_count():,}")

    # Create prefix-cache wrapper around the same model.
    prefix_model = RCT100MPrefixCache(
        model,
        cache_capacity=4,
    )

    prefix_model.eval()

    # Test configuration.
    batch_size = 4
    prefix_length = 16
    suffix_length = 16
    sequence_length = prefix_length + suffix_length

    torch.manual_seed(20261106)
    torch.cuda.manual_seed_all(20261106)

    # All requests share exactly the same prefix.
    shared_prefix = torch.randint(
        0,
        config.vocab_size,
        (1, prefix_length),
        device=device,
        dtype=torch.long,
    )

    suffix_a = torch.randint(
        0,
        config.vocab_size,
        (batch_size, suffix_length),
        device=device,
        dtype=torch.long,
    )

    suffix_b = torch.randint(
        0,
        config.vocab_size,
        (batch_size, suffix_length),
        device=device,
        dtype=torch.long,
    )

    prefix_batch = shared_prefix.expand(
        batch_size, -1
    )

    prompts_a = torch.cat(
        [prefix_batch, suffix_a],
        dim=1,
    )

    prompts_b = torch.cat(
        [prefix_batch, suffix_b],
        dim=1,
    )

    print("\nTest configuration:")
    print("Batch size:", batch_size)
    print("Prefix length:", prefix_length)
    print("Suffix length:", suffix_length)
    print("Sequence length:", sequence_length)

    print("\nRunning standard inference and prefix-cache inference...")

    with torch.inference_mode():

        # Standard model reference for request group A.
        standard_a = model(prompts_a)

        # First prefix-cache call: should build the cache.
        cached_a = prefix_model.forward_cached(
            prompts_a,
            prefix_length,
        )

        # Standard model reference for request group B.
        standard_b = model(prompts_b)

        # Second call: same prefix, different suffixes.
        # This should reuse the cached prefix.
        cached_b = prefix_model.forward_cached(
            prompts_b,
            prefix_length,
        )

    torch.cuda.synchronize()

    # Numerical comparison.
    max_error_a = (
        standard_a - cached_a
    ).abs().max().item()

    max_error_b = (
        standard_b - cached_b
    ).abs().max().item()

    mean_error_a = (
        standard_a - cached_a
    ).abs().mean().item()

    mean_error_b = (
        standard_b - cached_b
    ).abs().mean().item()

    passed_a = torch.allclose(
        standard_a,
        cached_a,
        rtol=1e-4,
        atol=1e-4,
    )

    passed_b = torch.allclose(
        standard_b,
        cached_b,
        rtol=1e-4,
        atol=1e-4,
    )

    print("\nNumerical correctness:")
    print("Standard A shape:", tuple(standard_a.shape))
    print("Cached A shape:", tuple(cached_a.shape))
    print(f"A maximum absolute error: {max_error_a:.8e}")
    print(f"A mean absolute error: {mean_error_a:.8e}")
    print("A allclose:", passed_a)

    print("\nDifferent suffixes, same cached prefix:")
    print("Standard B shape:", tuple(standard_b.shape))
    print("Cached B shape:", tuple(cached_b.shape))
    print(f"B maximum absolute error: {max_error_b:.8e}")
    print(f"B mean absolute error: {mean_error_b:.8e}")
    print("B allclose:", passed_b)

    print("\nCache statistics:")
    print(prefix_model.cache_statistics())

    allocated = torch.cuda.memory_allocated() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\nGPU allocated memory: {allocated:.2f} MB")
    print(f"GPU peak allocated memory: {peak:.2f} MB")

    if not passed_a or not passed_b:
        raise RuntimeError(
            "PREFIX CACHE CORRECTNESS TEST FAILED."
        )

    stats = prefix_model.cache_statistics()

    if stats["hits"] != 1 or stats["misses"] != 1:
        raise RuntimeError(
            "Expected exactly one cache miss and one cache hit."
        )

    print("\nPREFIX CACHE CORRECTNESS TEST: PASS")


if __name__ == "__main__":
    main()
