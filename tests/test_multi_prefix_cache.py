
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model.transformer import RCTModel


def make_prompt(prefix, suffix):
    return torch.tensor(
        [prefix + suffix],
        dtype=torch.long
    )


def main():

    torch.manual_seed(42)
    torch.set_num_threads(4)

    model = RCTModel(
        vocab_size=500,
        hidden_size=64,
        layers=2,
        heads=4,
        max_seq_len=128,
        prefix_cache_capacity=2
    )

    baseline = RCTModel(
        vocab_size=500,
        hidden_size=64,
        layers=2,
        heads=4,
        max_seq_len=128,
        prefix_cache_capacity=2
    )

    baseline.load_state_dict(model.state_dict())

    model.eval()
    baseline.eval()

    # Disable FFN cache in both models.
    for block in model.blocks:
        block.ffn.train()

    for block in baseline.blocks:
        block.ffn.train()

    prefix_a = [10, 20, 30, 40]
    prefix_b = [11, 21, 31, 41]
    prefix_c = [12, 22, 32, 42]

    prompt_a = make_prompt(prefix_a, [50, 60])
    prompt_b = make_prompt(prefix_b, [51, 61, 71])
    prompt_c = make_prompt(prefix_c, [52, 62, 72, 82])

    with torch.inference_mode():

        # Insert A and B.
        model.forward_cached(prompt_a, 4)
        model.forward_cached(prompt_b, 4)

        # Touch A, making B the least recently used.
        model.forward_cached(prompt_a, 4)

        # Insert C. B should be evicted.
        cached_c = model.forward_cached(prompt_c, 4)

        # A should still hit.
        cached_a = model.forward_cached(prompt_a, 4)

        # B should miss and be rebuilt.
        cached_b = model.forward_cached(prompt_b, 4)

        reference_a = baseline(prompt_a)
        reference_b = baseline(prompt_b)
        reference_c = baseline(prompt_c)

        diff_a = (
            cached_a - reference_a
        ).abs().max().item()

        diff_b = (
            cached_b - reference_b
        ).abs().max().item()

        diff_c = (
            cached_c - reference_c
        ).abs().max().item()

        print("\n===== RCT 0.8 MULTI-PREFIX TEST =====")

        print("Prompt A maximum difference:", diff_a)
        print("Prompt B maximum difference:", diff_b)
        print("Prompt C maximum difference:", diff_c)

        correct_a = torch.allclose(
            cached_a, reference_a,
            rtol=1e-4, atol=1e-4
        )

        correct_b = torch.allclose(
            cached_b, reference_b,
            rtol=1e-4, atol=1e-4
        )

        correct_c = torch.allclose(
            cached_c, reference_c,
            rtol=1e-4, atol=1e-4
        )

        print("Prompt A correct:", correct_a)
        print("Prompt B correct:", correct_b)
        print("Prompt C correct:", correct_c)

        print("\n===== CACHE STATISTICS =====")

        stats = model.prefix_cache_statistics()

        print(stats)

        assert correct_a
        assert correct_b
        assert correct_c

        assert stats["entries"] == 2
        assert stats["capacity"] == 2
        assert stats["evictions"] == 2
        assert stats["hits"] == 2
        assert stats["misses"] == 4

        print("\nLRU eviction test: PASSED")
        print("Numerical correctness test: PASSED")
        print("\n===== TEST COMPLETE =====")


if __name__ == "__main__":
    main()
