
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from model.transformer import RCTModel


def main():

    torch.manual_seed(42)
    torch.set_num_threads(4)

    model = RCTModel(
        vocab_size=1000,
        hidden_size=64,
        layers=2,
        heads=4,
        max_seq_len=128
    )

    model.eval()

    baseline = RCTModel(
        vocab_size=1000,
        hidden_size=64,
        layers=2,
        heads=4,
        max_seq_len=128
    )

    baseline.load_state_dict(model.state_dict())
    baseline.eval()

    # Keep attention in evaluation mode, but disable
    # caching only inside the reference FFN modules.
    for block in baseline.blocks:
        block.ffn.train()

    prompt_a = torch.tensor([
        [10, 20, 30, 40, 50, 60]
    ])

    prompt_b = torch.tensor([
        [10, 20, 30, 40, 70, 80]
    ])

    with torch.inference_mode():

        # RCT: caching enabled.
        output_a = model(prompt_a, reuse_prefix=True)
        output_b = model(prompt_b, reuse_prefix=True)

        # Reference: caching disabled, same token-wise
        # FFN execution and attention evaluation mode.
        reference_a = baseline(prompt_a, reuse_prefix=True)
        reference_b = baseline(prompt_b, reuse_prefix=True)

        difference_a = (
            output_a - reference_a
        ).abs().max().item()

        difference_b = (
            output_b - reference_b
        ).abs().max().item()

        prefix_difference = (
            output_a[:, :4, :] -
            output_b[:, :4, :]
        ).abs().max().item()

        print("\n===== RCT 0.5 CORRECTNESS TEST =====")

        print("Prompt A maximum difference:", difference_a)
        print("Prompt B maximum difference:", difference_b)

        print(
            "Prompt A numerically close:",
            torch.allclose(
                output_a,
                reference_a,
                rtol=1e-5,
                atol=2e-5
            )
        )

        print(
            "Prompt B numerically close:",
            torch.allclose(
                output_b,
                reference_b,
                rtol=1e-5,
                atol=2e-5
            )
        )

        print("\n===== PREFIX REUSE =====")

        print(
            "Shared prefix exactly equal:",
            prefix_difference == 0.0
        )

        print(
            "Shared prefix maximum difference:",
            prefix_difference
        )

        print("\n===== CACHE STATISTICS =====")

        for name, stats in model.cache_statistics().items():
            print(name, stats)

    print("\n===== TEST COMPLETE =====")


if __name__ == "__main__":
    main()
