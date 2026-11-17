import sys
import gc
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from model.rct_100m import RCT100M, RCT100MConfig


def main():

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    print("=" * 78)
    print("RCT 100M REAL ACTIVATION SIMILARITY ANALYSIS")
    print("=" * 78)

    checkpoint_path = (
        PROJECT_DIR
        / "checkpoints"
        / "rct100m_continued"
        / "best_model.pt"
    )

    validation_path = (
        PROJECT_DIR
        / "data"
        / "rct30"
        / "validation_expanded_tokens.bin"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    model = RCT100M(RCT100MConfig())

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print("Checkpoint step:", checkpoint["step"])
    print("Model parameters:", model.parameter_count())

    # ------------------------------------------------
    # Select eight validation windows from across
    # the validation corpus.
    # ------------------------------------------------

    data = np.memmap(
        validation_path,
        dtype=np.uint16,
        mode="r"
    )

    sequence_length = 32
    number_of_sequences = 8

    max_start = len(data) - sequence_length

    starts = np.linspace(
        0,
        max_start,
        num=number_of_sequences,
        dtype=np.int64
    )

    sequences = np.stack([
        np.array(
            data[start:start + sequence_length],
            dtype=np.int64
        )
        for start in starts
    ])

    del data
    gc.collect()

    tokens = torch.from_numpy(sequences).long()

    print("Validation sequences:", number_of_sequences)
    print("Tokens per sequence:", sequence_length)
    print("Total activation vectors per layer:",
          number_of_sequences * sequence_length)

    # ------------------------------------------------
    # Capture FFN input activations.
    # ------------------------------------------------

    captured = {}
    handles = []

    def make_hook(layer_index):

        def hook(module, inputs):

            captured[layer_index] = (
                inputs[0]
                .detach()
                .reshape(-1, inputs[0].shape[-1])
                .clone()
            )

        return hook

    for index, block in enumerate(model.blocks):

        handle = block.ffn.register_forward_pre_hook(
            make_hook(index)
        )

        handles.append(handle)

    print("\nRunning model forward pass...")

    with torch.inference_mode():
        model(tokens)

    for handle in handles:
        handle.remove()

    # ------------------------------------------------
    # Similarity analysis.
    # ------------------------------------------------

    thresholds = [0.90, 0.95, 0.99, 0.999]

    layer_nearest_medians = []
    layer_nearest_p99 = []

    print("\nLAYER RESULTS")
    print("-" * 78)

    print(
        f"{'Layer':<8}"
        f"{'NN median':>12}"
        f"{'NN p95':>12}"
        f"{'NN p99':>12}"
        f"{'>=.95':>10}"
        f"{'>=.99':>10}"
        f"{'>=.999':>10}"
    )

    all_nearest = []
    total_pairs = 0
    threshold_counts = {
        threshold: 0 for threshold in thresholds
    }

    for index in range(len(model.blocks)):

        activations = captured[index].float()

        normalized = F.normalize(
            activations,
            p=2,
            dim=1,
            eps=1e-12
        )

        similarities = normalized @ normalized.T

        # Exclude each vector's self-similarity.
        similarities.fill_diagonal_(-float("inf"))

        nearest = similarities.max(dim=1).values

        nearest_quantiles = torch.quantile(
            nearest,
            torch.tensor(
                [0.50, 0.90, 0.95, 0.99, 1.00]
            )
        )

        layer_nearest_medians.append(
            nearest_quantiles[0].item()
        )

        layer_nearest_p99.append(
            nearest_quantiles[3].item()
        )

        all_nearest.append(nearest)

        # Count each unordered pair only once.
        upper = torch.triu(
            similarities,
            diagonal=1
        )

        pair_values = upper[
            torch.triu(
                torch.ones_like(
                    upper,
                    dtype=torch.bool
                ),
                diagonal=1
            )
        ]

        total_pairs += pair_values.numel()

        for threshold in thresholds:

            threshold_counts[threshold] += (
                pair_values >= threshold
            ).sum().item()

        print(
            f"{index + 1:<8}"
            f"{nearest_quantiles[0].item():>12.6f}"
            f"{nearest_quantiles[2].item():>12.6f}"
            f"{nearest_quantiles[3].item():>12.6f}"
            f"{(nearest >= 0.95).float().mean().item() * 100:>9.3f}%"
            f"{(nearest >= 0.99).float().mean().item() * 100:>9.3f}%"
            f"{(nearest >= 0.999).float().mean().item() * 100:>9.3f}%"
        )

        del activations, normalized, similarities
        del nearest, upper, pair_values
        gc.collect()

    # ------------------------------------------------
    # Overall nearest-neighbor distribution.
    # ------------------------------------------------

    all_nearest = torch.cat(all_nearest)

    overall_quantiles = torch.quantile(
        all_nearest,
        torch.tensor(
            [0.50, 0.90, 0.95, 0.99, 1.00]
        )
    )

    print("\n" + "=" * 78)
    print("OVERALL NEAREST-NEIGHBOR SIMILARITY")
    print("=" * 78)

    print("Activation vectors:", all_nearest.numel())

    print("Median:", overall_quantiles[0].item())
    print("90th percentile:", overall_quantiles[1].item())
    print("95th percentile:", overall_quantiles[2].item())
    print("99th percentile:", overall_quantiles[3].item())
    print("Maximum:", overall_quantiles[4].item())

    print("\nPAIRWISE SIMILARITY DISTRIBUTION")
    print("-" * 78)

    print("Total unordered pairs:", total_pairs)

    for threshold in thresholds:

        count = threshold_counts[threshold]

        percentage = (
            100.0 * count / total_pairs
            if total_pairs
            else 0.0
        )

        print(
            f"Pairs with cosine similarity >= {threshold}: "
            f"{count} ({percentage:.6f}%)"
        )

    print("\nLAYER VARIATION")
    print("-" * 78)

    print(
        "Lowest layer nearest-neighbor median:",
        min(layer_nearest_medians)
    )

    print(
        "Highest layer nearest-neighbor median:",
        max(layer_nearest_medians)
    )

    print(
        "Lowest layer nearest-neighbor p99:",
        min(layer_nearest_p99)
    )

    print(
        "Highest layer nearest-neighbor p99:",
        max(layer_nearest_p99)
    )

    print("\nANALYSIS COMPLETE")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
