import sys
import gc
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from model.rct_100m import RCT100M, RCT100MConfig


def quantile(values, q):

    if values.numel() == 0:
        return 0.0

    return torch.quantile(
        values.float(),
        q
    ).item()


def main():

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    thresholds = [0.95, 0.99, 0.999]

    print("=" * 85)
    print("RCT 100M APPROXIMATE FFN REUSE ERROR EXPERIMENT")
    print("=" * 85)

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
    print("Parameters:", model.parameter_count())

    # ------------------------------------------------
    # Load eight validation sequences from across
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

    # ------------------------------------------------
    # Capture actual FFN inputs from every layer.
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

        handles.append(
            block.ffn.register_forward_pre_hook(
                make_hook(index)
            )
        )

    print("\nCapturing real FFN activations...")

    with torch.inference_mode():
        model(tokens)

    for handle in handles:
        handle.remove()

    # ------------------------------------------------
    # Evaluate approximate reuse independently
    # at every layer.
    # ------------------------------------------------

    totals = {
        threshold: {
            "accepted": 0,
            "count": 0,
            "relative_error_sum": 0.0,
        }
        for threshold in thresholds
    }

    print("\nLAYER-LEVEL RESULTS")
    print("=" * 85)

    for threshold in thresholds:

        print(f"\nCOSINE THRESHOLD: {threshold}")
        print("-" * 85)

        print(
            f"{'Layer':<8}"
            f"{'Reuse %':>10}"
            f"{'Mean Err':>13}"
            f"{'Median Err':>13}"
            f"{'P95 Err':>13}"
            f"{'Output Cos':>13}"
        )

        for index in range(len(model.blocks)):

            activations = captured[index].float()

            normalized = F.normalize(
                activations,
                p=2,
                dim=1,
                eps=1e-12
            )

            similarities = normalized @ normalized.T

            similarities.fill_diagonal_(
                -float("inf")
            )

            best_similarity, nearest_indices = (
                similarities.max(dim=1)
            )

            accepted = (
                best_similarity >= threshold
            )

            with torch.inference_mode():

                exact_outputs = model.blocks[index].ffn(
                    activations
                )

                candidate_outputs = model.blocks[index].ffn(
                    activations.index_select(
                        0,
                        nearest_indices
                    )
                )

            approximate_outputs = exact_outputs.clone()

            approximate_outputs[accepted] = (
                candidate_outputs[accepted]
            )

            exact_norm = torch.linalg.vector_norm(
                exact_outputs,
                dim=1
            ).clamp_min(1e-12)

            error_norm = torch.linalg.vector_norm(
                approximate_outputs - exact_outputs,
                dim=1
            )

            relative_errors = error_norm / exact_norm

            reused_errors = relative_errors[accepted]

            if accepted.any():

                exact_reused = exact_outputs[accepted]
                approximate_reused = (
                    approximate_outputs[accepted]
                )

                output_cosines = F.cosine_similarity(
                    exact_reused,
                    approximate_reused,
                    dim=1,
                    eps=1e-12
                )

                output_cosine_mean = (
                    output_cosines.mean().item()
                )

            else:

                output_cosine_mean = 1.0

            reuse_percentage = (
                accepted.float().mean().item() * 100.0
            )

            mean_error = relative_errors.mean().item()

            median_error = quantile(
                reused_errors,
                0.50
            )

            p95_error = quantile(
                reused_errors,
                0.95
            )

            print(
                f"{index + 1:<8}"
                f"{reuse_percentage:>9.3f}%"
                f"{mean_error:>13.6f}"
                f"{median_error:>13.6f}"
                f"{p95_error:>13.6f}"
                f"{output_cosine_mean:>13.6f}"
            )

            totals[threshold]["accepted"] += (
                accepted.sum().item()
            )

            totals[threshold]["count"] += (
                accepted.numel()
            )

            totals[threshold]["relative_error_sum"] += (
                relative_errors.sum().item()
            )

            del activations
            del normalized
            del similarities
            del exact_outputs
            del candidate_outputs
            del approximate_outputs
            del relative_errors
            gc.collect()

    # ------------------------------------------------
    # Overall summary.
    # ------------------------------------------------

    print("\n" + "=" * 85)
    print("OVERALL RESULTS")
    print("=" * 85)

    for threshold in thresholds:

        result = totals[threshold]

        accepted = result["accepted"]
        count = result["count"]

        reuse_percentage = (
            100.0 * accepted / count
            if count
            else 0.0
        )

        overall_mean_error = (
            result["relative_error_sum"] / count
            if count
            else 0.0
        )

        print(f"\nThreshold: {threshold}")
        print(f"Total activations: {count}")
        print(f"Reuse candidates: {accepted}")
        print(f"Reuse percentage: {reuse_percentage:.4f}%")
        print(
            "Overall mean relative output error:",
            f"{overall_mean_error:.8f}"
        )

    print("\nEXPERIMENT COMPLETE")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
