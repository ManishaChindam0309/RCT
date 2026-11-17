import sys
import gc
import time
import statistics
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
import torch.nn.functional as F

from model.rct_100m import RCT100M, RCT100MConfig


def online_reuse(ffn, activations, threshold):
    """
    Sequential representative selection.

    Each activation is compared only with representatives
    selected from earlier activations.

    The FFN is evaluated only on the representatives.
    """

    normalized = F.normalize(
        activations,
        p=2,
        dim=1,
        eps=1e-12
    )

    representative_inputs = []
    representative_norms = []
    assignments = []

    for index in range(activations.shape[0]):

        current_norm = normalized[index]

        if representative_norms:

            similarities = (
                torch.stack(representative_norms)
                @ current_norm
            )

            best_similarity, best_index = (
                similarities.max(dim=0)
            )

            if best_similarity.item() >= threshold:

                assignments.append(
                    best_index.item()
                )

                continue

        new_index = len(representative_inputs)

        representative_inputs.append(
            activations[index]
        )

        representative_norms.append(
            current_norm
        )

        assignments.append(new_index)

    representatives = torch.stack(
        representative_inputs
    )

    representative_outputs = ffn(
        representatives
    )

    assignment_tensor = torch.tensor(
        assignments,
        dtype=torch.long,
        device=activations.device
    )

    approximate_outputs = (
        representative_outputs.index_select(
            0,
            assignment_tensor
        )
    )

    return approximate_outputs, len(representative_inputs)


def median_runtime(function, warmups=1, repetitions=3):

    for _ in range(warmups):
        function()

    measurements = []

    for _ in range(repetitions):

        start = time.perf_counter()

        function()

        measurements.append(
            time.perf_counter() - start
        )

    return statistics.median(measurements)


def main():

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    thresholds = [0.95, 0.99, 0.999]

    print("=" * 90)
    print("RCT 100M ONLINE APPROXIMATE FFN REUSE")
    print("=" * 90)

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

    # Eight validation sequences from across the corpus.
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

    # Capture FFN input activations.
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

    print("\nCapturing activations...")

    with torch.inference_mode():
        model(tokens)

    for handle in handles:
        handle.remove()

    # Aggregate results for each threshold.
    results = {
        threshold: {
            "total": 0,
            "computed": 0,
            "absolute_error_sum": 0.0,
            "relative_error_sum": 0.0,
            "reused_relative_errors": [],
            "standard_times": [],
            "reuse_times": [],
        }
        for threshold in thresholds
    }

    print("\nRunning sequential representative-cache tests...")
    print("Timing: 1 warmup and 3 measured repetitions per layer.")

    for layer_index in range(len(model.blocks)):

        ffn = model.blocks[layer_index].ffn

        activations = captured[layer_index].float()

        with torch.inference_mode():

            exact_outputs = ffn(activations)

            standard_time = median_runtime(
                lambda: ffn(activations)
            )

        total = activations.shape[0]

        exact_norms = torch.linalg.vector_norm(
            exact_outputs,
            dim=1
        ).clamp_min(1e-12)

        for threshold in thresholds:

            def run_reuse():

                return online_reuse(
                    ffn,
                    activations,
                    threshold
                )

            with torch.inference_mode():

                approximate_outputs, computed = (
                    run_reuse()
                )

                reuse_time = median_runtime(
                    run_reuse
                )

            absolute_errors = (
                approximate_outputs - exact_outputs
            ).abs()

            error_norms = torch.linalg.vector_norm(
                approximate_outputs - exact_outputs,
                dim=1
            )

            relative_errors = (
                error_norms / exact_norms
            )

            # Only the activations actually reused.
            reused_mask = torch.ones(
                total,
                dtype=torch.bool
            )

            # The representative outputs are exact.
            # Identify reused vectors by comparing their
            # output against the corresponding exact result.
            reused_mask = (
                error_norms > 0
            )

            reused_errors = relative_errors[
                reused_mask
            ]

            result = results[threshold]

            result["total"] += total
            result["computed"] += computed

            result["absolute_error_sum"] += (
                absolute_errors.sum().item()
            )

            result["relative_error_sum"] += (
                relative_errors.sum().item()
            )

            result["reused_relative_errors"].extend(
                reused_errors.tolist()
            )

            result["standard_times"].append(
                standard_time
            )

            result["reuse_times"].append(
                reuse_time
            )

            del approximate_outputs
            del absolute_errors
            del error_norms
            del relative_errors
            gc.collect()

        print(
            f"Completed layer {layer_index + 1:02d}/30"
        )

    # ------------------------------------------------
    # Final summary
    # ------------------------------------------------

    print("\n" + "=" * 90)
    print("ONLINE REUSE SUMMARY")
    print("=" * 90)

    print(
        f"{'Threshold':<12}"
        f"{'Reuse %':>12}"
        f"{'Mean Rel Err':>16}"
        f"{'P95 Reuse Err':>16}"
        f"{'Standard ms':>15}"
        f"{'Reuse ms':>13}"
        f"{'Speedup':>10}"
    )

    for threshold in thresholds:

        result = results[threshold]

        total = result["total"]
        computed = result["computed"]

        reused = total - computed

        reuse_percentage = (
            100.0 * reused / total
        )

        mean_relative_error = (
            result["relative_error_sum"] / total
        )

        reused_errors = torch.tensor(
            result["reused_relative_errors"],
            dtype=torch.float32
        )

        p95_reuse_error = (
            torch.quantile(reused_errors, 0.95).item()
            if reused_errors.numel()
            else 0.0
        )

        standard_ms = (
            statistics.median(
                result["standard_times"]
            ) * 1000
        )

        reuse_ms = (
            statistics.median(
                result["reuse_times"]
            ) * 1000
        )

        speedup = (
            standard_ms / reuse_ms
            if reuse_ms > 0
            else 0.0
        )

        print(
            f"{threshold:<12.3f}"
            f"{reuse_percentage:>11.2f}%"
            f"{mean_relative_error:>16.6f}"
            f"{p95_reuse_error:>16.6f}"
            f"{standard_ms:>15.3f}"
            f"{reuse_ms:>13.3f}"
            f"{speedup:>10.3f}x"
        )

    print("\nInterpretation:")
    print("- Reuse percentage measures avoided FFN evaluations.")
    print("- Search and representative-selection overhead is timed.")
    print("- Timing is a CPU microbenchmark, not end-to-end generation.")
    print("- Error is measured against exact FFN outputs.")

    print("\nEXPERIMENT COMPLETE")

    del model
    gc.collect()


if __name__ == "__main__":
    main()
