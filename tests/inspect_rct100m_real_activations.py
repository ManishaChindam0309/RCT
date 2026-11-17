import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import gc
import numpy as np
import torch

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_adaptive_ffn import RCT100MAdaptiveFFNExecutor


def main():

    torch.set_num_threads(4)
    torch.manual_seed(2026)

    print("=" * 75)
    print("RCT 100M REAL ACTIVATION DUPLICATION ANALYSIS")
    print("=" * 75)

    checkpoint_path = (
        PROJECT_DIR / "checkpoints" /
        "rct100m_continued" / "best_model.pt"
    )

    validation_path = (
        PROJECT_DIR / "data" / "rct30" /
        "validation_expanded_tokens.bin"
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

    # Eight different validation sequences, 32 tokens each.
    tokens_data = np.memmap(
        validation_path,
        dtype=np.uint16,
        mode="r"
    )

    batch_size = 8
    sequence_length = 32

    required_tokens = batch_size * sequence_length

    tokens = torch.from_numpy(
        np.array(
            tokens_data[:required_tokens],
            dtype=np.int64
        ).reshape(batch_size, sequence_length)
    ).long()

    del tokens_data
    gc.collect()

    captured = {}

    handles = []

    def make_hook(layer_index):

        def hook(module, inputs):

            captured[layer_index] = (
                inputs[0].detach().reshape(
                    -1,
                    inputs[0].shape[-1]
                ).clone()
            )

        return hook

    for index, block in enumerate(model.blocks):

        handle = block.ffn.register_forward_pre_hook(
            make_hook(index)
        )

        handles.append(handle)

    print("\nRunning actual model forward pass...")

    with torch.inference_mode():
        model(tokens)

    for handle in handles:
        handle.remove()

    print("\nRESULTS")
    print("-" * 75)

    total_vectors = 0
    total_unique = 0

    for index in range(len(model.blocks)):

        activations = captured[index]

        unique = torch.unique(
            activations,
            dim=0
        )

        count = activations.shape[0]
        unique_count = unique.shape[0]

        duplicate_ratio = (
            1.0 - unique_count / count
        )

        total_vectors += count
        total_unique += unique_count

        executor = RCT100MAdaptiveFFNExecutor(
            model.blocks[index].ffn,
            duplicate_threshold=0.25,
            sample_size=32
        )

        estimated_ratio = (
            executor._estimate_duplicate_ratio(
                activations
            )
        )

        print(
            f"Layer {index + 1:02d}: "
            f"vectors={count:4d}, "
            f"unique={unique_count:4d}, "
            f"actual_duplicates={duplicate_ratio:.4f}, "
            f"estimated_duplicates={estimated_ratio:.4f}"
        )

        del activations, unique, executor

    overall_duplicate_ratio = (
        1.0 - total_unique / total_vectors
    )

    print("-" * 75)
    print("Total activation vectors:", total_vectors)
    print("Total unique vectors:", total_unique)
    print(
        "Overall exact duplicate ratio:",
        f"{overall_duplicate_ratio:.6f}"
    )

    print("\nANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
