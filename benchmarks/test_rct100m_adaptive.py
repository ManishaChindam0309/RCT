
from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_adaptive import RCT100MAdaptive


def load_model(device):
    checkpoint_path = (
        PROJECT_DIR
        / "checkpoints"
        / "rct100m_continued"
        / "best_model.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    config = RCT100MConfig(
        vocab_size=8192,
        hidden_size=512,
        num_layers=30,
        num_heads=8,
        intermediate_size=2048,
        max_seq_len=512
    )

    base_model = RCT100M(config)

    base_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    base_model.to(device)
    base_model.eval()

    return base_model, checkpoint


def make_prompts(
    batch_size,
    sequence_length,
    duplicate_rate,
    device
):
    unique_count = max(
        1,
        round(batch_size * (1.0 - duplicate_rate))
    )

    unique_prompts = torch.randint(
        low=0,
        high=8192,
        size=(unique_count, sequence_length),
        dtype=torch.long,
        device=device
    )

    repeat_indices = (
        torch.arange(batch_size, device=device)
        % unique_count
    )

    prompts = unique_prompts.index_select(
        0,
        repeat_indices
    )

    return prompts, unique_count


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This test expects your NVIDIA GPU."
        )

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    base_model, checkpoint = load_model(device)

    print("Checkpoint step:", checkpoint.get("step"))
    print("Checkpoint validation loss:", checkpoint.get("val_loss"))
    print("Parameters:", sum(
        parameter.numel()
        for parameter in base_model.parameters()
    ))

    model = RCT100MAdaptive(
        base_model,
        duplicate_threshold=0.25,
        fingerprint_columns=4
    )

    model.eval()

    batch_size = 8
    sequence_length = 24
    max_new_tokens = 4

    duplicate_rates = [
        0.0,
        0.50,
        0.75,
        0.875
    ]

    for requested_rate in duplicate_rates:
        prompts, expected_unique = make_prompts(
            batch_size,
            sequence_length,
            requested_rate,
            device
        )

        model.reset_statistics()

        with torch.inference_mode():
            outputs = model.generate(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=1.0
            )

        actual_unique = torch.unique(
            prompts,
            dim=0
        ).size(0)

        actual_duplicate_rate = (
            batch_size - actual_unique
        ) / batch_size

        stats = model.statistics()

        assert outputs.shape == (
            batch_size,
            sequence_length + max_new_tokens
        ), (
            f"Unexpected output shape: {outputs.shape}"
        )

        # Identical prompts must have identical greedy? 
        # The cached generator samples, but the deduplicated
        # path generates each unique prompt only once.
        if stats["last_decision"] == "deduplicated":
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    if torch.equal(prompts[i], prompts[j]):
                        assert torch.equal(
                            outputs[i],
                            outputs[j]
                        ), "Duplicate prompts produced different outputs."

        print("\n------------------------------------")
        print("Requested duplicate rate:", requested_rate)
        print("Actual duplicate rate:", actual_duplicate_rate)
        print("Expected unique prompts:", expected_unique)
        print("Actual unique prompts:", actual_unique)

        print(
            "Fingerprint estimate:",
            stats["last_fingerprint_duplicate_rate"]
        )

        print("Decision:", stats["last_decision"])
        print(
            "Duplicate prompts saved:",
            stats["total_duplicate_prompts_saved"]
        )

        print("Output shape:", tuple(outputs.shape))

        print("Output checks passed.")

    print("\nAll adaptive correctness checks passed.")


if __name__ == "__main__":
    main()
