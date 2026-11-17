
from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m_adaptive import RCT100MAdaptive


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 64
    sequence_length = 24
    fingerprint_columns = 4

    prompts = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.long,
        device=device
    )

    # Create unique prompts at the positions that the
    # old evenly spaced detector would have sampled.
    old_sample_indices = torch.linspace(
        0,
        batch_size - 1,
        steps=8,
        device=device
    ).round().long()

    for index, position in enumerate(old_sample_indices):
        prompts[position, 0] = index + 1

    actual_unique = torch.unique(
        prompts,
        dim=0
    ).size(0)

    actual_duplicate_rate = (
        batch_size - actual_unique
    ) / batch_size

    # Create an instance without loading a model.
    detector = object.__new__(RCT100MAdaptive)
    detector.fingerprint_columns = fingerprint_columns

    estimated_duplicate_rate = (
        detector.estimate_duplicate_rate(prompts)
    )

    print("Device:", device)
    print("Batch size:", batch_size)
    print("Fingerprint columns:", fingerprint_columns)
    print(
        "Old sampled positions:",
        old_sample_indices.tolist()
    )
    print("Actual unique prompts:", actual_unique)

    print(
        "Actual duplicate rate:",
        f"{actual_duplicate_rate:.2%}"
    )

    print(
        "Fingerprint duplicate-rate estimate:",
        f"{estimated_duplicate_rate:.2%}"
    )

    if estimated_duplicate_rate >= actual_duplicate_rate:
        print("\nPASS: Fingerprint estimate did not miss duplicates.")
    else:
        print("\nFAIL: Fingerprint estimate was below actual duplication.")

    if estimated_duplicate_rate > 0:
        print("The adaptive policy should consider exact deduplication.")
    else:
        print("The adaptive policy would skip deduplication.")


if __name__ == "__main__":
    main()
