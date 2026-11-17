
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference import load_rct_inference


CHECKPOINT = "checkpoints/rct100m_continued/best_model.pt"
BATCH_SIZE = 16
PROMPT_LENGTH = 32
NEW_TOKENS = 8

# Test the configured default by not passing dedup_threshold.
WORKLOADS = [
    (16, 0.0),
    (12, 0.25),
    (8, 0.50),
    (4, 0.75),
    (2, 0.875),
]


def make_prompts(unique_count, vocab_size, device):
    unique = torch.randint(
        low=1,
        high=vocab_size,
        size=(unique_count, PROMPT_LENGTH),
        dtype=torch.long,
        device=device,
    )

    indices = torch.arange(
        BATCH_SIZE,
        device=device,
    ) % unique_count

    return unique.index_select(0, indices)


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    device = "cuda"

    print("Loading RCT inference...")

    # Intentionally omit dedup_threshold to test the real default.
    inference = load_rct_inference(
        CHECKPOINT,
        device=device,
    )

    threshold = inference.dedup_threshold
    vocab_size = inference.model.config.vocab_size

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Prompt length: {PROMPT_LENGTH}")
    print(f"Generated tokens: {NEW_TOKENS}")
    print(f"Deduplication threshold: {threshold:.2%}")

    assert abs(threshold - 0.75) < 1e-9, (
        f"Expected default threshold 75%, got {threshold:.2%}"
    )

    total_checks = 0

    for unique_count, expected_rate in WORKLOADS:

        prompts = make_prompts(
            unique_count,
            vocab_size,
            device,
        )

        actual_rate = (
            BATCH_SIZE - unique_count
        ) / BATCH_SIZE

        assert abs(actual_rate - expected_rate) < 1e-9

        print("\n" + "=" * 55)
        print(f"Unique prompts: {unique_count}/{BATCH_SIZE}")
        print(f"Duplicate rate: {actual_rate:.2%}")

        inference.reset_adaptive_statistics()

        reference = inference.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            mode="standard",
        )

        adaptive = inference.generate(
            prompts,
            max_new_tokens=NEW_TOKENS,
            mode="adaptive",
        )

        torch.cuda.synchronize()

        exact_match = torch.equal(reference, adaptive)

        stats = inference.adaptive_statistics()

        should_activate = actual_rate >= threshold

        activation_correct = (
            stats["dedup_activations"] == int(should_activate)
        )

        expected_saved = (
            BATCH_SIZE - unique_count
            if should_activate
            else 0
        )

        saved_correct = (
            stats["duplicate_prompts_saved"] == expected_saved
        )

        detected_correct = (
            stats["duplicate_prompts_detected"]
            == BATCH_SIZE - unique_count
        )

        print(f"Standard/adaptive exact match: {exact_match}")
        print(
            f"Deduplication activated: "
            f"{stats['dedup_activations']}"
        )
        print(
            f"Duplicates detected: "
            f"{stats['duplicate_prompts_detected']}"
        )
        print(
            f"Duplicates saved: "
            f"{stats['duplicate_prompts_saved']}"
        )

        assert exact_match, "Output mismatch."
        assert activation_correct, "Incorrect gate decision."
        assert saved_correct, "Incorrect saved-duplicate count."
        assert detected_correct, "Incorrect detected-duplicate count."

        print("WORKLOAD PASS")
        total_checks += 1

    print("\n" + "=" * 55)
    print(
        f"ADAPTIVE GATE TEST: PASS "
        f"({total_checks}/{len(WORKLOADS)})"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
