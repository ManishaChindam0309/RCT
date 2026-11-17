
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from inference import load_rct_inference


CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)


def assert_exact_match(reference, output, label):

    if not torch.equal(reference, output):
        raise AssertionError(
            f"Output mismatch: {label}"
        )

    print(f"{label}: exact match PASS")


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    torch.manual_seed(20261108)

    inference = load_rct_inference(
        CHECKPOINT,
        device=device,
    )

    batch_size = 4
    prompt_len = 32
    prefix_len = 16
    new_tokens = 8

    shared_prefix = torch.randint(
        0,
        inference.model.config.vocab_size,
        (1, prefix_len),
        device=device,
    )

    suffixes = torch.randint(
        0,
        inference.model.config.vocab_size,
        (batch_size, prompt_len - prefix_len),
        device=device,
    )

    prompt = torch.cat(
        [
            shared_prefix.expand(batch_size, -1),
            suffixes,
        ],
        dim=1,
    )

    print("Checkpoint:", CHECKPOINT)
    print("GPU:", torch.cuda.get_device_name(device))
    print("Batch size:", batch_size)
    print("Prompt length:", prompt_len)
    print("Expected shared prefix:", prefix_len)
    print("Generated tokens:", new_tokens)

    # --------------------------------------------------
    # Test 1: Automatic longest-common-prefix detection
    # --------------------------------------------------

    detected_prefix = inference.detect_shared_prefix(prompt)

    print("\nAutomatic prefix detection")
    print("Expected prefix length:", prefix_len)
    print("Detected prefix length:", detected_prefix)

    if detected_prefix != prefix_len:
        raise AssertionError(
            "Automatic prefix detection returned an incorrect length."
        )

    print("Automatic prefix detection: PASS")

    # --------------------------------------------------
    # Test 2: Standard and KV generation
    # --------------------------------------------------

    print("\nTesting standard generation")

    standard_output = inference.generate(
        prompt,
        max_new_tokens=new_tokens,
        mode="standard",
    )

    print("Standard output shape:", tuple(standard_output.shape))

    print("\nTesting KV generation")

    kv_output = inference.generate(
        prompt,
        max_new_tokens=new_tokens,
        mode="kv",
    )

    print("KV output shape:", tuple(kv_output.shape))

    assert_exact_match(
        standard_output,
        kv_output,
        "Standard vs KV",
    )

    # --------------------------------------------------
    # Test 3: Manual prefix generation
    # --------------------------------------------------

    print("\nTesting manual prefix generation")

    manual_prefix_output = inference.generate(
        prompt,
        max_new_tokens=new_tokens,
        mode="prefix",
        prefix_len=prefix_len,
    )

    print(
        "Manual prefix output shape:",
        tuple(manual_prefix_output.shape),
    )

    assert_exact_match(
        standard_output,
        manual_prefix_output,
        "Manual prefix",
    )

    # --------------------------------------------------
    # Test 4: Automatic prefix generation
    # --------------------------------------------------

    print("\nTesting automatic prefix generation")

    automatic_prefix_output = inference.generate(
        prompt,
        max_new_tokens=new_tokens,
        mode="prefix",
    )

    print(
        "Automatic prefix output shape:",
        tuple(automatic_prefix_output.shape),
    )

    assert_exact_match(
        standard_output,
        automatic_prefix_output,
        "Automatic prefix",
    )

    # --------------------------------------------------
    # Test 5: Auto mode with manual prefix
    # --------------------------------------------------

    print("\nTesting auto mode with manual prefix")

    auto_manual_output = inference.generate(
        prompt,
        max_new_tokens=new_tokens,
        mode="auto",
        prefix_len=prefix_len,
    )

    assert_exact_match(
        standard_output,
        auto_manual_output,
        "Auto mode with manual prefix",
    )

    # --------------------------------------------------
    # Test 6: Auto mode with automatic prefix detection
    # --------------------------------------------------

    print("\nTesting auto mode with automatic detection")

    auto_detected_output = inference.generate(
        prompt,
        max_new_tokens=new_tokens,
        mode="auto",
    )

    assert_exact_match(
        standard_output,
        auto_detected_output,
        "Auto mode with automatic prefix",
    )

    # --------------------------------------------------
    # Test 7: No shared prefix
    # --------------------------------------------------

    print("\nTesting prompts without a shared prefix")

    different_prefix_prompt = prompt.clone()

    different_prefix_prompt[1, 0] = (
        different_prefix_prompt[1, 0] + 1
    ) % inference.model.config.vocab_size

    no_shared_prefix_length = inference.detect_shared_prefix(
        different_prefix_prompt
    )

    print(
        "Detected common prefix length:",
        no_shared_prefix_length,
    )

    if no_shared_prefix_length != 0:
        raise AssertionError(
            "Expected no common prefix across the batch."
        )

    print("No shared prefix detection: PASS")

    # --------------------------------------------------
    # Test 8: Automatic fallback in auto mode
    # --------------------------------------------------

    print("\nTesting automatic fallback in auto mode")

    auto_fallback_output = inference.generate(
        different_prefix_prompt,
        max_new_tokens=new_tokens,
        mode="auto",
        fallback=True,
    )

    different_prompt_kv_output = inference.generate(
        different_prefix_prompt,
        max_new_tokens=new_tokens,
        mode="kv",
    )

    assert_exact_match(
        different_prompt_kv_output,
        auto_fallback_output,
        "Auto mode fallback",
    )

    # --------------------------------------------------
    # Test 9: Automatic fallback in prefix mode
    # --------------------------------------------------

    print("\nTesting automatic fallback in prefix mode")

    prefix_fallback_output = inference.generate(
        different_prefix_prompt,
        max_new_tokens=new_tokens,
        mode="prefix",
        fallback=True,
    )

    assert_exact_match(
        different_prompt_kv_output,
        prefix_fallback_output,
        "Prefix mode fallback",
    )

    # --------------------------------------------------
    # Test 10: Strict prefix validation
    # --------------------------------------------------

    print("\nTesting strict prefix validation")

    try:

        inference.generate(
            different_prefix_prompt,
            max_new_tokens=new_tokens,
            mode="prefix",
            prefix_len=prefix_len,
            fallback=False,
        )

    except ValueError:

        print("Strict manual prefix validation: PASS")

    else:

        raise AssertionError(
            "Strict prefix mode should reject mismatched prefixes."
        )

    # --------------------------------------------------
    # Test 11: Strict automatic prefix validation
    # --------------------------------------------------

    print("\nTesting strict automatic prefix validation")

    try:

        inference.generate(
            different_prefix_prompt,
            max_new_tokens=new_tokens,
            mode="prefix",
            fallback=False,
        )

    except ValueError:

        print("Strict automatic prefix validation: PASS")

    else:

        raise AssertionError(
            "Automatic prefix mode should reject a batch "
            "without a common prefix when fallback is disabled."
        )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RCT INFERENCE INTEGRATION TEST: PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
