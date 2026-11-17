
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_prefix_generation import RCT100MPrefixGeneration


CHECKPOINT = PROJECT_DIR / "checkpoints" / "rct100m_continued" / "best_model.pt"

BATCH_SIZE = 4
PREFIX_LEN = 16
PROMPT_LEN = 32
MAX_NEW_TOKENS = 8

SEED = 2026


def load_model(device):

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    config = RCT100MConfig(**checkpoint["model_config"])

    model = RCT100M(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.inference_mode()
def standard_greedy(model, prompt, max_new_tokens):

    generated = prompt.clone()

    for _ in range(max_new_tokens):

        logits = model(generated)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated


def check_exact_match(name, reference, actual):

    exact_match = torch.equal(reference, actual)

    matching_tokens = (
        reference[:, reference.size(1) - MAX_NEW_TOKENS:]
        == actual[:, actual.size(1) - MAX_NEW_TOKENS:]
    ).sum().item()

    total_tokens = reference.size(0) * MAX_NEW_TOKENS

    print(f"\n{name}")
    print("-" * 70)
    print(f"Generated tokens matching: {matching_tokens}/{total_tokens}")
    print(f"Exact sequence match: {exact_match}")

    if not exact_match:

        mismatch_count = (
            reference[:, -MAX_NEW_TOKENS:]
            != actual[:, -MAX_NEW_TOKENS:]
        ).sum().item()

        print(f"Mismatched tokens: {mismatch_count}")

        raise AssertionError(f"{name} failed.")

    print(f"{name}: PASS")


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("=" * 70)
    print("RCT100M AUTOMATIC PREFIX GENERATION TEST")
    print("=" * 70)

    model, checkpoint = load_model(device)

    print(f"Checkpoint step: {checkpoint['step']}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Prompt length: {PROMPT_LEN}")
    print(f"New tokens: {MAX_NEW_TOKENS}")

    prefix_model = RCT100MPrefixGeneration(
        model,
        cache_capacity=16,
    )
    prefix_model.eval()

    # -------------------------------------------------
    # Test 1: Different prompts with a shared prefix.
    # -------------------------------------------------

    print("\nTEST 1: Automatic shared-prefix detection")

    shared_prefix = torch.randint(
        0,
        model.config.vocab_size,
        (1, PREFIX_LEN),
        device=device,
    )

    suffixes = torch.randint(
        0,
        model.config.vocab_size,
        (BATCH_SIZE, PROMPT_LEN - PREFIX_LEN),
        device=device,
    )

    prompt = torch.cat(
        [
            shared_prefix.expand(BATCH_SIZE, -1),
            suffixes,
        ],
        dim=1,
    )

    detected_length = (
        prefix_model.longest_common_prefix_length(prompt)
    )

    print(f"Expected shared prefix: {PREFIX_LEN}")
    print(f"Detected shared prefix: {detected_length}")

    assert detected_length == PREFIX_LEN, (
        "Automatic prefix detection returned an incorrect length."
    )

    standard_output = standard_greedy(
        model,
        prompt,
        MAX_NEW_TOKENS,
    )

    # No prefix_len: automatic detection is used.
    automatic_output = prefix_model.generate_greedy(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    check_exact_match(
        "AUTOMATIC PREFIX CORRECTNESS",
        standard_output,
        automatic_output,
    )

    # -------------------------------------------------
    # Test 2: Entire prompt is shared.
    # -------------------------------------------------

    print("\nTEST 2: Entire prompt shared")

    identical_prompt = (
        prompt[:1]
        .expand(BATCH_SIZE, -1)
        .contiguous()
    )

    detected_length = (
        prefix_model.longest_common_prefix_length(
            identical_prompt
        )
    )

    print(f"Expected shared prefix: {PROMPT_LEN}")
    print(f"Detected shared prefix: {detected_length}")

    assert detected_length == PROMPT_LEN, (
        "Entire-prompt shared prefix was not detected."
    )

    standard_identical = standard_greedy(
        model,
        identical_prompt,
        MAX_NEW_TOKENS,
    )

    automatic_identical = prefix_model.generate_greedy(
        identical_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    check_exact_match(
        "ENTIRE-PROMPT PREFIX CORRECTNESS",
        standard_identical,
        automatic_identical,
    )

    # -------------------------------------------------
    # Final statistics.
    # -------------------------------------------------

    print("\nPREFIX CACHE STATISTICS")
    print("-" * 70)
    print(prefix_model.prefix_cache.cache_statistics())

    print("\n" + "=" * 70)
    print("AUTOMATIC PREFIX GENERATION TEST: PASS (2/2)")
    print("=" * 70)


if __name__ == "__main__":
    main()
