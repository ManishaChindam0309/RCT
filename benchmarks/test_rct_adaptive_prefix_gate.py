
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

PROMPT_LENGTH = 32
NEW_TOKENS = 8


def create_prompts(
    batch_size,
    prefix_len,
    prompt_length,
    vocab_size,
    device,
):
    if prefix_len > 0:
        shared_prefix = torch.randint(
            0,
            vocab_size,
            (1, prefix_len),
            device=device,
        )

        suffix = torch.randint(
            0,
            vocab_size,
            (
                batch_size,
                prompt_length - prefix_len,
            ),
            device=device,
        )

        return torch.cat(
            [
                shared_prefix.expand(batch_size, -1),
                suffix,
            ],
            dim=1,
        )

    return torch.randint(
        0,
        vocab_size,
        (batch_size, prompt_length),
        device=device,
    )


def check_workload(
    inference,
    batch_size,
    prefix_len,
    expected_decision,
    expected_reason,
    device,
):
    prompt = create_prompts(
        batch_size=batch_size,
        prefix_len=prefix_len,
        prompt_length=PROMPT_LENGTH,
        vocab_size=inference.model.config.vocab_size,
        device=device,
    )

    detected = inference.detect_shared_prefix(prompt)

    if detected != prefix_len:
        raise AssertionError(
            f"Expected prefix {prefix_len}, detected {detected}."
        )

    inference.reset_prefix_gate_statistics()

    expected = inference.generate(
        prompt,
        max_new_tokens=NEW_TOKENS,
        mode="kv",
    )

    actual = inference.generate(
        prompt,
        max_new_tokens=NEW_TOKENS,
        mode="auto",
    )

    if not torch.equal(expected, actual):
        raise AssertionError(
            "Adaptive prefix output differs from KV output."
        )

    stats = inference.prefix_gate_statistics()

    if stats["last_decision"] != expected_decision:
        raise AssertionError(
            f"Expected decision {expected_decision}, "
            f"got {stats['last_decision']}."
        )

    if stats["last_reason"] != expected_reason:
        raise AssertionError(
            f"Expected reason {expected_reason}, "
            f"got {stats['last_reason']}."
        )

    if expected_decision == "prefix":
        if stats["gate_activations"] != 1:
            raise AssertionError(
                "Prefix gate activation counter is incorrect."
            )

        if stats["gate_bypasses"] != 0:
            raise AssertionError(
                "Unexpected prefix gate bypass."
            )

    else:
        if stats["gate_bypasses"] != 1:
            raise AssertionError(
                "Prefix gate bypass counter is incorrect."
            )

        if stats["gate_activations"] != 0:
            raise AssertionError(
                "Unexpected prefix gate activation."
            )

    print(
        f"Batch={batch_size:2d} "
        f"Prefix={prefix_len:2d} "
        f"Detected={detected:2d} "
        f"Decision={stats['last_decision']:6s} "
        f"Reason={stats['last_reason']:36s} "
        f"Exact match=PASS"
    )


def check_unsupported_prompt_length(
    inference,
    device,
):
    prompt = create_prompts(
        batch_size=4,
        prefix_len=8,
        prompt_length=24,
        vocab_size=inference.model.config.vocab_size,
        device=device,
    )

    inference.reset_prefix_gate_statistics()

    expected = inference.generate(
        prompt,
        max_new_tokens=NEW_TOKENS,
        mode="kv",
    )

    actual = inference.generate(
        prompt,
        max_new_tokens=NEW_TOKENS,
        mode="auto",
    )

    if not torch.equal(expected, actual):
        raise AssertionError(
            "Unsupported prompt-length fallback mismatch."
        )

    stats = inference.prefix_gate_statistics()

    assert stats["last_decision"] == "kv"
    assert stats["last_reason"] == "untested_prompt_length"

    print(
        "Unsupported prompt length: KV fallback PASS"
    )


def check_unsupported_generation_length(
    inference,
    device,
):
    prompt = create_prompts(
        batch_size=4,
        prefix_len=8,
        prompt_length=PROMPT_LENGTH,
        vocab_size=inference.model.config.vocab_size,
        device=device,
    )

    inference.reset_prefix_gate_statistics()

    expected = inference.generate(
        prompt,
        max_new_tokens=4,
        mode="kv",
    )

    actual = inference.generate(
        prompt,
        max_new_tokens=4,
        mode="auto",
    )

    if not torch.equal(expected, actual):
        raise AssertionError(
            "Unsupported generation-length fallback mismatch."
        )

    stats = inference.prefix_gate_statistics()

    assert stats["last_decision"] == "kv"
    assert stats["last_reason"] == "untested_generation_length"

    print(
        "Unsupported generation length: KV fallback PASS"
    )


def check_no_shared_prefix(
    inference,
    device,
):
    prompt = create_prompts(
        batch_size=4,
        prefix_len=0,
        prompt_length=PROMPT_LENGTH,
        vocab_size=inference.model.config.vocab_size,
        device=device,
    )

    inference.reset_prefix_gate_statistics()

    detected = inference.detect_shared_prefix(prompt)

    if detected != 0:
        raise AssertionError(
            f"Expected no common prefix, detected {detected}."
        )

    expected = inference.generate(
        prompt,
        max_new_tokens=NEW_TOKENS,
        mode="kv",
    )

    actual = inference.generate(
        prompt,
        max_new_tokens=NEW_TOKENS,
        mode="auto",
    )

    if not torch.equal(expected, actual):
        raise AssertionError(
            "No-shared-prefix fallback mismatch."
        )

    stats = inference.prefix_gate_statistics()

    assert stats["last_decision"] == "kv"
    assert stats["last_reason"] == "no_shared_prefix"

    print(
        "No shared prefix: KV fallback PASS"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this test."
        )

    device = torch.device("cuda")

    torch.manual_seed(20261106)
    torch.cuda.manual_seed_all(20261106)

    print("=" * 78)
    print("RCT ADAPTIVE PREFIX GATE TEST")
    print("=" * 78)

    print("Checkpoint:", CHECKPOINT)
    print("GPU:", torch.cuda.get_device_name(device))

    inference = load_rct_inference(
        CHECKPOINT,
        device=device,
    )

    print(
        "Model parameters:",
        sum(
            parameter.numel()
            for parameter in inference.model.parameters()
        ),
    )

    print("\nTesting benchmark-supported workloads")

    workloads = [
        # Batch 2: no demonstrated benefit.
        (2, 8, "kv", "untested_batch_size"),
        (2, 16, "kv", "untested_batch_size"),
        (2, 24, "kv", "untested_batch_size"),

        # Batch 4: only prefix 24 is enabled.
        (4, 8, "kv", "unprofitable_or_untested_prefix_length"),
        (4, 16, "kv", "unprofitable_or_untested_prefix_length"),
        (4, 24, "prefix", "benchmark_supported_workload"),

        # Batch 8: only prefix 24 is enabled.
        (8, 8, "kv", "unprofitable_or_untested_prefix_length"),
        (8, 16, "kv", "unprofitable_or_untested_prefix_length"),
        (8, 24, "prefix", "benchmark_supported_workload"),
    ]

    passed = 0

    for batch_size, prefix_len, decision, reason in workloads:
        check_workload(
            inference=inference,
            batch_size=batch_size,
            prefix_len=prefix_len,
            expected_decision=decision,
            expected_reason=reason,
            device=device,
        )

        passed += 1

    print("\nTesting unsupported workloads")

    check_unsupported_prompt_length(
        inference,
        device,
    )
    passed += 1

    check_unsupported_generation_length(
        inference,
        device,
    )
    passed += 1

    check_no_shared_prefix(
        inference,
        device,
    )
    passed += 1

    print("\nFinal gate statistics:")
    print(inference.prefix_gate_statistics())

    print("\nFinal adaptive deduplication statistics:")
    print(inference.adaptive_statistics())

    print("\n" + "=" * 78)
    print(
        f"RCT ADAPTIVE PREFIX GATE TEST: PASS ({passed}/12)"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
