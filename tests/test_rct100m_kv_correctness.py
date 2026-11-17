
import gc
import sys
from pathlib import Path

# Add the project root to Python's import path.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import psutil
import torch

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV


# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

VALIDATION_PATH = (
    PROJECT_DIR
    / "data"
    / "rct30"
    / "validation_expanded_tokens.bin"
)

VOCAB_SIZE = 8192
HIDDEN_SIZE = 512
NUM_LAYERS = 30
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048
MAX_SEQ_LEN = 512

CPU_THREADS = 4

PROMPT_LENGTH = 12
DECODE_STEPS = 8

# Numerical tolerance for comparing FP32 implementations.
ABSOLUTE_TOLERANCE = 1e-3


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def report_memory(label):

    memory = psutil.virtual_memory()
    process = psutil.Process()

    print(f"\n[{label}]")
    print(
        f"Available RAM: "
        f"{memory.available / (1024 ** 3):.2f} GB"
    )
    print(
        f"Process RAM: "
        f"{process.memory_info().rss / (1024 ** 3):.2f} GB"
    )


# ============================================================
# MODEL LOADING
# ============================================================

def create_model():

    config = RCT100MConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
        tie_embeddings=True,
    )

    return RCT100M(config)


def load_checkpoint():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("\nLoading continued step-1500 checkpoint...")

    model = create_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    elif "model" in checkpoint:
        state_dict = checkpoint["model"]

    else:
        raise KeyError(
            "Could not find model weights. "
            f"Checkpoint keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print("Checkpoint loaded successfully.")

    if "step" in checkpoint:
        print(f"Training step: {checkpoint['step']}")

    if "val_loss" in checkpoint:
        print(
            f"Saved validation loss: "
            f"{checkpoint['val_loss']:.6f}"
        )

    print(
        f"Parameter count: "
        f"{model.parameter_count():,}"
    )

    del checkpoint
    gc.collect()

    return model


# ============================================================
# VALIDATION INPUT
# ============================================================

def load_test_tokens():

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}"
        )

    data = np.memmap(
        VALIDATION_PATH,
        dtype=np.uint16,
        mode="r",
    )

    required_tokens = PROMPT_LENGTH + DECODE_STEPS

    if len(data) < required_tokens:
        raise ValueError(
            "Validation dataset is too small."
        )

    # Deterministic real validation tokens.
    tokens = np.array(
        data[:required_tokens],
        dtype=np.int64,
        copy=True,
    )

    del data

    return torch.from_numpy(tokens).unsqueeze(0)


# ============================================================
# LOGIT COMPARISON
# ============================================================

def compare_logits(
    standard_logits,
    cached_logits,
    label,
):

    standard_logits = standard_logits.float()
    cached_logits = cached_logits.float()

    difference = torch.abs(
        standard_logits - cached_logits
    )

    max_difference = difference.max().item()
    mean_difference = difference.mean().item()

    standard_top1 = standard_logits.argmax(dim=-1)
    cached_top1 = cached_logits.argmax(dim=-1)

    top1_matches = (
        standard_top1 == cached_top1
    ).float().mean().item()

    passed = (
        max_difference <= ABSOLUTE_TOLERANCE
    )

    print(f"\n{label}")
    print(f"Maximum absolute difference: {max_difference:.8f}")
    print(f"Mean absolute difference:    {mean_difference:.8f}")
    print(f"Top-1 token agreement:       {top1_matches * 100:.2f}%")
    print(f"Tolerance:                   {ABSOLUTE_TOLERANCE}")

    print(
        "Numerical comparison:       "
        + ("PASS" if passed else "CHECK")
    )

    return {
        "max_difference": max_difference,
        "mean_difference": mean_difference,
        "top1_agreement": top1_matches,
        "passed": passed,
    }


# ============================================================
# MAIN CORRECTNESS TEST
# ============================================================

def main():

    print("=" * 70)
    print("RCT 100M — KV CACHE CORRECTNESS TEST")
    print("=" * 70)

    torch.set_num_threads(CPU_THREADS)

    print(f"PyTorch: {torch.__version__}")
    print("Device: CPU")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Prompt length: {PROMPT_LENGTH}")
    print(f"Incremental decoding steps: {DECODE_STEPS}")

    report_memory("INITIAL MEMORY")

    # --------------------------------------------------------
    # Load standard model
    # --------------------------------------------------------

    standard_model = load_checkpoint()

    # --------------------------------------------------------
    # Construct KV model using exactly the same weights
    # --------------------------------------------------------

    print("\nCreating KV-cache model...")

    kv_model = RCT100MKV(standard_model)
    kv_model.eval()

    print("KV-cache model created.")

    # Both models share the same underlying trained parameters.
    print(
        f"Standard parameters: "
        f"{standard_model.parameter_count():,}"
    )

    print(
        f"KV parameters: "
        f"{kv_model.parameter_count():,}"
    )

    # --------------------------------------------------------
    # Prepare deterministic test tokens
    # --------------------------------------------------------

    tokens = load_test_tokens()

    prompt = tokens[:, :PROMPT_LENGTH]

    continuation = tokens[
        :,
        PROMPT_LENGTH:PROMPT_LENGTH + DECODE_STEPS
    ]

    print("\nTest input prepared.")
    print(f"Prompt shape: {tuple(prompt.shape)}")
    print(f"Continuation shape: {tuple(continuation.shape)}")

    results = []

    # --------------------------------------------------------
    # Test 1: Initial prompt prefill
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 1 — INITIAL PROMPT PREFILL")
    print("=" * 70)

    with torch.inference_mode():

        standard_logits = standard_model(prompt)

        cache = kv_model.create_cache(
            batch_size=prompt.size(0)
        )

        cached_logits = kv_model.decode_step(
            prompt,
            cache,
        )

        result = compare_logits(
            standard_logits,
            cached_logits,
            "Initial prompt logits",
        )

        results.append(result)

    print(f"KV cache position: {cache.position}")

    # --------------------------------------------------------
    # Test 2: Incremental decoding
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEST 2 — INCREMENTAL DECODING")
    print("=" * 70)

    current_prefix = prompt.clone()

    with torch.inference_mode():

        for step in range(DECODE_STEPS):

            next_input_token = continuation[
                :,
                step:step + 1
            ]

            # Append the same real token to both paths.
            current_prefix = torch.cat(
                [
                    current_prefix,
                    next_input_token,
                ],
                dim=1,
            )

            # Standard path recomputes the entire prefix.
            standard_logits = standard_model(
                current_prefix
            )

            # Cached path processes only the new token.
            cached_logits = kv_model.decode_step(
                next_input_token,
                cache,
            )

            # Compare the last-position logits.
            result = compare_logits(
                standard_logits[:, -1, :],
                cached_logits[:, -1, :],
                f"Incremental decoding step {step + 1}",
            )

            results.append(result)

            print(
                f"Cache position: {cache.position}"
            )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL CORRECTNESS REPORT")
    print("=" * 70)

    passed_count = sum(
        result["passed"]
        for result in results
    )

    total_count = len(results)

    overall_max_difference = max(
        result["max_difference"]
        for result in results
    )

    average_difference = sum(
        result["mean_difference"]
        for result in results
    ) / total_count

    average_top1 = sum(
        result["top1_agreement"]
        for result in results
    ) / total_count

    print(f"Comparisons: {total_count}")
    print(f"Numerical comparisons passing: {passed_count}")
    print(
        f"Overall maximum absolute difference: "
        f"{overall_max_difference:.8f}"
    )
    print(
        f"Average mean absolute difference: "
        f"{average_difference:.8f}"
    )
    print(
        f"Average top-1 agreement: "
        f"{average_top1 * 100:.2f}%"
    )

    print(
        "\nOverall result: "
        + (
            "PASS"
            if passed_count == total_count
            else "REVIEW REQUIRED"
        )
    )

    print("\nKV cache memory:")
    print(
        f"{cache.memory_bytes() / (1024 ** 2):.2f} MiB"
    )

    report_memory("FINAL MEMORY")

    del cache
    del kv_model
    del standard_model

    gc.collect()

    print("\nCorrectness test completed.")


if __name__ == "__main__":
    main()
