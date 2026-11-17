
import gc
import sys
import time
import statistics
from pathlib import Path

import numpy as np
import psutil
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV


# ============================================================
# CONFIGURATION
# ============================================================

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
NEW_TOKENS = 8

WARMUP_RUNS = 1
BENCHMARK_RUNS = 3


# ============================================================
# MEMORY
# ============================================================

def report_memory(label):

    memory = psutil.virtual_memory()
    process = psutil.Process()

    print(f"\n{label}")
    print(
        f"Available RAM: "
        f"{memory.available / (1024 ** 3):.2f} GB"
    )
    print(
        f"Process RAM: "
        f"{process.memory_info().rss / (1024 ** 3):.2f} GB"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

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

    model = RCT100M(config)

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
            f"Unknown checkpoint format: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Parameters: {model.parameter_count():,}")

    del checkpoint
    gc.collect()

    return model


# ============================================================
# TEST PROMPT
# ============================================================

def load_prompt():

    data = np.memmap(
        VALIDATION_PATH,
        dtype=np.uint16,
        mode="r",
    )

    prompt = np.array(
        data[:PROMPT_LENGTH],
        dtype=np.int64,
        copy=True,
    )

    del data

    return torch.from_numpy(prompt).unsqueeze(0)


# ============================================================
# STANDARD GENERATION
# ============================================================

@torch.inference_mode()
def generate_standard(model, prompt):

    generated = prompt.clone()

    for _ in range(NEW_TOKENS):

        context = generated[:, -MAX_SEQ_LEN:]

        logits = model(context)

        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated


# ============================================================
# KV-CACHED GENERATION
# ============================================================

@torch.inference_mode()
def generate_cached(kv_model, prompt):

    # Cache creation is deliberately included in timing.
    cache = kv_model.create_cache(
        batch_size=prompt.size(0)
    )

    generated = prompt.clone()

    # Initial prompt prefill.
    logits = kv_model.decode_step(
        prompt,
        cache,
    )

    for step in range(NEW_TOKENS):

        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        # Process only the new token when another prediction
        # is required. No unnecessary final decoding step.
        if step + 1 < NEW_TOKENS:

            logits = kv_model.decode_step(
                next_token,
                cache,
            )

    return generated


# ============================================================
# TIMING
# ============================================================

def measure(function):

    start = time.perf_counter()

    output = function()

    elapsed = time.perf_counter() - start

    return elapsed, output


def summarize(label, timings):

    median_time = statistics.median(timings)
    mean_time = statistics.mean(timings)

    throughput = NEW_TOKENS / median_time

    print(f"\n{label}")
    print(f"Individual runs: {[round(x, 4) for x in timings]}")
    print(f"Mean latency: {mean_time:.4f} seconds")
    print(f"Median latency: {median_time:.4f} seconds")
    print(f"Generated tokens/second: {throughput:.3f}")

    return median_time


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RCT 100M — STANDARD VS KV-CACHE BENCHMARK")
    print("=" * 70)

    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(42)

    print(f"PyTorch: {torch.__version__}")
    print("Device: CPU")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Prompt length: {PROMPT_LENGTH}")
    print(f"Generated tokens: {NEW_TOKENS}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Benchmark runs: {BENCHMARK_RUNS}")

    report_memory("INITIAL MEMORY")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(VALIDATION_PATH)

    model = load_model()

    kv_model = RCT100MKV(model)
    kv_model.eval()

    prompt = load_prompt()

    print(f"\nPrompt shape: {tuple(prompt.shape)}")

    report_memory("AFTER MODEL LOADING")

    # --------------------------------------------------------
    # Correctness check before timing
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRE-BENCHMARK OUTPUT CORRECTNESS")
    print("=" * 70)

    standard_output = generate_standard(model, prompt)
    cached_output = generate_cached(kv_model, prompt)

    outputs_match = torch.equal(
        standard_output,
        cached_output,
    )

    print(f"Standard output: {standard_output.tolist()}")
    print(f"Cached output:   {cached_output.tolist()}")
    print(f"Exact output match: {outputs_match}")

    if not outputs_match:
        raise RuntimeError(
            "Generation outputs differ. "
            "Performance benchmark aborted."
        )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WARM-UP")
    print("=" * 70)

    for index in range(WARMUP_RUNS):

        generate_standard(model, prompt)
        generate_cached(kv_model, prompt)

        print(f"Warm-up {index + 1}/{WARMUP_RUNS} completed.")

    # --------------------------------------------------------
    # Timed benchmark
    # Alternate execution order to reduce order bias.
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TIMED BENCHMARK")
    print("=" * 70)

    standard_times = []
    cached_times = []

    for index in range(BENCHMARK_RUNS):

        print(f"\nBenchmark repetition {index + 1}")

        if index % 2 == 0:

            standard_time, standard_output = measure(
                lambda: generate_standard(model, prompt)
            )

            cached_time, cached_output = measure(
                lambda: generate_cached(kv_model, prompt)
            )

        else:

            cached_time, cached_output = measure(
                lambda: generate_cached(kv_model, prompt)
            )

            standard_time, standard_output = measure(
                lambda: generate_standard(model, prompt)
            )

        if not torch.equal(standard_output, cached_output):
            raise RuntimeError(
                "Output mismatch during benchmark."
            )

        standard_times.append(standard_time)
        cached_times.append(cached_time)

        print(f"Standard: {standard_time:.4f} seconds")
        print(f"KV cache: {cached_time:.4f} seconds")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 70)

    standard_median = summarize(
        "STANDARD TRANSFORMER",
        standard_times,
    )

    cached_median = summarize(
        "RCT WITH KV CACHE",
        cached_times,
    )

    speedup = standard_median / cached_median

    latency_reduction = (
        (standard_median - cached_median)
        / standard_median
    ) * 100

    print("\n" + "-" * 70)
    print(f"Median latency speedup: {speedup:.3f}x")
    print(f"Median latency reduction: {latency_reduction:.2f}%")
    print("Exact output agreement: 100%")
    print("-" * 70)

    print("\nKV-cache allocation:")
    cache = kv_model.create_cache(1)
    print(
        f"{cache.memory_bytes() / (1024 ** 2):.2f} MiB"
    )
    del cache

    report_memory("FINAL MEMORY")

    del kv_model
    del model

    gc.collect()

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
