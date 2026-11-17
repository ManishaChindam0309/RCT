
import gc
import sys
import time
import statistics
from pathlib import Path

import numpy as np
import torch
import psutil

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

PROMPT_LENGTHS = [8, 16, 24]
NEW_TOKENS = 8

NUM_PROMPTS = 30
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10

SEED = 20261102
BOOTSTRAP_ITERATIONS = 20000


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
# LOAD DISTINCT VALIDATION WINDOWS
# ============================================================

def load_prompts():

    data = np.memmap(
        VALIDATION_PATH,
        dtype=np.uint16,
        mode="r",
    )

    max_prompt_length = max(PROMPT_LENGTHS)

    # Reserve a complete non-overlapping window for each
    # selected location, including generation space.
    window_size = max_prompt_length + NEW_TOKENS

    total_windows = len(data) // window_size

    if NUM_PROMPTS > total_windows:
        raise ValueError(
            f"Requested {NUM_PROMPTS} windows, "
            f"but only {total_windows} are available."
        )

    rng = np.random.default_rng(SEED)

    # Sample distinct window indices without replacement.
    window_indices = rng.choice(
        total_windows,
        size=NUM_PROMPTS,
        replace=False,
    )

    window_indices.sort()

    starts = (
        window_indices * window_size
    ).astype(np.int64)

    prompts = {}

    for prompt_length in PROMPT_LENGTHS:

        prompts[prompt_length] = []

        for start in starts:

            tokens = np.array(
                data[
                    start:start + prompt_length
                ],
                dtype=np.int64,
                copy=True,
            )

            prompts[prompt_length].append(
                torch.from_numpy(tokens).unsqueeze(0)
            )

    del data

    return prompts, starts.tolist(), window_size


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

    # Include cache allocation in end-to-end latency.
    cache = kv_model.create_cache(
        batch_size=prompt.size(0)
    )

    generated = prompt.clone()

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

        if step + 1 < NEW_TOKENS:

            logits = kv_model.decode_step(
                next_token,
                cache,
            )

    return generated


# ============================================================
# TIMING
# ============================================================

def timed_call(function):

    start = time.perf_counter()

    output = function()

    elapsed = time.perf_counter() - start

    return elapsed, output


# ============================================================
# CLUSTER BOOTSTRAP
# ============================================================

def cluster_bootstrap_ci(
    speedup_matrix,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=SEED,
):

    """
    speedup_matrix:
        Shape [number_of_prompt_locations, number_of_lengths].

    Resample complete prompt locations, retaining all
    prompt lengths belonging to each selected location.

    The statistic is the median of the resampled
    workload-level speedups.
    """

    matrix = np.asarray(
        speedup_matrix,
        dtype=np.float64,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Speedup matrix must be two-dimensional."
        )

    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            "Speedup matrix contains non-finite values."
        )

    rng = np.random.default_rng(seed)

    num_clusters = matrix.shape[0]

    bootstrap_medians = np.empty(
        iterations,
        dtype=np.float64,
    )

    for iteration in range(iterations):

        selected = rng.integers(
            0,
            num_clusters,
            size=num_clusters,
        )

        sampled_matrix = matrix[selected, :]

        bootstrap_medians[iteration] = np.median(
            sampled_matrix
        )

    lower, upper = np.percentile(
        bootstrap_medians,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("RCT 100M — LARGE CLUSTER-AWARE KV BENCHMARK")
    print("=" * 76)

    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)

    print(f"PyTorch: {torch.__version__}")
    print("Device: CPU")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Prompt lengths: {PROMPT_LENGTHS}")
    print(f"Generated tokens: {NEW_TOKENS}")
    print(f"Distinct prompt locations: {NUM_PROMPTS}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Timed repetitions: {BENCHMARK_RUNS}")
    print(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")

    report_memory("INITIAL MEMORY")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(VALIDATION_PATH)

    # --------------------------------------------------------
    # Load model and prompts
    # --------------------------------------------------------

    model = load_model()

    kv_model = RCT100MKV(model)
    kv_model.eval()

    prompts, starts, window_size = load_prompts()

    print(
        f"\nDistinct validation prompt locations: "
        f"{len(starts)}"
    )

    print(
        f"Non-overlapping window size: "
        f"{window_size} tokens"
    )

    print(f"Selected starts: {starts}")

    report_memory("AFTER MODEL LOADING")

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {
        length: {
            "standard": [],
            "cached": [],
            "paired_speedups": [],
            "prompt_speedups": [],
        }
        for length in PROMPT_LENGTHS
    }

    # One row per prompt location; one column per length.
    speedup_matrix = np.empty(
        (NUM_PROMPTS, len(PROMPT_LENGTHS)),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Correctness and warm-up
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("CORRECTNESS CHECKS AND WARM-UP")
    print("=" * 76)

    correctness_checks = 0

    for length in PROMPT_LENGTHS:

        for prompt_index, prompt in enumerate(
            prompts[length]
        ):

            standard_output = generate_standard(
                model,
                prompt,
            )

            cached_output = generate_cached(
                kv_model,
                prompt,
            )

            if not torch.equal(
                standard_output,
                cached_output,
            ):

                raise RuntimeError(
                    f"Output mismatch at "
                    f"length={length}, "
                    f"prompt={prompt_index + 1}"
                )

            correctness_checks += 1

            for _ in range(WARMUP_RUNS):

                generate_standard(model, prompt)
                generate_cached(kv_model, prompt)

            print(
                f"Length {length}, "
                f"prompt {prompt_index + 1:02d}: "
                f"exact match; warm-up complete"
            )

    expected_checks = NUM_PROMPTS * len(PROMPT_LENGTHS)

    print(
        f"\nCorrectness checks: "
        f"{correctness_checks}/{expected_checks}"
    )

    # --------------------------------------------------------
    # Timed benchmark
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("TIMED BENCHMARK")
    print("=" * 76)

    for length_index, length in enumerate(PROMPT_LENGTHS):

        for prompt_index, prompt in enumerate(
            prompts[length]
        ):

            standard_times = []
            cached_times = []

            for repetition in range(BENCHMARK_RUNS):

                # Alternate order for timing balance.
                if repetition % 2 == 0:

                    standard_time, standard_output = timed_call(
                        lambda: generate_standard(
                            model,
                            prompt,
                        )
                    )

                    cached_time, cached_output = timed_call(
                        lambda: generate_cached(
                            kv_model,
                            prompt,
                        )
                    )

                else:

                    cached_time, cached_output = timed_call(
                        lambda: generate_cached(
                            kv_model,
                            prompt,
                        )
                    )

                    standard_time, standard_output = timed_call(
                        lambda: generate_standard(
                            model,
                            prompt,
                        )
                    )

                if not torch.equal(
                    standard_output,
                    cached_output,
                ):

                    raise RuntimeError(
                        "Output mismatch during timed benchmark."
                    )

                standard_times.append(standard_time)
                cached_times.append(cached_time)

                results[length]["standard"].append(
                    standard_time
                )

                results[length]["cached"].append(
                    cached_time
                )

                print(
                    f"Length={length}, "
                    f"prompt={prompt_index + 1:02d}, "
                    f"run={repetition + 1:02d}: "
                    f"standard={standard_time:.4f}s, "
                    f"cached={cached_time:.4f}s"
                )

            standard_median = statistics.median(
                standard_times
            )

            cached_median = statistics.median(
                cached_times
            )

            paired_speedup = (
                standard_median / cached_median
            )

            speedup_matrix[
                prompt_index,
                length_index,
            ] = paired_speedup

            results[length]["paired_speedups"].append(
                paired_speedup
            )

            results[length]["prompt_speedups"].append(
                paired_speedup
            )

    # --------------------------------------------------------
    # Per-length summary
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("RESULTS BY PROMPT LENGTH")
    print("=" * 76)

    for length in PROMPT_LENGTHS:

        standard_times = results[length]["standard"]
        cached_times = results[length]["cached"]
        paired = results[length]["paired_speedups"]

        standard_median = statistics.median(
            standard_times
        )

        cached_median = statistics.median(
            cached_times
        )

        aggregate_speedup = (
            standard_median / cached_median
        )

        reduction = (
            (standard_median - cached_median)
            / standard_median
        ) * 100

        print(f"\nPrompt length: {length}")

        print(
            f"Measurements per method: "
            f"{len(standard_times)}"
        )

        print(
            f"Standard median: "
            f"{standard_median:.4f}s"
        )

        print(
            f"KV median: "
            f"{cached_median:.4f}s"
        )

        print(
            f"Aggregate median speedup: "
            f"{aggregate_speedup:.3f}x"
        )

        print(
            f"Latency reduction: "
            f"{reduction:.2f}%"
        )

        print(
            f"Median per-prompt speedup: "
            f"{statistics.median(paired):.3f}x"
        )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    all_standard = [
        value
        for length in PROMPT_LENGTHS
        for value in results[length]["standard"]
    ]

    all_cached = [
        value
        for length in PROMPT_LENGTHS
        for value in results[length]["cached"]
    ]

    all_speedups = speedup_matrix.reshape(-1)

    overall_standard = statistics.median(all_standard)
    overall_cached = statistics.median(all_cached)

    overall_aggregate = (
        overall_standard / overall_cached
    )

    paired_median = statistics.median(
        all_speedups
    )

    ci_lower, ci_upper = cluster_bootstrap_ci(
        speedup_matrix
    )

    print("\n" + "=" * 76)
    print("OVERALL RESULTS")
    print("=" * 76)

    print(
        f"Prompt locations: {NUM_PROMPTS}"
    )

    print(
        f"Prompt lengths: {len(PROMPT_LENGTHS)}"
    )

    print(
        f"Total workload combinations: "
        f"{NUM_PROMPTS * len(PROMPT_LENGTHS)}"
    )

    print(
        f"Timed measurements per method: "
        f"{len(all_standard)}"
    )

    print(
        f"Standard overall median: "
        f"{overall_standard:.4f}s"
    )

    print(
        f"KV overall median: "
        f"{overall_cached:.4f}s"
    )

    print(
        f"Aggregate median speedup: "
        f"{overall_aggregate:.3f}x"
    )

    print(
        f"Paired workload median speedup: "
        f"{paired_median:.3f}x"
    )

    print(
        f"Cluster bootstrap 95% CI: "
        f"[{ci_lower:.3f}x, {ci_upper:.3f}x]"
    )

    print(
        f"Exact output agreement: "
        f"{correctness_checks}/{expected_checks}"
    )

    # --------------------------------------------------------
    # Per-prompt results
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("PER-PROMPT SPEEDUPS")
    print("=" * 76)

    for prompt_index, start in enumerate(starts):

        values = speedup_matrix[prompt_index]

        print(
            f"Prompt {prompt_index + 1:02d}, "
            f"start={start}: "
            + ", ".join(
                f"L{length}={values[index]:.3f}x"
                for index, length in enumerate(PROMPT_LENGTHS)
            )
        )

    # --------------------------------------------------------
    # Cache memory
    # --------------------------------------------------------

    cache = kv_model.create_cache(1)

    print("\n" + "=" * 76)
    print("MEMORY")
    print("=" * 76)

    print(
        f"KV cache memory, batch 1: "
        f"{cache.memory_bytes() / (1024 ** 2):.2f} MiB"
    )

    del cache

    report_memory("FINAL MEMORY")

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del kv_model
    del model

    gc.collect()

    print("\nLarge benchmark completed.")


if __name__ == "__main__":
    main()
