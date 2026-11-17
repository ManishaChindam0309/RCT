
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
NUM_PROMPTS = 5

WARMUP_RUNS = 2
BENCHMARK_RUNS = 10

SEED = 2026
BOOTSTRAP_ITERATIONS = 10000


# ============================================================
# MEMORY REPORTING
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
# MODEL LOADING
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

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        f"Checkpoint step: "
        f"{checkpoint.get('step', 'unknown')}"
    )

    print(
        f"Parameters: {model.parameter_count():,}"
    )

    del checkpoint
    gc.collect()

    return model


# ============================================================
# VALIDATION PROMPT LOADING
# ============================================================

def load_prompts():

    data = np.memmap(
        VALIDATION_PATH,
        dtype=np.uint16,
        mode="r",
    )

    rng = np.random.default_rng(SEED)

    max_start = (
        len(data)
        - max(PROMPT_LENGTHS)
        - NEW_TOKENS
    )

    starts = rng.integers(
        0,
        max_start,
        size=NUM_PROMPTS,
    )

    prompts = {}

    for prompt_length in PROMPT_LENGTHS:

        prompts[prompt_length] = []

        for start in starts:

            tokens = np.array(
                data[start:start + prompt_length],
                dtype=np.int64,
                copy=True,
            )

            prompt = torch.from_numpy(
                tokens
            ).unsqueeze(0)

            prompts[prompt_length].append(prompt)

    del data

    return prompts, starts.tolist()


# ============================================================
# STANDARD TRANSFORMER GENERATION
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

    # Cache allocation is included in measured latency.
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

        # Decode the newly generated token only if
        # another prediction is required.
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
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_speedup_ci(
    speedups,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=SEED,
):

    rng = np.random.default_rng(seed)

    values = np.asarray(
        speedups,
        dtype=np.float64,
    )

    samples = rng.choice(
        values,
        size=(iterations, len(values)),
        replace=True,
    )

    bootstrap_medians = np.median(
        samples,
        axis=1,
    )

    lower, upper = np.percentile(
        bootstrap_medians,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


# ============================================================
# MAIN BENCHMARK
# ============================================================

def main():

    print("=" * 72)
    print("RCT 100M — ROBUST KV-CACHE BENCHMARK")
    print("=" * 72)

    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)

    print(f"PyTorch: {torch.__version__}")
    print("Device: CPU")
    print(f"CPU threads: {CPU_THREADS}")

    print(f"Prompt lengths: {PROMPT_LENGTHS}")
    print(f"Generated tokens: {NEW_TOKENS}")
    print(f"Prompts per length: {NUM_PROMPTS}")

    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Timed repetitions: {BENCHMARK_RUNS}")

    print(
        f"Bootstrap iterations: "
        f"{BOOTSTRAP_ITERATIONS}"
    )

    report_memory("INITIAL MEMORY")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(VALIDATION_PATH)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    kv_model = RCT100MKV(model)
    kv_model.eval()

    prompts, starts = load_prompts()

    print(
        f"\nValidation prompt start positions: {starts}"
    )

    report_memory("AFTER MODEL LOADING")

    # --------------------------------------------------------
    # Results storage
    # --------------------------------------------------------

    results = {
        length: {
            "standard": [],
            "cached": [],
            "paired_speedups": [],
            "prompt_results": [],
        }
        for length in PROMPT_LENGTHS
    }

    # --------------------------------------------------------
    # Correctness checks and warm-up
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("CORRECTNESS CHECKS AND WARM-UP")
    print("=" * 72)

    total_correctness_checks = 0

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
                    f"Output mismatch: "
                    f"length={length}, "
                    f"prompt={prompt_index + 1}"
                )

            total_correctness_checks += 1

            for _ in range(WARMUP_RUNS):

                generate_standard(
                    model,
                    prompt,
                )

                generate_cached(
                    kv_model,
                    prompt,
                )

            print(
                f"Length {length}, "
                f"prompt {prompt_index + 1}: "
                f"exact match; "
                f"{WARMUP_RUNS} warm-ups completed"
            )

    print(
        f"\nCorrectness checks passed: "
        f"{total_correctness_checks}/"
        f"{total_correctness_checks}"
    )

    # --------------------------------------------------------
    # Timed measurements
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("TIMED MEASUREMENTS")
    print("=" * 72)

    for length in PROMPT_LENGTHS:

        for prompt_index, prompt in enumerate(
            prompts[length]
        ):

            paired_standard = []
            paired_cached = []

            for repetition in range(BENCHMARK_RUNS):

                # Alternate order to reduce systematic
                # execution-order bias.

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

                paired_standard.append(
                    standard_time
                )

                paired_cached.append(
                    cached_time
                )

                results[length]["standard"].append(
                    standard_time
                )

                results[length]["cached"].append(
                    cached_time
                )

                print(
                    f"Length={length}, "
                    f"prompt={prompt_index + 1}, "
                    f"run={repetition + 1}: "
                    f"standard={standard_time:.4f}s, "
                    f"cached={cached_time:.4f}s"
                )

            # Compute one speedup for this workload
            # using its paired median latencies.

            standard_median = statistics.median(
                paired_standard
            )

            cached_median = statistics.median(
                paired_cached
            )

            paired_speedup = (
                standard_median / cached_median
            )

            results[length]["paired_speedups"].append(
                paired_speedup
            )

            results[length]["prompt_results"].append(
                {
                    "prompt": prompt_index + 1,
                    "standard_median": standard_median,
                    "cached_median": cached_median,
                    "speedup": paired_speedup,
                }
            )

    # --------------------------------------------------------
    # Results by prompt length
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("RESULTS BY PROMPT LENGTH")
    print("=" * 72)

    all_standard = []
    all_cached = []
    all_paired_speedups = []

    for length in PROMPT_LENGTHS:

        standard_times = results[length]["standard"]
        cached_times = results[length]["cached"]
        paired_speedups = results[length]["paired_speedups"]

        standard_median = statistics.median(
            standard_times
        )

        cached_median = statistics.median(
            cached_times
        )

        aggregate_speedup = (
            standard_median / cached_median
        )

        latency_reduction = (
            (standard_median - cached_median)
            / standard_median
        ) * 100

        paired_median = statistics.median(
            paired_speedups
        )

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
            f"Aggregate median-based speedup: "
            f"{aggregate_speedup:.3f}x"
        )

        print(
            f"Latency reduction: "
            f"{latency_reduction:.2f}%"
        )

        print(
            f"Median per-prompt speedup: "
            f"{paired_median:.3f}x"
        )

        all_standard.extend(
            standard_times
        )

        all_cached.extend(
            cached_times
        )

        all_paired_speedups.extend(
            paired_speedups
        )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    overall_standard = statistics.median(
        all_standard
    )

    overall_cached = statistics.median(
        all_cached
    )

    overall_aggregate_speedup = (
        overall_standard / overall_cached
    )

    overall_paired_speedup = statistics.median(
        all_paired_speedups
    )

    ci_lower, ci_upper = bootstrap_speedup_ci(
        all_paired_speedups
    )

    print("\n" + "=" * 72)
    print("OVERALL SUMMARY")
    print("=" * 72)

    print(
        f"Total timed comparisons: "
        f"{len(all_standard)}"
    )

    print(
        f"Independent prompt workloads: "
        f"{len(all_paired_speedups)}"
    )

    print(
        f"Standard median: "
        f"{overall_standard:.4f}s"
    )

    print(
        f"KV median: "
        f"{overall_cached:.4f}s"
    )

    print(
        f"Aggregate median-based speedup: "
        f"{overall_aggregate_speedup:.3f}x"
    )

    print(
        f"Paired workload median speedup: "
        f"{overall_paired_speedup:.3f}x"
    )

    print(
        f"Bootstrap 95% CI: "
        f"[{ci_lower:.3f}x, {ci_upper:.3f}x]"
    )

    print(
        f"Exact output agreement: "
        f"{total_correctness_checks}/"
        f"{total_correctness_checks} workloads"
    )

    # --------------------------------------------------------
    # Per-prompt detail
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("PER-PROMPT SPEEDUPS")
    print("=" * 72)

    for length in PROMPT_LENGTHS:

        print(f"\nPrompt length: {length}")

        for item in results[length]["prompt_results"]:

            print(
                f"Prompt {item['prompt']}: "
                f"{item['speedup']:.3f}x "
                f"(standard "
                f"{item['standard_median']:.4f}s, "
                f"cached "
                f"{item['cached_median']:.4f}s)"
            )

    # --------------------------------------------------------
    # Cache memory
    # --------------------------------------------------------

    cache = kv_model.create_cache(1)

    print("\n" + "=" * 72)
    print("MEMORY")
    print("=" * 72)

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

    print("\nRobust benchmark completed.")


if __name__ == "__main__":
    main()
