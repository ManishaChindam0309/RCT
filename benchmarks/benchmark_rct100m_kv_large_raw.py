
import gc
import sys
import time
import csv
import json
import statistics
import platform
from pathlib import Path
from datetime import datetime, timezone

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
    PROJECT_DIR / "checkpoints" / "rct100m_continued" / "best_model.pt"
)

VALIDATION_PATH = (
    PROJECT_DIR / "data" / "rct30" / "validation_expanded_tokens.bin"
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

RESULTS_ROOT = PROJECT_DIR / "results" / "rct100m_kv_large"


# ============================================================
# MEMORY
# ============================================================

def get_memory_info():
    memory = psutil.virtual_memory()
    process = psutil.Process()

    return {
        "available_ram_gb": round(
            memory.available / (1024 ** 3), 3
        ),
        "process_ram_gb": round(
            process.memory_info().rss / (1024 ** 3), 3
        ),
    }


def report_memory(label):
    info = get_memory_info()

    print(f"\n{label}")
    print(f"Available RAM: {info['available_ram_gb']:.2f} GB")
    print(f"Process RAM: {info['process_ram_gb']:.2f} GB")

    return info


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

    checkpoint_step = checkpoint.get("step", "unknown")

    print(f"Checkpoint step: {checkpoint_step}")
    print(f"Parameters: {model.parameter_count():,}")

    del checkpoint
    gc.collect()

    return model, checkpoint_step


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
    window_size = max_prompt_length + NEW_TOKENS

    total_windows = len(data) // window_size

    if NUM_PROMPTS > total_windows:
        raise ValueError(
            f"Requested {NUM_PROMPTS} windows, "
            f"but only {total_windows} are available."
        )

    rng = np.random.default_rng(SEED)

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
                data[start:start + prompt_length],
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

    # Cache allocation is included in end-to-end latency.
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
# SAVE WORKLOAD SUMMARY CSV
# ============================================================

def save_workload_summary(
    output_path,
    starts,
    speedup_matrix,
    workload_medians,
):

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "prompt_number",
            "validation_start_token",
            "prompt_length",
            "standard_median_seconds",
            "cached_median_seconds",
            "speedup",
            "latency_reduction_percent",
        ])

        for prompt_index, start in enumerate(starts):

            for length_index, length in enumerate(PROMPT_LENGTHS):

                standard_median, cached_median = (
                    workload_medians[prompt_index][length_index]
                )

                speedup = speedup_matrix[
                    prompt_index,
                    length_index,
                ]

                reduction = (
                    1.0 - cached_median / standard_median
                ) * 100

                writer.writerow([
                    prompt_index + 1,
                    start,
                    length,
                    f"{standard_median:.8f}",
                    f"{cached_median:.8f}",
                    f"{speedup:.8f}",
                    f"{reduction:.4f}",
                ])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 76)
    print("RCT 100M — LARGE KV BENCHMARK WITH RAW DATA EXPORT")
    print("=" * 76)

    # Create a unique folder for this run.
    run_timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    run_dir = RESULTS_ROOT / f"run_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    raw_csv_path = run_dir / "raw_timing_records.csv"
    workload_csv_path = run_dir / "per_workload_summary.csv"
    json_path = run_dir / "experiment_results.json"

    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)

    print(f"PyTorch: {torch.__version__}")
    print(f"Python: {platform.python_version()}")
    print(f"Device: CPU")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Prompt lengths: {PROMPT_LENGTHS}")
    print(f"Generated tokens: {NEW_TOKENS}")
    print(f"Distinct prompt locations: {NUM_PROMPTS}")
    print(f"Warm-up runs: {WARMUP_RUNS}")
    print(f"Timed repetitions: {BENCHMARK_RUNS}")
    print(f"Bootstrap iterations: {BOOTSTRAP_ITERATIONS}")
    print(f"Output folder: {run_dir}")

    initial_memory = report_memory("INITIAL MEMORY")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(VALIDATION_PATH)

    # --------------------------------------------------------
    # Load model and prompts
    # --------------------------------------------------------

    model, checkpoint_step = load_model()

    kv_model = RCT100MKV(model)
    kv_model.eval()

    prompts, starts, window_size = load_prompts()

    print(f"\nDistinct validation locations: {len(starts)}")
    print(f"Non-overlapping window size: {window_size} tokens")
    print(f"Selected starts: {starts}")

    after_loading_memory = report_memory(
        "AFTER MODEL LOADING"
    )

    # --------------------------------------------------------
    # Correctness checks and warm-up
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("CORRECTNESS CHECKS AND WARM-UP")
    print("=" * 76)

    correctness_checks = 0
    expected_checks = NUM_PROMPTS * len(PROMPT_LENGTHS)

    for length in PROMPT_LENGTHS:

        for prompt_index, prompt in enumerate(prompts[length]):

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
                    f"Output mismatch: length={length}, "
                    f"prompt={prompt_index + 1}"
                )

            correctness_checks += 1

            for _ in range(WARMUP_RUNS):

                generate_standard(model, prompt)
                generate_cached(kv_model, prompt)

            print(
                f"Length {length}, prompt {prompt_index + 1:02d}: "
                f"exact match; warm-up complete"
            )

    print(
        f"\nCorrectness checks: "
        f"{correctness_checks}/{expected_checks}"
    )

    # --------------------------------------------------------
    # Initialize result storage
    # --------------------------------------------------------

    results = {
        length: {
            "standard": [],
            "cached": [],
            "paired_speedups": [],
        }
        for length in PROMPT_LENGTHS
    }

    speedup_matrix = np.empty(
        (NUM_PROMPTS, len(PROMPT_LENGTHS)),
        dtype=np.float64,
    )

    workload_medians = [
        [None for _ in PROMPT_LENGTHS]
        for _ in range(NUM_PROMPTS)
    ]

    timing_records = 0
    timed_correctness_checks = 0

    # --------------------------------------------------------
    # Timed benchmark and immediate CSV export
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("TIMED BENCHMARK")
    print("=" * 76)

    with raw_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as raw_file:

        writer = csv.writer(raw_file)

        writer.writerow([
            "prompt_length",
            "prompt_number",
            "validation_start_token",
            "repetition",
            "first_method",
            "standard_seconds",
            "cached_seconds",
            "output_match",
        ])

        for length_index, length in enumerate(PROMPT_LENGTHS):

            for prompt_index, prompt in enumerate(prompts[length]):

                standard_times = []
                cached_times = []

                for repetition in range(BENCHMARK_RUNS):

                    if repetition % 2 == 0:

                        first_method = "standard"

                        standard_time, standard_output = timed_call(
                            lambda: generate_standard(model, prompt)
                        )

                        cached_time, cached_output = timed_call(
                            lambda: generate_cached(kv_model, prompt)
                        )

                    else:

                        first_method = "cached"

                        cached_time, cached_output = timed_call(
                            lambda: generate_cached(kv_model, prompt)
                        )

                        standard_time, standard_output = timed_call(
                            lambda: generate_standard(model, prompt)
                        )

                    output_match = torch.equal(
                        standard_output,
                        cached_output,
                    )

                    if not output_match:
                        raise RuntimeError(
                            f"Timed output mismatch: "
                            f"length={length}, "
                            f"prompt={prompt_index + 1}, "
                            f"run={repetition + 1}"
                        )

                    timed_correctness_checks += 1

                    standard_times.append(standard_time)
                    cached_times.append(cached_time)

                    results[length]["standard"].append(
                        standard_time
                    )

                    results[length]["cached"].append(
                        cached_time
                    )

                    writer.writerow([
                        length,
                        prompt_index + 1,
                        starts[prompt_index],
                        repetition + 1,
                        first_method,
                        f"{standard_time:.8f}",
                        f"{cached_time:.8f}",
                        output_match,
                    ])

                    # Preserve completed measurements even if
                    # the process is interrupted later.
                    raw_file.flush()

                    timing_records += 1

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

                workload_medians[
                    prompt_index
                ][
                    length_index
                ] = (
                    standard_median,
                    cached_median,
                )

                results[length]["paired_speedups"].append(
                    paired_speedup
                )

    # --------------------------------------------------------
    # Per-length summary
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("RESULTS BY PROMPT LENGTH")
    print("=" * 76)

    length_summaries = []

    for length in PROMPT_LENGTHS:

        standard_times = results[length]["standard"]
        cached_times = results[length]["cached"]
        paired = results[length]["paired_speedups"]

        standard_median = statistics.median(standard_times)
        cached_median = statistics.median(cached_times)

        aggregate_speedup = (
            standard_median / cached_median
        )

        reduction = (
            (standard_median - cached_median)
            / standard_median
        ) * 100

        paired_median = statistics.median(paired)

        summary = {
            "prompt_length": length,
            "measurements_per_method": len(standard_times),
            "standard_median_seconds": standard_median,
            "cached_median_seconds": cached_median,
            "aggregate_median_speedup": aggregate_speedup,
            "latency_reduction_percent": reduction,
            "median_per_prompt_speedup": paired_median,
        }

        length_summaries.append(summary)

        print(f"\nPrompt length: {length}")
        print(f"Measurements per method: {len(standard_times)}")
        print(f"Standard median: {standard_median:.4f}s")
        print(f"KV median: {cached_median:.4f}s")
        print(f"Aggregate median speedup: {aggregate_speedup:.3f}x")
        print(f"Latency reduction: {reduction:.2f}%")
        print(f"Median per-prompt speedup: {paired_median:.3f}x")

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

    paired_median = statistics.median(all_speedups)

    ci_lower, ci_upper = cluster_bootstrap_ci(
        speedup_matrix
    )

    print("\n" + "=" * 76)
    print("OVERALL RESULTS")
    print("=" * 76)

    print(f"Prompt locations: {NUM_PROMPTS}")
    print(f"Prompt lengths: {len(PROMPT_LENGTHS)}")
    print(f"Total workload combinations: {NUM_PROMPTS * len(PROMPT_LENGTHS)}")
    print(f"Timed paired records: {timing_records}")
    print(f"Timed comparisons per method: {len(all_standard)}")
    print(f"Standard overall median: {overall_standard:.4f}s")
    print(f"KV overall median: {overall_cached:.4f}s")
    print(f"Aggregate median speedup: {overall_aggregate:.3f}x")
    print(f"Paired workload median speedup: {paired_median:.3f}x")
    print(f"Cluster bootstrap 95% CI: [{ci_lower:.3f}x, {ci_upper:.3f}x]")
    print(f"Timed output agreement: {timed_correctness_checks}/{timing_records}")

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

    cache_memory_mib = (
        cache.memory_bytes() / (1024 ** 2)
    )

    print("\n" + "=" * 76)
    print("MEMORY")
    print("=" * 76)
    print(f"KV cache memory, batch 1: {cache_memory_mib:.2f} MiB")

    del cache

    final_memory = report_memory("FINAL MEMORY")

    # --------------------------------------------------------
    # Save workload summary CSV
    # --------------------------------------------------------

    save_workload_summary(
        workload_csv_path,
        starts,
        speedup_matrix,
        workload_medians,
    )

    # --------------------------------------------------------
    # Save structured experiment JSON
    # --------------------------------------------------------

    experiment = {
        "experiment": "RCT 100M large KV-cache benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_folder": str(run_dir.relative_to(PROJECT_DIR)),
        "configuration": {
            "checkpoint": str(CHECKPOINT_PATH.relative_to(PROJECT_DIR)),
            "checkpoint_step": checkpoint_step,
            "parameter_count": model.parameter_count(),
            "device": "CPU",
            "cpu_threads": CPU_THREADS,
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
            "prompt_lengths": PROMPT_LENGTHS,
            "generated_tokens": NEW_TOKENS,
            "number_of_prompt_locations": NUM_PROMPTS,
            "warmup_runs": WARMUP_RUNS,
            "timed_repetitions": BENCHMARK_RUNS,
            "seed": SEED,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "bootstrap_unit": "prompt-location cluster",
            "window_size_tokens": window_size,
            "validation_file": str(VALIDATION_PATH.relative_to(PROJECT_DIR)),
            "selected_validation_starts": starts,
        },
        "correctness": {
            "warmup_correctness_checks": correctness_checks,
            "warmup_expected_checks": expected_checks,
            "timed_correctness_checks": timed_correctness_checks,
            "timed_expected_checks": timing_records,
            "all_outputs_matched": (
                correctness_checks == expected_checks
                and timed_correctness_checks == timing_records
            ),
        },
        "results_by_prompt_length": length_summaries,
        "overall_results": {
            "prompt_location_clusters": NUM_PROMPTS,
            "workload_combinations": NUM_PROMPTS * len(PROMPT_LENGTHS),
            "timed_paired_records": timing_records,
            "timed_measurements_per_method": len(all_standard),
            "standard_overall_median_seconds": overall_standard,
            "cached_overall_median_seconds": overall_cached,
            "aggregate_median_speedup": overall_aggregate,
            "paired_workload_median_speedup": paired_median,
            "cluster_bootstrap_95_ci": [
                ci_lower,
                ci_upper,
            ],
            "cache_memory_batch_1_mib": cache_memory_mib,
        },
        "memory": {
            "initial": initial_memory,
            "after_model_loading": after_loading_memory,
            "final": final_memory,
        },
        "limitations": [
            "One CPU machine and one checkpoint were tested.",
            "Only prompt lengths 8, 16, and 24 were evaluated.",
            "Eight greedy-generated tokens were evaluated.",
            "The model was trained with sequence length 32.",
            "KV caching is an established technique.",
            "The experiment does not establish a novel KV-cache algorithm.",
        ],
        "files": {
            "raw_timing_records": raw_csv_path.name,
            "per_workload_summary": workload_csv_path.name,
            "experiment_results": json_path.name,
        },
    }

    json_path.write_text(
        json.dumps(experiment, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 76)
    print("RAW BENCHMARK DATA SAVED")
    print("=" * 76)

    print(f"Run folder: {run_dir}")
    print(f"Raw paired timing records: {raw_csv_path.name}")
    print(f"Per-workload summary: {workload_csv_path.name}")
    print(f"Experiment JSON: {json_path.name}")
    print(f"Raw records written: {timing_records}")
    print(f"Warm-up correctness: {correctness_checks}/{expected_checks}")
    print(f"Timed correctness: {timed_correctness_checks}/{timing_records}")

    del kv_model
    del model
    gc.collect()

    print("\nLarge benchmark completed and archived.")


if __name__ == "__main__":
    main()
