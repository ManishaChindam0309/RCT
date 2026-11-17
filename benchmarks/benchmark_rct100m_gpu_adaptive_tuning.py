
from pathlib import Path
import sys
import csv
import json
import time
import random
from datetime import datetime, timezone

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_adaptive import RCT100MAdaptive


# =========================================================
# 1. CONFIGURATION
# =========================================================

BATCH_SIZES = [8, 16]
PROMPT_LENGTHS = [24, 64]
DUPLICATE_RATES = [0.0, 0.50, 0.75, 0.875]

THRESHOLDS = [0.10, 0.25]
FINGERPRINT_COLUMNS = [2, 4, 8]

CLUSTERS = 10
TIMED_REPETITIONS = 5
WARMUP_REPETITIONS = 2
BOOTSTRAP_SAMPLES = 10000

MAX_NEW_TOKENS = 8
TEMPERATURE = 1.0
VOCAB_SIZE = 8192

SEED = 20261105

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

RESULTS_ROOT = (
    PROJECT_DIR
    / "results"
    / "rct100m_gpu_adaptive_tuning"
)


# =========================================================
# 2. GREEDY DECODING
# =========================================================

def greedy_multinomial(
    probabilities,
    num_samples,
    replacement=False,
    *,
    generator=None,
    out=None
):

    if num_samples != 1:
        raise ValueError(
            "Only num_samples=1 is supported."
        )

    result = torch.argmax(
        probabilities,
        dim=-1,
        keepdim=True
    )

    if out is not None:
        out.copy_(result)
        return out

    return result


# =========================================================
# 3. MODEL LOADING
# =========================================================

def load_base_model(device):

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    config = RCT100MConfig(
        vocab_size=8192,
        hidden_size=512,
        num_layers=30,
        num_heads=8,
        intermediate_size=2048,
        max_seq_len=512
    )

    model = RCT100M(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint


# =========================================================
# 4. WORKLOAD GENERATION
# =========================================================

def create_prompts(
    batch_size,
    prompt_length,
    duplicate_rate,
    device
):

    unique_count = max(
        1,
        round(
            batch_size * (1.0 - duplicate_rate)
        )
    )

    unique_prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (unique_count, prompt_length),
        dtype=torch.long,
        device=device
    )

    repeat_indices = (
        torch.arange(batch_size, device=device)
        % unique_count
    )

    prompts = unique_prompts.index_select(
        0,
        repeat_indices
    )

    actual_unique = torch.unique(
        prompts,
        dim=0
    ).size(0)

    actual_rate = (
        batch_size - actual_unique
    ) / batch_size

    return prompts, actual_unique, actual_rate


# =========================================================
# 5. OUTPUT VALIDATION
# =========================================================

def validate_output(
    output,
    batch_size,
    prompt_length
):

    expected_shape = (
        batch_size,
        prompt_length + MAX_NEW_TOKENS
    )

    return (
        isinstance(output, torch.Tensor)
        and tuple(output.shape) == expected_shape
        and output.dtype == torch.long
    )


def duplicate_outputs_consistent(
    prompts,
    outputs
):

    for i in range(prompts.size(0)):

        for j in range(i + 1, prompts.size(0)):

            if torch.equal(prompts[i], prompts[j]):

                if not torch.equal(outputs[i], outputs[j]):
                    return False

    return True


# =========================================================
# 6. METHODS
# =========================================================

def build_methods(base_model):

    methods = {
        "kv_only": RCT100MKV(base_model)
    }

    for threshold in THRESHOLDS:

        for columns in FINGERPRINT_COLUMNS:

            name = (
                f"adaptive_t{threshold:.2f}_c{columns}"
            )

            methods[name] = RCT100MAdaptive(
                base_model,
                duplicate_threshold=threshold,
                fingerprint_columns=columns
            )

    for method in methods.values():
        method.eval()

    return methods


def run_method(
    method,
    method_name,
    prompts
):

    if method_name == "kv_only":

        return method.generate_cached(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )

    return method.generate(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE
    )


# =========================================================
# 7. TIMING AND MEMORY
# =========================================================

def measure_once(
    method,
    method_name,
    prompts,
    device,
    batch_size,
    prompt_length
):

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    baseline_allocated = (
        torch.cuda.memory_allocated(device)
    )

    torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    with torch.inference_mode():

        output = run_method(
            method,
            method_name,
            prompts
        )

    torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
    )

    if not validate_output(
        output,
        batch_size,
        prompt_length
    ):

        raise AssertionError(
            f"Invalid output from {method_name}"
        )

    return {
        "latency_seconds": elapsed,
        "peak_allocated_mb": (
            peak_allocated / (1024 ** 2)
        ),
        "peak_incremental_mb": (
            max(
                0,
                peak_allocated - baseline_allocated
            ) / (1024 ** 2)
        ),
        "output": output
    }


# =========================================================
# 8. BOOTSTRAP
# =========================================================

def paired_bootstrap_ci(
    baseline,
    comparison,
    seed
):

    baseline = np.asarray(
        baseline,
        dtype=np.float64
    )

    comparison = np.asarray(
        comparison,
        dtype=np.float64
    )

    if len(baseline) != len(comparison):
        raise ValueError(
            "Paired measurements must have equal lengths."
        )

    ratios = baseline / comparison

    rng = np.random.default_rng(seed)

    samples = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64
    )

    for i in range(BOOTSTRAP_SAMPLES):

        selected = rng.integers(
            0,
            len(ratios),
            size=len(ratios)
        )

        samples[i] = np.median(
            ratios[selected]
        )

    return {
        "median_speedup": float(
            np.median(ratios)
        ),
        "ci_lower": float(
            np.quantile(samples, 0.025)
        ),
        "ci_upper": float(
            np.quantile(samples, 0.975)
        )
    }


# =========================================================
# 9. CSV
# =========================================================

def save_csv(path, records):

    if not records:
        raise ValueError(
            f"No records to save: {path}"
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )

        writer.writeheader()
        writer.writerows(records)


# =========================================================
# 10. MAIN
# =========================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    device = torch.device("cuda")

    random.seed(SEED)
    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    run_id = datetime.now(
        timezone.utc
    ).strftime("run_%Y%m%dT%H%M%SZ")

    output_dir = RESULTS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RCT100M ADAPTIVE INFERENCE TUNING")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Batch sizes:", BATCH_SIZES)
    print("Prompt lengths:", PROMPT_LENGTHS)
    print("Duplicate rates:", DUPLICATE_RATES)
    print("Thresholds:", THRESHOLDS)
    print("Fingerprint columns:", FINGERPRINT_COLUMNS)
    print("Clusters:", CLUSTERS)
    print("Output:", output_dir)

    print("\nLoading checkpoint...")

    base_model, checkpoint = load_base_model(device)

    parameter_count = sum(
        p.numel()
        for p in base_model.parameters()
    )

    print("Checkpoint step:", checkpoint.get("step"))
    print("Validation loss:", checkpoint.get("val_loss"))
    print("Parameters:", f"{parameter_count:,}")

    methods = build_methods(base_model)

    timing_records = []
    correctness_records = []
    cluster_results = {}

    rng = random.Random(SEED)

    total_configs = (
        len(BATCH_SIZES)
        * len(PROMPT_LENGTHS)
        * len(DUPLICATE_RATES)
    )

    config_number = 0

    original_multinomial = torch.multinomial

    try:

        torch.multinomial = greedy_multinomial

        for batch_size in BATCH_SIZES:

            for prompt_length in PROMPT_LENGTHS:

                for requested_rate in DUPLICATE_RATES:

                    config_number += 1

                    config_name = (
                        f"B{batch_size}"
                        f"_P{prompt_length}"
                        f"_D{int(requested_rate * 100)}"
                    )

                    print("\n" + "-" * 70)
                    print(
                        f"Configuration "
                        f"{config_number}/{total_configs}: "
                        f"{config_name}"
                    )

                    cluster_results[config_name] = {
                        name: []
                        for name in methods
                    }

                    for cluster_id in range(CLUSTERS):

                        prompts, actual_unique, actual_rate = (
                            create_prompts(
                                batch_size,
                                prompt_length,
                                requested_rate,
                                device
                            )
                        )

                        # ---------------------------------
                        # Correctness checks
                        # ---------------------------------

                        outputs = {}

                        for name, method in methods.items():

                            result = measure_once(
                                method,
                                name,
                                prompts,
                                device,
                                batch_size,
                                prompt_length
                            )

                            output = result["output"].clone()

                            outputs[name] = output

                            duplicate_ok = (
                                duplicate_outputs_consistent(
                                    prompts,
                                    output
                                )
                            )

                            correctness_records.append({
                                "config": config_name,
                                "cluster": cluster_id,
                                "method": name,
                                "shape_valid": True,
                                "duplicate_outputs_consistent": (
                                    duplicate_ok
                                ),
                                "matches_kv_only": None
                            })

                            if not duplicate_ok:
                                raise AssertionError(
                                    f"Duplicate output mismatch: {name}"
                                )

                            del result

                        kv_output = outputs["kv_only"]

                        for name in methods:

                            if name == "kv_only":
                                continue

                            matches = torch.equal(
                                kv_output,
                                outputs[name]
                            )

                            correctness_records.append({
                                "config": config_name,
                                "cluster": cluster_id,
                                "method": name,
                                "shape_valid": True,
                                "duplicate_outputs_consistent": True,
                                "matches_kv_only": matches
                            })

                            if not matches:
                                raise AssertionError(
                                    "Exact output disagreement: "
                                    f"{config_name}, "
                                    f"{cluster_id}, {name}"
                                )

                        del outputs
                        del kv_output

                        # ---------------------------------
                        # Warm-up
                        # ---------------------------------

                        for name, method in methods.items():

                            for _ in range(WARMUP_REPETITIONS):

                                warmup = measure_once(
                                    method,
                                    name,
                                    prompts,
                                    device,
                                    batch_size,
                                    prompt_length
                                )

                                del warmup

                        # ---------------------------------
                        # Timed measurements
                        # ---------------------------------

                        cluster_latencies = {
                            name: []
                            for name in methods
                        }

                        cluster_memory = {
                            name: []
                            for name in methods
                        }

                        for repetition in range(TIMED_REPETITIONS):

                            order = list(methods.keys())
                            rng.shuffle(order)

                            for name in order:

                                result = measure_once(
                                    methods[name],
                                    name,
                                    prompts,
                                    device,
                                    batch_size,
                                    prompt_length
                                )

                                latency = result["latency_seconds"]

                                peak_memory = (
                                    result["peak_allocated_mb"]
                                )

                                cluster_latencies[name].append(
                                    latency
                                )

                                cluster_memory[name].append(
                                    peak_memory
                                )

                                timing_records.append({
                                    "config": config_name,
                                    "batch_size": batch_size,
                                    "prompt_length": prompt_length,
                                    "requested_duplicate_rate": (
                                        requested_rate
                                    ),
                                    "actual_duplicate_rate": actual_rate,
                                    "actual_unique_prompts": actual_unique,
                                    "cluster": cluster_id,
                                    "repetition": repetition,
                                    "method": name,
                                    "latency_seconds": latency,
                                    "peak_allocated_mb": peak_memory,
                                    "peak_incremental_mb": (
                                        result["peak_incremental_mb"]
                                    )
                                })

                                del result

                        for name in methods:

                            cluster_results[config_name][name].append({
                                "median_latency": float(
                                    np.median(cluster_latencies[name])
                                ),
                                "median_memory": float(
                                    np.median(cluster_memory[name])
                                )
                            })

                        del prompts

                    print(
                        f"Completed {CLUSTERS} clusters."
                    )

    finally:

        torch.multinomial = original_multinomial

    # =====================================================
    # 11. AGGREGATE RESULTS
    # =====================================================

    summaries = []

    for config_name, config_data in cluster_results.items():

        batch_size = int(
            config_name.split("_")[0][1:]
        )

        prompt_length = int(
            config_name.split("_")[1][1:]
        )

        duplicate_rate = int(
            config_name.split("_")[2][1:]
        ) / 100.0

        kv_values = [
            item["median_latency"]
            for item in config_data["kv_only"]
        ]

        kv_memory = [
            item["median_memory"]
            for item in config_data["kv_only"]
        ]

        for name in methods:

            if name == "kv_only":
                continue

            adaptive_values = [
                item["median_latency"]
                for item in config_data[name]
            ]

            adaptive_memory = [
                item["median_memory"]
                for item in config_data[name]
            ]

            comparison = paired_bootstrap_ci(
                kv_values,
                adaptive_values,
                SEED + batch_size + prompt_length + len(name)
            )

            summaries.append({
                "config": config_name,
                "method": name,
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "duplicate_rate": duplicate_rate,
                "kv_only_median_seconds": float(
                    np.median(kv_values)
                ),
                "adaptive_median_seconds": float(
                    np.median(adaptive_values)
                ),
                "speedup": comparison,
                "kv_only_peak_memory_mb": float(
                    np.median(kv_memory)
                ),
                "adaptive_peak_memory_mb": float(
                    np.median(adaptive_memory)
                ),
                "memory_saved_mb": float(
                    np.median(kv_memory)
                    - np.median(adaptive_memory)
                )
            })

    # =====================================================
    # 12. SAVE RESULTS
    # =====================================================

    timing_path = output_dir / "raw_timings.csv"
    correctness_path = output_dir / "correctness.csv"
    summary_path = output_dir / "summary.csv"
    report_path = output_dir / "report.json"

    save_csv(timing_path, timing_records)
    save_csv(correctness_path, correctness_records)
    save_csv(summary_path, summaries)

    report = {
        "run_id": run_id,
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_step": checkpoint.get("step"),
        "validation_loss": checkpoint.get("val_loss"),
        "parameter_count": parameter_count,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "FP32",
        "decoding": "greedy_argmax",
        "batch_sizes": BATCH_SIZES,
        "prompt_lengths": PROMPT_LENGTHS,
        "duplicate_rates": DUPLICATE_RATES,
        "thresholds": THRESHOLDS,
        "fingerprint_columns": FINGERPRINT_COLUMNS,
        "clusters": CLUSTERS,
        "timed_repetitions": TIMED_REPETITIONS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "generated_tokens": MAX_NEW_TOKENS,
        "timing_records": len(timing_records),
        "correctness_records": len(correctness_records),
        "all_exact_output_checks_passed": True,
        "summaries": summaries
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(report, file, indent=2)

    # =====================================================
    # 13. PRINT SUMMARY
    # =====================================================

    print("\n" + "=" * 95)
    print("ADAPTIVE INFERENCE TUNING RESULTS")
    print("=" * 95)

    print(
        f"{'CONFIG':<13}"
        f"{'METHOD':<22}"
        f"{'SPEEDUP':>10}"
        f"{'CI LOWER':>11}"
        f"{'CI UPPER':>11}"
        f"{'MEM SAVED':>13}"
    )

    for item in summaries:

        speedup = item["speedup"]

        print(
            f"{item['config']:<13}"
            f"{item['method']:<22}"
            f"{speedup['median_speedup']:>9.3f}x"
            f"{speedup['ci_lower']:>11.3f}"
            f"{speedup['ci_upper']:>11.3f}"
            f"{item['memory_saved_mb']:>11.1f} MB"
        )

    print("\nAll exact output checks passed: YES")

    print("\nBENCHMARK COMPLETED")
    print("Output:", output_dir)
    print("Timings:", timing_path)
    print("Correctness:", correctness_path)
    print("Summary:", summary_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
