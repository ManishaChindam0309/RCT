
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
from model.rct_100m_combined import RCT100MCombined
from model.rct_100m_adaptive import RCT100MAdaptive


# =========================================================
# 1. BENCHMARK CONFIGURATION
# =========================================================

BATCH_SIZES = [4, 8, 16]
PROMPT_LENGTHS = [8, 24, 64]
DUPLICATE_RATES = [0.0, 0.50, 0.75]

CLUSTERS_PER_CONFIG = 20
TIMED_REPETITIONS = 5
WARMUP_REPETITIONS = 2

MAX_NEW_TOKENS = 8
TEMPERATURE = 1.0

DUPLICATE_THRESHOLD = 0.25
FINGERPRINT_COLUMNS = 4

SEED = 20261103
BOOTSTRAP_SAMPLES = 10000

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

RESULTS_ROOT = (
    PROJECT_DIR
    / "results"
    / "rct100m_gpu_adaptive"
)


# =========================================================
# 2. DETERMINISTIC GREEDY DECODING
# =========================================================

def greedy_multinomial(
    probabilities,
    num_samples,
    replacement=False,
    *,
    generator=None,
    out=None
):
    """
    Replace multinomial sampling with deterministic argmax.

    This allows exact output comparisons between methods.
    """

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
        low=0,
        high=8192,
        size=(unique_count, prompt_length),
        dtype=torch.long,
        device=device
    )

    repeat_indices = (
        torch.arange(
            batch_size,
            device=device
        ) % unique_count
    )

    prompts = unique_prompts.index_select(
        0,
        repeat_indices
    )

    actual_unique = torch.unique(
        prompts,
        dim=0
    ).size(0)

    actual_duplicate_rate = (
        batch_size - actual_unique
    ) / batch_size

    return (
        prompts,
        actual_unique,
        actual_duplicate_rate
    )


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


def check_duplicate_outputs(
    prompts,
    outputs
):

    batch_size = prompts.size(0)

    for i in range(batch_size):

        for j in range(i + 1, batch_size):

            if torch.equal(
                prompts[i],
                prompts[j]
            ):

                if not torch.equal(
                    outputs[i],
                    outputs[j]
                ):
                    return False

    return True


# =========================================================
# 6. METHOD-SPECIFIC INFERENCE
# =========================================================

def run_method(
    method,
    method_name,
    prompts
):
    """
    Invoke the correct inference API for each implementation.
    """

    if method_name == "kv_only":

        return method.generate_cached(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )

    if method_name == "fixed_rct":

        return method.generate(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )

    if method_name == "adaptive_rct":

        return method.generate(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        )

    raise ValueError(
        f"Unknown benchmark method: {method_name}"
    )


# =========================================================
# 7. TIMING AND GPU MEMORY
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
            f"Invalid output shape from {method_name}: "
            f"{tuple(output.shape)}"
        )

    return {
        "latency_seconds": elapsed,

        "baseline_allocated_mb": (
            baseline_allocated / (1024 ** 2)
        ),

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
# 8. PAIRED CLUSTER BOOTSTRAP
# =========================================================

def paired_bootstrap_ci(
    numerator,
    denominator,
    seed
):

    numerator = np.asarray(
        numerator,
        dtype=np.float64
    )

    denominator = np.asarray(
        denominator,
        dtype=np.float64
    )

    if len(numerator) != len(denominator):
        raise ValueError(
            "Paired samples must have equal lengths."
        )

    if np.any(denominator <= 0):
        raise ValueError(
            "Latency values must be positive."
        )

    ratios = numerator / denominator

    rng = np.random.default_rng(seed)

    bootstrap_medians = np.empty(
        BOOTSTRAP_SAMPLES,
        dtype=np.float64
    )

    sample_count = len(ratios)

    for index in range(BOOTSTRAP_SAMPLES):

        selected = rng.integers(
            0,
            sample_count,
            size=sample_count
        )

        bootstrap_medians[index] = np.median(
            ratios[selected]
        )

    return {
        "median_speedup": float(
            np.median(ratios)
        ),

        "ci_lower": float(
            np.quantile(
                bootstrap_medians,
                0.025
            )
        ),

        "ci_upper": float(
            np.quantile(
                bootstrap_medians,
                0.975
            )
        )
    }


# =========================================================
# 9. CSV EXPORT
# =========================================================

def save_csv(path, records):

    if not records:
        raise ValueError(
            f"No records available for {path}"
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
# 10. MAIN BENCHMARK
# =========================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This benchmark requires an NVIDIA GPU."
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
    ).strftime(
        "run_%Y%m%dT%H%M%SZ"
    )

    output_dir = RESULTS_ROOT / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("RCT100M ADAPTIVE GPU BENCHMARK")
    print("=" * 65)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print("Precision: FP32")
    print("Decoding: Greedy")
    print("Threshold:", DUPLICATE_THRESHOLD)
    print("Fingerprint columns:", FINGERPRINT_COLUMNS)

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Results:",
        output_dir
    )

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    print("\nLoading checkpoint...")

    base_model, checkpoint = load_base_model(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in base_model.parameters()
    )

    print(
        "Checkpoint step:",
        checkpoint.get("step")
    )

    print(
        "Validation loss:",
        checkpoint.get("val_loss")
    )

    print(
        "Parameters:",
        f"{parameter_count:,}"
    )

    # -----------------------------------------------------
    # Construct three methods.
    # All wrappers share the same base-model weights.
    # -----------------------------------------------------

    kv_only = RCT100MKV(base_model)

    fixed_rct = RCT100MCombined(base_model)

    adaptive_rct = RCT100MAdaptive(
        base_model,
        duplicate_threshold=DUPLICATE_THRESHOLD,
        fingerprint_columns=FINGERPRINT_COLUMNS
    )

    kv_only.eval()
    fixed_rct.eval()
    adaptive_rct.eval()

    methods = {
        "kv_only": kv_only,
        "fixed_rct": fixed_rct,
        "adaptive_rct": adaptive_rct
    }

    # -----------------------------------------------------
    # Record collections
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Temporarily use deterministic greedy decoding.
    # -----------------------------------------------------

    original_multinomial = torch.multinomial

    try:

        torch.multinomial = greedy_multinomial

        # =================================================
        # CONFIGURATION LOOP
        # =================================================

        for batch_size in BATCH_SIZES:

            for prompt_length in PROMPT_LENGTHS:

                for requested_duplicate_rate in DUPLICATE_RATES:

                    config_number += 1

                    duplicate_percent = int(
                        requested_duplicate_rate * 100
                    )

                    config_name = (
                        f"B{batch_size}"
                        f"_P{prompt_length}"
                        f"_D{duplicate_percent}"
                    )

                    print("\n" + "-" * 65)

                    print(
                        f"Configuration "
                        f"{config_number}/{total_configs}: "
                        f"{config_name}"
                    )

                    cluster_results[config_name] = {
                        name: []
                        for name in methods
                    }

                    # -------------------------------------
                    # Cluster loop
                    # -------------------------------------

                    for cluster_id in range(
                        CLUSTERS_PER_CONFIG
                    ):

                        prompts, actual_unique, actual_duplicate_rate = (
                            create_prompts(
                                batch_size,
                                prompt_length,
                                requested_duplicate_rate,
                                device
                            )
                        )

                        # =================================
                        # CORRECTNESS CHECKS
                        # =================================

                        outputs = {}

                        for method_name, method in methods.items():

                            result = measure_once(
                                method,
                                method_name,
                                prompts,
                                device,
                                batch_size,
                                prompt_length
                            )

                            output = result["output"]

                            outputs[method_name] = (
                                output.clone()
                            )

                            duplicate_consistency = (
                                check_duplicate_outputs(
                                    prompts,
                                    output
                                )
                            )

                            correctness_records.append({
                                "config": config_name,
                                "cluster": cluster_id,
                                "method": method_name,
                                "shape_valid": True,
                                "duplicate_outputs_consistent": (
                                    duplicate_consistency
                                )
                            })

                            if not duplicate_consistency:

                                raise AssertionError(
                                    "Duplicate output mismatch: "
                                    f"{config_name}, "
                                    f"{method_name}"
                                )

                            del result
                            del output

                        # ---------------------------------
                        # Cross-method exact agreement
                        # ---------------------------------

                        kv_output = outputs["kv_only"]

                        for method_name in (
                            "fixed_rct",
                            "adaptive_rct"
                        ):

                            if not torch.equal(
                                kv_output,
                                outputs[method_name]
                            ):

                                raise AssertionError(
                                    "Cross-method output mismatch: "
                                    f"{config_name}, "
                                    f"cluster={cluster_id}, "
                                    f"method={method_name}"
                                )

                        correctness_records.append({
                            "config": config_name,
                            "cluster": cluster_id,
                            "method": "cross_method",
                            "shape_valid": True,
                            "duplicate_outputs_consistent": True
                        })

                        del outputs
                        del kv_output

                        # =================================
                        # WARM-UP
                        # =================================

                        for method_name, method in methods.items():

                            for _ in range(
                                WARMUP_REPETITIONS
                            ):

                                warmup_result = measure_once(
                                    method,
                                    method_name,
                                    prompts,
                                    device,
                                    batch_size,
                                    prompt_length
                                )

                                del warmup_result

                        # =================================
                        # TIMED REPETITIONS
                        # =================================

                        cluster_latencies = {
                            name: []
                            for name in methods
                        }

                        for repetition in range(
                            TIMED_REPETITIONS
                        ):

                            order = list(
                                methods.keys()
                            )

                            rng.shuffle(order)

                            for method_name in order:

                                method = methods[method_name]

                                result = measure_once(
                                    method,
                                    method_name,
                                    prompts,
                                    device,
                                    batch_size,
                                    prompt_length
                                )

                                latency = result[
                                    "latency_seconds"
                                ]

                                cluster_latencies[
                                    method_name
                                ].append(latency)

                                timing_records.append({
                                    "config": config_name,
                                    "batch_size": batch_size,
                                    "prompt_length": prompt_length,
                                    "requested_duplicate_rate": (
                                        requested_duplicate_rate
                                    ),
                                    "actual_duplicate_rate": (
                                        actual_duplicate_rate
                                    ),
                                    "actual_unique_prompts": (
                                        actual_unique
                                    ),
                                    "cluster": cluster_id,
                                    "repetition": repetition,
                                    "method": method_name,
                                    "latency_seconds": latency,
                                    "baseline_allocated_mb": (
                                        result["baseline_allocated_mb"]
                                    ),
                                    "peak_allocated_mb": (
                                        result["peak_allocated_mb"]
                                    ),
                                    "peak_incremental_mb": (
                                        result["peak_incremental_mb"]
                                    )
                                })

                                del result

                        # ---------------------------------
                        # Save cluster-level medians
                        # ---------------------------------

                        for method_name in methods:

                            cluster_results[config_name][
                                method_name
                            ].append({
                                "median_latency": float(
                                    np.median(
                                        cluster_latencies[
                                            method_name
                                        ]
                                    )
                                )
                            })

                        del prompts

                        if (cluster_id + 1) % 5 == 0:

                            print(
                                f"  Completed "
                                f"{cluster_id + 1}/"
                                f"{CLUSTERS_PER_CONFIG} clusters"
                            )

                    print("Configuration completed.")

    finally:

        # Restore PyTorch's original function.
        torch.multinomial = original_multinomial

    # =====================================================
    # 11. SAVE RAW DATA
    # =====================================================

    timing_path = output_dir / "raw_timings.csv"

    save_csv(
        timing_path,
        timing_records
    )

    correctness_path = output_dir / "correctness.csv"

    save_csv(
        correctness_path,
        correctness_records
    )

    # =====================================================
    # 12. AGGREGATE RESULTS
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

        fixed_values = [
            item["median_latency"]
            for item in config_data["fixed_rct"]
        ]

        adaptive_values = [
            item["median_latency"]
            for item in config_data["adaptive_rct"]
        ]

        adaptive_vs_kv = paired_bootstrap_ci(
            kv_values,
            adaptive_values,
            SEED + batch_size + prompt_length
        )

        adaptive_vs_fixed = paired_bootstrap_ci(
            fixed_values,
            adaptive_values,
            SEED + batch_size + prompt_length + 100
        )

        fixed_vs_kv = paired_bootstrap_ci(
            kv_values,
            fixed_values,
            SEED + batch_size + prompt_length + 200
        )

        summary = {
            "config": config_name,
            "batch_size": batch_size,
            "prompt_length": prompt_length,
            "duplicate_rate": duplicate_rate,

            "kv_only_median_seconds": float(
                np.median(kv_values)
            ),

            "fixed_rct_median_seconds": float(
                np.median(fixed_values)
            ),

            "adaptive_rct_median_seconds": float(
                np.median(adaptive_values)
            ),

            "adaptive_vs_kv": adaptive_vs_kv,

            "adaptive_vs_fixed": adaptive_vs_fixed,

            "fixed_vs_kv": fixed_vs_kv
        }

        summaries.append(summary)

    # =====================================================
    # 13. JSON REPORT
    # =====================================================

    report = {
        "run_id": run_id,

        "checkpoint": str(CHECKPOINT_PATH),

        "checkpoint_step": checkpoint.get("step"),

        "checkpoint_validation_loss": (
            checkpoint.get("val_loss")
        ),

        "parameter_count": parameter_count,

        "gpu": torch.cuda.get_device_name(0),

        "pytorch_version": torch.__version__,

        "cuda_version": torch.version.cuda,

        "precision": "FP32",

        "decoding": "greedy_argmax",

        "batch_sizes": BATCH_SIZES,

        "prompt_lengths": PROMPT_LENGTHS,

        "duplicate_rates": DUPLICATE_RATES,

        "clusters_per_configuration": (
            CLUSTERS_PER_CONFIG
        ),

        "timed_repetitions": TIMED_REPETITIONS,

        "warmup_repetitions": WARMUP_REPETITIONS,

        "generated_tokens": MAX_NEW_TOKENS,

        "adaptive_threshold": DUPLICATE_THRESHOLD,

        "fingerprint_columns": FINGERPRINT_COLUMNS,

        "timing_record_count": len(timing_records),

        "correctness_record_count": len(correctness_records),

        "all_correctness_checks_passed": True,

        "summaries": summaries
    }

    report_path = output_dir / "report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    # =====================================================
    # 14. HUMAN-READABLE REPORT
    # =====================================================

    lines = []

    lines.append(
        "RCT100M ADAPTIVE GPU BENCHMARK REPORT"
    )

    lines.append("=" * 95)

    lines.append(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    lines.append(
        f"Checkpoint step: {checkpoint.get('step')}"
    )

    lines.append(
        f"Timing records: {len(timing_records)}"
    )

    lines.append(
        f"Correctness records: {len(correctness_records)}"
    )

    lines.append(
        "All exact output checks passed: YES"
    )

    lines.append("")

    header = (
        "CONFIG       KV(s)    FIXED(s) ADAPT(s) "
        "ADAPT/KV       ADAPT/FIXED"
    )

    lines.append(header)
    lines.append("-" * len(header))

    for item in summaries:

        lines.append(
            f"{item['config']:<12} "
            f"{item['kv_only_median_seconds']:.4f}   "
            f"{item['fixed_rct_median_seconds']:.4f}   "
            f"{item['adaptive_rct_median_seconds']:.4f}   "
            f"{item['adaptive_vs_kv']['median_speedup']:.3f}x "
            f"[{item['adaptive_vs_kv']['ci_lower']:.3f}, "
            f"{item['adaptive_vs_kv']['ci_upper']:.3f}]   "
            f"{item['adaptive_vs_fixed']['median_speedup']:.3f}x"
        )

    lines.append("")

    lines.append(
        "Speedup = baseline latency / comparison latency."
    )

    lines.append(
        "Confidence intervals are paired cluster-bootstrap 95% intervals."
    )

    lines.append(
        "Intervals are exploratory and are not adjusted for 27 comparisons."
    )

    lines.append(
        "Latency includes fingerprint detection and exact deduplication."
    )

    report_text = "\n".join(lines)

    report_text_path = output_dir / "REPORT.txt"

    report_text_path.write_text(
        report_text,
        encoding="utf-8"
    )

    # =====================================================
    # 15. PRINT FINAL REPORT
    # =====================================================

    print("\n" + report_text)

    print("\n" + "=" * 65)
    print("BENCHMARK COMPLETED")
    print("=" * 65)

    print("Output directory:", output_dir)
    print("Raw timings:", timing_path)
    print("Correctness:", correctness_path)
    print("JSON report:", report_path)
    print("Text report:", report_text_path)


if __name__ == "__main__":
    main()
