
from pathlib import Path
import sys
import csv
import json
import time
import random
import statistics
from datetime import datetime, timezone

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_adaptive import RCT100MAdaptive


# =========================================================
# CONFIGURATION
# =========================================================

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

RESULTS_ROOT = (
    PROJECT_DIR
    / "results"
    / "rct100m_gpu_collision_overhead"
)

BATCH_SIZES = [4, 8, 16]
PROMPT_LENGTHS = [8, 24, 64]
FINGERPRINT_COLUMNS = [2, 4, 8]

THRESHOLD = 0.25
GENERATED_TOKENS = 8
VOCAB_SIZE = 8192

CLUSTERS = 3
WARMUP_REPETITIONS = 1
TIMED_REPETITIONS = 5

SEED = 20261103


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(device):

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
# GREEDY DECODING
# =========================================================

def greedy_multinomial(
    input,
    num_samples,
    replacement=False,
    *,
    generator=None,
    out=None
):
    """
    Replace sampling with deterministic argmax.
    This makes KV-only and adaptive outputs comparable.
    """

    indices = torch.argmax(
        input,
        dim=-1,
        keepdim=True
    )

    if num_samples > 1:
        indices = indices.expand(
            *indices.shape[:-1],
            num_samples
        )

    if out is not None:
        out.copy_(indices)
        return out

    return indices


# =========================================================
# FINGERPRINT POSITION SELECTION
# =========================================================

def get_fingerprint_positions(
    prompt_length,
    fingerprint_columns
):

    return sorted(set(
        round(
            i * (prompt_length - 1)
            / max(fingerprint_columns - 1, 1)
        )
        for i in range(fingerprint_columns)
    ))


# =========================================================
# CREATE UNIQUE PROMPTS
# =========================================================

def create_unique_prompts(
    batch_size,
    prompt_length,
    device
):

    prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, prompt_length),
        dtype=torch.long,
        device=device
    )

    # Ensure every complete prompt is unique.
    # This is a defensive check for this small workload.
    while torch.unique(
        prompts,
        dim=0
    ).size(0) != batch_size:

        prompts = torch.randint(
            0,
            VOCAB_SIZE,
            (batch_size, prompt_length),
            dtype=torch.long,
            device=device
        )

    return prompts


# =========================================================
# CREATE DELIBERATE COLLISION PROMPTS
# =========================================================

def create_collision_prompts(
    batch_size,
    prompt_length,
    fingerprint_columns,
    device
):

    selected_positions = get_fingerprint_positions(
        prompt_length,
        fingerprint_columns
    )

    selected_set = set(selected_positions)

    unselected_positions = [
        position
        for position in range(prompt_length)
        if position not in selected_set
    ]

    if not unselected_positions:
        return None

    prompts = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, prompt_length),
        dtype=torch.long,
        device=device
    )

    # Identical fingerprints across the whole batch.
    prompts[:, selected_positions] = 1

    # Different complete prompts.
    unique_position = unselected_positions[0]

    prompts[:, unique_position] = (
        torch.arange(
            batch_size,
            device=device
        ) + 100
    )

    return prompts


# =========================================================
# VERIFY WORKLOAD
# =========================================================

def inspect_workload(
    prompts,
    detector,
    expected_collision
):

    batch_size = prompts.size(0)

    unique_count = torch.unique(
        prompts,
        dim=0
    ).size(0)

    actual_duplicate_rate = (
        batch_size - unique_count
    ) / batch_size

    fingerprint_rate = (
        detector.estimate_duplicate_rate(prompts)
    )

    if isinstance(fingerprint_rate, torch.Tensor):
        fingerprint_rate = fingerprint_rate.item()

    if actual_duplicate_rate != 0.0:
        raise RuntimeError(
            "Workload unexpectedly contains duplicate prompts."
        )

    if expected_collision:

        if fingerprint_rate < THRESHOLD:
            raise RuntimeError(
                "Collision workload did not trigger the detector."
            )

    else:

        if fingerprint_rate != 0.0:
            raise RuntimeError(
                "Unique control workload unexpectedly has "
                "fingerprint collisions."
            )

    return float(actual_duplicate_rate), float(fingerprint_rate)


# =========================================================
# GENERATION METHODS
# =========================================================

def run_kv_only(
    kv_model,
    prompts
):

    return kv_model.generate_cached(
        prompts,
        max_new_tokens=GENERATED_TOKENS,
        temperature=1.0
    )


def run_adaptive(
    adaptive_model,
    prompts
):

    return adaptive_model.generate(
        prompts,
        max_new_tokens=GENERATED_TOKENS,
        temperature=1.0
    )


# =========================================================
# TIMING
# =========================================================

def measure_method(
    function,
    prompts,
    device
):

    torch.cuda.synchronize(device)

    start = time.perf_counter()

    output = function(prompts)

    torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    return output, elapsed


# =========================================================
# BENCHMARK ONE WORKLOAD
# =========================================================

def benchmark_workload(
    prompts,
    kv_model,
    adaptive_model,
    device,
    batch_size,
    prompt_length,
    fingerprint_columns,
    scenario,
    cluster_id
):

    methods = {
        "kv_only": lambda x: run_kv_only(
            kv_model,
            x
        ),
        "adaptive_rct": lambda x: run_adaptive(
            adaptive_model,
            x
        )
    }

    # Verify correctness before collecting timings.
    kv_output = methods["kv_only"](prompts)

    adaptive_output = methods["adaptive_rct"](prompts)

    torch.cuda.synchronize(device)

    if not torch.equal(
        kv_output,
        adaptive_output
    ):
        raise RuntimeError(
            "Correctness failure: KV-only and adaptive outputs differ."
        )

    del kv_output
    del adaptive_output

    # Warm-up.
    for _ in range(WARMUP_REPETITIONS):

        for method in methods.values():
            method(prompts)

        torch.cuda.synchronize(device)

    measurements = {
        "kv_only": [],
        "adaptive_rct": []
    }

    # Randomize method order to reduce systematic timing bias.
    method_names = list(methods.keys())

    for _ in range(TIMED_REPETITIONS):

        order = method_names.copy()
        random.shuffle(order)

        for method_name in order:

            _, elapsed = measure_method(
                methods[method_name],
                prompts,
                device
            )

            measurements[method_name].append(
                elapsed * 1000
            )

    kv_median = statistics.median(
        measurements["kv_only"]
    )

    adaptive_median = statistics.median(
        measurements["adaptive_rct"]
    )

    speedup = (
        kv_median / adaptive_median
        if adaptive_median > 0
        else 0.0
    )

    return {
        "scenario": scenario,
        "cluster_id": cluster_id,
        "batch_size": batch_size,
        "prompt_length": prompt_length,
        "fingerprint_columns": fingerprint_columns,
        "kv_only_median_ms": kv_median,
        "adaptive_median_ms": adaptive_median,
        "adaptive_speedup_vs_kv": speedup,
        "kv_only_measurements_ms": measurements["kv_only"],
        "adaptive_measurements_ms": measurements["adaptive_rct"],
        "correctness_passed": True
    }


# =========================================================
# MAIN
# =========================================================

def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark."
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)

    device = torch.device("cuda")

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

    print("=" * 70)
    print("RCT100M END-TO-END FINGERPRINT COLLISION OVERHEAD")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Threshold:", THRESHOLD)
    print("Generated tokens:", GENERATED_TOKENS)
    print("Fingerprint columns:", FINGERPRINT_COLUMNS)
    print("Clusters:", CLUSTERS)
    print("Warmups:", WARMUP_REPETITIONS)
    print("Timed repetitions:", TIMED_REPETITIONS)

    print("\nLoading checkpoint...")

    base_model, checkpoint = load_model(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in base_model.parameters()
    )

    print("Checkpoint step:", checkpoint.get("step"))
    print("Parameters:", f"{parameter_count:,}")

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    records = []
    correctness_count = 0
    total_workloads = 0

    original_multinomial = torch.multinomial

    # Make generation deterministic for correctness comparisons.
    torch.multinomial = greedy_multinomial

    try:

        # =================================================
        # MAIN EXPERIMENT
        # =================================================

        for batch_size in BATCH_SIZES:

            for prompt_length in PROMPT_LENGTHS:

                for columns in FINGERPRINT_COLUMNS:

                    adaptive_model = RCT100MAdaptive(
                        base_model,
                        duplicate_threshold=THRESHOLD,
                        fingerprint_columns=columns
                    )

                    adaptive_model.eval()

                    collision_prompts = create_collision_prompts(
                        batch_size,
                        prompt_length,
                        columns,
                        device
                    )

                    if collision_prompts is None:

                        print(
                            f"\nSkipping impossible collision: "
                            f"B{batch_size} P{prompt_length} "
                            f"F{columns}"
                        )

                        continue

                    # Two scenarios use the same configuration:
                    # 1. Unique prompts with no fingerprint collision.
                    # 2. Unique prompts with deliberate collisions.

                    scenarios = [
                        (
                            "unique_control",
                            create_unique_prompts(
                                batch_size,
                                prompt_length,
                                device
                            ),
                            False
                        ),
                        (
                            "deliberate_collision",
                            collision_prompts,
                            True
                        )
                    ]

                    for scenario, prompts, expected_collision in scenarios:

                        actual_rate, fingerprint_rate = (
                            inspect_workload(
                                prompts,
                                adaptive_model,
                                expected_collision
                            )
                        )

                        for cluster_id in range(CLUSTERS):

                            # New prompts for each cluster.
                            if cluster_id > 0:

                                if expected_collision:

                                    prompts = create_collision_prompts(
                                        batch_size,
                                        prompt_length,
                                        columns,
                                        device
                                    )

                                else:

                                    prompts = create_unique_prompts(
                                        batch_size,
                                        prompt_length,
                                        device
                                    )

                                actual_rate, fingerprint_rate = (
                                    inspect_workload(
                                        prompts,
                                        adaptive_model,
                                        expected_collision
                                    )
                                )

                            total_workloads += 1

                            result = benchmark_workload(
                                prompts=prompts,
                                kv_model=kv_model,
                                adaptive_model=adaptive_model,
                                device=device,
                                batch_size=batch_size,
                                prompt_length=prompt_length,
                                fingerprint_columns=columns,
                                scenario=scenario,
                                cluster_id=cluster_id
                            )

                            result["actual_duplicate_rate"] = actual_rate
                            result["fingerprint_duplicate_rate"] = (
                                fingerprint_rate
                            )

                            records.append(result)
                            correctness_count += 1

                            print(
                                f"[{total_workloads}] "
                                f"{scenario} "
                                f"B{batch_size} "
                                f"P{prompt_length} "
                                f"F{columns} "
                                f"Cluster{cluster_id + 1}: "
                                f"KV={result['kv_only_median_ms']:.3f} ms, "
                                f"Adaptive={result['adaptive_median_ms']:.3f} ms, "
                                f"Speedup={result['adaptive_speedup_vs_kv']:.3f}x"
                            )

                            del prompts

                    del adaptive_model
                    del collision_prompts

                    torch.cuda.empty_cache()

    finally:

        torch.multinomial = original_multinomial

    # =====================================================
    # SAVE RAW CSV
    # =====================================================

    csv_path = output_dir / "raw_results.csv"

    csv_fields = [
        "scenario",
        "cluster_id",
        "batch_size",
        "prompt_length",
        "fingerprint_columns",
        "actual_duplicate_rate",
        "fingerprint_duplicate_rate",
        "kv_only_median_ms",
        "adaptive_median_ms",
        "adaptive_speedup_vs_kv",
        "correctness_passed"
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields
        )

        writer.writeheader()

        for record in records:

            writer.writerow({
                key: record[key]
                for key in csv_fields
            })

    # =====================================================
    # SUMMARY
    # =====================================================

    summaries = []

    for scenario in [
        "unique_control",
        "deliberate_collision"
    ]:

        subset = [
            row
            for row in records
            if row["scenario"] == scenario
        ]

        if not subset:
            continue

        speedups = [
            row["adaptive_speedup_vs_kv"]
            for row in subset
        ]

        kv_latencies = [
            row["kv_only_median_ms"]
            for row in subset
        ]

        adaptive_latencies = [
            row["adaptive_median_ms"]
            for row in subset
        ]

        summaries.append({
            "scenario": scenario,
            "workload_count": len(subset),
            "mean_speedup": statistics.mean(speedups),
            "median_speedup": statistics.median(speedups),
            "mean_kv_latency_ms": statistics.mean(kv_latencies),
            "mean_adaptive_latency_ms": statistics.mean(
                adaptive_latencies
            )
        })

    report = {
        "run_id": run_id,
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "checkpoint_step": checkpoint.get("step"),
        "parameter_count": parameter_count,
        "threshold": THRESHOLD,
        "generated_tokens": GENERATED_TOKENS,
        "batch_sizes": BATCH_SIZES,
        "prompt_lengths": PROMPT_LENGTHS,
        "fingerprint_columns": FINGERPRINT_COLUMNS,
        "clusters_per_scenario": CLUSTERS,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "timed_repetitions": TIMED_REPETITIONS,
        "total_workloads": total_workloads,
        "correctness_passed": correctness_count,
        "all_correctness_checks_passed": (
            correctness_count == total_workloads
        ),
        "summaries": summaries,
        "records": records
    }

    json_path = output_dir / "report.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    # =====================================================
    # PRINT FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("END-TO-END COLLISION OVERHEAD SUMMARY")
    print("=" * 70)

    print("Total workloads:", total_workloads)
    print("Correctness checks passed:", correctness_count)

    for summary in summaries:

        print("\nScenario:", summary["scenario"])
        print("Workloads:", summary["workload_count"])

        print(
            "Mean adaptive speedup:",
            f"{summary['mean_speedup']:.4f}x"
        )

        print(
            "Median adaptive speedup:",
            f"{summary['median_speedup']:.4f}x"
        )

        print(
            "Mean KV-only latency:",
            f"{summary['mean_kv_latency_ms']:.4f} ms"
        )

        print(
            "Mean adaptive latency:",
            f"{summary['mean_adaptive_latency_ms']:.4f} ms"
        )

    print("\nSaved results:")
    print(csv_path)
    print(json_path)

    print("\nBenchmark completed.")


if __name__ == "__main__":
    main()
