
import csv
import random
import statistics
from pathlib import Path
from collections import defaultdict

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "benchmarks"
    / "rct_249_holdout_results.csv"
)

OUTPUT_CSV = (
    ROOT / "benchmarks"
    / "rct_2410_statistical_results.csv"
)

OUTPUT_TXT = (
    ROOT / "benchmarks"
    / "rct_2410_summary.txt"
)

THRESHOLDS = [0.25, 0.375, 0.50]

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 2410


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 126:
        raise ValueError(
            f"Expected 126 records, found {len(rows)}"
        )

    required = [
        "seed",
        "batch_size",
        "unique_prompts",
        "duplicate_rate",
        "threshold",
        "kv_only_ms",
        "adaptive_ms",
        "adaptive_speedup_vs_kv",
        "adaptive_latency_reduction_pct",
        "adaptive_correct",
        "decision_correct",
        "combined_correct",
        "kv_reproducible",
    ]

    for row in rows:
        for field in required:
            if field not in row:
                raise ValueError(
                    f"Missing CSV column: {field}"
                )

    for row in rows:

        if row["adaptive_correct"] != "True":
            raise ValueError(
                "Adaptive correctness failure found."
            )

        if row["decision_correct"] != "True":
            raise ValueError(
                "Decision correctness failure found."
            )

        if row["combined_correct"] != "True":
            raise ValueError(
                "Combined correctness failure found."
            )

        if row["kv_reproducible"] != "True":
            raise ValueError(
                "KV reproducibility failure found."
            )

    print("Data validation passed.")
    print("Records:", len(rows))

    return rows


# ============================================================
# CONVERT AND GROUP DATA
# ============================================================

def prepare_data(rows):

    grouped = defaultdict(dict)
    workloads = {}

    for row in rows:

        threshold = float(row["threshold"])

        workload_key = (
            int(row["batch_size"]),
            int(row["unique_prompts"]),
            round(float(row["duplicate_rate"]), 6),
        )

        paired_key = (
            int(row["seed"]),
            workload_key,
        )

        if threshold in grouped[paired_key]:
            raise ValueError(
                f"Duplicate threshold record: {paired_key}"
            )

        record = {
            "speedup": float(
                row["adaptive_speedup_vs_kv"]
            ),
            "latency": float(
                row["adaptive_ms"]
            ),
            "kv_latency": float(
                row["kv_only_ms"]
            ),
            "reduction": float(
                row["adaptive_latency_reduction_pct"]
            ),
        }

        grouped[paired_key][threshold] = record
        workloads[workload_key] = True

    expected_thresholds = set(THRESHOLDS)

    for key, values in grouped.items():

        if set(values.keys()) != expected_thresholds:
            raise ValueError(
                f"Incomplete paired observations: {key}"
            )

    if len(grouped) != 42:
        raise ValueError(
            f"Expected 42 workload/seed pairs, got {len(grouped)}"
        )

    return grouped, list(workloads.keys())


# ============================================================
# CLUSTER BOOTSTRAP
# ============================================================

def build_cluster_arrays(grouped, workload_keys):

    workload_index = {
        key: index
        for index, key in enumerate(workload_keys)
    }

    arrays = {
        threshold: np.zeros(
            (len(workload_keys), 3, 3),
            dtype=np.float64,
        )
        for threshold in THRESHOLDS
    }

    seed_order = [2491, 2492, 2493]

    for paired_key, values in grouped.items():

        seed, workload_key = paired_key

        i = workload_index[workload_key]
        j = seed_order.index(seed)

        for threshold in THRESHOLDS:

            record = values[threshold]

            arrays[threshold][i, j, 0] = (
                record["speedup"]
            )

            arrays[threshold][i, j, 1] = (
                record["latency"]
            )

            arrays[threshold][i, j, 2] = (
                record["reduction"]
            )

    return arrays


def bootstrap_means(values, rng):

    """
    Cluster bootstrap.

    Resample workload configurations, retaining all
    three seed observations inside each selected cluster.
    """

    cluster_count = values.shape[0]

    indices = rng.integers(
        0,
        cluster_count,
        size=(BOOTSTRAP_SAMPLES, cluster_count),
    )

    sampled = values[indices]

    return sampled.mean(axis=(1, 2))


def bootstrap_paired_difference(
    values_a,
    values_b,
    rng,
):

    """
    Paired cluster bootstrap.

    Each resampled workload includes the same seeds
    for both thresholds.
    """

    differences = values_a - values_b

    return bootstrap_means(
        differences,
        rng,
    )


def confidence_interval(values):

    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


# ============================================================
# MAIN STATISTICAL ANALYSIS
# ============================================================

def main():

    print("=" * 75)
    print("RCT 2.4.10 - STATISTICAL THRESHOLD VALIDATION")
    print("=" * 75)

    rows = load_data()

    grouped, workload_keys = prepare_data(rows)

    print("Paired workload/seed observations:", len(grouped))
    print("Independent workload clusters:", len(workload_keys))
    print("Thresholds:", THRESHOLDS)

    arrays = build_cluster_arrays(
        grouped,
        workload_keys,
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    # --------------------------------------------------------
    # Threshold summaries
    # --------------------------------------------------------

    summaries = {}

    print("\nTHRESHOLD RESULTS")
    print("-" * 75)

    for threshold in THRESHOLDS:

        values = arrays[threshold]

        speedups = values[:, :, 0]
        latencies = values[:, :, 1]
        reductions = values[:, :, 2]

        bootstrap_speedup = bootstrap_means(
            speedups,
            rng,
        )

        bootstrap_reduction = bootstrap_means(
            reductions,
            rng,
        )

        speedup_ci = confidence_interval(
            bootstrap_speedup
        )

        reduction_ci = confidence_interval(
            bootstrap_reduction
        )

        item = {
            "threshold": threshold,
            "mean_speedup": float(speedups.mean()),
            "median_speedup": float(
                np.median(speedups)
            ),
            "mean_latency_ms": float(
                latencies.mean()
            ),
            "mean_reduction_pct": float(
                reductions.mean()
            ),
            "speedup_ci_low": speedup_ci[0],
            "speedup_ci_high": speedup_ci[1],
            "reduction_ci_low": reduction_ci[0],
            "reduction_ci_high": reduction_ci[1],
        }

        summaries[threshold] = item

        print(
            f"\nThreshold: {threshold:.3f}"
        )

        print(
            f"Mean speedup: "
            f"{item['mean_speedup']:.4f}x"
        )

        print(
            f"Median speedup: "
            f"{item['median_speedup']:.4f}x"
        )

        print(
            f"Mean latency: "
            f"{item['mean_latency_ms']:.3f} ms"
        )

        print(
            f"Mean latency reduction: "
            f"{item['mean_reduction_pct']:.2f}%"
        )

        print(
            f"Speedup 95% CI: "
            f"[{speedup_ci[0]:.4f}, "
            f"{speedup_ci[1]:.4f}]"
        )

        print(
            f"Reduction 95% CI: "
            f"[{reduction_ci[0]:.2f}%, "
            f"{reduction_ci[1]:.2f}%]"
        )

    # --------------------------------------------------------
    # Paired threshold comparisons
    # --------------------------------------------------------

    comparisons = []

    print("\nPAIRED THRESHOLD COMPARISONS")
    print("-" * 75)

    comparison_pairs = [
        (0.25, 0.375),
        (0.25, 0.50),
        (0.375, 0.50),
    ]

    for threshold_a, threshold_b in comparison_pairs:

        a = arrays[threshold_a][:, :, 0]
        b = arrays[threshold_b][:, :, 0]

        observed_difference = float(
            (a - b).mean()
        )

        bootstrap_difference = (
            bootstrap_paired_difference(
                a,
                b,
                rng,
            )
        )

        difference_ci = confidence_interval(
            bootstrap_difference
        )

        # Ratio of mean speedups, descriptive only.
        ratio = (
            float(a.mean())
            / float(b.mean())
        )

        comparison = {
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            "mean_speedup_difference": observed_difference,
            "difference_ci_low": difference_ci[0],
            "difference_ci_high": difference_ci[1],
            "ratio_of_mean_speedups": ratio,
            "clear_difference": (
                difference_ci[0] > 0
                or difference_ci[1] < 0
            ),
        }

        comparisons.append(comparison)

        print(
            f"\n{threshold_a:.3f} vs {threshold_b:.3f}"
        )

        print(
            f"Mean paired speedup difference: "
            f"{observed_difference:+.5f}"
        )

        print(
            f"95% CI: "
            f"[{difference_ci[0]:+.5f}, "
            f"{difference_ci[1]:+.5f}]"
        )

        print(
            f"Ratio of mean speedups: {ratio:.5f}"
        )

        print(
            "Evidence of a difference:",
            "YES" if comparison["clear_difference"]
            else "NO",
        )

    # --------------------------------------------------------
    # Duplicate-rate analysis
    # --------------------------------------------------------

    print("\nPERFORMANCE BY DUPLICATE RATE")
    print("-" * 75)

    duplicate_summaries = []

    for threshold in THRESHOLDS:

        print(
            f"\nThreshold: {threshold:.3f}"
        )

        for duplicate_rate in sorted(
            set(key[2] for key in workload_keys)
        ):

            selected = []

            for workload_key in workload_keys:

                if workload_key[2] != duplicate_rate:
                    continue

                for seed in [2491, 2492, 2493]:

                    selected.append(
                        grouped[
                            (seed, workload_key)
                        ][threshold]["speedup"]
                    )

            if not selected:
                continue

            mean_speedup = statistics.mean(
                selected
            )

            print(
                f"Duplicate rate {duplicate_rate:.1%}: "
                f"{mean_speedup:.3f}x"
            )

            duplicate_summaries.append({
                "threshold": threshold,
                "duplicate_rate": duplicate_rate,
                "observations": len(selected),
                "mean_speedup": mean_speedup,
            })

    # --------------------------------------------------------
    # Save statistical CSV
    # --------------------------------------------------------

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fields = list(summaries[THRESHOLDS[0]].keys())

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for threshold in THRESHOLDS:
            writer.writerow(
                summaries[threshold]
            )

    # --------------------------------------------------------
    # Save full text report
    # --------------------------------------------------------

    lines = [
        "RCT 2.4.10 - STATISTICAL THRESHOLD VALIDATION",
        "",
        f"Input records: {len(rows)}",
        f"Paired observations: {len(grouped)}",
        f"Workload clusters: {len(workload_keys)}",
        f"Bootstrap samples: {BOOTSTRAP_SAMPLES}",
        "",
        "THRESHOLD RESULTS",
        "",
    ]

    for threshold in THRESHOLDS:

        item = summaries[threshold]

        lines.extend([
            f"Threshold: {threshold:.3f}",
            f"Mean speedup: {item['mean_speedup']:.5f}x",
            f"Median speedup: {item['median_speedup']:.5f}x",
            f"Mean latency: {item['mean_latency_ms']:.4f} ms",
            (
                "Mean latency reduction: "
                f"{item['mean_reduction_pct']:.3f}%"
            ),
            (
                "Speedup 95% CI: "
                f"[{item['speedup_ci_low']:.5f}, "
                f"{item['speedup_ci_high']:.5f}]"
            ),
            "",
        ])

    lines.extend([
        "PAIRED COMPARISONS",
        "",
    ])

    for item in comparisons:

        lines.extend([
            (
                f"{item['threshold_a']:.3f} vs "
                f"{item['threshold_b']:.3f}"
            ),
            (
                "Mean paired difference: "
                f"{item['mean_speedup_difference']:+.6f}"
            ),
            (
                "95% CI: "
                f"[{item['difference_ci_low']:+.6f}, "
                f"{item['difference_ci_high']:+.6f}]"
            ),
            (
                "Clear difference: "
                f"{item['clear_difference']}"
            ),
            "",
        ])

    lines.extend([
        "METHODOLOGICAL LIMITATIONS",
        "",
        "Bootstrap resampling uses 14 workload clusters.",
        "Three seeds are nested within each workload cluster.",
        "The bootstrap does not create new independent workloads.",
        "The holdout prompts use five shared templates.",
        "The model is randomly initialized and untrained.",
        "Results apply to this CPU and tested workload.",
        "No general language quality conclusion is possible.",
        "",
    ])

    OUTPUT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final interpretation
    # --------------------------------------------------------

    best_threshold = max(
        THRESHOLDS,
        key=lambda t: summaries[t]["mean_speedup"],
    )

    print("\n" + "=" * 75)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 75)

    print(
        f"Highest observed mean speedup: "
        f"{best_threshold:.3f}"
    )

    print(
        "This is a point-estimate winner, "
        "not necessarily a statistically superior threshold."
    )

    print("\nSaved files:")
    print(OUTPUT_CSV)
    print(OUTPUT_TXT)


if __name__ == "__main__":
    main()
