
import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

# ============================================================
# RCT 2.4.8 - Paired Statistical Threshold Analysis
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "benchmarks" / "rct_247_results.csv"
OUTPUT_CSV = ROOT / "benchmarks" / "rct_248_statistical_results.csv"
SUMMARY_TXT = ROOT / "benchmarks" / "rct_248_summary.txt"

THRESHOLDS = [0.25, 0.375, 0.50]
BOOTSTRAP_SAMPLES = 20000
RANDOM_SEED = 2480


def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")

    position = (len(values) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def bootstrap_mean_ci(cluster_values, rng):
    """Bootstrap a mean by resampling workload clusters."""
    n = len(cluster_values)

    if n < 2:
        return float("nan"), float("nan")

    estimates = []

    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [
            cluster_values[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(statistics.mean(sample))

    return (
        percentile(estimates, 0.025),
        percentile(estimates, 0.975),
    )


def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found: {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("The input CSV contains no records.")

    required = {
        "seed",
        "batch_size",
        "unique_prompts",
        "threshold",
        "kv_only_ms",
        "adaptive_ms",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}"
        )

    records = []

    for row in rows:
        kv_ms = float(row["kv_only_ms"])
        adaptive_ms = float(row["adaptive_ms"])

        if kv_ms <= 0 or adaptive_ms <= 0:
            raise ValueError("Invalid latency value in CSV.")

        batch_size = int(row["batch_size"])
        unique_prompts = int(row["unique_prompts"])

        record = {
            "seed": int(row["seed"]),
            "batch_size": batch_size,
            "unique_prompts": unique_prompts,
            "duplicate_rate": (
                batch_size - unique_prompts
            ) / batch_size,
            "threshold": float(row["threshold"]),
            "kv_ms": kv_ms,
            "adaptive_ms": adaptive_ms,
            "speedup": kv_ms / adaptive_ms,
            "latency_reduction_pct": (
                1 - adaptive_ms / kv_ms
            ) * 100,
        }

        records.append(record)

    return records


def build_clusters(records):
    """
    Each workload configuration is a cluster.
    Its three seeds remain together during bootstrap.
    """
    clusters = defaultdict(dict)

    for record in records:
        workload = (
            record["batch_size"],
            record["unique_prompts"],
        )

        threshold = record["threshold"]
        seed = record["seed"]

        clusters[workload].setdefault(
            threshold, {}
        )[seed] = record

    return clusters


def validate_data(records, clusters):
    expected_thresholds = set(THRESHOLDS)

    actual_thresholds = {
        round(record["threshold"], 3)
        for record in records
    }

    if actual_thresholds != {
        round(value, 3) for value in expected_thresholds
    }:
        raise ValueError(
            f"Unexpected thresholds: {actual_thresholds}"
        )

    if len(records) != 126:
        print(
            f"WARNING: Expected 126 records, "
            f"found {len(records)}."
        )

    for workload, threshold_data in clusters.items():
        for threshold in THRESHOLDS:
            if threshold not in threshold_data:
                raise ValueError(
                    f"Missing threshold {threshold} "
                    f"for workload {workload}"
                )

            if len(threshold_data[threshold]) != 3:
                raise ValueError(
                    f"Expected 3 seeds for workload "
                    f"{workload}, threshold {threshold}"
                )

    print("Data validation passed.")
    print(f"Records: {len(records)}")
    print(f"Workload clusters: {len(clusters)}")
    print(f"Thresholds: {THRESHOLDS}")


def threshold_statistics(clusters, rng):
    results = []

    for threshold in THRESHOLDS:
        workload_speedups = []
        workload_reductions = []
        all_speedups = []
        all_reductions = []

        for threshold_data in clusters.values():
            seed_records = list(
                threshold_data[threshold].values()
            )

            speeds = [
                item["speedup"]
                for item in seed_records
            ]

            reductions = [
                item["latency_reduction_pct"]
                for item in seed_records
            ]

            workload_speedups.append(
                statistics.mean(speeds)
            )

            workload_reductions.append(
                statistics.mean(reductions)
            )

            all_speedups.extend(speeds)
            all_reductions.extend(reductions)

        speed_ci = bootstrap_mean_ci(
            workload_speedups, rng
        )

        reduction_ci = bootstrap_mean_ci(
            workload_reductions, rng
        )

        results.append({
            "threshold": threshold,
            "records": len(all_speedups),
            "workloads": len(workload_speedups),
            "mean_speedup": statistics.mean(all_speedups),
            "median_speedup": statistics.median(all_speedups),
            "geometric_mean_speedup": math.exp(
                statistics.mean(
                    math.log(value)
                    for value in all_speedups
                )
            ),
            "speedup_ci_low": speed_ci[0],
            "speedup_ci_high": speed_ci[1],
            "mean_reduction_pct": statistics.mean(
                all_reductions
            ),
            "reduction_ci_low": reduction_ci[0],
            "reduction_ci_high": reduction_ci[1],
        })

    return results


def paired_comparisons(clusters, rng):
    comparisons = []

    pairs = [
        (0.25, 0.375),
        (0.25, 0.50),
        (0.375, 0.50),
    ]

    for threshold_a, threshold_b in pairs:
        differences = []
        log_ratios = []

        for threshold_data in clusters.values():
            speeds_a = [
                item["speedup"]
                for item in threshold_data[
                    threshold_a
                ].values()
            ]

            speeds_b = [
                item["speedup"]
                for item in threshold_data[
                    threshold_b
                ].values()
            ]

            # Match the same seeds within each workload.
            seeds_a = threshold_data[threshold_a]
            seeds_b = threshold_data[threshold_b]

            common_seeds = sorted(
                set(seeds_a) & set(seeds_b)
            )

            paired_a = [
                seeds_a[seed]["speedup"]
                for seed in common_seeds
            ]

            paired_b = [
                seeds_b[seed]["speedup"]
                for seed in common_seeds
            ]

            mean_a = statistics.mean(paired_a)
            mean_b = statistics.mean(paired_b)

            differences.append(mean_a - mean_b)

            log_ratios.append(
                math.log(mean_a / mean_b)
            )

        difference_ci = bootstrap_mean_ci(
            differences, rng
        )

        log_ratio_ci = bootstrap_mean_ci(
            log_ratios, rng
        )

        mean_difference = statistics.mean(
            differences
        )

        ratio = math.exp(
            statistics.mean(log_ratios)
        )

        ratio_ci = (
            math.exp(log_ratio_ci[0]),
            math.exp(log_ratio_ci[1]),
        )

        if difference_ci[0] > 0:
            conclusion = (
                f"{threshold_a:.3f} has a higher "
                f"mean speedup in this workload sample"
            )
        elif difference_ci[1] < 0:
            conclusion = (
                f"{threshold_b:.3f} has a higher "
                f"mean speedup in this workload sample"
            )
        else:
            conclusion = (
                "No clear difference: paired 95% CI "
                "includes zero"
            )

        comparisons.append({
            "threshold_a": threshold_a,
            "threshold_b": threshold_b,
            "mean_speedup_difference": mean_difference,
            "difference_ci_low": difference_ci[0],
            "difference_ci_high": difference_ci[1],
            "geometric_speedup_ratio_a_over_b": ratio,
            "ratio_ci_low": ratio_ci[0],
            "ratio_ci_high": ratio_ci[1],
            "conclusion": conclusion,
        })

    return comparisons


def save_results(results, comparisons):
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fields = list(results[0].keys())
        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "RCT 2.4.8 - PAIRED STATISTICAL ANALYSIS",
        "",
        "Input: benchmarks/rct_247_results.csv",
        f"Bootstrap samples: {BOOTSTRAP_SAMPLES}",
        f"Workload clusters: {results[0]['workloads']}",
        "",
        "THRESHOLD RESULTS",
        "",
    ]

    for result in results:
        lines.extend([
            f"Threshold: {result['threshold']:.3f}",
            f"Mean speedup: {result['mean_speedup']:.4f}x",
            (
                "95% bootstrap CI: "
                f"[{result['speedup_ci_low']:.4f}, "
                f"{result['speedup_ci_high']:.4f}]"
            ),
            (
                "Mean latency reduction: "
                f"{result['mean_reduction_pct']:.2f}%"
            ),
            (
                "95% reduction CI: "
                f"[{result['reduction_ci_low']:.2f}%, "
                f"{result['reduction_ci_high']:.2f}%]"
            ),
            "",
        ])

    lines.extend([
        "PAIRED THRESHOLD COMPARISONS",
        "",
    ])

    for comparison in comparisons:
        lines.extend([
            (
                f"{comparison['threshold_a']:.3f} "
                f"versus {comparison['threshold_b']:.3f}"
            ),
            (
                "Mean speedup difference: "
                f"{comparison['mean_speedup_difference']:.4f}"
            ),
            (
                "95% difference CI: "
                f"[{comparison['difference_ci_low']:.4f}, "
                f"{comparison['difference_ci_high']:.4f}]"
            ),
            (
                "Geometric speedup ratio A/B: "
                f"{comparison['geometric_speedup_ratio_a_over_b']:.4f}"
            ),
            (
                "95% ratio CI: "
                f"[{comparison['ratio_ci_low']:.4f}, "
                f"{comparison['ratio_ci_high']:.4f}]"
            ),
            comparison["conclusion"],
            "",
        ])

    lines.extend([
        "IMPORTANT LIMITATIONS",
        "",
        "This analysis reuses the existing RCT 2.4.7 results.",
        "It does not add new seeds or benchmark repetitions.",
        "Bootstrap resampling is clustered by workload configuration.",
        "Only 14 workload configurations are available.",
        "Confidence intervals describe this sampled workload set;",
        "they do not establish production performance.",
        "The model is randomly initialized and untrained.",
        "",
    ])

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    print("=" * 70)
    print("RCT 2.4.8 - PAIRED STATISTICAL ANALYSIS")
    print("=" * 70)

    rng = random.Random(RANDOM_SEED)

    records = load_data()
    clusters = build_clusters(records)

    validate_data(records, clusters)

    results = threshold_statistics(
        clusters, rng
    )

    comparisons = paired_comparisons(
        clusters, rng
    )

    print("\nTHRESHOLD RESULTS")
    print("-" * 70)

    for result in results:
        print(
            f"\nThreshold: {result['threshold']:.3f}"
        )
        print(
            f"Mean speedup: "
            f"{result['mean_speedup']:.4f}x"
        )
        print(
            f"Median speedup: "
            f"{result['median_speedup']:.4f}x"
        )
        print(
            f"95% bootstrap CI: "
            f"[{result['speedup_ci_low']:.4f}, "
            f"{result['speedup_ci_high']:.4f}]"
        )
        print(
            f"Mean latency reduction: "
            f"{result['mean_reduction_pct']:.2f}%"
        )

    print("\nPAIRED COMPARISONS")
    print("-" * 70)

    for comparison in comparisons:
        print(
            f"\n{comparison['threshold_a']:.3f} "
            f"versus {comparison['threshold_b']:.3f}"
        )
        print(
            "Mean speedup difference: "
            f"{comparison['mean_speedup_difference']:.4f}"
        )
        print(
            "95% difference CI: "
            f"[{comparison['difference_ci_low']:.4f}, "
            f"{comparison['difference_ci_high']:.4f}]"
        )
        print(
            "Geometric speedup ratio A/B: "
            f"{comparison['geometric_speedup_ratio_a_over_b']:.4f}"
        )
        print(
            "95% ratio CI: "
            f"[{comparison['ratio_ci_low']:.4f}, "
            f"{comparison['ratio_ci_high']:.4f}]"
        )
        print(comparison["conclusion"])

    save_results(results, comparisons)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Summary: {SUMMARY_TXT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
