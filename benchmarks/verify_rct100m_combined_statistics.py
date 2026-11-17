import csv
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "results" / "rct100m_combined_robust" / "run_20261103T053200Z"
CSV_FILE = RUN_DIR / "raw_timing_records.csv"

BOOTSTRAP_ITERATIONS = 20000
SEED = 20261103

METHODS = [
    "standard",
    "kv_cache",
    "combined_no_dedup",
    "combined_dedup",
]

COMPARISONS = {
    "kv_vs_standard": ("standard", "kv_cache"),
    "combined_no_dedup_vs_standard": ("standard", "combined_no_dedup"),
    "combined_dedup_vs_standard": ("standard", "combined_dedup"),
    "dedup_vs_no_dedup": ("combined_no_dedup", "combined_dedup"),
}


def percentile(values, percentage):
    values = sorted(values)
    position = (len(values) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def bootstrap_ci(values, rng):
    estimates = []
    count = len(values)

    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [values[rng.randrange(count)] for _ in range(count)]
        estimates.append(statistics.median(sample))

    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def load_records():
    if not CSV_FILE.exists():
        raise FileNotFoundError(CSV_FILE)

    with CSV_FILE.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "workload",
            "unique_prompts",
            "duplicate_prompts",
            "cluster",
            "repetition",
            "method",
            "elapsed_seconds",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {required - set(reader.fieldnames or [])}")

        rows = list(reader)

    if len(rows) != 2400:
        raise ValueError(f"Expected 2400 records, found {len(rows)}")

    return rows


def calculate_cluster_medians(rows):
    grouped = defaultdict(list)

    for row in rows:
        workload = row["workload"]
        cluster = int(row["cluster"])
        repetition = int(row["repetition"])
        method = row["method"]
        elapsed = float(row["elapsed_seconds"])

        if method not in METHODS:
            raise ValueError(f"Unexpected method: {method}")

        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid timing: {elapsed}")

        grouped[(workload, cluster, method)].append((repetition, elapsed))

    workloads = sorted({row["workload"] for row in rows})
    results = {}

    for workload in workloads:
        clusters = sorted({
            cluster
            for (name, cluster, method) in grouped
            if name == workload
        })

        if len(clusters) != 30:
            raise ValueError(f"{workload}: expected 30 clusters, found {len(clusters)}")

        for cluster in clusters:
            for method in METHODS:
                values = grouped[(workload, cluster, method)]

                if len(values) != 5:
                    raise ValueError(
                        f"{workload}, cluster {cluster}, {method}: "
                        f"expected 5 repetitions, found {len(values)}"
                    )

                repetition_ids = sorted(rep for rep, _ in values)
                if len(set(repetition_ids)) != 5:
                    raise ValueError("Duplicate repetition IDs detected")

                results[(workload, cluster, method)] = statistics.median(
                    elapsed for _, elapsed in values
                )

    return workloads, results


def summarize(workload, cluster_ids, medians, rng):
    print(f"\n{workload}")
    print("-" * 88)
    print(f"{'Comparison':43s} {'Median':>12s} {'95% bootstrap CI':>25s}")

    for label, (baseline, optimized) in COMPARISONS.items():
        ratios = [
            medians[(workload, cluster, baseline)]
            / medians[(workload, cluster, optimized)]
            for cluster in cluster_ids
        ]

        estimate = statistics.median(ratios)
        low, high = bootstrap_ci(ratios, rng)

        print(
            f"{label:43s} "
            f"{estimate:9.3f}x "
            f"[{low:.3f}x, {high:.3f}x]"
        )

    print("\nMedian latency across clusters:")
    for method in METHODS:
        latencies = [
            medians[(workload, cluster, method)]
            for cluster in cluster_ids
        ]
        print(f"{method:25s} {statistics.median(latencies):.6f}s")


def main():
    print("=" * 88)
    print("INDEPENDENT RCT 100M COMBINED BENCHMARK STATISTICS VERIFICATION")
    print("=" * 88)

    rows = load_records()
    print("Raw timing records:", len(rows))

    workloads, medians = calculate_cluster_medians(rows)
    print("Workloads:", len(workloads))
    print("Independent clusters per workload: 30")
    print("Repetitions per method and cluster: 5")
    print("Bootstrap iterations:", BOOTSTRAP_ITERATIONS)

    rng = random.Random(SEED)
    cluster_ids = {
        workload: sorted({
            cluster
            for (name, cluster, method) in medians
            if name == workload
        })
        for workload in workloads
    }

    for workload in workloads:
        summarize(workload, cluster_ids[workload], medians, rng)

    print("\n" + "=" * 88)
    print("POOLED EXPLORATORY RESULTS")
    print("=" * 88)
    print("Warning: the same underlying prompt clusters occur in multiple workloads.")

    pooled = {}

    for label, (baseline, optimized) in {
        "kv_vs_standard": ("standard", "kv_cache"),
        "combined_dedup_vs_standard": ("standard", "combined_dedup"),
        "dedup_vs_no_dedup": ("combined_no_dedup", "combined_dedup"),
    }.items():
        ratios = []

        for workload in workloads:
            for cluster in cluster_ids[workload]:
                ratios.append(
                    medians[(workload, cluster, baseline)]
                    / medians[(workload, cluster, optimized)]
                )

        estimate = statistics.median(ratios)
        low, high = bootstrap_ci(ratios, rng)
        pooled[label] = (estimate, low, high)

        print(
            f"{label:38s} "
            f"{estimate:.3f}x [{low:.3f}x, {high:.3f}x]"
        )

    print("\n" + "=" * 88)
    print("VERIFICATION COMPLETE")
    print("All timing records and paired cluster calculations passed validation.")
    print("=" * 88)


if __name__ == "__main__":
    main()
