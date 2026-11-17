import csv
import random
import statistics
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "results" / "rct100m_generalization" / "run_20261103T062842Z"

INPUT = RUN_DIR / "raw_timing_records.csv"
OUTPUT = RUN_DIR / "independent_config_summary.csv"

BOOTSTRAP_ITERATIONS = 20000
SEED = 20261103

random.seed(SEED)
rng = np.random.default_rng(SEED)

with INPUT.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

# Organize raw timings.
data = defaultdict(list)

for row in rows:
    config = row["configuration"]
    cluster = int(row["cluster"])
    method = row["method"]
    elapsed = float(row["elapsed_seconds"])

    data[(config, cluster, method)].append(elapsed)

configs = sorted(set(row["configuration"] for row in rows))

methods = [
    "standard",
    "kv_cache",
    "combined_no_dedup",
    "combined_dedup",
]

comparisons = {
    "kv_vs_standard": ("standard", "kv_cache"),
    "no_dedup_vs_standard": ("standard", "combined_no_dedup"),
    "dedup_vs_standard": ("standard", "combined_dedup"),
    "dedup_vs_no_dedup": ("combined_no_dedup", "combined_dedup"),
}

def bootstrap_median_ci(values):
    values = np.asarray(values, dtype=float)
    samples = rng.integers(
        0, len(values),
        size=(BOOTSTRAP_ITERATIONS, len(values))
    )
    bootstrap_medians = np.median(values[samples], axis=1)

    return (
        float(np.median(values)),
        float(np.quantile(bootstrap_medians, 0.025)),
        float(np.quantile(bootstrap_medians, 0.975)),
    )

results = []

for config in configs:
    clusters = sorted({
        cluster
        for cfg, cluster, method in data
        if cfg == config
    })

    cluster_latency = {}

    for cluster in clusters:
        for method in methods:
            timings = data[(config, cluster, method)]

            if len(timings) != 3:
                raise ValueError(
                    f"Expected 3 repetitions: {config}, "
                    f"cluster {cluster}, {method}"
                )

            cluster_latency[(cluster, method)] = statistics.median(timings)

    result = {
        "configuration": config,
        "clusters": len(clusters),
    }

    for method in methods:
        all_times = [
            t
            for (cfg, cluster, m), timings in data.items()
            if cfg == config and m == method
            for t in timings
        ]
        result[f"{method}_median_seconds"] = statistics.median(all_times)

    for name, (baseline, optimized) in comparisons.items():
        ratios = [
            cluster_latency[(cluster, baseline)]
            / cluster_latency[(cluster, optimized)]
            for cluster in clusters
        ]

        estimate, low, high = bootstrap_median_ci(ratios)

        result[f"{name}_speedup"] = estimate
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high

    results.append(result)

fields = list(results[0].keys())

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print("=" * 70)
print("INDEPENDENT GENERALIZATION ANALYSIS")
print("=" * 70)
print("Raw records:", len(rows))
print("Configurations:", len(configs))
print("Bootstrap iterations:", BOOTSTRAP_ITERATIONS)
print("Clusters per configuration:", sorted(set(r["clusters"] for r in results)))

print("\nDEDUPLICATION SPEEDUPS")
print("-" * 70)

for r in results:
    print(
        f'{r["configuration"]:30s} '
        f'{r["dedup_vs_no_dedup_speedup"]:.3f}x '
        f'[{r["dedup_vs_no_dedup_ci_low"]:.3f}, '
        f'{r["dedup_vs_no_dedup_ci_high"]:.3f}]'
    )

print("\nSaved:", OUTPUT)
print("ANALYSIS COMPLETE")