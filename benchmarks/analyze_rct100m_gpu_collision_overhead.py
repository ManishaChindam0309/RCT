
from pathlib import Path
import csv
import random
import statistics
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_DIR / "results" / "rct100m_gpu_collision_overhead"

RUN_DIRS = sorted(
    [p for p in RESULTS_ROOT.glob("run_*") if p.is_dir()],
    key=lambda p: p.name
)

if not RUN_DIRS:
    raise FileNotFoundError("No benchmark result directory found.")

RUN_DIR = RUN_DIRS[-1]
CSV_PATH = RUN_DIR / "raw_results.csv"

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Each row is already a paired measurement:
# KV-only and adaptive RCT were timed on the same prompt batch.

data = {}

for row in rows:
    scenario = row["scenario"]
    config = (
        int(row["batch_size"]),
        int(row["prompt_length"]),
        int(row["fingerprint_columns"])
    )
    cluster = int(row["cluster_id"])

    kv = float(row["kv_only_median_ms"])
    adaptive = float(row["adaptive_median_ms"])

    speedup = kv / adaptive

    data[(scenario, config, cluster)] = {
        "speedup": speedup,
        "kv": kv,
        "adaptive": adaptive
    }

scenarios = ["unique_control", "deliberate_collision"]

configs = sorted({
    key[1] for key in data
})

# Use a stratified paired bootstrap:
# resample the three clusters within each configuration.
# This preserves the benchmark's configuration mix.

BOOTSTRAPS = 20000
rng = np.random.default_rng(20261103)


def bootstrap_mean_ci(values_by_config):
    estimates = []

    for _ in range(BOOTSTRAPS):
        sample = []

        for values in values_by_config:
            indices = rng.integers(
                0, len(values), size=len(values)
            )
            sample.extend(values[i] for i in indices)

        estimates.append(float(np.mean(sample)))

    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5))
    )


print("=" * 72)
print("RCT GPU COLLISION OVERHEAD: PAIRED BOOTSTRAP ANALYSIS")
print("=" * 72)
print("Source:", CSV_PATH)
print("Bootstrap replicates:", BOOTSTRAPS)

for scenario in scenarios:

    grouped = []

    all_speedups = []
    all_kv = []
    all_adaptive = []

    for config in configs:
        values = []

        for cluster in range(3):
            key = (scenario, config, cluster)

            if key not in data:
                continue

            record = data[key]
            values.append(record["speedup"])

            all_speedups.append(record["speedup"])
            all_kv.append(record["kv"])
            all_adaptive.append(record["adaptive"])

        if values:
            grouped.append(values)

    ci_low, ci_high = bootstrap_mean_ci(grouped)

    print("\nScenario:", scenario)
    print("Workload clusters:", len(all_speedups))
    print("Mean speedup:", f"{statistics.mean(all_speedups):.4f}x")
    print("Median speedup:", f"{statistics.median(all_speedups):.4f}x")
    print(
        "95% bootstrap CI:",
        f"[{ci_low:.4f}, {ci_high:.4f}]"
    )
    print(
        "Mean KV latency:",
        f"{statistics.mean(all_kv):.3f} ms"
    )
    print(
        "Mean adaptive latency:",
        f"{statistics.mean(all_adaptive):.3f} ms"
    )

# Compare collision and control speedups for matching
# configuration and cluster IDs.
difference_groups = []
all_differences = []

for config in configs:
    differences = []

    for cluster in range(3):
        control_key = ("unique_control", config, cluster)
        collision_key = ("deliberate_collision", config, cluster)

        if control_key not in data or collision_key not in data:
            continue

        difference = (
            data[collision_key]["speedup"]
            - data[control_key]["speedup"]
        )

        differences.append(difference)
        all_differences.append(difference)

    if differences:
        difference_groups.append(differences)

diff_low, diff_high = bootstrap_mean_ci(difference_groups)

print("\n" + "=" * 72)
print("PAIRED COLLISION EFFECT")
print("=" * 72)
print(
    "Mean speedup difference (collision - control):",
    f"{statistics.mean(all_differences):+.4f}x"
)
print(
    "95% bootstrap CI:",
    f"[{diff_low:+.4f}, {diff_high:+.4f}]"
)

if diff_low <= 0 <= diff_high:
    print("Interpretation: no clear difference between scenarios.")
elif diff_low > 0:
    print("Interpretation: collision scenario was faster in this workload mix.")
else:
    print("Interpretation: collision scenario was slower in this workload mix.")

print("\nAnalysis complete.")
