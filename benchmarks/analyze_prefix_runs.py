
from pathlib import Path
import csv
import random
import statistics
from collections import defaultdict

ROOT = Path("results/rct_adaptive_prefix_gate_confirmation")
FILES = sorted(ROOT.rglob("raw_measurements.csv"))

if not FILES:
    raise SystemExit(f"No raw_measurements.csv files found under {ROOT}")

# Store paired speedups separately for each workload and run.
runs = defaultdict(dict)

for file in FILES:
    run_name = file.parent.name

    with file.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        # Also support ordinary comma-separated CSV files.
        if reader.fieldnames and len(reader.fieldnames) == 1:
            f.seek(0)
            reader = csv.DictReader(f)

        for row in reader:
            batch = int(row["batch_size"])
            prefix = int(row["prefix_length"])
            kv = float(row["kv_ms"])
            auto = float(row["auto_ms"])

            key = (batch, prefix)
            runs[key].setdefault(run_name, []).append(kv / auto)

def bootstrap_ci(values, iterations=10000, seed=42):
    rng = random.Random(seed)
    estimates = []

    for _ in range(iterations):
        sample = [rng.choice(values) for _ in values]
        estimates.append(statistics.median(sample))

    estimates.sort()
    return estimates[int(0.025 * iterations)], estimates[int(0.975 * iterations)]

print(f"\nFound {len(FILES)} benchmark run files.")

for key in sorted(runs):
    batch, prefix = key
    run_data = runs[key]
    pooled = [x for values in run_data.values() for x in values]

    print(f"\nBatch {batch}, prefix {prefix}")
    print(f"  Runs: {len(run_data)}")
    print(f"  Measurements: {len(pooled)}")

    for run_name, values in sorted(run_data.items()):
        print(
            f"  {run_name}: "
            f"median speedup={statistics.median(values):.4f}x "
            f"({sum(x > 1 for x in values)}/{len(values)} wins)"
        )

    low, high = bootstrap_ci(pooled)
    median = statistics.median(pooled)

    print(f"  Pooled median speedup: {median:.4f}x")
    print(f"  Pooled 95% bootstrap CI: [{low:.4f}, {high:.4f}]")
