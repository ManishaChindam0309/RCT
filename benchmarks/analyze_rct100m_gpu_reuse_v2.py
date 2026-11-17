

from pathlib import Path
import csv
import statistics

PROJECT_DIR = Path(__file__).resolve().parent.parent

RUN_DIR = (
    PROJECT_DIR
    / "results"
    / "rct100m_gpu_reuse_v2"
    / "run_20261103_143329"
)

CSV_PATH = RUN_DIR / "memory_records.csv"

records = []

with open(CSV_PATH, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        records.append({
            "duplicate_rate": row["duplicate_rate"],
            "method": row["method"],
            "peak_allocated_mb": float(
                row["peak_allocated_mb"]
            ),
        })

rates = ["0%", "50%", "75%", "87.5%"]
methods = [
    "standard_no_cache",
    "kv_cache",
    "rct_dedup_kv",
]

print("=" * 75)
print("RCT GPU MEMORY ANALYSIS")
print("=" * 75)

for rate in rates:
    print(f"\nDuplicate rate: {rate}")

    for method in methods:
        values = [
            row["peak_allocated_mb"]
            for row in records
            if row["duplicate_rate"] == rate
            and row["method"] == method
        ]

        if not values:
            print(method, "NO DATA")
            continue

        print(
            f"{method:22s}"
            f" median={statistics.median(values):.2f} MB"
            f" mean={statistics.mean(values):.2f} MB"
            f" min={min(values):.2f} MB"
            f" max={max(values):.2f} MB"
            f" samples={len(values)}"
        )

print("\nTotal memory records:", len(records))
