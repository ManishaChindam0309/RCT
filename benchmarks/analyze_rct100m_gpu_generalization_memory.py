
import csv
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

RUN_DIR = (
    PROJECT_DIR
    / "results"
    / "rct100m_gpu_generalization"
    / "run_20261103_151314"
)

INPUT_FILE = RUN_DIR / "memory_records.csv"
OUTPUT_CSV = RUN_DIR / "memory_analysis.csv"
OUTPUT_REPORT = RUN_DIR / "memory_analysis.txt"

METHODS = [
    "standard_no_cache",
    "kv_cache",
    "rct_dedup_kv",
]


def median(values):
    return statistics.median(values) if values else float("nan")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    records = []

    with INPUT_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "configuration",
            "batch_size",
            "prompt_length",
            "duplicate_rate",
            "cluster",
            "repeat",
            "method",
            "peak_allocated_mb",
        }

        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing columns. Found: {reader.fieldnames}"
            )

        for row in reader:
            row["batch_size"] = int(row["batch_size"])
            row["prompt_length"] = int(row["prompt_length"])
            row["cluster"] = int(row["cluster"])
            row["repeat"] = int(row["repeat"])
            row["peak_allocated_mb"] = float(
                row["peak_allocated_mb"]
            )
            records.append(row)

    print(f"Loaded memory records: {len(records)}")

    if len(records) != 8100:
        print(
            "WARNING: Expected 8100 records. "
            f"Found {len(records)}."
        )

    # Group measurements by configuration and method.
    grouped = defaultdict(list)

    # Pair methods by configuration, cluster and repeat.
    paired = {}

    for row in records:
        key = (
            row["configuration"],
            row["method"],
        )
        grouped[key].append(row["peak_allocated_mb"])

        pair_key = (
            row["configuration"],
            row["cluster"],
            row["repeat"],
        )

        paired.setdefault(pair_key, {})[
            row["method"]
        ] = row["peak_allocated_mb"]

    configurations = sorted({
        row["configuration"] for row in records
    })

    summary = []

    for config in configurations:
        config_rows = [
            row for row in records
            if row["configuration"] == config
        ]

        first = config_rows[0]

        values = {}

        for method in METHODS:
            method_values = grouped.get((config, method), [])
            values[method] = median(method_values)

        kv_mb = values["kv_cache"]
        rct_mb = values["rct_dedup_kv"]
        standard_mb = values["standard_no_cache"]

        if kv_mb > 0:
            saved_mb = kv_mb - rct_mb
            saved_percent = saved_mb / kv_mb * 100
        else:
            saved_mb = 0
            saved_percent = 0

        summary.append({
            "configuration": config,
            "batch_size": first["batch_size"],
            "prompt_length": first["prompt_length"],
            "duplicate_rate": first["duplicate_rate"],
            "standard_mb": standard_mb,
            "kv_mb": kv_mb,
            "rct_mb": rct_mb,
            "saved_mb": saved_mb,
            "saved_percent": saved_percent,
        })

    # Save per-configuration results.
    fields = list(summary[0].keys())

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    # Analyze paired RCT/KV memory measurements.
    paired_savings = defaultdict(list)

    for key, methods in paired.items():
        if (
            "kv_cache" in methods
            and "rct_dedup_kv" in methods
        ):
            config = key[0]
            kv = methods["kv_cache"]
            rct = methods["rct_dedup_kv"]

            if kv > 0:
                paired_savings[config].append(
                    (kv - rct) / kv * 100
                )

    # Aggregate by batch size and duplication rate.
    aggregate = defaultdict(list)

    for row in summary:
        aggregate[
            (row["batch_size"], row["duplicate_rate"])
        ].append(row["saved_percent"])

    lines = []

    lines.append("RCT GPU MEMORY GENERALIZATION ANALYSIS")
    lines.append("=" * 72)
    lines.append(f"Input records: {len(records)}")
    lines.append(f"Configurations: {len(configurations)}")
    lines.append("")
    lines.append(
        "Peak allocated memory in MB; RCT savings are relative "
        "to KV caching."
    )
    lines.append("")

    lines.append("PER-CONFIGURATION RESULTS")
    lines.append("-" * 72)

    header = (
        f"{'Config':<14}"
        f"{'Standard':>12}"
        f"{'KV':>12}"
        f"{'RCT':>12}"
        f"{'Saved MB':>12}"
        f"{'Saved %':>11}"
    )
    lines.append(header)

    for row in summary:
        lines.append(
            f"{row['configuration']:<14}"
            f"{row['standard_mb']:>12.2f}"
            f"{row['kv_mb']:>12.2f}"
            f"{row['rct_mb']:>12.2f}"
            f"{row['saved_mb']:>12.2f}"
            f"{row['saved_percent']:>10.2f}%"
        )

    lines.append("")
    lines.append("AGGREGATED MEMORY SAVINGS")
    lines.append("-" * 72)
    lines.append(
        "Median percentage saved across prompt lengths "
        "for each batch/duplicate combination."
    )

    for batch in (4, 8, 16):
        for duplicate in ("0%", "50%", "75%"):
            vals = aggregate.get((batch, duplicate), [])
            if vals:
                lines.append(
                    f"Batch {batch:>2}, duplicates {duplicate:>3}: "
                    f"{median(vals):.2f}% median memory saved"
                )

    lines.append("")
    lines.append("PAIRED MEASUREMENT COUNTS")
    lines.append("-" * 72)

    for config in configurations:
        count = len(paired_savings.get(config, []))
        lines.append(f"{config}: {count} paired measurements")

    lines.append("")
    lines.append("INTERPRETATION NOTES")
    lines.append("-" * 72)
    lines.append(
        "1. Values represent peak PyTorch allocated GPU memory, "
        "not total GPU VRAM usage."
    )
    lines.append(
        "2. RCT memory savings are measured against KV caching."
    )
    lines.append(
        "3. The standard method does not allocate the same "
        "persistent KV cache."
    )
    lines.append(
        "4. Results describe this model, checkpoint, GPU and "
        "benchmark workload."
    )
    lines.append(
        "5. Negative savings mean RCT allocated more memory "
        "than the KV baseline."
    )

    report_text = "\n".join(lines)

    OUTPUT_REPORT.write_text(
        report_text,
        encoding="utf-8"
    )

    print("\n" + report_text)

    print("\nSaved files:")
    print(OUTPUT_CSV)
    print(OUTPUT_REPORT)


if __name__ == "__main__":
    main()
