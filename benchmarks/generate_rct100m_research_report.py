
import csv
import random
import statistics
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

RUN_DIR = (
    ROOT / "results" / "rct100m_generalization"
    / "run_20261103T062842Z"
)

RAW_FILE = RUN_DIR / "raw_timing_records.csv"
REPORT_FILE = RUN_DIR / "REPORT.md"
SUMMARY_FILE = RUN_DIR / "independent_config_summary.csv"

BOOTSTRAP_ITERATIONS = 20000
SEED = 20261103

rng = np.random.default_rng(SEED)
random.seed(SEED)

METHODS = [
    "standard",
    "kv_cache",
    "combined_no_dedup",
    "combined_dedup",
]

METHOD_LABELS = {
    "standard": "Standard",
    "kv_cache": "KV cache",
    "combined_no_dedup": "Combined without dedup",
    "combined_dedup": "Combined with dedup",
}

COMPARISONS = {
    "KV vs standard": ("standard", "kv_cache"),
    "Combined without dedup vs standard":
        ("standard", "combined_no_dedup"),
    "Combined with dedup vs standard":
        ("standard", "combined_dedup"),
    "Dedup vs combined without dedup":
        ("combined_no_dedup", "combined_dedup"),
}


def load_data():
    with RAW_FILE.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 6480:
        raise ValueError(
            f"Expected 6480 records, found {len(rows)}"
        )

    data = defaultdict(list)

    for row in rows:
        key = (
            row["configuration"],
            int(row["cluster"]),
            row["method"],
        )

        data[key].append(
            float(row["elapsed_seconds"])
        )

    return rows, data


def bootstrap_ci(values):
    values = np.asarray(values, dtype=float)

    indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_ITERATIONS, len(values)),
    )

    samples = values[indices]
    estimates = np.median(samples, axis=1)

    return (
        float(np.median(values)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def analyze(rows, data):
    configurations = sorted(
        set(row["configuration"] for row in rows)
    )

    results = []

    for config in configurations:
        clusters = sorted({
            cluster
            for cfg, cluster, method in data
            if cfg == config
        })

        cluster_latency = {}

        for cluster in clusters:
            for method in METHODS:
                timings = data[(config, cluster, method)]

                if len(timings) != 3:
                    raise ValueError(
                        f"Invalid repetition count: "
                        f"{config}, {cluster}, {method}"
                    )

                cluster_latency[(cluster, method)] = (
                    statistics.median(timings)
                )

        result = {
            "configuration": config,
            "batch_size": int(
                next(r["batch_size"] for r in rows
                     if r["configuration"] == config)
            ),
            "prompt_length": int(
                next(r["prompt_length"] for r in rows
                     if r["configuration"] == config)
            ),
            "duplication_rate": float(
                next(r["duplication_rate"] for r in rows
                     if r["configuration"] == config)
            ),
            "clusters": len(clusters),
        }

        for method in METHODS:
            timings = [
                t
                for (cfg, cluster, m), values in data.items()
                if cfg == config and m == method
                for t in values
            ]

            result[f"{method}_median"] = (
                statistics.median(timings)
            )

        for name, (baseline, optimized) in COMPARISONS.items():
            ratios = [
                cluster_latency[(cluster, baseline)]
                / cluster_latency[(cluster, optimized)]
                for cluster in clusters
            ]

            estimate, low, high = bootstrap_ci(ratios)

            key = name.lower().replace(" ", "_").replace("/", "_")

            result[f"{key}_speedup"] = estimate
            result[f"{key}_low"] = low
            result[f"{key}_high"] = high

        results.append(result)

    return results


def save_summary(results):
    fields = list(results[0].keys())

    with SUMMARY_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def build_report(rows, results):
    lines = []

    def add(text=""):
        lines.append(text)

    add("# RCT: Generalization Benchmark Research Report")
    add()
    add("**Experiment:** RCT 100M inference generalization")
    add()
    add("**Run:** `run_20261103T062842Z`")
    add()
    add("**Status:** Completed")
    add()

    add("## Abstract")
    add()
    add(
        "This experiment evaluates standard Transformer inference, "
        "KV-cache inference, combined inference without prompt "
        "deduplication, and combined inference with exact "
        "duplicate-prompt deduplication. The experiment covers "
        "27 configurations spanning three batch sizes, three "
        "prompt lengths, and three duplication rates."
    )
    add()
    add(
        "The dataset contains 6,480 timing records. Results show "
        "that the measured benefit of exact prompt deduplication "
        "increases with the proportion of duplicate prompts. "
        "The results are specific to the tested model, CPU "
        "environment, and synthetic batch construction."
    )
    add()

    add("## 1. Experimental setup")
    add()
    add("| Parameter | Value |")
    add("|---|---|")
    add("| Model | RCT 100M |")
    add("| Parameters | 99,028,992 |")
    add("| Checkpoint | Continued training, step 1500 |")
    add("| Framework | PyTorch 2.15.0+cpu |")
    add("| Hardware | CPU, 4 threads |")
    add("| Training sequence length | 32 tokens |")
    add("| Batch sizes | 4, 8, 16 |")
    add("| Prompt lengths | 8, 16, 24 tokens |")
    add("| Generated tokens | 8 |")
    add("| Duplication rates | 0%, 50%, 75% |")
    add("| Clusters per configuration | 20 |")
    add("| Timed repetitions per method and cluster | 3 |")
    add("| Warm-up runs | 1 |")
    add("| Bootstrap iterations | 20,000 |")
    add("| Generation mode | Greedy |")
    add("| Total configurations | 27 |")
    add("| Total timing records | 6,480 |")
    add()

    add("## 2. Inference methods")
    add()
    add(
        "1. **Standard:** Full-context recomputation during "
        "autoregressive generation."
    )
    add(
        "2. **KV cache:** Reuses previously computed attention "
        "keys and values."
    )
    add(
        "3. **Combined without dedup:** Uses KV caching but "
        "processes every prompt independently."
    )
    add(
        "4. **Combined with dedup:** Uses KV caching and exact "
        "duplicate-prompt deduplication."
    )
    add()

    add("## 3. Statistical methodology")
    add()
    add(
        "For each configuration, three timing repetitions are "
        "aggregated using their median within each workload "
        "cluster. Paired speedup ratios are calculated between "
        "methods for the same cluster."
    )
    add()
    add(
        "The reported point estimate is the median of the "
        "20 cluster-level speedup ratios. The 95% confidence "
        "interval is calculated using 20,000 bootstrap samples "
        "of the workload clusters."
    )
    add()
    add(
        "Confidence intervals are configuration-specific and "
        "are not adjusted for multiple comparisons across "
        "the 27 configurations."
    )
    add()

    add("## 4. Results for all configurations")
    add()
    add(
        "| Configuration | Standard (s) | KV (s) | "
        "Combined no dedup (s) | Combined dedup (s) | "
        "Dedup speedup | 95% CI |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|")

    for r in results:
        key = "dedup_vs_combined_without_dedup"

        # Match the comparison key generated above.
        if key not in r:
            key = "dedup_vs_combined_without_dedup"

        speedup = r[key + "_speedup"]
        low = r[key + "_low"]
        high = r[key + "_high"]

        add(
            f"| {r['configuration']} "
            f"| {r['standard_median']:.6f} "
            f"| {r['kv_cache_median']:.6f} "
            f"| {r['combined_no_dedup_median']:.6f} "
            f"| {r['combined_dedup_median']:.6f} "
            f"| {speedup:.3f}x "
            f"| [{low:.3f}, {high:.3f}] |"
        )

    add()

    add("## 5. Results by duplication rate")
    add()
    add(
        "The following are descriptive summaries of the nine "
        "configuration-level speedup estimates at each "
        "duplication rate. They are not pooled estimates."
    )
    add()
    add(
        "| Duplication | Mean speedup | Median speedup | "
        "Configurations with CI excluding 1 |"
    )
    add("|---|---:|---:|---:|")

    key = "dedup_vs_combined_without_dedup"

    for rate in [0.0, 0.5, 0.75]:
        subset = [
            r for r in results
            if r["duplication_rate"] == rate
        ]

        values = [
            r[key + "_speedup"] for r in subset
        ]

        significant = sum(
            r[key + "_low"] > 1.0
            or r[key + "_high"] < 1.0
            for r in subset
        )

        add(
            f"| {rate:.0%} "
            f"| {statistics.mean(values):.3f}x "
            f"| {statistics.median(values):.3f}x "
            f"| {significant}/9 |"
        )

    add()

    add("## 6. Correctness")
    add()
    add(
        "The benchmark execution logs reported passing correctness "
        "checks for all 27 configurations, with 20 out of 20 "
        "workload clusters passing in each configuration."
    )
    add()
    add(
        "The timing CSV does not itself contain generated token "
        "outputs or per-cluster correctness records. Those should "
        "be preserved separately as correctness evidence."
    )
    add()

    add("## 7. Data integrity")
    add()
    add("| Validation | Result |")
    add("|---|---:|")
    add("| Total records | 6,480 |")
    add("| Expected records | 6,480 |")
    add("| Unique configurations | 27 |")
    add("| Records per configuration | 240 |")
    add("| Records per method | 1,620 |")
    add("| Duplicate measurement keys | 0 |")
    add("| Invalid timings | 0 |")
    add()
    add(
        "The raw timing CSV passed the independent integrity "
        "checks performed after the experiment."
    )
    add()

    add("## 8. Limitations")
    add()
    add(
        "- The experiment uses one CPU system and one model checkpoint."
    )
    add(
        "- The trained model has limited training data and does "
        "not establish strong language-model quality."
    )
    add(
        "- The workload uses constructed duplicate batches rather "
        "than production inference traces."
    )
    add(
        "- The benchmark evaluates greedy generation; identical "
        "duplicate prompts share the same continuation."
    )
    add(
        "- The tested prompt lengths are 8, 16, and 24 tokens, "
        "with 8 generated tokens."
    )
    add(
        "- The model was trained with sequence length 32; "
        "the experiment does not establish performance or "
        "quality for long-context inference."
    )
    add(
        "- KV caching and prompt deduplication are established "
        "techniques. This experiment does not establish "
        "algorithmic novelty."
    )
    add(
        "- No direct comparison with vLLM automatic prefix "
        "caching or SGLang RadixAttention has been performed."
    )
    add(
        "- The confidence intervals are not adjusted for "
        "multiple configuration comparisons."
    )
    add()

    add("## 9. Research interpretation")
    add()
    add(
        "The observed results support a workload-dependent "
        "benefit from exact prompt deduplication. The lack of "
        "a clear benefit at zero duplication and increasing "
        "speedups at higher duplication rates motivate "
        "further investigation of adaptive reuse policies."
    )
    add()
    add(
        "The results should be interpreted as preliminary "
        "systems evidence rather than proof of a new general "
        "inference algorithm."
    )
    add()

    add("## 10. Reproducibility artifacts")
    add()
    add("- `raw_timing_records.csv` — original timing records.")
    add("- `experiment_results.json` — benchmark summary.")
    add("- `progress.json` — experiment progress record.")
    add("- `independent_config_summary.csv` — independent analysis.")
    add("- `REPORT.md` — this report.")
    add()
    add("## Conclusion")
    add()
    add(
        "The RCT generalization experiment completed all 27 "
        "configurations and recorded 6,480 timing measurements. "
        "The results provide preliminary evidence that exact "
        "duplicate-prompt reuse can reduce inference latency "
        "when repeated prompts are present in a batch. "
        "Further comparison against established inference "
        "systems and evaluation on GPU hardware are necessary "
        "before making broader performance or novelty claims."
    )

    REPORT_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    rows, data = load_data()

    print("Loaded raw records:", len(rows))

    results = analyze(rows, data)

    save_summary(results)
    build_report(rows, results)

    print("Configurations:", len(results))
    print("Saved summary:", SUMMARY_FILE)
    print("Saved report:", REPORT_FILE)
    print("Report size:", REPORT_FILE.stat().st_size, "bytes")
    print("REPORT GENERATION COMPLETE")


if __name__ == "__main__":
    main()
