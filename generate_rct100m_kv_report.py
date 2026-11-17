
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

PROJECT_DIR = Path(__file__).resolve().parent

RUN_DIR = (
    PROJECT_DIR
    / "results"
    / "rct100m_kv_large"
    / "run_20261102T173833Z"
)

JSON_PATH = RUN_DIR / "experiment_results.json"
RAW_CSV_PATH = RUN_DIR / "raw_timing_records.csv"
SUMMARY_CSV_PATH = RUN_DIR / "per_workload_summary.csv"
REPORT_PATH = RUN_DIR / "REPORT.md"


def load_data():
    for path in [JSON_PATH, RAW_CSV_PATH, SUMMARY_CSV_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing archive file: {path}")

    with open(JSON_PATH, encoding="utf-8") as f:
        metadata = json.load(f)

    with open(RAW_CSV_PATH, newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    with open(SUMMARY_CSV_PATH, newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    return metadata, raw_rows, summary_rows


def validate_data(metadata, raw_rows, summary_rows):
    config = metadata["configuration"]
    overall = metadata["overall_results"]
    correctness = metadata["correctness"]

    expected_records = (
        len(config["prompt_lengths"])
        * config["number_of_prompt_locations"]
        * config["timed_repetitions"]
    )

    assert len(raw_rows) == expected_records, (
        f"Expected {expected_records} raw records, "
        f"found {len(raw_rows)}"
    )

    assert len(summary_rows) == (
        len(config["prompt_lengths"])
        * config["number_of_prompt_locations"]
    )

    assert correctness["all_outputs_matched"] is True
    assert correctness["timed_correctness_checks"] == expected_records

    assert overall["timed_paired_records"] == expected_records

    assert all(
        row["output_match"].lower() == "true"
        for row in raw_rows
    )

    return expected_records


def build_report(metadata, raw_rows, summary_rows):
    config = metadata["configuration"]
    overall = metadata["overall_results"]
    correctness = metadata["correctness"]
    memory = metadata["memory"]

    expected_records = validate_data(
        metadata, raw_rows, summary_rows
    )

    lines = []

    def add(text=""):
        lines.append(text)

    add("# RCT 100M KV-Cache Benchmark Report")
    add()
    add("## 1. Experiment overview")
    add()
    add(f"- Experiment: {metadata['experiment']}")
    add(f"- Run timestamp (UTC): {metadata['created_utc']}")
    add(f"- Checkpoint: `{config['checkpoint']}`")
    add(f"- Checkpoint step: {config['checkpoint_step']}")
    add(f"- Model parameters: {config['parameter_count']:,}")
    add(f"- Device: {config['device']}")
    add(f"- CPU threads: {config['cpu_threads']}")
    add(f"- PyTorch: {config['pytorch_version']}")
    add(f"- Python: {config['python_version']}")
    add()

    add("## 2. Benchmark configuration")
    add()
    add(f"- Prompt lengths: {config['prompt_lengths']}")
    add(f"- Generated tokens: {config['generated_tokens']}")
    add(
        "- Distinct prompt locations: "
        f"{config['number_of_prompt_locations']}"
    )
    add(f"- Warm-up repetitions: {config['warmup_runs']}")
    add(f"- Timed repetitions: {config['timed_repetitions']}")
    add(f"- Random seed: {config['seed']}")
    add(
        "- Bootstrap iterations: "
        f"{config['bootstrap_iterations']:,}"
    )
    add(f"- Bootstrap unit: {config['bootstrap_unit']}")
    add(f"- Window size: {config['window_size_tokens']} tokens")
    add(f"- Validation file: `{config['validation_file']}`")
    add()

    add("## 3. Results by prompt length")
    add()
    add(
        "| Prompt length | Standard median (s) | "
        "KV median (s) | Speedup | Latency reduction |"
    )
    add("|---:|---:|---:|---:|---:|")

    for result in metadata["results_by_prompt_length"]:
        add(
            f"| {result['prompt_length']} "
            f"| {result['standard_median_seconds']:.4f} "
            f"| {result['cached_median_seconds']:.4f} "
            f"| {result['aggregate_median_speedup']:.3f}x "
            f"| {result['latency_reduction_percent']:.2f}% |"
        )

    add()
    add("## 4. Overall performance")
    add()
    add(
        f"- Workload combinations: "
        f"{overall['workload_combinations']}"
    )
    add(
        f"- Paired timing records: "
        f"{overall['timed_paired_records']}"
    )
    add(
        f"- Measurements per method: "
        f"{overall['timed_measurements_per_method']}"
    )
    add(
        f"- Standard overall median: "
        f"{overall['standard_overall_median_seconds']:.4f} seconds"
    )
    add(
        f"- KV overall median: "
        f"{overall['cached_overall_median_seconds']:.4f} seconds"
    )
    add(
        f"- Aggregate median speedup: "
        f"{overall['aggregate_median_speedup']:.3f}x"
    )
    add(
        f"- Paired workload median speedup: "
        f"{overall['paired_workload_median_speedup']:.3f}x"
    )

    ci = overall["cluster_bootstrap_95_ci"]
    add(
        f"- Cluster-bootstrap 95% CI: "
        f"[{ci[0]:.3f}x, {ci[1]:.3f}x]"
    )
    add()

    add("## 5. Correctness verification")
    add()
    add(
        f"- Warm-up checks: "
        f"{correctness['warmup_correctness_checks']}/"
        f"{correctness['warmup_expected_checks']}"
    )
    add(
        f"- Timed checks: "
        f"{correctness['timed_correctness_checks']}/"
        f"{correctness['timed_expected_checks']}"
    )
    add(
        f"- All outputs matched: "
        f"{correctness['all_outputs_matched']}"
    )
    add(
        f"- Independently verified raw records: "
        f"{expected_records}"
    )
    add()

    add("## 6. Per-prompt workload results")
    add()
    add(
        "| Prompt | Validation start | Length | "
        "Standard (s) | KV (s) | Speedup |"
    )
    add("|---:|---:|---:|---:|---:|---:|")

    for row in summary_rows:
        add(
            f"| {int(row['prompt_number'])} "
            f"| {int(row['validation_start_token'])} "
            f"| {int(row['prompt_length'])} "
            f"| {float(row['standard_median_seconds']):.4f} "
            f"| {float(row['cached_median_seconds']):.4f} "
            f"| {float(row['speedup']):.3f}x |"
        )

    add()
    add("## 7. Memory measurements")
    add()
    for stage, values in memory.items():
        add(f"### {stage.replace('_', ' ').title()}")
        add()
        if isinstance(values, dict):
            for key, value in values.items():
                add(f"- {key}: {value}")
        else:
            add(f"- {values}")
        add()

    add(
        f"- KV cache memory, batch size 1: "
        f"{overall['cache_memory_batch_1_mib']:.2f} MiB"
    )
    add()

    add("## 8. Limitations")
    add()
    for limitation in metadata["limitations"]:
        add(f"- {limitation}")
    add()

    add("## 9. Interpretation")
    add()
    add(
        "The benchmark measures autoregressive decoding with "
        "standard full-prefix recomputation versus an implementation "
        "using a key-value cache."
    )
    add()
    add(
        "The measured speedup is specific to the tested checkpoint, "
        "CPU environment, validation prompts, and short generation "
        "length. The increase in speedup with prompt length is "
        "consistent with avoiding repeated computation over the "
        "already processed prefix."
    )
    add()
    add(
        "Exact greedy output agreement was verified for every "
        "timed comparison. This supports numerical and behavioral "
        "correctness for the tested workloads, but does not establish "
        "equivalence for all possible inputs or configurations."
    )
    add()
    add(
        "KV caching is an established inference optimization. "
        "These results validate this implementation and measure "
        "its performance; they do not establish KV caching itself "
        "as a novel RCT contribution."
    )
    add()

    add("## 10. Reproducibility files")
    add()
    add("- `experiment_results.json`: configuration and aggregate results")
    add("- `raw_timing_records.csv`: paired raw timing measurements")
    add("- `per_workload_summary.csv`: per-workload median measurements")
    add("- `REPORT.md`: this report")
    add()
    add("---")
    add()
    add(
        "Report generated from archived experiment data on "
        f"{datetime.now(timezone.utc).isoformat()}."
    )

    return "\n".join(lines) + "\n"


def main():
    print("Loading archived benchmark data...")

    metadata, raw_rows, summary_rows = load_data()

    print("Validating raw measurements...")
    report = build_report(metadata, raw_rows, summary_rows)

    REPORT_PATH.write_text(report, encoding="utf-8")

    print()
    print("RESEARCH REPORT GENERATED")
    print(f"Report: {REPORT_PATH}")
    print(f"Raw timing records: {len(raw_rows)}")
    print(f"Per-workload summaries: {len(summary_rows)}")
    print(f"Report size: {REPORT_PATH.stat().st_size:,} bytes")
    print()
    print("Report generation complete.")


if __name__ == "__main__":
    main()
