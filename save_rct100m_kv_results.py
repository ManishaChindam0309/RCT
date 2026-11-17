
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rct100m_kv_large"
OUT.mkdir(parents=True, exist_ok=True)

# Experiment configuration
config = {
    "experiment": "RCT 100M KV-cache large benchmark",
    "checkpoint": "checkpoints/rct100m_continued/best_model.pt",
    "checkpoint_step": 1500,
    "parameters": 99028992,
    "device": "CPU",
    "cpu_threads": 4,
    "prompt_lengths": [8, 16, 24],
    "generated_tokens": 8,
    "distinct_prompt_locations": 30,
    "workload_combinations": 90,
    "warmup_runs_per_method_per_workload": 2,
    "timed_runs_per_method_per_workload": 10,
    "timed_measurements_per_method": 900,
    "bootstrap_iterations": 20000,
    "bootstrap_unit": "prompt-location cluster",
    "validation_corpus": "data/rct30/validation_expanded_tokens.bin",
    "cache_memory_batch_1_mib": 60.00,
    "exact_output_agreement": "90/90",
    "created_utc": datetime.now(timezone.utc).isoformat(),
}

# Reported aggregate measurements
summary = [
    {
        "prompt_length": 8,
        "standard_median_seconds": 0.2998,
        "kv_median_seconds": 0.2603,
        "aggregate_median_speedup": 1.152,
        "latency_reduction_percent": 13.17,
        "median_per_prompt_speedup": 1.150,
    },
    {
        "prompt_length": 16,
        "standard_median_seconds": 0.2677,
        "kv_median_seconds": 0.2266,
        "aggregate_median_speedup": 1.182,
        "latency_reduction_percent": 15.37,
        "median_per_prompt_speedup": 1.184,
    },
    {
        "prompt_length": 24,
        "standard_median_seconds": 0.2987,
        "kv_median_seconds": 0.2293,
        "aggregate_median_speedup": 1.303,
        "latency_reduction_percent": 23.24,
        "median_per_prompt_speedup": 1.318,
    },
]

overall = {
    "standard_overall_median_seconds": 0.2835,
    "kv_overall_median_seconds": 0.2306,
    "aggregate_median_speedup": 1.229,
    "paired_workload_median_speedup": 1.191,
    "cluster_bootstrap_95_ci_lower": 1.183,
    "cluster_bootstrap_95_ci_upper": 1.208,
    "exact_output_agreement": "90/90",
}

# Per-prompt paired speedups.
# Each prompt location was evaluated at all three prompt lengths.
prompt_results = [
    (1, 7232, 1.187, 1.166, 1.373),
    (2, 73056, 1.112, 1.165, 1.297),
    (3, 182720, 1.128, 1.186, 1.228),
    (4, 344512, 1.172, 1.200, 1.353),
    (5, 373696, 1.269, 1.232, 1.226),
    (6, 412224, 1.096, 1.171, 1.367),
    (7, 481216, 1.186, 1.224, 1.345),
    (8, 482368, 1.189, 1.196, 1.318),
    (9, 493152, 1.069, 1.125, 1.254),
    (10, 542144, 0.984, 1.210, 1.317),
    (11, 788672, 1.117, 1.111, 1.380),
    (12, 889760, 1.153, 1.148, 1.327),
    (13, 952416, 1.184, 1.186, 1.285),
    (14, 959264, 1.147, 1.158, 1.345),
    (15, 982304, 1.160, 1.182, 1.450),
    (16, 1080960, 1.044, 1.112, 1.545),
    (17, 1120224, 1.276, 1.198, 1.403),
    (18, 1176736, 1.178, 1.194, 1.341),
    (19, 1200128, 1.169, 1.151, 1.376),
    (20, 1485408, 1.148, 1.121, 1.340),
    (21, 1517472, 1.161, 1.178, 1.316),
    (22, 1612000, 1.182, 1.098, 1.339),
    (23, 1642784, 1.274, 1.281, 1.189),
    (24, 1657312, 1.127, 1.201, 1.270),
    (25, 1830624, 1.144, 1.169, 1.204),
    (26, 1852384, 1.132, 1.228, 1.301),
    (27, 1852704, 1.260, 1.220, 1.218),
    (28, 1856128, 1.141, 1.186, 1.238),
    (29, 1978176, 1.124, 1.193, 1.292),
    (30, 2111744, 1.132, 1.144, 1.244),
]

# 1. Save per-prompt CSV
csv_path = OUT / "rct100m_kv_per_prompt_speedups.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "prompt_number",
        "validation_start_token",
        "speedup_length_8",
        "speedup_length_16",
        "speedup_length_24",
    ])
    writer.writerows(prompt_results)

# 2. Save structured JSON
json_data = {
    "configuration": config,
    "results_by_prompt_length": summary,
    "overall_results": overall,
    "per_prompt_speedups": [
        {
            "prompt_number": row[0],
            "validation_start_token": row[1],
            "speedup_length_8": row[2],
            "speedup_length_16": row[3],
            "speedup_length_24": row[4],
        }
        for row in prompt_results
    ],
    "limitations": [
        "One CPU machine and one model checkpoint were tested.",
        "The benchmark tested prompt lengths 8, 16, and 24.",
        "Eight greedy-generated tokens were evaluated.",
        "The model was trained using sequence length 32.",
        "The reported archive does not contain individual run-level timings.",
        "KV caching is an established technique; this benchmark does not establish novelty."
    ],
}

json_path = OUT / "rct100m_kv_large_results.json"
json_path.write_text(
    json.dumps(json_data, indent=2),
    encoding="utf-8"
)

# 3. Save readable Markdown report
report_path = OUT / "REPORT.md"

lines = [
    "# RCT 100M KV-Cache Large Benchmark",
    "",
    f"Archive created (UTC): {config['created_utc']}",
    "",
    "## Experiment configuration",
    "",
    f"- Checkpoint: `{config['checkpoint']}`",
    f"- Checkpoint step: {config['checkpoint_step']}",
    f"- Parameters: {config['parameters']:,}",
    f"- Hardware: {config['device']}, {config['cpu_threads']} CPU threads",
    f"- Generated tokens: {config['generated_tokens']}",
    f"- Prompt locations: {config['distinct_prompt_locations']}",
    f"- Prompt lengths: {config['prompt_lengths']}",
    f"- Workload combinations: {config['workload_combinations']}",
    f"- Warm-up runs per method/workload: {config['warmup_runs_per_method_per_workload']}",
    f"- Timed runs per method/workload: {config['timed_runs_per_method_per_workload']}",
    f"- Timed measurements per method: {config['timed_measurements_per_method']}",
    f"- Bootstrap iterations: {config['bootstrap_iterations']}",
    f"- Bootstrap unit: {config['bootstrap_unit']}",
    "",
    "## Results by prompt length",
    "",
    "| Prompt length | Standard median (s) | KV median (s) | Speedup | Latency reduction |",
    "|---:|---:|---:|---:|---:|",
]

for row in summary:
    lines.append(
        f"| {row['prompt_length']} | "
        f"{row['standard_median_seconds']:.4f} | "
        f"{row['kv_median_seconds']:.4f} | "
        f"{row['aggregate_median_speedup']:.3f}x | "
        f"{row['latency_reduction_percent']:.2f}% |"
    )

lines += [
    "",
    "## Overall results",
    "",
    f"- Standard overall median: {overall['standard_overall_median_seconds']:.4f} s",
    f"- KV overall median: {overall['kv_overall_median_seconds']:.4f} s",
    f"- Aggregate median-based speedup: {overall['aggregate_median_speedup']:.3f}x",
    f"- Paired workload median speedup: {overall['paired_workload_median_speedup']:.3f}x",
    f"- Cluster-bootstrap 95% CI: "
    f"[{overall['cluster_bootstrap_95_ci_lower']:.3f}x, "
    f"{overall['cluster_bootstrap_95_ci_upper']:.3f}x]",
    f"- Exact greedy output agreement: {overall['exact_output_agreement']}",
    f"- KV cache memory, batch 1: {config['cache_memory_batch_1_mib']:.2f} MiB",
    "",
    "## Interpretation",
    "",
    "The KV-cached implementation was faster in the aggregate for all three "
    "tested prompt lengths. The measured speedup was larger at prompt length "
    "24 than at lengths 8 and 16.",
    "",
    "All 90 workload combinations passed exact greedy-output comparison.",
    "The 30 prompt locations were evaluated at each of the three lengths. "
    "They are therefore 30 prompt-location clusters, not 90 independent "
    "prompt locations.",
    "",
    "## Limitations",
    "",
    "- Results apply to this CPU machine, checkpoint, corpus, and workload.",
    "- Only short prompts and eight generated tokens were evaluated.",
    "- The model was trained with sequence length 32; this is not a long-context test.",
    "- KV caching is an established technique. These results validate this "
    "implementation and its measured performance, not a novel caching algorithm.",
    "- Individual timing repetitions were not preserved in this archive.",
    "",
    "## Reproducibility files",
    "",
    "- `rct100m_kv_per_prompt_speedups.csv`",
    "- `rct100m_kv_large_results.json`",
    "- `REPORT.md`",
    "",
]

report_path.write_text("\n".join(lines), encoding="utf-8")

print("\nRESULTS ARCHIVED")
print(f"Folder: {OUT}")
print(f"CSV:    {csv_path.name}")
print(f"JSON:   {json_path.name}")
print(f"Report: {report_path.name}")
print(f"Prompt records: {len(prompt_results)}")
print(f"Length summaries: {len(summary)}")
print("\nArchive creation complete.")
