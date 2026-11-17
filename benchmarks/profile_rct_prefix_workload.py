
import csv
import gc
import random
from datetime import datetime, timezone
from pathlib import Path
import sys

import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from inference import load_rct_inference


# ---------------------------------------------------------
# Configuration: strongest validated workload
# ---------------------------------------------------------

CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

BATCH_SIZE = 8
PROMPT_LENGTH = 32
PREFIX_LENGTH = 24
NEW_TOKENS = 8

WARMUP_RUNS = 3
PROFILE_RUNS = 3

SEED = 20261117

MODES = ["kv", "prefix", "auto"]

SHAPE_REPORT_ROWS = 60


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def synchronize():
    torch.cuda.synchronize()


def create_prompt(inference, device):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    shared_prefix = torch.randint(
        0,
        inference.model.config.vocab_size,
        (1, PREFIX_LENGTH),
        device=device,
    )

    suffix = torch.randint(
        0,
        inference.model.config.vocab_size,
        (
            BATCH_SIZE,
            PROMPT_LENGTH - PREFIX_LENGTH,
        ),
        device=device,
    )

    return torch.cat(
        [
            shared_prefix.expand(BATCH_SIZE, -1),
            suffix,
        ],
        dim=1,
    )


def generate(inference, prompt, mode):
    if mode == "prefix":
        return inference.generate(
            prompt,
            max_new_tokens=NEW_TOKENS,
            mode="prefix",
            prefix_len=PREFIX_LENGTH,
        )

    return inference.generate(
        prompt,
        max_new_tokens=NEW_TOKENS,
        mode=mode,
    )


# ---------------------------------------------------------
# Profile one execution path
# ---------------------------------------------------------

def profile_mode(
    inference,
    prompt,
    mode,
    output_dir,
    device,
    expected_output,
):
    print("\n" + "=" * 70)
    print(f"PROFILING MODE: {mode.upper()}")
    print("=" * 70)

    gc.collect()
    synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    allocated_before = torch.cuda.memory_allocated(device)

    trace_path = output_dir / f"rct_{mode}_trace.json"
    report_path = output_dir / f"rct_{mode}_operators.txt"
    shape_report_path = (
        output_dir / f"rct_{mode}_operator_shapes.txt"
    )

    activities = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:

        for iteration in range(PROFILE_RUNS):

            with record_function(f"RCT::{mode}::generate"):

                output = generate(
                    inference,
                    prompt,
                    mode,
                )

                synchronize()

            prof.step()

    # -----------------------------------------------------
    # Correctness verification
    # -----------------------------------------------------

    if not torch.equal(output, expected_output):
        raise AssertionError(
            f"{mode} output differs from KV reference."
        )

    print(f"{mode.upper()} correctness: PASS")

    # -----------------------------------------------------
    # Export Chrome trace
    # -----------------------------------------------------

    prof.export_chrome_trace(str(trace_path))

    # -----------------------------------------------------
    # GPU memory statistics
    # -----------------------------------------------------

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    # -----------------------------------------------------
    # Standard operator statistics
    # -----------------------------------------------------

    averages = prof.key_averages()

    cuda_table = averages.table(
        sort_by="self_cuda_time_total",
        row_limit=40,
    )

    cpu_table = averages.table(
        sort_by="self_cpu_time_total",
        row_limit=40,
    )

    # -----------------------------------------------------
    # NEW: Group operators by input tensor shape
    # -----------------------------------------------------

    shape_averages = prof.key_averages(
        group_by_input_shape=True
    )

    shape_table = shape_averages.table(
        sort_by="self_cuda_time_total",
        row_limit=SHAPE_REPORT_ROWS,
    )

    # -----------------------------------------------------
    # Save standard operator report
    # -----------------------------------------------------

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            f"RCT PROFILING REPORT: {mode.upper()}\n"
        )

        report.write("=" * 70 + "\n\n")

        report.write("CUDA operators\n")
        report.write("=" * 70 + "\n")
        report.write(cuda_table)

        report.write("\n\nCPU operators\n")
        report.write("=" * 70 + "\n")
        report.write(cpu_table)

    # -----------------------------------------------------
    # NEW: Save shape-grouped operator report
    # -----------------------------------------------------

    with shape_report_path.open(
        "w",
        encoding="utf-8",
    ) as report:

        report.write(
            f"RCT OPERATOR SHAPE REPORT: {mode.upper()}\n"
        )

        report.write("=" * 70 + "\n\n")

        report.write(
            "Operators grouped by input tensor shape\n"
        )

        report.write("=" * 70 + "\n\n")

        report.write(shape_table)

    # -----------------------------------------------------
    # Results
    # -----------------------------------------------------

    result = {
        "mode": mode,
        "allocated_before_mb": (
            allocated_before / (1024 ** 2)
        ),
        "peak_allocated_mb": (
            peak_allocated / (1024 ** 2)
        ),
        "peak_reserved_mb": (
            peak_reserved / (1024 ** 2)
        ),
        "trace_file": trace_path.name,
        "report_file": report_path.name,
        "shape_report_file": shape_report_path.name,
    }

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------

    print("\nGPU memory:")

    print(
        f"Allocated before: "
        f"{result['allocated_before_mb']:.2f} MB"
    )

    print(
        f"Peak allocated:   "
        f"{result['peak_allocated_mb']:.2f} MB"
    )

    print(
        f"Peak reserved:    "
        f"{result['peak_reserved_mb']:.2f} MB"
    )

    print("\nTop CUDA operators:")
    print(cuda_table)

    print("\nTop CUDA operators grouped by input shape:")
    print(shape_table)

    print("\nSaved trace:", trace_path)
    print("Saved operator report:", report_path)
    print("Saved shape report:", shape_report_path)

    return result


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda")

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct_gpu_profiling"
        / f"profile_{timestamp}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("RCT100M GPU PROFILING EXPERIMENT")
    print("=" * 70)

    print("GPU:", torch.cuda.get_device_name(device))
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Batch size:", BATCH_SIZE)
    print("Prompt length:", PROMPT_LENGTH)
    print("Shared prefix:", PREFIX_LENGTH)
    print("New tokens:", NEW_TOKENS)
    print("Warmup runs:", WARMUP_RUNS)
    print("Profile runs:", PROFILE_RUNS)

    print("\nLoading RCT100M...")

    inference = load_rct_inference(
        CHECKPOINT,
        device=device,
    )

    prompt = create_prompt(
        inference,
        device,
    )

    # -----------------------------------------------------
    # Verify shared prefix detection
    # -----------------------------------------------------

    detected = inference.detect_shared_prefix(prompt)

    if detected != PREFIX_LENGTH:
        raise AssertionError(
            f"Expected prefix {PREFIX_LENGTH}, "
            f"detected {detected}"
        )

    print("Shared prefix detection: PASS")

    # -----------------------------------------------------
    # Generate KV correctness reference
    # -----------------------------------------------------

    print("\nGenerating KV reference...")

    prompt_output_reference = generate(
        inference,
        prompt,
        "kv",
    )

    # -----------------------------------------------------
    # Confirm all execution paths match
    # -----------------------------------------------------

    print("\nChecking output correctness...")

    for mode in MODES:

        output = generate(
            inference,
            prompt,
            mode,
        )

        if not torch.equal(
            output,
            prompt_output_reference,
        ):
            raise AssertionError(
                f"Output mismatch in {mode} mode."
            )

        print(f"{mode.upper()}: PASS")

    # -----------------------------------------------------
    # Warm up every execution path
    # -----------------------------------------------------

    print("\nWarming up execution paths...")

    for iteration in range(WARMUP_RUNS):

        for mode in MODES:

            generate(
                inference,
                prompt,
                mode,
            )

    synchronize()

    print("Warmup complete.")

    # -----------------------------------------------------
    # Profile each execution path separately
    # -----------------------------------------------------

    results = []

    for mode in MODES:

        result = profile_mode(
            inference=inference,
            prompt=prompt,
            mode=mode,
            output_dir=output_dir,
            device=device,
            expected_output=prompt_output_reference,
        )

        results.append(result)

    # -----------------------------------------------------
    # Save memory and file summary
    # -----------------------------------------------------

    summary_path = output_dir / "memory_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)

    print("\nResults directory:")
    print(output_dir)

    print("\nMemory summary:")
    print(summary_path)

    print("\nGenerated shape reports:")

    for mode in MODES:
        print(
            output_dir
            / f"rct_{mode}_operator_shapes.txt"
        )

    print("\nCorrectness: all execution paths matched.")
    print("Original benchmark files were not modified.")


if __name__ == "__main__":
    main()
