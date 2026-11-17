
import gc
import csv
import json
import time
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import psutil
import torch

from model.rct_100m import RCT100M, RCT100MConfig


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

VALIDATION_PATH = (
    PROJECT_DIR
    / "data"
    / "rct30"
    / "validation_expanded_tokens.bin"
)

THREAD_COUNTS = [1, 2, 4, 8]

NUM_PROMPTS = 20
PROMPT_LENGTH = 24
FORWARD_LENGTH = 32
GENERATION_TOKENS = 8

WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

SEED = 20261103
BOOTSTRAP_ITERATIONS = 10000

VOCAB_SIZE = 8192
HIDDEN_SIZE = 512
NUM_LAYERS = 30
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048
MAX_SEQ_LEN = 512

OUTPUT_ROOT = PROJECT_DIR / "results" / "rct100m_cpu_threads"


# ============================================================
# UTILITIES
# ============================================================

def report_memory():

    memory = psutil.virtual_memory()
    process = psutil.Process()

    print(
        f"Available RAM: "
        f"{memory.available / (1024 ** 3):.2f} GB"
    )

    print(
        f"Process RAM: "
        f"{process.memory_info().rss / (1024 ** 3):.2f} GB"
    )


def check_memory():

    available = psutil.virtual_memory().available / (1024 ** 3)

    if available < 1.5:
        raise MemoryError(
            f"Only {available:.2f} GB RAM available. "
            "Stopping before benchmark."
        )


# ============================================================
# MODEL
# ============================================================

def create_model():

    config = RCT100MConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
        tie_embeddings=True,
    )

    return RCT100M(config)


def extract_state_dict(checkpoint):

    for key in [
        "model_state_dict",
        "model_state",
        "state_dict",
        "model",
    ]:

        if key in checkpoint:
            return checkpoint[key]

    raise KeyError(
        f"Model weights not found. "
        f"Checkpoint keys: {list(checkpoint.keys())}"
    )


def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    print("\nLoading checkpoint:")
    print(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = create_model()

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    step = checkpoint.get("step", "Unknown")

    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    print("Checkpoint step:", step)
    print(f"Parameters: {parameter_count:,}")

    if parameter_count != 99028992:
        raise ValueError(
            f"Unexpected parameter count: {parameter_count}"
        )

    del checkpoint
    del state_dict
    gc.collect()

    return model, step


# ============================================================
# FIXED INPUTS
# ============================================================

def load_prompts():

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(VALIDATION_PATH)

    data = np.memmap(
        VALIDATION_PATH,
        dtype=np.uint16,
        mode="r",
    )

    required_length = FORWARD_LENGTH + 1

    if len(data) <= required_length:
        raise ValueError("Validation dataset is too small.")

    rng = np.random.default_rng(SEED)

    starts = rng.choice(
        len(data) - required_length,
        size=NUM_PROMPTS,
        replace=False,
    )

    prompts = []

    for start in starts:

        chunk = np.array(
            data[start:start + required_length],
            dtype=np.int64,
            copy=True,
        )

        forward_input = torch.from_numpy(
            chunk[:FORWARD_LENGTH].copy()
        ).unsqueeze(0)

        generation_input = torch.from_numpy(
            chunk[:PROMPT_LENGTH].copy()
        ).unsqueeze(0)

        prompts.append({
            "start": int(start),
            "forward": forward_input,
            "generation": generation_input,
        })

    del data

    return prompts


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================

@torch.inference_mode()
def forward_once(model, tokens):

    return model(tokens)


@torch.inference_mode()
def greedy_generate(model, prompt):

    generated = prompt.clone()

    for _ in range(GENERATION_TOKENS):

        context = generated[:, -MAX_SEQ_LEN:]

        logits = model(context)

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

    return generated


# ============================================================
# CORRECTNESS CHECK
# ============================================================

def check_thread_correctness(model, prompts):

    print("\n" + "=" * 65)
    print("CROSS-THREAD OUTPUT CORRECTNESS")
    print("=" * 65)

    reference = {}

    # Reference results use one CPU thread.
    torch.set_num_threads(1)

    with torch.inference_mode():

        for index, item in enumerate(prompts):

            logits = forward_once(
                model,
                item["forward"],
            )

            generated = greedy_generate(
                model,
                item["generation"],
            )

            reference[index] = {
                "logits": logits.clone(),
                "generated": generated.clone(),
            }

    correctness = {}

    for thread_count in THREAD_COUNTS:

        torch.set_num_threads(thread_count)

        maximum_difference = 0.0
        matching_tokens = 0
        total_tokens = 0

        for index, item in enumerate(prompts):

            with torch.inference_mode():

                logits = forward_once(
                    model,
                    item["forward"],
                )

                generated = greedy_generate(
                    model,
                    item["generation"],
                )

            difference = (
                logits - reference[index]["logits"]
            ).abs().max().item()

            maximum_difference = max(
                maximum_difference,
                difference,
            )

            reference_tokens = reference[index]["generated"]

            matching_tokens += (
                generated == reference_tokens
            ).sum().item()

            total_tokens += generated.numel()

        agreement = matching_tokens / total_tokens * 100

        passed = (
            maximum_difference <= 1e-4
            and matching_tokens == total_tokens
        )

        correctness[thread_count] = {
            "max_logit_difference": maximum_difference,
            "token_agreement_percent": agreement,
            "passed": passed,
        }

        print(
            f"Threads {thread_count}: "
            f"max logit difference={maximum_difference:.8f}, "
            f"token agreement={agreement:.2f}%, "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if not passed:
            raise RuntimeError(
                f"Correctness check failed at {thread_count} threads."
            )

    del reference
    gc.collect()

    return correctness


# ============================================================
# TIMING
# ============================================================

def benchmark(model, prompts, output_dir):

    raw_path = output_dir / "raw_timing_records.csv"

    records = []

    print("\n" + "=" * 65)
    print("CPU THREAD PERFORMANCE BENCHMARK")
    print("=" * 65)

    print("Prompts:", NUM_PROMPTS)
    print("Warmups:", WARMUP_RUNS)
    print("Timed repetitions:", BENCHMARK_RUNS)

    rng = random.Random(SEED)

    # Warm-up each thread configuration.
    print("\nWARM-UP")

    for thread_count in THREAD_COUNTS:

        torch.set_num_threads(thread_count)

        for item in prompts[:WARMUP_RUNS]:

            forward_once(
                model,
                item["forward"],
            )

            greedy_generate(
                model,
                item["generation"],
            )

        print(f"Threads {thread_count}: warm-up complete")

    print("\nTIMED RUNS")

    for repetition in range(1, BENCHMARK_RUNS + 1):

        method_order = THREAD_COUNTS.copy()
        rng.shuffle(method_order)

        for thread_count in method_order:

            torch.set_num_threads(thread_count)

            prompt_order = list(range(NUM_PROMPTS))
            rng.shuffle(prompt_order)

            for prompt_index in prompt_order:

                item = prompts[prompt_index]

                # Forward pass timing.
                start = time.perf_counter()

                forward_once(
                    model,
                    item["forward"],
                )

                forward_seconds = (
                    time.perf_counter() - start
                )

                # Greedy generation timing.
                start = time.perf_counter()

                generated = greedy_generate(
                    model,
                    item["generation"],
                )

                generation_seconds = (
                    time.perf_counter() - start
                )

                records.append({
                    "thread_count": thread_count,
                    "prompt_index": prompt_index,
                    "validation_start": item["start"],
                    "repetition": repetition,
                    "forward_seconds": forward_seconds,
                    "generation_seconds": generation_seconds,
                    "generated_tokens": GENERATION_TOKENS,
                })

        print(
            f"Completed repetition "
            f"{repetition}/{BENCHMARK_RUNS}"
        )

    fieldnames = [
        "thread_count",
        "prompt_index",
        "validation_start",
        "repetition",
        "forward_seconds",
        "generation_seconds",
        "generated_tokens",
    ]

    with open(
        raw_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nRaw timing records:", len(records))
    print("Saved:", raw_path)

    return records


# ============================================================
# STATISTICS
# ============================================================

def bootstrap_speedup(
    baseline_values,
    comparison_values,
    seed,
):

    rng = np.random.default_rng(seed)

    baseline_values = np.asarray(baseline_values)
    comparison_values = np.asarray(comparison_values)

    ratios = baseline_values / comparison_values

    observed = float(np.median(ratios))

    samples = rng.integers(
        0,
        len(ratios),
        size=(BOOTSTRAP_ITERATIONS, len(ratios)),
    )

    bootstrap_values = np.median(
        ratios[samples],
        axis=1,
    )

    lower = float(np.percentile(bootstrap_values, 2.5))
    upper = float(np.percentile(bootstrap_values, 97.5))

    return observed, lower, upper


def analyze_records(records):

    summary = {}

    for metric in [
        "forward_seconds",
        "generation_seconds",
    ]:

        print("\n" + "=" * 65)
        print("RESULTS:", metric)
        print("=" * 65)

        by_thread = {}

        for thread_count in THREAD_COUNTS:

            per_prompt = []

            for prompt_index in range(NUM_PROMPTS):

                values = [
                    row[metric]
                    for row in records
                    if row["thread_count"] == thread_count
                    and row["prompt_index"] == prompt_index
                ]

                if len(values) != BENCHMARK_RUNS:
                    raise ValueError(
                        "Unexpected number of timing repetitions."
                    )

                per_prompt.append(
                    float(np.median(values))
                )

            by_thread[thread_count] = per_prompt

        baseline = by_thread[1]

        metric_results = {}

        print(
            f"{'Threads':<12}"
            f"{'Median latency':>18}"
            f"{'Speedup':>14}"
            f"{'95% CI':>24}"
        )

        for thread_count in THREAD_COUNTS:

            values = by_thread[thread_count]

            median_latency = float(np.median(values))

            if thread_count == 1:

                speedup = 1.0
                lower = 1.0
                upper = 1.0

            else:

                speedup, lower, upper = bootstrap_speedup(
                    baseline,
                    values,
                    SEED + thread_count,
                )

            metric_results[str(thread_count)] = {
                "median_latency_seconds": median_latency,
                "speedup_vs_one_thread": speedup,
                "bootstrap_95_ci": [lower, upper],
            }

            print(
                f"{thread_count:<12}"
                f"{median_latency:>18.6f}"
                f"{speedup:>13.3f}x"
                f"{'[' + format(lower, '.3f') + ', ' + format(upper, '.3f') + ']':>24}"
            )

        summary[metric] = metric_results

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("RCT 100M CPU THREAD BENCHMARK")
    print("=" * 65)

    print("PyTorch:", torch.__version__)
    print("Device: CPU")
    print("Thread configurations:", THREAD_COUNTS)

    check_memory()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    torch.set_num_threads(1)

    print("\n[INITIAL MEMORY]")
    report_memory()

    model, checkpoint_step = load_model()

    print("\n[MODEL MEMORY]")
    report_memory()

    prompts = load_prompts()

    print("\nFixed validation prompts:", len(prompts))

    check_memory()

    output_dir = OUTPUT_ROOT / (
        "run_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    correctness = check_thread_correctness(
        model,
        prompts,
    )

    records = benchmark(
        model,
        prompts,
        output_dir,
    )

    summary = analyze_records(records)

    results = {
        "experiment": "RCT100M CPU thread comparison",
        "checkpoint_step": checkpoint_step,
        "checkpoint": str(CHECKPOINT_PATH),
        "parameter_count": sum(
            p.numel() for p in model.parameters()
        ),
        "pytorch_version": torch.__version__,
        "thread_counts": THREAD_COUNTS,
        "num_prompts": NUM_PROMPTS,
        "prompt_length": PROMPT_LENGTH,
        "forward_length": FORWARD_LENGTH,
        "generation_tokens": GENERATION_TOKENS,
        "warmup_runs": WARMUP_RUNS,
        "benchmark_runs": BENCHMARK_RUNS,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "correctness": correctness,
        "summary": summary,
        "raw_records": len(records),
    }

    json_path = output_dir / "experiment_results.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print("\nSaved results:", json_path)

    print("\n" + "=" * 65)
    print("CPU THREAD BENCHMARK COMPLETED")
    print("=" * 65)

    print("\n[FINAL MEMORY]")
    report_memory()

    del model
    del prompts
    gc.collect()


if __name__ == "__main__":
    main()
