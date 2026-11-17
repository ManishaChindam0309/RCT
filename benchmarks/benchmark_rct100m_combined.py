import sys
import gc
import time
import random
import statistics
from pathlib import Path

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined


CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

VALIDATION_DATA = (
    PROJECT_DIR
    / "data"
    / "rct30"
    / "validation_expanded_tokens.bin"
)

BATCH_SIZE = 8
PROMPT_LENGTH = 12
NEW_TOKENS = 8

WARMUP_RUNS = 2
BENCHMARK_RUNS = 15

SEED = 2026


def deterministic_sampling():

    """
    Temporarily replace multinomial sampling with argmax.

    This allows exact output comparisons between generation
    implementations without changing model weights.
    """

    original_multinomial = torch.multinomial

    def argmax_sampling(probabilities, num_samples, *args, **kwargs):

        if num_samples != 1:
            raise ValueError(
                "Benchmark expects num_samples=1."
            )

        return probabilities.argmax(
            dim=-1,
            keepdim=True
        )

    torch.multinomial = argmax_sampling

    return original_multinomial


def restore_sampling(original_multinomial):
    torch.multinomial = original_multinomial


def median_runtime(function):

    for _ in range(WARMUP_RUNS):
        function()

    measurements = []

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        function()

        elapsed = time.perf_counter() - start

        measurements.append(elapsed)

    return statistics.median(measurements), measurements


def load_prompts():

    data = np.memmap(
        VALIDATION_DATA,
        dtype=np.uint16,
        mode="r"
    )

    max_start = len(data) - PROMPT_LENGTH

    rng = np.random.default_rng(SEED)

    starts = rng.choice(
        max_start,
        size=100,
        replace=False
    )

    prompts = np.stack([
        np.array(
            data[start:start + PROMPT_LENGTH],
            dtype=np.int64
        )
        for start in starts
    ])

    del data

    return torch.from_numpy(prompts).long()


def make_workloads(prompts):

    unique = prompts[:BATCH_SIZE]

    # Eight unique prompts.
    all_unique = unique.clone()

    # Four unique prompts, each repeated twice.
    duplicate_heavy = torch.cat(
        [unique[:4], unique[:4]],
        dim=0
    )

    # Two unique prompts, each repeated four times.
    quarter_unique = unique[:2].repeat(4, 1)

    # One unique prompt repeated eight times.
    all_same = unique[:1].repeat(BATCH_SIZE, 1)

    return {
        "all_unique": all_unique,
        "duplicate_heavy": duplicate_heavy,
        "quarter_unique": quarter_unique,
        "all_same": all_same
    }


def main():

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 90)
    print("RCT 100M END-TO-END GENERATION BENCHMARK")
    print("=" * 90)

    print("PyTorch:", torch.__version__)
    print("CPU threads:", torch.get_num_threads())
    print("Batch size:", BATCH_SIZE)
    print("Prompt length:", PROMPT_LENGTH)
    print("Generated tokens:", NEW_TOKENS)
    print("Warmups:", WARMUP_RUNS)
    print("Measured runs:", BENCHMARK_RUNS)

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    base_model = RCT100M(RCT100MConfig())

    base_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    base_model.eval()

    print("Checkpoint step:", checkpoint["step"])
    print("Parameters:", base_model.parameter_count())

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    combined_model = RCT100MCombined(base_model)
    combined_model.eval()

    prompts = load_prompts()
    workloads = make_workloads(prompts)

    original_multinomial = deterministic_sampling()

    results = []

    try:

        with torch.inference_mode():

            for workload_name, batch in workloads.items():

                print("\n" + "=" * 90)
                print("WORKLOAD:", workload_name)
                print("=" * 90)

                unique_count = torch.unique(
                    batch,
                    dim=0
                ).shape[0]

                print("Batch size:", batch.shape[0])
                print("Unique prompts:", unique_count)
                print(
                    "Duplicate prompts:",
                    batch.shape[0] - unique_count
                )

                functions = {
                    "standard": lambda: base_model.generate(
                        batch,
                        max_new_tokens=NEW_TOKENS,
                        temperature=1.0
                    ),

                    "kv_cache": lambda: kv_model.generate_cached(
                        batch,
                        max_new_tokens=NEW_TOKENS,
                        temperature=1.0
                    ),

                    "combined_no_dedup": lambda: combined_model.generate(
                        batch,
                        max_new_tokens=NEW_TOKENS,
                        temperature=1.0,
                        deduplicate=False
                    ),

                    "combined_dedup": lambda: combined_model.generate(
                        batch,
                        max_new_tokens=NEW_TOKENS,
                        temperature=1.0,
                        deduplicate=True
                    )
                }

                outputs = {}

                for name, function in functions.items():

                    output = function()

                    outputs[name] = output

                    print(
                        f"{name:20s} output shape: "
                        f"{tuple(output.shape)}"
                    )

                reference = outputs["standard"]

                print("\nExact output comparisons:")

                for name, output in outputs.items():

                    matches = torch.equal(
                        reference,
                        output
                    )

                    print(
                        f"{name:20s} "
                        f"{'PASS' if matches else 'FAIL'}"
                    )

                # Verify duplicate prompts receive identical
                # restored outputs in the deduplicated result.
                if workload_name == "duplicate_heavy":

                    dedup_output = outputs["combined_dedup"]

                    duplicate_match = torch.equal(
                        dedup_output[0],
                        dedup_output[4]
                    )

                    print(
                        "Duplicate restoration: ",
                        "PASS" if duplicate_match else "FAIL"
                    )

                print("\nTiming generation methods...")

                workload_times = {}

                for name, function in functions.items():

                    median_time, raw_times = median_runtime(
                        function
                    )

                    workload_times[name] = median_time

                    print(
                        f"{name:20s} "
                        f"median={median_time:.6f}s "
                        f"runs={raw_times}"
                    )

                standard_time = workload_times["standard"]
                kv_time = workload_times["kv_cache"]
                combined_no_dedup_time = workload_times[
                    "combined_no_dedup"
                ]
                combined_dedup_time = workload_times[
                    "combined_dedup"
                ]

                print("\nSpeedup relative to standard:")

                for name, elapsed in workload_times.items():

                    print(
                        f"{name:20s} "
                        f"{standard_time / elapsed:.3f}x"
                    )

                print("\nDeduplication effect:")

                print(
                    "Combined without dedup:",
                    f"{combined_no_dedup_time:.6f}s"
                )

                print(
                    "Combined with dedup:",
                    f"{combined_dedup_time:.6f}s"
                )

                print(
                    "Deduplication speedup:",
                    f"{combined_no_dedup_time / combined_dedup_time:.3f}x"
                )

                results.append({
                    "workload": workload_name,
                    "standard": standard_time,
                    "kv_cache": kv_time,
                    "combined_no_dedup": combined_no_dedup_time,
                    "combined_dedup": combined_dedup_time
                })

                gc.collect()

    finally:

        restore_sampling(original_multinomial)

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    print(
        f"{'Workload':20s}"
        f"{'Standard':>13s}"
        f"{'KV Cache':>13s}"
        f"{'Combined':>13s}"
        f"{'KV Speedup':>13s}"
        f"{'RCT Speedup':>13s}"
    )

    for result in results:

        standard = result["standard"]
        kv = result["kv_cache"]
        combined = result["combined_dedup"]

        print(
            f"{result['workload']:20s}"
            f"{standard:13.4f}"
            f"{kv:13.4f}"
            f"{combined:13.4f}"
            f"{standard / kv:12.3f}x"
            f"{standard / combined:12.3f}x"
        )

    print("\nBenchmark limitations:")
    print("- CPU-only measurements.")
    print("- Greedy argmax decoding, not stochastic sampling.")
    print("- Eight prompts and eight generated tokens.")
    print("- One machine and one trained checkpoint.")
    print("- Timing includes prompt deduplication and output restoration.")
    print("- No statistical confidence interval in this first benchmark.")

    print("\nBENCHMARK COMPLETE")

    del combined_model
    del kv_model
    del base_model
    del checkpoint

    gc.collect()


if __name__ == "__main__":
    main()
