
from pathlib import Path
from datetime import datetime
import sys
import csv
import json
import gc
import time
import statistics

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

BATCH_SIZE = 8
PROMPT_LENGTH = 24
NEW_TOKENS = 8

CLUSTERS = 4
REPEATS = 5
WARMUPS = 1

DUPLICATE_RATES = {
    "0%": 8,
    "50%": 4,
    "75%": 2,
    "87.5%": 1,
}


def synchronize():
    torch.cuda.synchronize()


@torch.inference_mode()
def greedy_standard(model, prompts):
    generated = prompts

    for _ in range(NEW_TOKENS):
        logits = model(generated)
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True
        )
        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

    return generated


@torch.inference_mode()
def greedy_kv(kv_model, prompts):
    cache = kv_model.create_cache(prompts.size(0))

    logits = kv_model.decode_step(prompts, cache)
    generated = prompts

    for step in range(NEW_TOKENS):
        next_token = logits[:, -1, :].argmax(
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )

        if step + 1 < NEW_TOKENS:
            logits = kv_model.decode_step(
                next_token,
                cache
            )

    return generated


@torch.inference_mode()
def greedy_rct_dedup(combined, prompts):
    unique_prompts, inverse_indices = (
        combined._stable_deduplicate(prompts)
    )

    unique_outputs = greedy_kv(
        combined.kv_model,
        unique_prompts
    )

    return unique_outputs.index_select(
        0,
        inverse_indices
    )


def make_workload(base_prompts, unique_count):
    indices = torch.arange(BATCH_SIZE, device="cuda")
    indices = indices % unique_count

    return base_prompts.index_select(0, indices)


def timed_run(function, model, prompts):
    synchronize()
    start = time.perf_counter()

    output = function(model, prompts)

    synchronize()
    elapsed = time.perf_counter() - start

    return output, elapsed


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    torch.manual_seed(2026)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Batch size:", BATCH_SIZE)
    print("Prompt length:", PROMPT_LENGTH)
    print("Generated tokens:", NEW_TOKENS)

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    config = RCT100MConfig(**checkpoint["model_config"])

    base_model = RCT100M(config)
    base_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    base_model = base_model.cuda()
    base_model.eval()

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    combined = RCT100MCombined(base_model)
    combined.eval()

    print("Parameters:", base_model.parameter_count())
    print("Checkpoint step:", checkpoint["step"])

    del checkpoint
    gc.collect()

    raw_results = []
    correctness_results = []

    for rate_index, (rate, unique_count) in enumerate(
        DUPLICATE_RATES.items()
    ):

        print("\n" + "=" * 75)
        print(
            f"Duplicate rate: {rate} | "
            f"Unique prompts: {unique_count}/{BATCH_SIZE}"
        )
        print("=" * 75)

        for cluster in range(CLUSTERS):

            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                50000 + rate_index * 1000 + cluster
            )

            base_prompts = torch.randint(
                low=0,
                high=config.vocab_size,
                size=(BATCH_SIZE, PROMPT_LENGTH),
                generator=generator,
                dtype=torch.long
            ).cuda()

            prompts = make_workload(
                base_prompts,
                unique_count
            )

            # Verify exact duplicate count.
            actual_unique = torch.unique(
                prompts,
                dim=0
            ).size(0)

            if actual_unique != unique_count:
                raise RuntimeError(
                    f"Unexpected unique prompt count: {actual_unique}"
                )

            methods = {
                "standard_no_cache": (
                    greedy_standard,
                    base_model
                ),
                "kv_cache": (
                    greedy_kv,
                    kv_model
                ),
                "rct_dedup_kv": (
                    greedy_rct_dedup,
                    combined
                ),
            }

            # Run correctness checks before timing.
            standard_output = greedy_standard(
                base_model,
                prompts
            )

            kv_output = greedy_kv(
                kv_model,
                prompts
            )

            rct_output = greedy_rct_dedup(
                combined,
                prompts
            )

            kv_agreement = (
                standard_output == kv_output
            ).float().mean().item()

            rct_agreement = (
                kv_output == rct_output
            ).float().mean().item()

            correctness = {
                "duplicate_rate": rate,
                "cluster": cluster,
                "unique_prompts": unique_count,
                "standard_vs_kv_token_agreement": kv_agreement,
                "kv_vs_rct_token_agreement": rct_agreement,
                "standard_vs_kv_exact": torch.equal(
                    standard_output,
                    kv_output
                ),
                "kv_vs_rct_exact": torch.equal(
                    kv_output,
                    rct_output
                ),
            }

            correctness_results.append(correctness)

            if not correctness["standard_vs_kv_exact"]:
                raise RuntimeError(
                    f"Standard/KV output mismatch: {rate}, cluster {cluster}"
                )

            if not correctness["kv_vs_rct_exact"]:
                raise RuntimeError(
                    f"KV/RCT output mismatch: {rate}, cluster {cluster}"
                )

            print(
                f"Cluster {cluster + 1}/{CLUSTERS}: "
                f"correctness passed"
            )

            # Warm-up every method.
            for function, model in methods.values():
                function(model, prompts)

            # Measure repeated, paired workloads.
            for repeat in range(REPEATS):

                for method_name, (function, model) in methods.items():

                    output, elapsed = timed_run(
                        function,
                        model,
                        prompts
                    )

                    raw_results.append({
                        "duplicate_rate": rate,
                        "cluster": cluster,
                        "repeat": repeat + 1,
                        "method": method_name,
                        "latency_seconds": elapsed,
                        "output_tokens_per_second": (
                            BATCH_SIZE * NEW_TOKENS / elapsed
                        ),
                        "unique_computation_tokens_per_second": (
                            unique_count * NEW_TOKENS / elapsed
                        ),
                        "unique_prompts": unique_count,
                    })

            del prompts, base_prompts
            del standard_output, kv_output, rct_output
            gc.collect()

    # Save experiment results.
    output_dir = (
        PROJECT_DIR
        / "results"
        / "rct100m_gpu_reuse"
        / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "raw_timings.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(raw_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(raw_results)

    # Aggregate medians per method and duplicate rate.
    summary = {}

    for rate in DUPLICATE_RATES:

        summary[rate] = {}

        for method in (
            "standard_no_cache",
            "kv_cache",
            "rct_dedup_kv"
        ):

            rows = [
                row for row in raw_results
                if row["duplicate_rate"] == rate
                and row["method"] == method
            ]

            summary[rate][method] = {
                "median_latency_seconds": statistics.median(
                    row["latency_seconds"] for row in rows
                ),
                "median_output_tokens_per_second": statistics.median(
                    row["output_tokens_per_second"] for row in rows
                ),
                "median_unique_computation_tokens_per_second": (
                    statistics.median(
                        row["unique_computation_tokens_per_second"]
                        for row in rows
                    )
                ),
            }

        kv_latency = summary[rate]["kv_cache"][
            "median_latency_seconds"
        ]

        rct_latency = summary[rate]["rct_dedup_kv"][
            "median_latency_seconds"
        ]

        summary[rate]["rct_speedup_vs_kv"] = (
            kv_latency / rct_latency
        )

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint_step": 1500,
        "precision": "FP32",
        "batch_size": BATCH_SIZE,
        "prompt_length": PROMPT_LENGTH,
        "generated_tokens": NEW_TOKENS,
        "clusters_per_rate": CLUSTERS,
        "timed_repetitions": REPEATS,
        "warmups_per_method": WARMUPS,
        "correctness": correctness_results,
        "summary": summary,
        "raw_records": len(raw_results),
    }

    json_path = output_dir / "report.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 75)
    print("FINAL GPU REUSE BENCHMARK")
    print("=" * 75)

    for rate, methods in summary.items():

        print(f"\nDuplicate rate: {rate}")

        for method in (
            "standard_no_cache",
            "kv_cache",
            "rct_dedup_kv"
        ):

            result = methods[method]

            print(
                f"{method:22s} "
                f"latency={result['median_latency_seconds']:.5f}s "
                f"output_tokens/s="
                f"{result['median_output_tokens_per_second']:.2f}"
            )

        print(
            "RCT speedup vs KV:",
            f"{methods['rct_speedup_vs_kv']:.3f}x"
        )

    print("\nCorrectness checks:", len(correctness_results))
    print("Raw timing records:", len(raw_results))
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
