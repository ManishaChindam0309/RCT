
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import benchmark_rct_249 as b

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined
from model.rct_100m_adaptive import RCT100MAdaptive


def compare(name, actual, expected):
    same_shape = actual.shape == expected.shape
    equal = same_shape and torch.equal(actual, expected)

    if same_shape:
        mismatches = int(
            (actual != expected).sum().item()
        )
        total = actual.numel()

        if mismatches:
            locations = torch.nonzero(
                actual != expected,
                as_tuple=False,
            )
            first = locations[0].tolist()

            print(
                f"  {name}: FAIL | "
                f"mismatches={mismatches}/{total} | "
                f"first_difference={first} | "
                f"actual_token={actual[tuple(first)].item()} | "
                f"expected_token={expected[tuple(first)].item()}"
            )
        else:
            print(f"  {name}: PASS")
    else:
        mismatches = "shape mismatch"
        print(
            f"  {name}: FAIL | "
            f"actual_shape={tuple(actual.shape)} | "
            f"expected_shape={tuple(expected.shape)}"
        )

    return equal


def main():
    torch.set_num_threads(b.CPU_THREADS)

    _, prompt_pool = b.load_prompt_tokens()

    base_model = RCT100M()
    base_model.eval()

    kv_model = RCT100MKV(base_model)
    kv_model.eval()

    combined_model = RCT100MCombined(base_model)
    combined_model.eval()

    adaptive_models = {
        threshold: RCT100MAdaptive(
            base_model,
            duplicate_threshold=threshold,
        )
        for threshold in b.THRESHOLDS
    }

    for model in adaptive_models.values():
        model.eval()

    cases = [
        (4, 0.0),
        (4, 0.50),
        (4, 0.75),
        (8, 0.875),
    ]

    for case_number, (batch_size, duplicate_rate) in enumerate(cases, 1):
        seed = 2491

        workload_seed = (
            seed
            + batch_size * 100
            + int(duplicate_rate * 1000)
        )

        prompts, unique_prompts, inverse, actual_rate = (
            b.make_workload(
                prompt_pool,
                batch_size,
                duplicate_rate,
                workload_seed,
            )
        )

        duplicate_count = (
            batch_size - unique_prompts.shape[0]
        )

        print("\n" + "=" * 70)
        print(
            f"CASE {case_number}: batch={batch_size}, "
            f"duplicate_rate={actual_rate:.1%}"
        )
        print("=" * 70)

        print(
            "Prompt shapes:",
            tuple(prompts.shape),
            tuple(unique_prompts.shape),
        )

        kv_full = b.generate_kv(
            kv_model,
            prompts,
            seed,
        )

        kv_unique = b.generate_kv(
            kv_model,
            unique_prompts,
            seed,
        )

        expected_deduplicated = kv_unique.index_select(
            0,
            inverse,
        )

        print(
            "KV output shapes:",
            tuple(kv_full.shape),
            tuple(kv_unique.shape),
        )

        compare(
            "KV full reproducibility",
            b.generate_kv(kv_model, prompts, seed),
            kv_full,
        )

        compare(
            "Unique KV mapped to full batch",
            expected_deduplicated,
            expected_deduplicated,
        )

        combined_output = b.generate_combined(
            combined_model,
            prompts,
            seed,
        )

        compare(
            "Combined vs unique KV reference",
            combined_output,
            expected_deduplicated,
        )

        for threshold, adaptive_model in adaptive_models.items():
            expected_dedup = b.should_deduplicate(
                batch_size,
                duplicate_count,
                threshold,
            )

            adaptive_output = b.generate_adaptive(
                adaptive_model,
                prompts,
                seed,
            )

            if expected_dedup:
                expected_output = expected_deduplicated
                decision = "deduplicated"
            else:
                expected_output = kv_full
                decision = "kv_only"

            stats = adaptive_model.statistics()

            print(
                f"\nThreshold={threshold:.3f} | "
                f"expected_decision={decision} | "
                f"reported_decision={stats.get('last_decision')}"
            )

            compare(
                "Adaptive vs expected reference",
                adaptive_output,
                expected_output,
            )


if __name__ == "__main__":
    main()
