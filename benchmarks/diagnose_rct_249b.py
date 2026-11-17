
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

import benchmark_rct_249 as benchmark

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined


def generate_kv(model, prompts, seed):
    torch.manual_seed(seed)
    with torch.inference_mode():
        return model.generate_cached(
            prompts,
            max_new_tokens=benchmark.NEW_TOKENS,
            temperature=benchmark.TEMPERATURE,
        )


def generate_combined(model, prompts, seed, deduplicate):
    torch.manual_seed(seed)
    with torch.inference_mode():
        return model.generate(
            prompts,
            max_new_tokens=benchmark.NEW_TOKENS,
            temperature=benchmark.TEMPERATURE,
            deduplicate=deduplicate,
        )


def check(label, a, b):
    same = (
        a.shape == b.shape
        and torch.equal(a, b)
    )

    print(
        f"{label}: {'PASS' if same else 'FAIL'}"
    )

    if a.shape == b.shape and not same:
        print(
            "  Mismatches:",
            int((a != b).sum().item()),
            "/",
            a.numel(),
        )

    return same


def main():
    torch.set_num_threads(benchmark.CPU_THREADS)

    _, pool = benchmark.load_prompt_tokens()

    base = RCT100M()
    base.eval()

    kv = RCT100MKV(base)
    kv.eval()

    combined = RCT100MCombined(base)
    combined.eval()

    cases = [
        (4, 0.0),
        (4, 0.50),
        (4, 0.75),
    ]

    for batch, rate in cases:
        seed = 2491

        prompts, unique, inverse, actual_rate = (
            benchmark.make_workload(
                pool,
                batch,
                rate,
                seed + batch * 100 + int(rate * 1000),
            )
        )

        print("\n" + "=" * 65)
        print(
            f"Batch={batch}, duplicate rate={actual_rate:.1%}"
        )
        print("=" * 65)

        stable_unique, stable_inverse = (
            combined._stable_deduplicate(prompts)
        )

        print(
            "Stable unique prompts match benchmark:",
            torch.equal(stable_unique, unique),
        )

        print(
            "Stable inverse indices match benchmark:",
            torch.equal(stable_inverse, inverse),
        )

        print(
            "No-reordering case:",
            torch.equal(prompts, stable_unique)
            if prompts.shape == stable_unique.shape
            else False,
        )

        # 1. Ordinary KV model.
        reference = generate_kv(
            kv,
            prompts,
            seed,
        )

        # 2. KV model inside combined wrapper.
        inner_kv = generate_kv(
            combined.kv_model,
            prompts,
            seed,
        )

        check(
            "Independent KV vs combined inner KV",
            reference,
            inner_kv,
        )

        # 3. Combined wrapper with deduplication disabled.
        combined_no_dedup = generate_combined(
            combined,
            prompts,
            seed,
            False,
        )

        check(
            "Combined deduplicate=False vs KV",
            combined_no_dedup,
            reference,
        )

        # 4. Combined wrapper with deduplication enabled.
        combined_dedup = generate_combined(
            combined,
            prompts,
            seed,
            True,
        )

        unique_reference = generate_kv(
            kv,
            unique,
            seed,
        )

        expected = unique_reference.index_select(
            0,
            inverse,
        )

        check(
            "Unique KV reference vs expected restoration",
            expected,
            expected,
        )

        check(
            "Combined deduplicate=True vs unique KV",
            combined_dedup,
            expected,
        )


if __name__ == "__main__":
    main()
