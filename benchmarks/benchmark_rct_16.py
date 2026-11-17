
import sys
import time
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer import RCTModel


# ============================================================
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
HIDDEN_SIZE = 256
LAYERS = 4
HEADS = 4
MAX_SEQ_LEN = 256

PREFIX_LENGTH = 64
SUFFIX_LENGTH = 32

PREFIX_GROUPS = 4
PROMPTS_PER_GROUP = 3

BENCHMARK_RUNS = 15
WARMUP_RUNS = 2
CPU_THREADS = 4


# ============================================================
# TIMING
# ============================================================

def median_ms(values):
    return statistics.median(values) * 1000


def benchmark(function):
    for _ in range(WARMUP_RUNS):
        function()

    samples = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)

    return median_ms(samples)


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        layers=LAYERS,
        heads=HEADS,
        max_seq_len=MAX_SEQ_LEN
    )

    model.eval()

    # Disable independent FFN result caching.
    # Prefix caching remains enabled.
    for block in model.blocks:
        block.ffn.train()

    return model


# ============================================================
# PROMPT GENERATION
# ============================================================

def create_prompt_groups():

    groups = []

    for group_index in range(PREFIX_GROUPS):

        generator = torch.Generator(device="cpu")
        generator.manual_seed(16000 + group_index)

        prefix = torch.randint(
            0,
            VOCAB_SIZE,
            (1, PREFIX_LENGTH),
            generator=generator
        )

        prompts = []

        for prompt_index in range(PROMPTS_PER_GROUP):

            suffix_generator = torch.Generator(device="cpu")
            suffix_generator.manual_seed(
                17000 + group_index * 100 + prompt_index
            )

            suffix = torch.randint(
                0,
                VOCAB_SIZE,
                (1, SUFFIX_LENGTH),
                generator=suffix_generator
            )

            prompts.append(
                torch.cat([prefix, suffix], dim=1)
            )

        groups.append(prompts)

    return groups


# ============================================================
# CACHE UTILITIES
# ============================================================

def clear_cache(model):

    model._prefix_cache.clear()

    model._prefix_cache_hits = 0
    model._prefix_cache_misses = 0
    model._prefix_cache_evictions = 0


def cache_statistics(model):

    hits = model._prefix_cache_hits
    misses = model._prefix_cache_misses
    evictions = model._prefix_cache_evictions

    total = hits + misses

    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "entries": len(model._prefix_cache),
        "capacity": model._prefix_cache_capacity,
        "hit_rate": hits / total if total else 0.0
    }


def cache_memory(model):

    total_bytes = 0
    tensor_count = 0

    for entry in model._prefix_cache.values():

        for tensor_group in ("keys", "values"):

            for tensor in entry[tensor_group]:

                total_bytes += (
                    tensor.numel() * tensor.element_size()
                )

                tensor_count += 1

        hidden = entry["hidden"]

        total_bytes += (
            hidden.numel() * hidden.element_size()
        )

        tensor_count += 1

    return {
        "bytes": total_bytes,
        "mib": total_bytes / (1024 * 1024),
        "tensor_count": tensor_count
    }


# ============================================================
# CORRECTNESS
# ============================================================

def test_all_prompt_correctness(model, groups):

    print("\n1. NUMERICAL CORRECTNESS")
    print("-" * 65)

    clear_cache(model)

    total_prompts = 0
    maximum_difference = 0.0

    with torch.inference_mode():

        for group_index, prompts in enumerate(groups):

            for prompt_index, prompt in enumerate(prompts):

                standard = model(prompt)

                cached = model.forward_cached(
                    prompt,
                    PREFIX_LENGTH
                )

                difference = (
                    standard - cached
                ).abs().max().item()

                maximum_difference = max(
                    maximum_difference,
                    difference
                )

                correct = torch.allclose(
                    standard,
                    cached,
                    rtol=1e-4,
                    atol=1e-4
                )

                if not correct:
                    raise AssertionError(
                        f"Numerical mismatch: group {group_index}, "
                        f"prompt {prompt_index}, "
                        f"difference {difference}"
                    )

                total_prompts += 1

    print("Prefix groups       :", len(groups))
    print("Prompts per group   :", PROMPTS_PER_GROUP)
    print("Total prompts       :", total_prompts)

    print(
        f"Maximum difference  : {maximum_difference:.10f}"
    )

    print("All prompts correct : True")
    print("NUMERICAL CORRECTNESS PASSED")


# ============================================================
# STANDARD WORKLOAD
# ============================================================

def run_standard_workload(model, groups):

    results = []

    with torch.inference_mode():

        for prompts in groups:

            for prompt in prompts:

                results.append(model(prompt))

    return results


def benchmark_standard_workload(model, groups):

    return benchmark(
        lambda: run_standard_workload(model, groups)
    )


# ============================================================
# CACHED WORKLOAD
# ============================================================

def run_cached_workload(model, groups):

    results = []

    for prompts in groups:

        for prompt in prompts:

            results.append(
                model.forward_cached(
                    prompt,
                    PREFIX_LENGTH
                )
            )

    return results


def benchmark_cached_workload(model, groups):

    clear_cache(model)

    with torch.inference_mode():

        # First call per group builds the prefix entry.
        # Later calls reuse that prefix.
        start = time.perf_counter()

        run_cached_workload(model, groups)

        cold_workload_ms = (
            time.perf_counter() - start
        ) * 1000

        # All four prefixes are now cached.
        # Measure repeated warm workloads.
        warm_workload_ms = benchmark(
            lambda: run_cached_workload(model, groups)
        )

    return cold_workload_ms, warm_workload_ms


# ============================================================
# PER-PROMPT WARM LATENCY
# ============================================================

def measure_warm_latency(model, groups):

    clear_cache(model)

    measurements = []

    with torch.inference_mode():

        # Populate all four prefix entries.
        for prompts in groups:

            model.forward_cached(
                prompts[0],
                PREFIX_LENGTH
            )

        # Time only warm requests.
        for prompts in groups:

            for prompt in prompts[1:]:

                measurements.append(
                    benchmark(
                        lambda: model.forward_cached(
                            prompt,
                            PREFIX_LENGTH
                        )
                    )
                )

    return statistics.median(measurements)


# ============================================================
# MULTI-PREFIX ISOLATION
# ============================================================

def test_prefix_isolation(model, groups):

    print("\n6. PREFIX ISOLATION TEST")
    print("-" * 65)

    clear_cache(model)

    with torch.inference_mode():

        for prompts in groups:

            model.forward_cached(
                prompts[0],
                PREFIX_LENGTH
            )

    stats = cache_statistics(model)

    print("Distinct prefixes inserted :", len(groups))
    print("Cache entries              :", stats["entries"])
    print("Cache misses               :", stats["misses"])
    print("Cache hits                 :", stats["hits"])

    passed = (
        stats["entries"] == PREFIX_GROUPS
        and stats["misses"] == PREFIX_GROUPS
    )

    print("Prefixes stored separately :", passed)

    if not passed:
        raise AssertionError(
            "Distinct prefix isolation test failed."
        )

    print("PREFIX ISOLATION PASSED")


# ============================================================
# LRU EVICTION
# ============================================================

def test_lru_eviction(model, groups):

    print("\n7. LRU EVICTION TEST")
    print("-" * 65)

    original_capacity = model._prefix_cache_capacity

    try:

        model._prefix_cache_capacity = 3

        clear_cache(model)

        with torch.inference_mode():

            # Insert groups 0, 1, 2.
            for prompts in groups[:3]:

                model.forward_cached(
                    prompts[0],
                    PREFIX_LENGTH
                )

            # Access group 0 again.
            # Group 0 becomes most recently used.
            model.forward_cached(
                groups[0][1],
                PREFIX_LENGTH
            )

            # Insert group 3.
            # Group 1 should now be evicted.
            model.forward_cached(
                groups[3][0],
                PREFIX_LENGTH
            )

        stats = cache_statistics(model)

        print("Test capacity    :", model._prefix_cache_capacity)
        print("Current entries  :", stats["entries"])
        print("Evictions        :", stats["evictions"])

        # Accessing group 0 should hit.
        with torch.inference_mode():

            model.forward_cached(
                groups[0][2],
                PREFIX_LENGTH
            )

        hits_after_group_0 = model._prefix_cache_hits

        # Accessing group 1 should miss because it was evicted.
        with torch.inference_mode():

            model.forward_cached(
                groups[1][1],
                PREFIX_LENGTH
            )

        final_stats = cache_statistics(model)

        print("Group 0 retained :", model._prefix_cache_hits > hits_after_group_0 - 1)
        print("Final hits       :", final_stats["hits"])
        print("Final misses     :", final_stats["misses"])
        print("Final evictions  :", final_stats["evictions"])

        passed = (
            final_stats["entries"] <= 3
            and final_stats["evictions"] >= 2
            and final_stats["misses"] >= 5
        )

        if not passed:
            raise AssertionError(
                "LRU eviction test failed."
            )

        print("LRU EVICTION TEST PASSED")

    finally:

        model._prefix_cache_capacity = original_capacity
        clear_cache(model)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("RCT 1.6 - MULTIPLE SHARED PREFIX REUSE")
    print("=" * 65)

    torch.set_num_threads(CPU_THREADS)

    model = create_model()

    groups = create_prompt_groups()

    print("\nCONFIGURATION")
    print("-" * 65)

    print("Device           : CPU")
    print("Layers           :", LAYERS)
    print("Hidden size      :", HIDDEN_SIZE)
    print("Prefix length    :", PREFIX_LENGTH)
    print("Suffix length    :", SUFFIX_LENGTH)
    print("Prefix groups    :", PREFIX_GROUPS)
    print("Prompts per group:", PROMPTS_PER_GROUP)
    print("Total prompts    :", PREFIX_GROUPS * PROMPTS_PER_GROUP)
    print("Cache capacity   :", model._prefix_cache_capacity)

    # 1. Correctness
    test_all_prompt_correctness(model, groups)

    # 2. Standard workload
    print("\n2. STANDARD WORKLOAD")
    print("-" * 65)

    standard_ms = benchmark_standard_workload(
        model,
        groups
    )

    print(f"All prompts standard median: {standard_ms:.4f} ms")

    # 3. Cached workload
    print("\n3. MULTI-PREFIX CACHED WORKLOAD")
    print("-" * 65)

    cold_ms, warm_ms = benchmark_cached_workload(
        model,
        groups
    )

    print(f"Cold workload time: {cold_ms:.4f} ms")
    print(f"Warm workload median: {warm_ms:.4f} ms")

    # 4. Warm per-prompt latency
    print("\n4. WARM PER-PROMPT LATENCY")
    print("-" * 65)

    warm_per_prompt_ms = measure_warm_latency(
        model,
        groups
    )

    standard_per_prompt_ms = standard_ms / (
        PREFIX_GROUPS * PROMPTS_PER_GROUP
    )

    print(
        f"Standard average per prompt: "
        f"{standard_per_prompt_ms:.4f} ms"
    )

    print(
        f"Warm cached per prompt: "
        f"{warm_per_prompt_ms:.4f} ms"
    )

    # 5. Cache statistics
    print("\n5. CACHE STATISTICS")
    print("-" * 65)

    clear_cache(model)

    with torch.inference_mode():

        for prompts in groups:

            for prompt in prompts:

                model.forward_cached(
                    prompt,
                    PREFIX_LENGTH
                )

    stats = cache_statistics(model)

    print("Hits     :", stats["hits"])
    print("Misses   :", stats["misses"])
    print("Evictions:", stats["evictions"])
    print("Entries  :", stats["entries"])
    print("Capacity :", stats["capacity"])
    print(f"Hit rate : {stats['hit_rate']:.4f}")

    expected_hits = (
        PREFIX_GROUPS * (PROMPTS_PER_GROUP - 1)
    )

    expected_misses = PREFIX_GROUPS

    if (
        stats["hits"] != expected_hits
        or stats["misses"] != expected_misses
    ):
        raise AssertionError(
            "Unexpected multi-prefix hit/miss statistics."
        )

    print("MULTI-PREFIX REUSE PASSED")

    # 6. Prefix isolation
    test_prefix_isolation(model, groups)

    # 7. LRU eviction
    test_lru_eviction(model, groups)

    # 8. Memory
    print("\n8. CACHE MEMORY")
    print("-" * 65)

    clear_cache(model)

    with torch.inference_mode():

        for prompts in groups:

            model.forward_cached(
                prompts[0],
                PREFIX_LENGTH
            )

    memory = cache_memory(model)

    print("Cached prefixes :", len(model._prefix_cache))
    print("Stored tensors  :", memory["tensor_count"])
    print(f"Memory          : {memory['mib']:.4f} MiB")
    print(f"Memory bytes    : {memory['bytes']:,}")

    # 9. Final summary
    print("\n")
    print("=" * 65)
    print("RCT 1.6 FINAL RESULTS")
    print("=" * 65)

    speedup = (
        standard_ms / warm_ms
        if warm_ms > 0 else 0
    )

    reduction = (
        (1 - warm_ms / standard_ms) * 100
        if standard_ms > 0 else 0
    )

    print(f"Standard workload : {standard_ms:.4f} ms")
    print(f"Cold cached       : {cold_ms:.4f} ms")
    print(f"Warm cached       : {warm_ms:.4f} ms")

    print(f"Warm speedup      : {speedup:.3f}x")
    print(f"Latency reduction : {reduction:.2f}%")

    print("Numerical correctness : PASSED")
    print("Multi-prefix reuse    : PASSED")
    print("Prefix isolation      : PASSED")
    print("LRU eviction          : PASSED")

    print("\nRCT 1.6 BENCHMARK COMPLETED")
    print("=" * 65)

    print(
        "\nNote: Synthetic random-weight model, CPU only. "
        "The workload uses repeated identical prefixes and "
        "does not establish language-model quality."
    )


if __name__ == "__main__":
    main()
