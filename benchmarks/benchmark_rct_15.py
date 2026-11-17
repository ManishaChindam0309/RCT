
import sys
import time
import math
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.transformer import RCTModel


# ============================================================
# RCT 1.5 CONFIGURATION
# ============================================================

VOCAB_SIZE = 8000
HIDDEN_SIZE = 256
LAYERS = 4
HEADS = 4
MAX_SEQ_LEN = 256

PREFIX_LENGTH = 64
SUFFIX_LENGTH = 32

WARMUP_RUNS = 2
BENCHMARK_RUNS = 15
CPU_THREADS = 4

DEVICE = torch.device("cpu")


# ============================================================
# TIMING
# ============================================================

def median_ms(values):
    return statistics.median(values) * 1000


def percentile_95_ms(values):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index] * 1000


def benchmark(function):
    for _ in range(WARMUP_RUNS):
        function()

    measurements = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - start)

    return {
        "median_ms": median_ms(measurements),
        "p95_ms": percentile_95_ms(measurements),
        "samples": measurements
    }


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

    model = model.to(DEVICE)
    model.eval()

    # Disable the independent FFN result cache.
    # Keep the overall model in eval mode because
    # forward_cached requires model.eval().
    for block in model.blocks:
        block.ffn.train()

    return model


# ============================================================
# INPUT GENERATION
# ============================================================

def create_prompts():
    torch.manual_seed(1500)

    shared_prefix = torch.randint(
        0,
        VOCAB_SIZE,
        (1, PREFIX_LENGTH),
        dtype=torch.long,
        device=DEVICE
    )

    suffix_a = torch.randint(
        0,
        VOCAB_SIZE,
        (1, SUFFIX_LENGTH),
        dtype=torch.long,
        device=DEVICE
    )

    suffix_b = torch.randint(
        0,
        VOCAB_SIZE,
        (1, SUFFIX_LENGTH),
        dtype=torch.long,
        device=DEVICE
    )

    prompt_a = torch.cat(
        [shared_prefix, suffix_a],
        dim=1
    )

    prompt_b = torch.cat(
        [shared_prefix, suffix_b],
        dim=1
    )

    return shared_prefix, prompt_a, prompt_b


# ============================================================
# CACHE UTILITIES
# ============================================================

def clear_prefix_cache(model):
    model._prefix_cache.clear()


def reset_prefix_statistics(model):
    model._prefix_cache_hits = 0
    model._prefix_cache_misses = 0
    model._prefix_cache_evictions = 0


def get_prefix_statistics(model):
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


def prefix_cache_memory(model):
    total_bytes = 0
    tensor_count = 0

    for entry in model._prefix_cache.values():

        for group in ("keys", "values"):

            for tensor in entry[group]:
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

def test_correctness(model, prompt_a, prompt_b):

    print("\n1. NUMERICAL CORRECTNESS")
    print("-" * 65)

    with torch.inference_mode():

        standard_a = model(prompt_a)

        standard_b = model(prompt_b)

        clear_prefix_cache(model)

        cached_a = model.forward_cached(
            prompt_a,
            PREFIX_LENGTH
        )

        cached_b = model.forward_cached(
            prompt_b,
            PREFIX_LENGTH
        )

    difference_a = (
        standard_a - cached_a
    ).abs().max().item()

    difference_b = (
        standard_b - cached_b
    ).abs().max().item()

    correct_a = torch.allclose(
        standard_a,
        cached_a,
        rtol=1e-4,
        atol=1e-4
    )

    correct_b = torch.allclose(
        standard_b,
        cached_b,
        rtol=1e-4,
        atol=1e-4
    )

    prefix_identical = torch.equal(
        prompt_a[:, :PREFIX_LENGTH],
        prompt_b[:, :PREFIX_LENGTH]
    )

    print("Shared prefix identical :", prefix_identical)
    print(f"Prompt A max difference : {difference_a:.10f}")
    print(f"Prompt B max difference : {difference_b:.10f}")
    print("Prompt A correct        :", correct_a)
    print("Prompt B correct        :", correct_b)

    if not (correct_a and correct_b and prefix_identical):
        raise AssertionError(
            "Cross-prompt prefix reuse failed numerical correctness."
        )

    print("NUMERICAL CORRECTNESS PASSED")


# ============================================================
# STANDARD INFERENCE
# ============================================================

def benchmark_standard(model, prompt):

    with torch.inference_mode():
        return benchmark(
            lambda: model(prompt)
        )


# ============================================================
# COLD PREFIX CACHE
# ============================================================

def benchmark_cold(model, prompt):

    measurements = []

    with torch.inference_mode():

        for _ in range(BENCHMARK_RUNS):

            # Clear outside the timed region.
            clear_prefix_cache(model)

            start = time.perf_counter()

            model.forward_cached(
                prompt,
                PREFIX_LENGTH
            )

            measurements.append(
                time.perf_counter() - start
            )

    return {
        "median_ms": median_ms(measurements),
        "p95_ms": percentile_95_ms(measurements),
        "samples": measurements
    }


# ============================================================
# WARM PREFIX CACHE
# ============================================================

def benchmark_warm(model, prompt_a, prompt_b):

    with torch.inference_mode():

        # Build the prefix entry using prompt A.
        clear_prefix_cache(model)

        model.forward_cached(
            prompt_a,
            PREFIX_LENGTH
        )

        # Prompt B has the same prefix but a different suffix.
        # Its request should reuse the existing prefix.
        return benchmark(
            lambda: model.forward_cached(
                prompt_b,
                PREFIX_LENGTH
            )
        )


# ============================================================
# BREAK-EVEN ANALYSIS
# ============================================================

def calculate_break_even(standard_ms, cold_ms, warm_ms):

    saving_per_request = standard_ms - warm_ms

    extra_initial_cost = cold_ms - warm_ms

    if saving_per_request <= 0:
        return {
            "break_even": None,
            "saving": saving_per_request,
            "extra_cost": extra_initial_cost
        }

    requests = max(
        1,
        math.ceil(
            extra_initial_cost / saving_per_request
        )
    )

    return {
        "break_even": requests,
        "saving": saving_per_request,
        "extra_cost": extra_initial_cost
    }


# ============================================================
# LRU CAPACITY TEST
# ============================================================

def test_lru_capacity(model):

    print("\n7. LRU CACHE CAPACITY TEST")
    print("-" * 65)

    clear_prefix_cache(model)
    reset_prefix_statistics(model)

    capacity = model._prefix_cache_capacity

    with torch.inference_mode():

        # Insert one more distinct prefix than the cache can hold.
        for index in range(capacity + 1):

            generator = torch.Generator(device="cpu")
            generator.manual_seed(9000 + index)

            prefix = torch.randint(
                0,
                VOCAB_SIZE,
                (1, PREFIX_LENGTH),
                generator=generator,
                dtype=torch.long
            )

            suffix = torch.randint(
                0,
                VOCAB_SIZE,
                (1, SUFFIX_LENGTH),
                generator=generator,
                dtype=torch.long
            )

            prompt = torch.cat(
                [prefix, suffix],
                dim=1
            )

            model.forward_cached(
                prompt,
                PREFIX_LENGTH
            )

    stats = get_prefix_statistics(model)

    passed = (
        stats["entries"] == capacity
        and stats["evictions"] >= 1
    )

    print("Configured capacity :", capacity)
    print("Current entries     :", stats["entries"])
    print("Cache misses        :", stats["misses"])
    print("Cache evictions     :", stats["evictions"])
    print("Capacity respected  :", passed)

    if not passed:
        raise AssertionError(
            "LRU capacity test failed."
        )

    print("LRU CAPACITY TEST PASSED")

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("RCT 1.5 - CROSS-PROMPT PREFIX REUSE BENCHMARK")
    print("=" * 65)

    print("\nConfiguration")
    print("-" * 65)
    print("Vocabulary size :", VOCAB_SIZE)
    print("Hidden size     :", HIDDEN_SIZE)
    print("Layers          :", LAYERS)
    print("Heads           :", HEADS)
    print("Prefix length   :", PREFIX_LENGTH)
    print("Suffix length   :", SUFFIX_LENGTH)
    print("Total length    :", PREFIX_LENGTH + SUFFIX_LENGTH)
    print("Benchmark runs  :", BENCHMARK_RUNS)
    print("CPU threads     :", CPU_THREADS)

    torch.set_num_threads(CPU_THREADS)

    model = create_model()

    _, prompt_a, prompt_b = create_prompts()

    # --------------------------------------------------------
    # 1. CORRECTNESS
    # --------------------------------------------------------

    test_correctness(
        model,
        prompt_a,
        prompt_b
    )

    # --------------------------------------------------------
    # 2. STANDARD INFERENCE
    # --------------------------------------------------------

    print("\n2. STANDARD INFERENCE")
    print("-" * 65)

    standard_a = benchmark_standard(
        model,
        prompt_a
    )

    standard_b = benchmark_standard(
        model,
        prompt_b
    )

    standard_ms = statistics.median([
        standard_a["median_ms"],
        standard_b["median_ms"]
    ])

    print(f"Prompt A median : {standard_a['median_ms']:.4f} ms")
    print(f"Prompt B median : {standard_b['median_ms']:.4f} ms")
    print(f"Average baseline: {standard_ms:.4f} ms")

    # --------------------------------------------------------
    # 3. COLD REQUEST
    # --------------------------------------------------------

    print("\n3. COLD PREFIX CACHE")
    print("-" * 65)

    cold = benchmark_cold(
        model,
        prompt_a
    )

    print(f"Cold median : {cold['median_ms']:.4f} ms")
    print(f"Cold P95    : {cold['p95_ms']:.4f} ms")

    # --------------------------------------------------------
    # 4. WARM REQUEST
    # --------------------------------------------------------

    print("\n4. WARM CROSS-PROMPT CACHE")
    print("-" * 65)

    warm = benchmark_warm(
        model,
        prompt_a,
        prompt_b
    )

    print(f"Warm median : {warm['median_ms']:.4f} ms")
    print(f"Warm P95    : {warm['p95_ms']:.4f} ms")

    # --------------------------------------------------------
    # 5. CACHE STATISTICS
    # --------------------------------------------------------

    print("\n5. CACHE STATISTICS")
    print("-" * 65)

    reset_prefix_statistics(model)

    with torch.inference_mode():

        clear_prefix_cache(model)

        # First prompt: miss.
        model.forward_cached(
            prompt_a,
            PREFIX_LENGTH
        )

        # Second prompt: same prefix, different suffix.
        # This should hit.
        model.forward_cached(
            prompt_b,
            PREFIX_LENGTH
        )

        # Third request: same prefix again.
        model.forward_cached(
            prompt_a,
            PREFIX_LENGTH
        )

    stats = get_prefix_statistics(model)

    print("Cache hits     :", stats["hits"])
    print("Cache misses   :", stats["misses"])
    print("Cache evictions:", stats["evictions"])
    print("Cache entries  :", stats["entries"])
    print("Cache capacity :", stats["capacity"])
    print(f"Hit rate       : {stats['hit_rate']:.4f}")

    if stats["hits"] != 2 or stats["misses"] != 1:
        raise AssertionError(
            "Expected two hits and one miss for shared prefix."
        )

    print("CROSS-PROMPT CACHE REUSE PASSED")

    # --------------------------------------------------------
    # 6. MEMORY
    # --------------------------------------------------------

    print("\n6. PREFIX CACHE MEMORY")
    print("-" * 65)

    memory = prefix_cache_memory(model)

    print("Cached prefixes :", len(model._prefix_cache))
    print("Stored tensors  :", memory["tensor_count"])
    print(f"Memory          : {memory['mib']:.4f} MiB")
    print(f"Memory bytes    : {memory['bytes']:,}")

    # --------------------------------------------------------
    # 7. BREAK-EVEN
    # --------------------------------------------------------

    print("\n7. BREAK-EVEN ANALYSIS")
    print("-" * 65)

    break_even = calculate_break_even(
        standard_ms,
        cold["median_ms"],
        warm["median_ms"]
    )

    print(f"Standard request : {standard_ms:.4f} ms")
    print(f"Cold request     : {cold['median_ms']:.4f} ms")
    print(f"Warm request     : {warm['median_ms']:.4f} ms")

    print(
        f"Extra initial cost: "
        f"{break_even['extra_cost']:.4f} ms"
    )

    print(
        f"Saving per warm request: "
        f"{break_even['saving']:.4f} ms"
    )

    if break_even["break_even"] is None:
        print("Break-even       : Not achieved with current timings.")
    else:
        print(
            "Total requests needed to break even:",
            break_even["break_even"]
        )

    # --------------------------------------------------------
    # 8. LRU
    # --------------------------------------------------------

    lru_stats = test_lru_capacity(model)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 65)
    print("RCT 1.5 FINAL RESULTS")
    print("=" * 65)

    cold_speedup = (
        standard_ms / cold["median_ms"]
        if cold["median_ms"] > 0 else 0
    )

    warm_speedup = (
        standard_ms / warm["median_ms"]
        if warm["median_ms"] > 0 else 0
    )

    print(f"Standard inference : {standard_ms:.4f} ms")
    print(f"Cold cache request : {cold['median_ms']:.4f} ms")
    print(f"Warm cache request : {warm['median_ms']:.4f} ms")
    print(f"Cold speedup       : {cold_speedup:.3f}x")
    print(f"Warm speedup       : {warm_speedup:.3f}x")

    print("Shared-prefix correctness: PASSED")
    print("Cross-prompt reuse       : PASSED")
    print("LRU capacity             : PASSED")

    print("\nRCT 1.5 BENCHMARK COMPLETED")
    print("=" * 65)

    print(
        "\nNote: Synthetic random-weight model on CPU. "
        "Results do not establish language-model quality "
        "or general performance across hardware."
    )


if __name__ == "__main__":
    main()
