
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

CONTEXT_LENGTHS = [64, 128, 192]
BATCH_SIZES = [1, 2]

GENERATED_TOKENS = 16
WARMUP_RUNS = 2
BENCHMARK_RUNS = 7
CPU_THREADS = 4

DEVICE = torch.device("cpu")


# ============================================================
# TIMING UTILITIES
# ============================================================

def median_ms(values):
    return statistics.median(values) * 1000


def percentile_95_ms(values):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[index] * 1000


def benchmark_function(function):
    for _ in range(WARMUP_RUNS):
        function()

    measurements = []

    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - start)

    return {
        "median_ms": median_ms(measurements),
        "p95_ms": percentile_95_ms(measurements)
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

    return model


# ============================================================
# CACHE API
# ============================================================

def create_cache(model, prompt):
    capacity = prompt.size(1) + GENERATED_TOKENS

    logits, cache = model.create_generation_cache(
        prompt,
        max_cache_len=capacity
    )

    return logits, cache


def select_token(logits):
    return logits.argmax(dim=-1, keepdim=True)


# ============================================================
# INCREMENTAL DECODING
# ============================================================

def decode_remaining_tokens(model, logits, cache):
    """
    The prefill logits predict the first generated token.

    That token is already predicted, so we feed it into
    decode_step to predict the second token.

    Therefore, generating N tokens requires N-1 decode calls.
    """

    next_token = select_token(logits)

    for _ in range(GENERATED_TOKENS - 1):
        logits, cache = model.decode_step(
            next_token,
            cache
        )

        next_token = select_token(logits)

    return next_token, cache


# ============================================================
# CACHE MEMORY
# ============================================================

def cache_memory(cache):
    allocated_bytes = 0
    populated_bytes = 0
    tensor_count = 0

    for group in ("keys", "values"):
        for tensor in cache[group]:

            tensor_count += 1

            allocated_bytes += (
                tensor.numel() * tensor.element_size()
            )

            capacity = tensor.shape[-2]
            populated_length = cache["length"]

            populated_bytes += (
                tensor.numel()
                * tensor.element_size()
                * populated_length
                // capacity
            )

    return {
        "allocated_bytes": allocated_bytes,
        "populated_bytes": populated_bytes,
        "tensor_count": tensor_count,
        "length": cache["length"],
        "capacity": cache["capacity"]
    }


def mib(value):
    return value / (1024 * 1024)


# ============================================================
# ONE WORKLOAD
# ============================================================

def run_workload(model, batch_size, context_length):

    print()
    print("=" * 65)
    print(
        f"BATCH SIZE {batch_size} | "
        f"CONTEXT LENGTH {context_length}"
    )
    print("=" * 65)

    torch.manual_seed(1000 + batch_size + context_length)

    prompt = torch.randint(
        0,
        VOCAB_SIZE,
        (batch_size, context_length),
        dtype=torch.long,
        device=DEVICE
    )

    # --------------------------------------------------------
    # 1. NUMERICAL CORRECTNESS
    # --------------------------------------------------------

    with torch.inference_mode():

        standard_output = model.generate(
            prompt,
            max_new_tokens=GENERATED_TOKENS
        )

        cached_output = model.generate_cached(
            prompt,
            max_new_tokens=GENERATED_TOKENS
        )

    correct = torch.equal(
        standard_output,
        cached_output
    )

    print("\n1. NUMERICAL CORRECTNESS")
    print("-" * 65)
    print("Standard output shape :", tuple(standard_output.shape))
    print("Cached output shape   :", tuple(cached_output.shape))
    print("Generated sequences identical:", correct)

    if not correct:
        raise AssertionError(
            "Standard and cached generation produced different tokens."
        )

    # --------------------------------------------------------
    # 2. STANDARD GENERATION
    # --------------------------------------------------------

    with torch.inference_mode():
        standard_result = benchmark_function(
            lambda: model.generate(
                prompt,
                max_new_tokens=GENERATED_TOKENS
            )
        )

    print("\n2. STANDARD GENERATION")
    print("-" * 65)
    print(f"Median latency : {standard_result['median_ms']:.4f} ms")
    print(f"P95 latency    : {standard_result['p95_ms']:.4f} ms")

    # --------------------------------------------------------
    # 3. PREFILL
    # --------------------------------------------------------

    with torch.inference_mode():

        prefill_result = benchmark_function(
            lambda: create_cache(model, prompt)
        )

        prefill_logits, cache = create_cache(model, prompt)

    print("\n3. PROMPT PREFILL")
    print("-" * 65)
    print(f"Median latency : {prefill_result['median_ms']:.4f} ms")
    print(f"P95 latency    : {prefill_result['p95_ms']:.4f} ms")

    # --------------------------------------------------------
    # 4. INCREMENTAL DECODE
    # --------------------------------------------------------

    decode_measurements = []

    with torch.inference_mode():

        for _ in range(WARMUP_RUNS + BENCHMARK_RUNS):

            # Each measurement gets a fresh, fully populated
            # prompt cache. Cache construction is outside timing.
            logits, decode_cache = create_cache(model, prompt)

            next_token = select_token(logits)

            start = time.perf_counter()

            for _ in range(GENERATED_TOKENS - 1):

                logits, decode_cache = model.decode_step(
                    next_token,
                    decode_cache
                )

                next_token = select_token(logits)

            elapsed = time.perf_counter() - start

            decode_measurements.append(elapsed)

    decode_measurements = decode_measurements[WARMUP_RUNS:]

    decode_total_ms = median_ms(decode_measurements)

    decode_per_step_ms = (
        decode_total_ms / (GENERATED_TOKENS - 1)
    )

    print("\n4. INCREMENTAL DECODE")
    print("-" * 65)
    print("Decode steps        :", GENERATED_TOKENS - 1)
    print(f"Decode total median : {decode_total_ms:.4f} ms")
    print(f"Decode total P95    : {percentile_95_ms(decode_measurements) :.4f} ms")
    print(f"Decode per step     : {decode_per_step_ms:.4f} ms")

    # --------------------------------------------------------
    # 5. COMPLETE CACHED GENERATION
    # --------------------------------------------------------

    def cached_generation():

        logits, generation_cache = create_cache(
            model,
            prompt
        )

        decode_remaining_tokens(
            model,
            logits,
            generation_cache
        )

    with torch.inference_mode():
        cached_result = benchmark_function(cached_generation)

    cached_total_ms = cached_result["median_ms"]

    print("\n5. COMPLETE CACHED GENERATION")
    print("-" * 65)
    print(f"Cached total median : {cached_total_ms:.4f} ms")
    print(f"Cached total P95    : {cached_result['p95_ms']:.4f} ms")

    # --------------------------------------------------------
    # 6. KV MEMORY
    # --------------------------------------------------------

    # Use the final decode cache, which contains the prompt
    # and all but the final predicted token.
    memory = cache_memory(decode_cache)

    print("\n6. KV CACHE MEMORY")
    print("-" * 65)
    print("Cache populated length :", memory["length"])
    print("Cache capacity         :", memory["capacity"])
    print("K/V tensor count       :", memory["tensor_count"])

    print(
        f"Allocated K/V memory   : "
        f"{mib(memory['allocated_bytes']):.4f} MiB"
    )

    print(
        f"Estimated populated K/V: "
        f"{mib(memory['populated_bytes']):.4f} MiB"
    )

    # --------------------------------------------------------
    # 7. PERFORMANCE COMPARISON
    # --------------------------------------------------------

    standard_ms = standard_result["median_ms"]

    speedup = standard_ms / cached_total_ms

    reduction = (
        (standard_ms - cached_total_ms)
        / standard_ms
    ) * 100

    print("\n7. PERFORMANCE COMPARISON")
    print("-" * 65)
    print(f"Standard total    : {standard_ms:.4f} ms")
    print(f"Cached total      : {cached_total_ms:.4f} ms")
    print(f"Speedup           : {speedup:.3f}x")
    print(f"Latency reduction : {reduction:.2f}%")

    return {
        "batch_size": batch_size,
        "context_length": context_length,
        "standard_ms": standard_ms,
        "cached_ms": cached_total_ms,
        "prefill_ms": prefill_result["median_ms"],
        "decode_ms": decode_total_ms,
        "speedup": speedup,
        "reduction": reduction,
        "allocated_mib": mib(memory["allocated_bytes"]),
        "populated_mib": mib(memory["populated_bytes"]),
        "correct": correct
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("RCT 1.4 - LONG-CONTEXT AND KV MEMORY BENCHMARK")
    print("=" * 65)

    print("\nConfiguration")
    print("-" * 65)
    print("Context lengths :", CONTEXT_LENGTHS)
    print("Batch sizes     :", BATCH_SIZES)
    print("Generated tokens:", GENERATED_TOKENS)
    print("Benchmark runs  :", BENCHMARK_RUNS)
    print("CPU threads     :", CPU_THREADS)

    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(42)

    model = create_model()

    results = []

    for batch_size in BATCH_SIZES:

        for context_length in CONTEXT_LENGTHS:

            if context_length + GENERATED_TOKENS > MAX_SEQ_LEN:
                continue

            result = run_workload(
                model,
                batch_size,
                context_length
            )

            results.append(result)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 105)
    print("RCT 1.4 FINAL WORKLOAD SUMMARY")
    print("=" * 105)

    print(
        f"{'Batch':>6} "
        f"{'Context':>8} "
        f"{'Standard':>12} "
        f"{'Cached':>12} "
        f"{'Speedup':>10} "
        f"{'Reduction':>12} "
        f"{'KV MiB':>10} "
        f"{'Correct':>9}"
    )

    print("-" * 105)

    for item in results:

        print(
            f"{item['batch_size']:>6} "
            f"{item['context_length']:>8} "
            f"{item['standard_ms']:>12.3f} "
            f"{item['cached_ms']:>12.3f} "
            f"{item['speedup']:>9.3f}x "
            f"{item['reduction']:>11.2f}% "
            f"{item['allocated_mib']:>10.3f} "
            f"{str(item['correct']):>9}"
        )

    print("\n")
    print("=" * 65)
    print("RCT 1.4 OVERALL RESULTS")
    print("=" * 65)

    if results:

        average_speedup = statistics.mean(
            item["speedup"] for item in results
        )

        average_reduction = statistics.mean(
            item["reduction"] for item in results
        )

        correct_count = sum(
            item["correct"] for item in results
        )

        print("Workloads tested    :", len(results))
        print("Numerically correct :", correct_count)

        print(f"Average speedup     : {average_speedup:.3f}x")
        print(f"Average reduction   : {average_reduction:.2f}%")

        print(
            "Maximum KV memory   : "
            f"{max(item['allocated_mib'] for item in results):.4f} MiB"
        )

    print("\nRCT 1.4 BENCHMARK COMPLETED")
    print("=" * 65)

    print(
        "\nNote: Synthetic random-weight model. "
        "Results do not establish language-model quality "
        "or general performance across hardware."
    )


if __name__ == "__main__":
    main()
