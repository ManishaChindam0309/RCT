
import copy
import statistics
import sys
import time
from pathlib import Path

import torch

# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import RCTModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

VOCAB_SIZE = 8000
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_HEADS = 4
MAX_SEQ_LEN = 256

PROMPT_LENGTH = 128
GENERATED_TOKENS = 32

WARMUP_RUNS = 3
BENCHMARK_RUNS = 15

CPU_THREADS = 4

SEED = 42


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def synchronize():
    """
    CPU execution does not require CUDA synchronization.
    This function keeps timing calls easy to adapt later.
    """
    pass


def percentile(values, percentile_value):
    """
    Calculate a percentile using sorted measurements.
    """
    ordered = sorted(values)

    if not ordered:
        return 0.0

    index = round(
        (percentile_value / 100.0) * (len(ordered) - 1)
    )

    return ordered[index]


def measure_time(function):
    """
    Measure one function execution in milliseconds.
    """
    synchronize()

    start = time.perf_counter()

    result = function()

    synchronize()

    elapsed = (time.perf_counter() - start) * 1000.0

    return elapsed, result


def benchmark(function, warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS):
    """
    Warm up and measure repeated executions.

    Returns median, P95, and all measurements.
    """

    for _ in range(warmup):
        function()

    measurements = []

    for _ in range(runs):
        elapsed, _ = measure_time(function)
        measurements.append(elapsed)

    return {
        "median": statistics.median(measurements),
        "p95": percentile(measurements, 95),
        "measurements": measurements
    }


def disable_ffn_cache(model):
    """
    Disable ReusableFFN caching so this benchmark isolates
    autoregressive KV-cache performance.
    """
    for block in model.blocks:
        block.ffn.eval()

        # Clear any previous entries.
        block.ffn.memory.clear()


def create_model():
    """
    Create a model with the benchmark configuration.
    """
    model = RCTModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=MAX_SEQ_LEN
    )

    model.eval()

    disable_ffn_cache(model)

    return model


# ---------------------------------------------------------
# Standard autoregressive generation
# ---------------------------------------------------------

@torch.inference_mode()
def standard_generation(model, prompt):
    """
    Generate tokens by recomputing the full sequence
    at every generation step.
    """
    return model.generate(
        prompt,
        max_new_tokens=GENERATED_TOKENS,
        temperature=0.0
    )


# ---------------------------------------------------------
# Preallocated KV-cache generation
# ---------------------------------------------------------

@torch.inference_mode()
def cached_generation(model, prompt):
    """
    Generate tokens using preallocated incremental KV cache.
    """
    return model.generate_cached(
        prompt,
        max_new_tokens=GENERATED_TOKENS,
        temperature=0.0
    )


# ---------------------------------------------------------
# Prefill benchmark
# ---------------------------------------------------------

@torch.inference_mode()
def measure_prefill(model, prompt):
    """
    Measure prompt prefill and KV-cache allocation.
    """
    _, cache = model.create_generation_cache(
        prompt,
        max_cache_len=PROMPT_LENGTH + GENERATED_TOKENS
    )

    return cache


# ---------------------------------------------------------
# Decode benchmark
# ---------------------------------------------------------

@torch.inference_mode()
def measure_decode(model, prompt):
    """
    Measure incremental decoding after prompt prefill.

    This measures GENERATED_TOKENS - 1 decode steps because
    the first generated token is sampled directly from
    the prompt's prefill logits.
    """

    logits, cache = model.create_generation_cache(
        prompt,
        max_cache_len=PROMPT_LENGTH + GENERATED_TOKENS
    )

    next_token = torch.argmax(
        logits,
        dim=-1,
        keepdim=True
    )

    decode_times = []

    for step in range(GENERATED_TOKENS - 1):

        elapsed, result = measure_time(
            lambda: model.decode_step(
                next_token,
                cache
            )
        )

        decode_times.append(elapsed)

        logits, cache = result

        next_token = torch.argmax(
            logits,
            dim=-1,
            keepdim=True
        )

    return {
        "median_step": statistics.median(decode_times),
        "total": sum(decode_times),
        "steps": len(decode_times),
        "cache": cache
    }


# ---------------------------------------------------------
# KV cache memory
# ---------------------------------------------------------

def calculate_cache_memory(cache):
    """
    Calculate memory occupied by populated K/V tensors
    and by the complete preallocated buffers.
    """

    populated_bytes = 0
    allocated_bytes = 0

    for key, value in zip(
        cache["keys"],
        cache["values"]
    ):
        populated_bytes += (
            key[:, :, :cache["length"], :].numel()
            * key.element_size()
        )

        populated_bytes += (
            value[:, :, :cache["length"], :].numel()
            * value.element_size()
        )

        allocated_bytes += key.numel() * key.element_size()
        allocated_bytes += value.numel() * value.element_size()

    return {
        "populated_bytes": populated_bytes,
        "allocated_bytes": allocated_bytes
    }


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("RCT 1.3.1 - PREALLOCATED AUTOREGRESSIVE KV CACHE BENCHMARK")
    print("=" * 65)

    print("\nConfiguration")
    print("-" * 65)
    print(f"Vocabulary size       : {VOCAB_SIZE}")
    print(f"Hidden size           : {HIDDEN_SIZE}")
    print(f"Transformer layers    : {NUM_LAYERS}")
    print(f"Attention heads       : {NUM_HEADS}")
    print(f"Maximum sequence      : {MAX_SEQ_LEN}")
    print(f"Prompt length         : {PROMPT_LENGTH}")
    print(f"Generated tokens      : {GENERATED_TOKENS}")
    print(f"Warmup runs           : {WARMUP_RUNS}")
    print(f"Benchmark runs        : {BENCHMARK_RUNS}")
    print(f"CPU threads           : {CPU_THREADS}")

    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)

    # Create baseline model.
    standard_model = create_model()

    # Create cached model with identical parameters.
    cached_model = create_model()

    cached_model.load_state_dict(
        copy.deepcopy(standard_model.state_dict())
    )

    standard_model.eval()
    cached_model.eval()

    disable_ffn_cache(standard_model)
    disable_ffn_cache(cached_model)

    # Generate a deterministic random prompt.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + 1)

    prompt = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(1, PROMPT_LENGTH),
        generator=generator,
        dtype=torch.long
    )

    # -----------------------------------------------------
    # Correctness test
    # -----------------------------------------------------

    print("\n1. NUMERICAL CORRECTNESS")
    print("-" * 65)

    standard_output = standard_generation(
        standard_model,
        prompt
    )

    cached_output = cached_generation(
        cached_model,
        prompt
    )

    print(f"Standard output shape : {tuple(standard_output.shape)}")
    print(f"Cached output shape   : {tuple(cached_output.shape)}")

    identical = torch.equal(
        standard_output,
        cached_output
    )

    print(f"Generated sequences identical: {identical}")

    if not identical:
        differences = (
            standard_output != cached_output
        ).sum().item()

        print(f"Different token positions: {differences}")

        raise AssertionError(
            "Standard and cached generation produced different tokens."
        )

    print("NUMERICAL GENERATION CORRECTNESS PASSED")

    # -----------------------------------------------------
    # Standard generation benchmark
    # -----------------------------------------------------

    print("\n2. STANDARD GENERATION BENCHMARK")
    print("-" * 65)

    standard_results = benchmark(
        lambda: standard_generation(
            standard_model,
            prompt
        )
    )

    print(
        f"Standard median latency : "
        f"{standard_results['median']:.4f} ms"
    )

    print(
        f"Standard P95 latency    : "
        f"{standard_results['p95']:.4f} ms"
    )

    # -----------------------------------------------------
    # Cached generation benchmark
    # -----------------------------------------------------

    print("\n3. PREALLOCATED KV CACHE BENCHMARK")
    print("-" * 65)

    cached_results = benchmark(
        lambda: cached_generation(
            cached_model,
            prompt
        )
    )

    print(
        f"Cached total median     : "
        f"{cached_results['median']:.4f} ms"
    )

    print(
        f"Cached total P95        : "
        f"{cached_results['p95']:.4f} ms"
    )

    # -----------------------------------------------------
    # Prefill benchmark
    # -----------------------------------------------------

    print("\n4. PROMPT PREFILL BENCHMARK")
    print("-" * 65)

    prefill_results = benchmark(
        lambda: measure_prefill(
            cached_model,
            prompt
        )
    )

    print(
        f"Prefill median latency  : "
        f"{prefill_results['median']:.4f} ms"
    )

    print(
        f"Prefill P95 latency     : "
        f"{prefill_results['p95']:.4f} ms"
    )

    # -----------------------------------------------------
    # Incremental decode benchmark
    # -----------------------------------------------------

    print("\n5. INCREMENTAL DECODE BENCHMARK")
    print("-" * 65)

    decode_results = []

    for _ in range(WARMUP_RUNS):
        measure_decode(cached_model, prompt)

    for _ in range(BENCHMARK_RUNS):
        result = measure_decode(
            cached_model,
            prompt
        )

        decode_results.append(result["total"])

    decode_median = statistics.median(decode_results)
    decode_p95 = percentile(decode_results, 95)

    print(
        f"Decode steps            : "
        f"{GENERATED_TOKENS - 1}"
    )

    print(
        f"Decode total median     : "
        f"{decode_median:.4f} ms"
    )

    print(
        f"Decode total P95        : "
        f"{decode_p95:.4f} ms"
    )

    print(
        f"Decode median per step  : "
        f"{decode_median / (GENERATED_TOKENS - 1):.4f} ms"
    )

    # -----------------------------------------------------
    # KV cache memory benchmark
    # -----------------------------------------------------

    print("\n6. KV CACHE MEMORY")
    print("-" * 65)

    _, memory_cache = cached_model.create_generation_cache(
        prompt,
        max_cache_len=PROMPT_LENGTH + GENERATED_TOKENS
    )

    memory = calculate_cache_memory(memory_cache)

    populated_mib = (
        memory["populated_bytes"] / (1024 ** 2)
    )

    allocated_mib = (
        memory["allocated_bytes"] / (1024 ** 2)
    )

    print(f"Cache populated length  : {memory_cache['length']}")
    print(f"Cache capacity          : {memory_cache['capacity']}")

    print(
        f"Populated K/V memory    : "
        f"{memory['populated_bytes']:,} bytes "
        f"({populated_mib:.4f} MiB)"
    )

    print(
        f"Allocated K/V memory    : "
        f"{memory['allocated_bytes']:,} bytes "
        f"({allocated_mib:.4f} MiB)"
    )

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    print("\n7. PERFORMANCE COMPARISON")
    print("-" * 65)

    standard_median = standard_results["median"]
    cached_median = cached_results["median"]

    speedup = (
        standard_median / cached_median
        if cached_median > 0 else 0.0
    )

    latency_reduction = (
        (standard_median - cached_median)
        / standard_median
        * 100.0
        if standard_median > 0 else 0.0
    )

    standard_ms_per_token = (
        standard_median / GENERATED_TOKENS
    )

    cached_ms_per_token = (
        cached_median / GENERATED_TOKENS
    )

    print(f"Standard total latency  : {standard_median:.4f} ms")
    print(f"Cached total latency    : {cached_median:.4f} ms")
    print(f"Speedup                 : {speedup:.3f}x")
    print(f"Latency reduction       : {latency_reduction:.2f}%")

    print(
        f"Standard ms/token      : "
        f"{standard_ms_per_token:.4f}"
    )

    print(
        f"Cached ms/token        : "
        f"{cached_ms_per_token:.4f}"
    )

    print("\n" + "=" * 65)
    print("RCT 1.3.1 BENCHMARK COMPLETED")
    print("=" * 65)

    print(
        "\nNote: Results are from a synthetic random-weight model "
        "on this machine. They do not establish language-model "
        "quality or general performance across hardware."
    )


if __name__ == "__main__":
    main()
