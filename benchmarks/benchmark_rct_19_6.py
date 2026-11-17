
import os
import sys
import time
import random
import hashlib
import statistics

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from model.reusable_block import ReusableFFN


def standard_ffn(ffn, x):
    return ffn.network(x)


def torch_deduplicated_ffn(ffn, x):

    unique_tokens, inverse = torch.unique(
        x,
        dim=1,
        return_inverse=True
    )

    unique_output = ffn.network(unique_tokens)

    return unique_output.index_select(1, inverse)


def hash_deduplicated_ffn(ffn, x):

    tokens = x[0].contiguous()

    groups = {}
    unique_rows = []
    inverse = []

    for index in range(tokens.shape[0]):

        row = tokens[index]

        raw_bytes = row.numpy().tobytes()

        fingerprint = hashlib.blake2b(
            raw_bytes,
            digest_size=16
        ).digest()

        group_index = groups.get(fingerprint)

        if group_index is None:

            group_index = len(unique_rows)

            groups[fingerprint] = group_index

            unique_rows.append(row)

        else:

            # Verify exact equality before reusing.
            if not torch.equal(
                unique_rows[group_index],
                row
            ):
                raise RuntimeError(
                    "Fingerprint collision detected"
                )

        inverse.append(group_index)

    unique_tokens = torch.stack(
        unique_rows,
        dim=0
    ).unsqueeze(0)

    inverse = torch.tensor(
        inverse,
        dtype=torch.long,
        device=x.device
    )

    unique_output = ffn.network(
        unique_tokens
    )

    return unique_output.index_select(
        1,
        inverse
    )


def measure(function, trials=40):

    timings = []

    for _ in range(trials):

        start = time.perf_counter()

        function()

        elapsed = (
            time.perf_counter() - start
        ) * 1000

        timings.append(elapsed)

    return statistics.median(timings)


def make_input(sequence_length, unique_count, hidden_size):

    unique_tokens = torch.randn(
        1,
        unique_count,
        hidden_size
    )

    indices = (
        torch.arange(sequence_length) % unique_count
    )

    indices = indices[
        torch.randperm(sequence_length)
    ]

    return unique_tokens.index_select(
        1,
        indices
    )


def main():

    torch.manual_seed(42)
    random.seed(42)

    torch.set_num_threads(4)

    hidden_size = 256

    sequence_lengths = [64, 128, 256]

    duplicate_ratios = [
        0.50,
        0.625,
        0.75,
        0.875
    ]

    rounds = 3
    trials = 40

    ffn = ReusableFFN(
        hidden_size=hidden_size,
        expansion=4
    )

    ffn.eval()

    print("=" * 85)
    print("RCT 1.9.6 EXACT HASH-BASED REUSE BENCHMARK")
    print("=" * 85)

    print("Hidden size:", hidden_size)
    print("CPU threads:", torch.get_num_threads())
    print("Rounds:", rounds)
    print("Trials per round:", trials)

    with torch.inference_mode():

        for sequence_length in sequence_lengths:

            for target_ratio in duplicate_ratios:

                unique_count = max(
                    1,
                    round(
                        sequence_length *
                        (1.0 - target_ratio)
                    )
                )

                round_results = []

                for round_index in range(rounds):

                    x = make_input(
                        sequence_length,
                        unique_count,
                        hidden_size
                    )

                    standard_output = standard_ffn(
                        ffn,
                        x
                    )

                    torch_output = torch_deduplicated_ffn(
                        ffn,
                        x
                    )

                    hash_output = hash_deduplicated_ffn(
                        ffn,
                        x
                    )

                    torch_correct = torch.allclose(
                        standard_output,
                        torch_output,
                        rtol=1e-5,
                        atol=1e-6
                    )

                    hash_correct = torch.allclose(
                        standard_output,
                        hash_output,
                        rtol=1e-5,
                        atol=1e-6
                    )

                    if not torch_correct or not hash_correct:
                        raise AssertionError(
                            "Numerical correctness failed"
                        )

                    # Warmup
                    for _ in range(10):

                        standard_ffn(ffn, x)

                        torch_deduplicated_ffn(
                            ffn,
                            x
                        )

                        hash_deduplicated_ffn(
                            ffn,
                            x
                        )

                    standard_times = []
                    torch_times = []
                    hash_times = []

                    for _ in range(trials):

                        order = [
                            "standard",
                            "torch",
                            "hash"
                        ]

                        random.shuffle(order)

                        for method in order:

                            if method == "standard":

                                elapsed = measure(
                                    lambda: standard_ffn(ffn, x),
                                    trials=1
                                )

                                standard_times.append(elapsed)

                            elif method == "torch":

                                elapsed = measure(
                                    lambda: torch_deduplicated_ffn(
                                        ffn,
                                        x
                                    ),
                                    trials=1
                                )

                                torch_times.append(elapsed)

                            else:

                                elapsed = measure(
                                    lambda: hash_deduplicated_ffn(
                                        ffn,
                                        x
                                    ),
                                    trials=1
                                )

                                hash_times.append(elapsed)

                    standard_ms = statistics.median(
                        standard_times
                    )

                    torch_ms = statistics.median(
                        torch_times
                    )

                    hash_ms = statistics.median(
                        hash_times
                    )

                    torch_speedup = (
                        standard_ms / torch_ms
                    )

                    hash_speedup = (
                        standard_ms / hash_ms
                    )

                    hash_vs_torch = (
                        torch_ms / hash_ms
                    )

                    round_results.append({
                        "torch_speedup": torch_speedup,
                        "hash_speedup": hash_speedup,
                        "hash_vs_torch": hash_vs_torch
                    })

                    print("\n" + "-" * 85)

                    print(
                        f"Sequence: {sequence_length} | "
                        f"Unique: {unique_count} | "
                        f"Duplicates: {target_ratio * 100:.1f}% | "
                        f"Round: {round_index + 1}"
                    )

                    print(
                        f"Standard FFN: {standard_ms:.5f} ms"
                    )

                    print(
                        f"PyTorch dedup: {torch_ms:.5f} ms"
                    )

                    print(
                        f"Hash dedup: {hash_ms:.5f} ms"
                    )

                    print(
                        f"PyTorch speedup: {torch_speedup:.3f}x"
                    )

                    print(
                        f"Hash speedup: {hash_speedup:.3f}x"
                    )

                    print(
                        f"Hash vs PyTorch: {hash_vs_torch:.3f}x"
                    )

                    print(
                        "Numerical correctness: PASSED"
                    )

                print("\nWORKLOAD SUMMARY")

                for key in [
                    "torch_speedup",
                    "hash_speedup",
                    "hash_vs_torch"
                ]:

                    values = [
                        result[key]
                        for result in round_results
                    ]

                    print(
                        f"{key}: "
                        f"{statistics.median(values):.3f}x"
                    )

    print("\n" + "=" * 85)
    print("RCT 1.9.6 BENCHMARK COMPLETED")
    print("=" * 85)


if __name__ == "__main__":
    main()
