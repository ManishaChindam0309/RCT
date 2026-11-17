
import random
import statistics
import time

import torch
import torch.nn as nn


HIDDEN = 256
EXPANSION = 4
THREADS = 4
ROUNDS = 5
TRIALS = 40

SEQUENCES = [64, 128, 256]

DUPLICATES = [
    0.40, 0.45, 0.50, 0.55, 0.60,
    0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]


class FFN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(HIDDEN, HIDDEN * EXPANSION),
            nn.GELU(),
            nn.Linear(HIDDEN * EXPANSION, HIDDEN)
        )

    def forward(self, x):
        return self.network(x)


class DedupFFN(FFN):
    def __init__(self, original):
        super().__init__()
        self.network.load_state_dict(
            original.network.state_dict()
        )

    def forward(self, x):
        batch, seq, hidden = x.shape

        unique, inverse = torch.unique(
            x.reshape(batch * seq, hidden),
            dim=0,
            return_inverse=True
        )

        output = self.network(unique)

        return output.index_select(
            0, inverse
        ).reshape(batch, seq, hidden)


def make_input(seq, duplicate_ratio, seed):

    generator = torch.Generator().manual_seed(seed)

    unique_count = max(
        1,
        round(seq * (1 - duplicate_ratio))
    )

    unique = torch.randn(
        unique_count,
        HIDDEN,
        generator=generator
    )

    indices = torch.randint(
        unique_count,
        (seq,),
        generator=generator
    )

    indices[:unique_count] = torch.arange(
        unique_count
    )

    permutation = torch.randperm(
        seq,
        generator=generator
    )

    indices = indices[permutation]

    return unique[indices].unsqueeze(0), unique_count


def measure(fn):
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def benchmark(standard, dedup, x):

    standard_times = []
    dedup_times = []

    with torch.inference_mode():

        for _ in range(10):
            standard(x)
            dedup(x)

        for trial in range(TRIALS):

            if random.random() < 0.5:

                standard_times.append(
                    measure(lambda: standard(x))
                )

                dedup_times.append(
                    measure(lambda: dedup(x))
                )

            else:

                dedup_times.append(
                    measure(lambda: dedup(x))
                )

                standard_times.append(
                    measure(lambda: standard(x))
                )

    return (
        statistics.median(standard_times),
        statistics.median(dedup_times)
    )


def main():

    torch.manual_seed(198)
    random.seed(198)
    torch.set_num_threads(THREADS)

    standard = FFN().eval()
    dedup = DedupFFN(standard).eval()

    print("=" * 85)
    print("RCT 1.9.8 DEDUPLICATION BREAK-EVEN EXPERIMENT")
    print("=" * 85)

    print("Hidden:", HIDDEN)
    print("Threads:", THREADS)
    print("Rounds:", ROUNDS)
    print("Trials:", TRIALS)

    for seq in SEQUENCES:

        print("\n" + "#" * 85)
        print("SEQUENCE LENGTH:", seq)
        print("#" * 85)

        for ratio in DUPLICATES:

            speedups = []
            reductions = []
            faster = 0

            for round_index in range(ROUNDS):

                x, unique_count = make_input(
                    seq,
                    ratio,
                    198000 + seq * 100
                    + round_index * 10
                    + int(ratio * 1000)
                )

                with torch.inference_mode():

                    expected = standard(x)
                    actual = dedup(x)

                if not torch.allclose(
                    expected,
                    actual,
                    rtol=1e-4,
                    atol=1e-5
                ):
                    raise RuntimeError(
                        "Numerical correctness failed"
                    )

                standard_ms, dedup_ms = benchmark(
                    standard,
                    dedup,
                    x
                )

                speedup = standard_ms / dedup_ms

                reduction = (
                    1 - dedup_ms / standard_ms
                ) * 100

                speedups.append(speedup)
                reductions.append(reduction)

                if speedup > 1:
                    faster += 1

            median_speedup = statistics.median(
                speedups
            )

            median_reduction = statistics.median(
                reductions
            )

            print(
                f"Duplicates {ratio * 100:5.1f}% | "
                f"Unique {round(seq * (1-ratio)):3d} | "
                f"Speedup {median_speedup:.3f}x | "
                f"Reduction {median_reduction:7.2f}% | "
                f"Faster {faster}/{ROUNDS}"
            )

    print("\n" + "=" * 85)
    print("RCT 1.9.8 COMPLETED")
    print("=" * 85)


if __name__ == "__main__":
    main()
