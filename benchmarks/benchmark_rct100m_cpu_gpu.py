
from pathlib import Path
import sys
import time
import statistics

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig


CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

CPU_THREADS = 4
SEQUENCE_LENGTHS = [32, 128, 256]
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_forward(model, tokens, device):
    timings = []

    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(tokens)

        synchronize(device)

        for _ in range(BENCHMARK_RUNS):
            synchronize(device)
            start = time.perf_counter()

            model(tokens)

            synchronize(device)
            timings.append(time.perf_counter() - start)

    return statistics.median(timings)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    torch.manual_seed(42)
    torch.set_num_threads(CPU_THREADS)

    print("Loading checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    config = RCT100MConfig(**checkpoint["model_config"])

    cpu_model = RCT100M(config)
    cpu_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    cpu_model.eval()

    print("Creating GPU model...")
    gpu_model = RCT100M(config)
    gpu_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    gpu_model = gpu_model.to("cuda")
    gpu_model.eval()

    print("Parameters:", sum(p.numel() for p in cpu_model.parameters()))
    print("CPU threads:", torch.get_num_threads())
    print("GPU:", torch.cuda.get_device_name(0))

    print("\nCPU vs GPU inference benchmark")
    print("=" * 85)

    print(
        f"{'Seq':>6} {'CPU(s)':>12} {'GPU(s)':>12} "
        f"{'Speedup':>12} {'CPU tok/s':>12} {'GPU tok/s':>12}"
    )

    for seq_len in SEQUENCE_LENGTHS:
        torch.manual_seed(100 + seq_len)

        cpu_tokens = torch.randint(
            0,
            config.vocab_size,
            (1, seq_len),
            dtype=torch.long,
        )

        gpu_tokens = cpu_tokens.to("cuda")

        # Compare outputs before timing.
        with torch.inference_mode():
            cpu_output = cpu_model(cpu_tokens)
            gpu_output = gpu_model(gpu_tokens).cpu()

        max_diff = (cpu_output - gpu_output).abs().max().item()

        cpu_top1 = cpu_output.argmax(dim=-1)
        gpu_top1 = gpu_output.argmax(dim=-1)
        agreement = (
            cpu_top1 == gpu_top1
        ).float().mean().item() * 100

        cpu_time = benchmark_forward(
            cpu_model,
            cpu_tokens,
            torch.device("cpu"),
        )

        gpu_time = benchmark_forward(
            gpu_model,
            gpu_tokens,
            torch.device("cuda"),
        )

        speedup = cpu_time / gpu_time
        cpu_tps = seq_len / cpu_time
        gpu_tps = seq_len / gpu_time

        print(
            f"{seq_len:>6} "
            f"{cpu_time:>12.4f} "
            f"{gpu_time:>12.4f} "
            f"{speedup:>11.2f}x "
            f"{cpu_tps:>12.2f} "
            f"{gpu_tps:>12.2f}"
        )

        print(
            f"       Max logit difference: {max_diff:.8f} | "
            f"Top-1 agreement: {agreement:.2f}%"
        )

    print("\nGPU memory allocated:",
          f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("GPU peak memory:",
          f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    print("\nInference benchmark completed.")


if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    main()
