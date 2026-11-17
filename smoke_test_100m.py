
import gc
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from model.rct_100m import RCT100M, RCT100MConfig


# Training configuration to test
VOCAB_SIZE = 8192
BATCH_SIZE = 1
SEQUENCE_LENGTH = 32
GRADIENT_ACCUMULATION_STEPS = 8
CPU_THREADS = 4

EXPECTED_PARAMETERS = 99_028_992
MIN_AVAILABLE_RAM_GB = 3.5

DATA_FILE = Path("data/rct30/train_expanded_tokens.bin")


def memory_report(label):
    memory = psutil.virtual_memory()
    process = psutil.Process()

    print(f"\n{label}")
    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    print(f"Process RAM:   {process.memory_info().rss / (1024**3):.2f} GB")


def main():
    print("=" * 68)
    print("RCT 100M — GRADIENT ACCUMULATION SMOKE TEST")
    print("=" * 68)

    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(30)
    np.random.seed(30)

    memory_report("INITIAL MEMORY")

    available = psutil.virtual_memory().available / (1024**3)

    if available < MIN_AVAILABLE_RAM_GB:
        raise RuntimeError(
            f"Only {available:.2f} GB RAM available. "
            "Close other applications and retry."
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    tokens = np.memmap(DATA_FILE, dtype=np.uint16, mode="r")

    config = RCT100MConfig(
        vocab_size=8192,
        hidden_size=512,
        num_layers=30,
        num_heads=8,
        intermediate_size=2048,
        max_seq_len=512,
        dropout=0.0,
        tie_embeddings=True,
    )
    config.validate()

    print("\nCreating RCT 100M...")
    model = RCT100M(config).to("cpu")
    model.train()

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameter_count:,}")

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError("Unexpected model parameter count.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        foreach=False,
    )

    optimizer.zero_grad(set_to_none=True)

    losses = []
    start_time = time.perf_counter()

    print("\nRunning 8 microbatches...")

    for microstep in range(GRADIENT_ACCUMULATION_STEPS):
        start = np.random.randint(
            0,
            len(tokens) - SEQUENCE_LENGTH - 1,
        )

        sample = np.array(
            tokens[start:start + SEQUENCE_LENGTH + 1],
            dtype=np.int64,
            copy=True,
        )

        x = torch.from_numpy(sample[:-1]).unsqueeze(0)
        y = torch.from_numpy(sample[1:]).unsqueeze(0)

        logits = model(x)

        raw_loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1),
        )

        losses.append(raw_loss.item())

        (raw_loss / GRADIENT_ACCUMULATION_STEPS).backward()

        print(
            f"Microbatch {microstep + 1}/"
            f"{GRADIENT_ACCUMULATION_STEPS} "
            f"| loss={raw_loss.item():.5f}"
        )

        del logits, raw_loss, x, y, sample

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 68)
    print("GRADIENT ACCUMULATION TEST COMPLETED")
    print("=" * 68)

    print(f"Parameters: {parameter_count:,}")
    print(f"Microbatch size: {BATCH_SIZE}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Average loss: {sum(losses) / len(losses):.6f}")
    print(f"Optimizer update time: {elapsed:.2f} seconds")

    memory_report("AFTER ACCUMULATED OPTIMIZER UPDATE")

    del optimizer, model, tokens
    gc.collect()

    print("\nSmoke test finished. No checkpoint was saved.")


if __name__ == "__main__":
    main()
