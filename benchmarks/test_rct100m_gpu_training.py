
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig


CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "rct100m_continued"
    / "best_model.pt"
)

TRAIN_DATA = PROJECT_DIR / "data" / "rct30" / "train_expanded_tokens.bin"

BATCH_SIZE = 1
SEQUENCE_LENGTH = 32
LEARNING_RATE = 1e-5


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    device = torch.device("cuda")
    torch.manual_seed(42)
    np.random.seed(42)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    config = RCT100MConfig(**checkpoint["model_config"])
    model = RCT100M(config)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model = model.to(device)
    model.train()

    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Starting checkpoint step:", checkpoint["step"])

    print("\nLoading training sample...")

    if not TRAIN_DATA.exists():
        raise FileNotFoundError(TRAIN_DATA)

    data = np.memmap(TRAIN_DATA, dtype=np.uint16, mode="r")

    if len(data) <= SEQUENCE_LENGTH:
        raise RuntimeError("Training dataset is too small.")

    start = 1000
    if start + SEQUENCE_LENGTH >= len(data):
        start = 0

    sample = np.array(
        data[start:start + SEQUENCE_LENGTH],
        dtype=np.int64,
    )

    tokens = torch.from_numpy(sample).unsqueeze(0).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scaler = torch.cuda.amp.GradScaler()

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("\nRunning one GPU training step...")
    start_time = time.perf_counter()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = model(tokens)

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, config.vocab_size),
            tokens[:, 1:].reshape(-1),
        )

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss: {loss.item()}")

    scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    print("\nGPU TRAINING STEP SUCCESSFUL")
    print("Input shape:", tuple(tokens.shape))
    print("Logits shape:", tuple(logits.shape))
    print("Training loss:", round(loss.item(), 6))
    print("Training step time:", round(elapsed, 4), "seconds")
    print(
        "GPU allocated memory:",
        round(torch.cuda.memory_allocated() / 1024**2, 2),
        "MB",
    )
    print(
        "GPU peak memory:",
        round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "MB",
    )
    print(
        "GPU reserved memory:",
        round(torch.cuda.memory_reserved() / 1024**2, 2),
        "MB",
    )

    print("\nThis was one experimental step.")
    print("The original checkpoint was not overwritten.")


if __name__ == "__main__":
    main()
