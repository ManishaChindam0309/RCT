
from pathlib import Path
import sys
import time

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    device = torch.device("cuda")
    checkpoint_path = (
        PROJECT_DIR
        / "checkpoints"
        / "rct100m_continued"
        / "best_model.pt"
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", checkpoint_path)

    print("\nLoading checkpoint...")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config_data = checkpoint["model_config"]
    print("Saved configuration:", config_data)

    # Use the saved architecture configuration.
    config = RCT100MConfig(**config_data)
    model = RCT100M(config)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {parameter_count:,}")
    print(f"Training step: {checkpoint['step']}")
    print(f"Validation loss: {checkpoint['val_loss']}")

    print("\nMoving model to GPU...")
    model = model.to(device)

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2

    print(f"GPU memory allocated: {allocated:.2f} MB")
    print(f"GPU memory reserved: {reserved:.2f} MB")

    # Short forward-pass test.
    sequence_length = 32
    tokens = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(1, sequence_length),
        device=device,
        dtype=torch.long,
    )

    print("\nRunning GPU inference...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        logits = model(tokens)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print("GPU inference: SUCCESS")
    print("Input shape:", tuple(tokens.shape))
    print("Output shape:", tuple(logits.shape))
    print(f"Forward-pass time: {elapsed:.4f} seconds")
    print(
        "Peak GPU memory:",
        f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB",
    )

    print("\nGPU test completed successfully.")


if __name__ == "__main__":
    main()
