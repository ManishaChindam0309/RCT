
from pathlib import Path
from datetime import datetime
import sys
import json
import time
import gc

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.rct_100m import RCT100M, RCT100MConfig


CHECKPOINT = (
    PROJECT_DIR / "checkpoints" /
    "rct100m_continued" / "best_model.pt"
)

TRAIN_DATA = (
    PROJECT_DIR / "data" / "rct30" /
    "train_expanded_tokens.bin"
)

CONFIGURATIONS = [
    {"batch_size": 1, "sequence_length": 32},
    {"batch_size": 1, "sequence_length": 64},
    {"batch_size": 1, "sequence_length": 128},
]

TOTAL_STEPS = 5
WARMUP_STEPS = 1
LEARNING_RATE = 1e-5


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def run_configuration(config_data, training_data, run_number):
    batch_size = config_data["batch_size"]
    seq_len = config_data["sequence_length"]

    print("\n" + "=" * 65)
    print(f"Configuration: batch={batch_size}, sequence={seq_len}")
    print("=" * 65)

    model = None
    optimizer = None
    scaler = None

    try:
        cleanup()

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        model_config = RCT100MConfig(**checkpoint["model_config"])
        model = RCT100M(model_config)
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        del checkpoint
        model = model.cuda()
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        scaler = torch.amp.GradScaler("cuda")

        torch.cuda.reset_peak_memory_stats()
        timings = []
        losses = []

        for step in range(TOTAL_STEPS):
            # Deterministic, different data window per step.
            start = (
                10000
                + run_number * 100000
                + step * (seq_len + 1)
            )

            batch = []

            for b in range(batch_size):
                offset = start + b * (seq_len + 1)

                sample = np.array(
                    training_data[offset:offset + seq_len + 1],
                    dtype=np.int64,
                )

                if len(sample) != seq_len + 1:
                    raise RuntimeError("Insufficient training data.")

                batch.append(sample)

            batch = torch.from_numpy(
                np.stack(batch)
            ).cuda()

            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_start = time.perf_counter()

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(inputs)

                loss = F.cross_entropy(
                    logits.reshape(-1, model_config.vocab_size),
                    targets.reshape(-1),
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step + 1}"
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - step_start

            timings.append(elapsed)
            losses.append(float(loss.detach().item()))

            print(
                f"Step {step + 1}/{TOTAL_STEPS} | "
                f"loss={losses[-1]:.5f} | "
                f"time={elapsed:.4f}s"
            )

        measured = timings[WARMUP_STEPS:]
        average_time = sum(measured) / len(measured)

        tokens_per_step = batch_size * seq_len
        tokens_per_second = tokens_per_step / average_time

        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024**2
        )

        reserved_memory = (
            torch.cuda.max_memory_reserved() / 1024**2
        )

        result = {
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "successful_steps": TOTAL_STEPS,
            "average_step_time_seconds": average_time,
            "tokens_per_second": tokens_per_second,
            "last_training_loss": losses[-1],
            "peak_gpu_memory_mb": peak_memory,
            "peak_reserved_memory_mb": reserved_memory,
            "status": "SUCCESS",
        }

        print("\nConfiguration successful.")
        print(f"Average measured step: {average_time:.4f}s")
        print(f"Throughput: {tokens_per_second:.2f} tokens/s")
        print(f"Peak GPU memory: {peak_memory:.2f} MB")

        return result

    except torch.cuda.OutOfMemoryError as error:
        print("\nCUDA OUT OF MEMORY:", error)
        return {
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "status": "OUT_OF_MEMORY",
        }

    except Exception as error:
        print("\nCONFIGURATION FAILED:", repr(error))
        return {
            "batch_size": batch_size,
            "sequence_length": seq_len,
            "status": "FAILED",
            "error": repr(error),
        }

    finally:
        del model, optimizer, scaler
        cleanup()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    if not TRAIN_DATA.exists():
        raise FileNotFoundError(TRAIN_DATA)

    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3, 2
    ), "GB")

    print("\nLoading training data...")
    data = np.memmap(
        TRAIN_DATA,
        dtype=np.uint16,
        mode="r",
    )

    results = []

    for index, config in enumerate(CONFIGURATIONS):
        result = run_configuration(
            config,
            data,
            index + 1,
        )
        results.append(result)

    output_dir = (
        PROJECT_DIR / "results" /
        "rct100m_gpu_training" /
        datetime.now().strftime("run_%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(CHECKPOINT),
        "precision": "FP16 mixed precision",
        "learning_rate": LEARNING_RATE,
        "total_steps_per_configuration": TOTAL_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "results": results,
    }

    report_path = output_dir / "results.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 65)
    print("FINAL RESULTS")
    print("=" * 65)

    for result in results:
        print(result)

    print("\nReport saved:", report_path)


if __name__ == "__main__":
    main()
