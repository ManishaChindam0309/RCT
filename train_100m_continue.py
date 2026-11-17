
import gc
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from model.rct_100m import RCT100M, RCT100MConfig


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data" / "rct30"

ORIGINAL_CHECKPOINT = (
    PROJECT_DIR / "checkpoints" / "rct100m" / "latest.pt"
)

CONTINUATION_DIR = (
    PROJECT_DIR / "checkpoints" / "rct100m_continued"
)

LATEST_CHECKPOINT = CONTINUATION_DIR / "latest.pt"
BEST_CHECKPOINT = CONTINUATION_DIR / "best_model.pt"

TRAIN_DATA_PATH = DATA_DIR / "train_expanded_tokens.bin"
VALIDATION_DATA_PATH = DATA_DIR / "validation_expanded_tokens.bin"


# Model architecture
VOCAB_SIZE = 8192
HIDDEN_SIZE = 512
NUM_LAYERS = 30
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048
MAX_SEQ_LEN = 512

EXPECTED_PARAMETERS = 99_028_992


# Training configuration
BATCH_SIZE = 1
SEQUENCE_LENGTH = 32
GRAD_ACCUM_STEPS = 8

START_LR = 2e-5
MIN_LR = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

STARTING_STEP = 500
MAX_TOTAL_STEPS = 1500

EVAL_INTERVAL = 100
EVAL_BATCHES = 64
CHECKPOINT_INTERVAL = 100

CPU_THREADS = 4
SEED = 30

MIN_AVAILABLE_RAM_GB = 1.0


# ============================================================
# MEMORY MANAGEMENT
# ============================================================

def memory_report(label="MEMORY"):

    memory = psutil.virtual_memory()
    process = psutil.Process()

    available = memory.available / (1024 ** 3)
    process_ram = process.memory_info().rss / (1024 ** 3)

    print(f"\n[{label}]")
    print(f"Available system RAM: {available:.2f} GB")
    print(f"Process RAM: {process_ram:.2f} GB")

    return available


def check_memory():

    available = psutil.virtual_memory().available / (1024 ** 3)

    if available < MIN_AVAILABLE_RAM_GB:
        raise MemoryError(
            f"Available RAM is too low: {available:.2f} GB. "
            "Training stopped to protect the system."
        )


# ============================================================
# RANDOM SEEDS
# ============================================================

def initialize_random_seeds():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    return np.random.default_rng(SEED)


# ============================================================
# MODEL
# ============================================================

def create_model():

    config = RCT100MConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        intermediate_size=INTERMEDIATE_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dropout=0.0,
        tie_embeddings=True,
    )

    model = RCT100M(config)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(
            f"Expected {EXPECTED_PARAMETERS:,} parameters, "
            f"got {parameter_count:,}."
        )

    return model


# ============================================================
# DATA
# ============================================================

def load_datasets():

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(TRAIN_DATA_PATH)

    if not VALIDATION_DATA_PATH.exists():
        raise FileNotFoundError(VALIDATION_DATA_PATH)

    train_data = np.memmap(
        TRAIN_DATA_PATH,
        dtype=np.uint16,
        mode="r",
    )

    validation_data = np.memmap(
        VALIDATION_DATA_PATH,
        dtype=np.uint16,
        mode="r",
    )

    if len(train_data) <= SEQUENCE_LENGTH:
        raise ValueError("Training dataset is too small.")

    if len(validation_data) <= SEQUENCE_LENGTH:
        raise ValueError("Validation dataset is too small.")

    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(validation_data):,}")

    return train_data, validation_data


def sample_batch(data, rng):

    start = int(
        rng.integers(
            0,
            len(data) - SEQUENCE_LENGTH - 1,
        )
    )

    chunk = np.array(
        data[start:start + SEQUENCE_LENGTH + 1],
        dtype=np.int64,
        copy=True,
    )

    x = torch.from_numpy(chunk[:-1]).unsqueeze(0)
    y = torch.from_numpy(chunk[1:]).unsqueeze(0)

    return x, y


# ============================================================
# VALIDATION
# ============================================================

def create_validation_starts(validation_data):

    rng = np.random.default_rng(2026)

    return rng.integers(
        0,
        len(validation_data) - SEQUENCE_LENGTH - 1,
        size=EVAL_BATCHES,
    )


def evaluate(model, validation_data, starts):

    model.eval()

    total_loss = 0.0
    total_positions = 0

    with torch.inference_mode():

        for start in starts:

            chunk = np.array(
                validation_data[
                    int(start):int(start) + SEQUENCE_LENGTH + 1
                ],
                dtype=np.int64,
                copy=True,
            )

            x = torch.from_numpy(chunk[:-1]).unsqueeze(0)
            y = torch.from_numpy(chunk[1:]).unsqueeze(0)

            logits = model(x)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if isinstance(logits, dict):
                logits = logits["logits"]

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_positions += y.numel()

    average_loss = total_loss / total_positions
    perplexity = math.exp(min(average_loss, 20))

    model.train()

    return average_loss, perplexity


# ============================================================
# LEARNING RATE
# ============================================================

def get_learning_rate(global_step):

    first_step = STARTING_STEP + 1
    total_steps = MAX_TOTAL_STEPS - STARTING_STEP

    progress = (
        (global_step - first_step) /
        max(1, total_steps - 1)
    )

    progress = min(max(progress, 0.0), 1.0)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return MIN_LR + (START_LR - MIN_LR) * cosine


def set_learning_rate(optimizer, learning_rate):

    for group in optimizer.param_groups:
        group["lr"] = learning_rate


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def atomic_save(data, path):

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(data, temporary_path)

    os.replace(temporary_path, path)


def get_checkpoint_value(checkpoint, possible_keys):

    for key in possible_keys:
        if key in checkpoint:
            return checkpoint[key]

    raise KeyError(
        f"Could not find any of these checkpoint keys: "
        f"{possible_keys}. Available keys: {list(checkpoint.keys())}"
    )


def save_best_model(model, step, val_loss):

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "model_config": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
        },
    }

    atomic_save(checkpoint, BEST_CHECKPOINT)

    print(f"Best continuation model saved: {BEST_CHECKPOINT}")


def save_training_checkpoint(
    model,
    optimizer,
    step,
    best_val_loss,
    rng,
):

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,

        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),

        "training_rng_state": rng.bit_generator.state,

        "training_config": {
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM_STEPS,
            "start_lr": START_LR,
            "min_lr": MIN_LR,
            "max_total_steps": MAX_TOTAL_STEPS,
        },
    }

    atomic_save(checkpoint, LATEST_CHECKPOINT)

    print(f"Continuation checkpoint saved: {LATEST_CHECKPOINT}")


def load_training_checkpoint(model, optimizer, rng):

    if LATEST_CHECKPOINT.exists():

        checkpoint_path = LATEST_CHECKPOINT
        print("\nResuming continuation from its latest checkpoint.")

    elif ORIGINAL_CHECKPOINT.exists():

        checkpoint_path = ORIGINAL_CHECKPOINT
        print("\nLoading original RCT 100M step-500 checkpoint.")

    else:

        raise FileNotFoundError(
            "Neither original nor continuation checkpoint exists."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_state = get_checkpoint_value(
        checkpoint,
        [
            "model_state_dict",
            "model_state",
            "state_dict",
            "model",
        ],
    )

    optimizer_state = get_checkpoint_value(
        checkpoint,
        [
            "optimizer_state_dict",
            "optimizer_state",
            "optimizer",
        ],
    )

    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

    step = int(checkpoint["step"])

    if step < STARTING_STEP:
        raise ValueError(
            f"Checkpoint step {step} is earlier than "
            f"the expected starting step {STARTING_STEP}."
        )

    if step > MAX_TOTAL_STEPS:
        raise ValueError(
            f"Checkpoint step {step} exceeds target "
            f"step {MAX_TOTAL_STEPS}."
        )

    if "training_rng_state" in checkpoint:
        rng.bit_generator.state = checkpoint["training_rng_state"]

    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])

    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])

    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    if checkpoint_path == LATEST_CHECKPOINT:
        best_val_loss = float(
            checkpoint.get("best_val_loss", float("inf"))
        )
    else:
        # The continuation uses its own validation baseline.
        best_val_loss = float("inf")

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Resuming at global step: {step}")
    print(f"Previous best continuation loss: {best_val_loss:.6f}")

    del checkpoint
    gc.collect()

    return step, best_val_loss


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 72)
    print("RCT 100M — CONTINUED TRAINING")
    print("=" * 72)

    torch.set_num_threads(CPU_THREADS)

    rng = initialize_random_seeds()

    print(f"PyTorch: {torch.__version__}")
    print("Device: CPU")
    print(f"CPU threads: {CPU_THREADS}")
    print(f"Starting step: {STARTING_STEP}")
    print(f"Target global step: {MAX_TOTAL_STEPS}")
    print(f"Additional steps: {MAX_TOTAL_STEPS - STARTING_STEP}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Continuation learning rate: {START_LR} to {MIN_LR}")

    CONTINUATION_DIR.mkdir(parents=True, exist_ok=True)

    memory_report("STARTUP")
    check_memory()

    train_data, validation_data = load_datasets()

    validation_starts = create_validation_starts(validation_data)

    print("\nInitializing RCT 100M model...")

    model = create_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=START_LR,
        weight_decay=WEIGHT_DECAY,
    )

    memory_report("AFTER MODEL INITIALIZATION")

    step, best_val_loss = load_training_checkpoint(
        model,
        optimizer,
        rng,
    )

    memory_report("AFTER CHECKPOINT LOADING")

    if step == STARTING_STEP and best_val_loss == float("inf"):

        print("\nEvaluating continuation baseline...")

        baseline_loss, baseline_perplexity = evaluate(
            model,
            validation_data,
            validation_starts,
        )

        best_val_loss = baseline_loss

        print(f"Baseline validation loss: {baseline_loss:.6f}")
        print(f"Baseline perplexity: {baseline_perplexity:.2f}")

        save_best_model(
            model,
            step,
            baseline_loss,
        )

    elif BEST_CHECKPOINT.exists():

        print(f"\nExisting best continuation model: {BEST_CHECKPOINT}")

    print("\nStarting continuation training...")

    model.train()

    start_time = time.time()
    recent_losses = []

    for global_step in range(step + 1, MAX_TOTAL_STEPS + 1):

        check_memory()

        learning_rate = get_learning_rate(global_step)
        set_learning_rate(optimizer, learning_rate)

        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0

        for micro_step in range(GRAD_ACCUM_STEPS):

            x, y = sample_batch(train_data, rng)

            logits = model(x)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if isinstance(logits, dict):
                logits = logits["logits"]

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

            (loss / GRAD_ACCUM_STEPS).backward()

            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        step_loss = accumulated_loss / GRAD_ACCUM_STEPS

        recent_losses.append(step_loss)

        if len(recent_losses) > 20:
            recent_losses.pop(0)

        if global_step % 10 == 0 or global_step == MAX_TOTAL_STEPS:

            recent_average = sum(recent_losses) / len(recent_losses)

            elapsed_minutes = (time.time() - start_time) / 60

            print(
                f"Step {global_step:4d}/{MAX_TOTAL_STEPS} | "
                f"train loss {step_loss:.5f} | "
                f"recent avg {recent_average:.5f} | "
                f"lr {learning_rate:.8f} | "
                f"elapsed {elapsed_minutes:.1f} min"
            )

        if (
            global_step % EVAL_INTERVAL == 0
            or global_step == MAX_TOTAL_STEPS
        ):

            print(f"\nValidation at step {global_step}...")

            val_loss, perplexity = evaluate(
                model,
                validation_data,
                validation_starts,
            )

            print(f"Validation loss: {val_loss:.6f}")
            print(f"Validation perplexity: {perplexity:.2f}")

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                print("New best continuation validation loss.")

                save_best_model(
                    model,
                    global_step,
                    val_loss,
                )

            memory_report(f"STEP {global_step}")

        if (
            global_step % CHECKPOINT_INTERVAL == 0
            or global_step == MAX_TOTAL_STEPS
        ):

            save_training_checkpoint(
                model,
                optimizer,
                global_step,
                best_val_loss,
                rng,
            )

            memory_report(f"CHECKPOINT STEP {global_step}")

    total_minutes = (time.time() - start_time) / 60

    print("\n" + "=" * 72)
    print("RCT 100M CONTINUATION FINISHED")
    print("=" * 72)

    print(f"Completed global step: {MAX_TOTAL_STEPS}")
    print(f"Best continuation validation loss: {best_val_loss:.6f}")
    print(f"Continuation time: {total_minutes:.2f} minutes")
    print(f"Best model: {BEST_CHECKPOINT}")
    print(f"Latest checkpoint: {LATEST_CHECKPOINT}")

    memory_report("FINAL MEMORY")

    del model
    del optimizer
    del train_data
    del validation_data

    gc.collect()


if __name__ == "__main__":
    main()
