
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


# =========================================================
# RCT 100M TRAINING CONFIGURATION
# =========================================================

VOCAB_SIZE = 8192
HIDDEN_SIZE = 512
NUM_LAYERS = 30
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048
MAX_SEQ_LEN = 512

BATCH_SIZE = 1
SEQUENCE_LENGTH = 32
GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0

MAX_STEPS = 500
WARMUP_STEPS = 20
EVAL_INTERVAL = 100
EVAL_BATCHES = 64
CHECKPOINT_INTERVAL = 100

CPU_THREADS = 4
SEED = 30

EXPECTED_PARAMETERS = 99_028_992
MIN_AVAILABLE_RAM_GB = 3.5

DATA_DIR = Path("data/rct30")

TRAIN_FILE = DATA_DIR / "train_expanded_tokens.bin"
VALIDATION_FILE = DATA_DIR / "validation_expanded_tokens.bin"

CHECKPOINT_DIR = Path("checkpoints/rct100m")
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"
BEST_CHECKPOINT = CHECKPOINT_DIR / "best_model.pt"


# =========================================================
# MEMORY MONITORING
# =========================================================

def memory_report(label):
    memory = psutil.virtual_memory()
    process = psutil.Process()

    print(f"\n[{label}]")
    print(
        f"Available system RAM: "
        f"{memory.available / (1024 ** 3):.2f} GB"
    )
    print(
        f"Process RAM: "
        f"{process.memory_info().rss / (1024 ** 3):.2f} GB"
    )


def check_memory():
    available = (
        psutil.virtual_memory().available / (1024 ** 3)
    )

    if available < 1.0:
        raise RuntimeError(
            f"Critically low system RAM: {available:.2f} GB. "
            "Stopping to protect the system."
        )


# =========================================================
# RANDOM SEEDS
# =========================================================

def initialize_random_seeds():
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    return np.random.default_rng(SEED)


# =========================================================
# LEARNING RATE SCHEDULER
# =========================================================

def set_learning_rate(optimizer, step):
    if step <= WARMUP_STEPS:
        lr = LEARNING_RATE * step / WARMUP_STEPS
    else:
        progress = (
            (step - WARMUP_STEPS)
            / max(1, MAX_STEPS - WARMUP_STEPS)
        )

        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        lr = MIN_LEARNING_RATE + (
            LEARNING_RATE - MIN_LEARNING_RATE
        ) * cosine

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


# =========================================================
# MODEL CREATION
# =========================================================

def make_model():
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

    config.validate()

    model = RCT100M(config).to("cpu")

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETERS:,} parameters, "
            f"found {parameter_count:,}."
        )

    return model


# =========================================================
# DATA LOADING
# =========================================================

def load_datasets():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(VALIDATION_FILE)

    train_data = np.memmap(
        TRAIN_FILE,
        dtype=np.uint16,
        mode="r",
    )

    validation_data = np.memmap(
        VALIDATION_FILE,
        dtype=np.uint16,
        mode="r",
    )

    if len(train_data) <= SEQUENCE_LENGTH + 1:
        raise ValueError("Training dataset is too small.")

    if len(validation_data) <= SEQUENCE_LENGTH + 1:
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

    sample = np.array(
        data[start:start + SEQUENCE_LENGTH + 1],
        dtype=np.int64,
        copy=True,
    )

    x = torch.from_numpy(sample[:-1]).unsqueeze(0)
    y = torch.from_numpy(sample[1:]).unsqueeze(0)

    return x, y


# =========================================================
# FIXED VALIDATION SAMPLES
# =========================================================

def create_validation_starts(validation_data):
    rng = np.random.default_rng(2026)

    starts = rng.integers(
        0,
        len(validation_data) - SEQUENCE_LENGTH - 1,
        size=EVAL_BATCHES,
    )

    return starts


def evaluate(model, validation_data, eval_starts):
    model.eval()

    losses = []

    with torch.inference_mode():
        for start in eval_starts:
            sample = np.array(
                validation_data[
                    start:start + SEQUENCE_LENGTH + 1
                ],
                dtype=np.int64,
                copy=True,
            )

            x = torch.from_numpy(
                sample[:-1]
            ).unsqueeze(0)

            y = torch.from_numpy(
                sample[1:]
            ).unsqueeze(0)

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

            losses.append(loss.item())

            del logits, loss, x, y, sample

    model.train()

    return float(np.mean(losses))


# =========================================================
# CHECKPOINT SAVING
# =========================================================

def atomic_save(state, destination):
    temporary_path = destination.with_name(
        destination.name + ".tmp"
    )

    torch.save(state, temporary_path)

    # Atomic replacement on the same filesystem.
    os.replace(temporary_path, destination)


def save_checkpoint(
    model,
    optimizer,
    step,
    best_val_loss,
    rng,
    destination,
):
    state = {
        "step": step,

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "best_val_loss": best_val_loss,

        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),

        # Save the independent training sampler RNG.
        "training_rng_state": rng.bit_generator.state,

        "config": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps":
                GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_steps": MAX_STEPS,
        },
    }

    atomic_save(state, destination)

    print(f"Checkpoint saved: {destination}")


def save_best_model(model, step, val_loss):
    state = {
        "step": step,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),

        "config": {
            "vocab_size": VOCAB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "intermediate_size": INTERMEDIATE_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
        },
    }

    atomic_save(state, BEST_CHECKPOINT)

    print(f"Best model saved: {BEST_CHECKPOINT}")


# =========================================================
# CHECKPOINT RESUMING
# =========================================================

def load_checkpoint(model, optimizer, rng):
    if not LATEST_CHECKPOINT.exists():
        print("\nNo previous checkpoint found.")
        print("Starting training from scratch.")

        return 0, float("inf")

    print(f"\nLoading checkpoint: {LATEST_CHECKPOINT}")

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    expected_config = {
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "intermediate_size": INTERMEDIATE_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "min_learning_rate": MIN_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_steps": MAX_STEPS,
    }

    if checkpoint["config"] != expected_config:
        raise RuntimeError(
            "Checkpoint configuration mismatch. "
            "Do not resume with different model or training settings."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    torch.set_rng_state(
        checkpoint["torch_rng_state"]
    )

    np.random.set_state(
        checkpoint["numpy_rng_state"]
    )

    random.setstate(
        checkpoint["python_rng_state"]
    )

    rng.bit_generator.state = (
        checkpoint["training_rng_state"]
    )

    step = int(checkpoint["step"])

    best_val_loss = float(
        checkpoint["best_val_loss"]
    )

    print(f"Resumed at step: {step}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    del checkpoint
    gc.collect()

    return step, best_val_loss


# =========================================================
# MAIN TRAINING
# =========================================================

def main():
    print("=" * 72)
    print("RCT 100M — CPU LANGUAGE MODEL TRAINING")
    print("=" * 72)

    rng = initialize_random_seeds()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"PyTorch: {torch.__version__}")
    print("Device: CPU")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Maximum steps: {MAX_STEPS}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Microbatch size: {BATCH_SIZE}")
    print(f"Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(
        f"Effective batch size: "
        f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
    )

    memory_report("STARTUP")

    available_gb = (
        psutil.virtual_memory().available / (1024 ** 3)
    )

    if available_gb < MIN_AVAILABLE_RAM_GB:
        raise RuntimeError(
            f"Only {available_gb:.2f} GB RAM available. "
            "Close other applications before training."
        )

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    train_data, validation_data = load_datasets()

    eval_starts = create_validation_starts(
        validation_data
    )

    # -----------------------------------------------------
    # CREATE MODEL
    # -----------------------------------------------------

    print("\nInitializing full RCT 100M model...")

    model = make_model()
    model.train()

    memory_report("AFTER MODEL INITIALIZATION")

    # -----------------------------------------------------
    # CREATE OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )

    # -----------------------------------------------------
    # RESUME CHECKPOINT
    # -----------------------------------------------------

    step, best_val_loss = load_checkpoint(
        model,
        optimizer,
        rng,
    )

    if step >= MAX_STEPS:
        print(
            f"\nTraining already reached step {step}. "
            "Increase MAX_STEPS deliberately to continue."
        )
        return

    # -----------------------------------------------------
    # INITIAL VALIDATION
    # -----------------------------------------------------

    if step == 0:
        print("\nEvaluating initial model...")

        initial_val_loss = evaluate(
            model,
            validation_data,
            eval_starts,
        )

        initial_perplexity = math.exp(
            min(initial_val_loss, 20.0)
        )

        print(
            f"Initial validation loss: "
            f"{initial_val_loss:.6f}"
        )

        print(
            f"Initial validation perplexity: "
            f"{initial_perplexity:.2f}"
        )

        best_val_loss = initial_val_loss

        save_best_model(
            model,
            step,
            initial_val_loss,
        )

    # -----------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------

    print("\nStarting training...")

    training_start = time.perf_counter()
    recent_losses = []

    while step < MAX_STEPS:
        check_memory()

        next_step = step + 1

        learning_rate = set_learning_rate(
            optimizer,
            next_step,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        microbatch_losses = []

        for microstep in range(
            GRADIENT_ACCUMULATION_STEPS
        ):
            x, y = sample_batch(
                train_data,
                rng,
            )

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1),
            )

            microbatch_losses.append(
                loss.item()
            )

            scaled_loss = (
                loss / GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

            del logits, loss, scaled_loss, x, y

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )

        optimizer.step()

        step = next_step

        average_loss = float(
            np.mean(microbatch_losses)
        )

        recent_losses.append(average_loss)

        if len(recent_losses) > 20:
            recent_losses.pop(0)

        if step == 1 or step % 10 == 0:
            elapsed = (
                time.perf_counter() - training_start
            )

            print(
                f"Step {step:4d}/{MAX_STEPS} | "
                f"train loss {average_loss:.5f} | "
                f"recent avg {np.mean(recent_losses):.5f} | "
                f"lr {learning_rate:.7f} | "
                f"elapsed {elapsed / 60:.1f} min"
            )

        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if (
            step % EVAL_INTERVAL == 0
            or step == MAX_STEPS
        ):
            print(f"\nValidation at step {step}...")

            val_loss = evaluate(
                model,
                validation_data,
                eval_starts,
            )

            perplexity = math.exp(
                min(val_loss, 20.0)
            )

            print(
                f"Validation loss: {val_loss:.6f}"
            )

            print(
                f"Validation perplexity: {perplexity:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                print("New best validation loss.")

                save_best_model(
                    model,
                    step,
                    val_loss,
                )

            memory_report(
                f"STEP {step}"
            )

        # -------------------------------------------------
        # PERIODIC CHECKPOINT
        # -------------------------------------------------

        if (
            step % CHECKPOINT_INTERVAL == 0
            or step == MAX_STEPS
        ):
            save_checkpoint(
                model,
                optimizer,
                step,
                best_val_loss,
                rng,
                LATEST_CHECKPOINT,
            )

            memory_report(
                f"CHECKPOINT STEP {step}"
            )

    # -----------------------------------------------------
    # FINISH
    # -----------------------------------------------------

    total_elapsed = (
        time.perf_counter() - training_start
    )

    print("\n" + "=" * 72)
    print("RCT 100M TRAINING FINISHED")
    print("=" * 72)

    print(f"Completed steps: {step}")
    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Training time: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print(f"Best model: {BEST_CHECKPOINT}")
    print(f"Latest checkpoint: {LATEST_CHECKPOINT}")

    memory_report("FINAL MEMORY")

    del optimizer
    del model
    del train_data
    del validation_data

    gc.collect()


if __name__ == "__main__":
    main()
