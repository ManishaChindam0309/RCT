
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import TrainingConfig
from model.rct_100m import RCT100M, RCT100MConfig


def load_tokens(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    tokens = np.fromfile(path, dtype=np.uint16).astype(np.int64)

    if len(tokens) < 100:
        raise ValueError(f"Not enough tokens in {path}")

    return torch.from_numpy(tokens)


def get_batch(tokens, batch_size, sequence_length):
    max_start = len(tokens) - sequence_length - 1

    if max_start <= 0:
        raise ValueError("Dataset is too small for requested sequence length.")

    starts = torch.randint(
        0,
        max_start,
        (batch_size,)
    )

    x = torch.stack([
        tokens[i:i + sequence_length]
        for i in starts.tolist()
    ])

    y = torch.stack([
        tokens[i + 1:i + sequence_length + 1]
        for i in starts.tolist()
    ])

    return x, y


@torch.inference_mode()
def evaluate(model, tokens, config):
    model.eval()

    losses = []

    for _ in range(config.eval_batches):
        x, y = get_batch(
            tokens,
            config.batch_size,
            config.sequence_length
        )

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1)
        )

        losses.append(loss.item())

    model.train()

    return sum(losses) / len(losses)


def save_checkpoint(path, model, optimizer, step, best_loss, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_loss": best_loss,
        "training_config": vars(config),
    }, path)


def main():
    config = TrainingConfig()
    config.validate()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    torch.set_num_threads(config.cpu_threads)

    train_tokens = load_tokens(config.train_data)
    validation_tokens = load_tokens(config.validation_data)

    print("=" * 65)
    print("RCT 3.0 CPU PILOT TRAINING")
    print("=" * 65)

    print("Training tokens:", f"{len(train_tokens):,}")
    print("Validation tokens:", f"{len(validation_tokens):,}")

    model_config = RCT100MConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        tie_embeddings=True,
    )

    model = RCT100M(model_config)
    model.train()

    parameters = model.parameter_count()

    print("Model parameters:", f"{parameters:,}")
    print("Batch size:", config.batch_size)
    print("Sequence length:", config.sequence_length)
    print("Maximum steps:", config.max_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_checkpoint = checkpoint_dir / "latest.pt"
    best_checkpoint = checkpoint_dir / "best.pt"

    start_step = 0
    best_validation_loss = float("inf")

    if latest_checkpoint.exists():
        print("\nExisting checkpoint detected.")
        print("For a fresh run, move or rename:", latest_checkpoint)
        print("This initial script starts a fresh run.")

    print("\nStarting training...")
    start_time = time.perf_counter()

    for step in range(1, config.max_steps + 1):

        optimizer.zero_grad(set_to_none=True)

        accumulated_loss = 0.0

        for _ in range(config.gradient_accumulation_steps):

            x, y = get_batch(
                train_tokens,
                config.batch_size,
                config.sequence_length,
            )

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1),
            )

            (loss / config.gradient_accumulation_steps).backward()

            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.gradient_clip,
        )

        optimizer.step()

        if step == 1 or step % config.eval_interval == 0:

            validation_loss = evaluate(
                model,
                validation_tokens,
                config,
            )

            elapsed = time.perf_counter() - start_time

            print(
                f"Step {step:4d}/{config.max_steps} | "
                f"Train loss: {accumulated_loss / config.gradient_accumulation_steps:.4f} | "
                f"Validation loss: {validation_loss:.4f} | "
                f"Elapsed: {elapsed:.1f}s"
            )

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

                save_checkpoint(
                    best_checkpoint,
                    model,
                    optimizer,
                    step,
                    best_validation_loss,
                    config,
                )

                print("  Best checkpoint saved.")

            save_checkpoint(
                latest_checkpoint,
                model,
                optimizer,
                step,
                best_validation_loss,
                config,
            )

    elapsed = time.perf_counter() - start_time

    print("\nTRAINING FINISHED")
    print("-" * 65)
    print("Final step:", config.max_steps)
    print("Best validation loss:", round(best_validation_loss, 4))
    print("Elapsed seconds:", round(elapsed, 2))
    print("Latest checkpoint:", latest_checkpoint)
    print("Best checkpoint:", best_checkpoint)


if __name__ == "__main__":
    main()
