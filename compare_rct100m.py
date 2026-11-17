
import gc
import math
import random
from pathlib import Path

import numpy as np
import psutil
import sentencepiece as spm
import torch
import torch.nn.functional as F

from model.rct_100m import RCT100M, RCT100MConfig


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINTS = {
    "Original step-500": (
        PROJECT_DIR / "checkpoints" / "rct100m" / "best_model.pt"
    ),
    "Continued step-1500": (
        PROJECT_DIR / "checkpoints" / "rct100m_continued" / "best_model.pt"
    ),
}

VALIDATION_PATH = (
    PROJECT_DIR / "data" / "rct30" / "validation_expanded_tokens.bin"
)

TOKENIZER_PATH = (
    PROJECT_DIR / "data" / "rct30" / "rct30_8k.model"
)

VOCAB_SIZE = 8192
HIDDEN_SIZE = 512
NUM_LAYERS = 30
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048
MAX_SEQ_LEN = 512

SEQUENCE_LENGTH = 32
EVALUATION_BATCHES = 256
CPU_THREADS = 4

GENERATION_TOKENS = 80
TEMPERATURE = 0.8
TOP_K = 40

SEED = 42

PROMPTS = [
    "Alice was beginning to get",
    "It was a dark and stormy night",
    "The detective entered the room",
    "Once upon a time",
    "The old man looked at the",
]


# ============================================================
# MEMORY
# ============================================================

def report_memory():

    memory = psutil.virtual_memory()
    process = psutil.Process()

    print(f"Available RAM: {memory.available / (1024**3):.2f} GB")
    print(f"Process RAM: {process.memory_info().rss / (1024**3):.2f} GB")


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

    return RCT100M(config)


def extract_state_dict(checkpoint):

    possible_keys = [
        "model_state_dict",
        "model_state",
        "state_dict",
        "model",
    ]

    for key in possible_keys:
        if key in checkpoint:
            return checkpoint[key]

    raise KeyError(
        f"Model weights not found. Checkpoint keys: {list(checkpoint.keys())}"
    )


def load_checkpoint(path):

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"\nLoading: {path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    model = create_model()

    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    step = checkpoint.get("step", "Unknown")
    saved_val_loss = checkpoint.get("val_loss", None)

    print(f"Checkpoint step: {step}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if saved_val_loss is not None:
        print(f"Saved validation loss: {saved_val_loss:.6f}")

    del checkpoint
    del state_dict
    gc.collect()

    return model, step


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_PATH)
    )

    if tokenizer.get_piece_size() != VOCAB_SIZE:
        raise ValueError("Tokenizer vocabulary mismatch.")

    return tokenizer


# ============================================================
# FIXED VALIDATION POSITIONS
# ============================================================

def prepare_validation_starts():

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_PATH}"
        )

    validation_data = np.memmap(
        VALIDATION_PATH,
        dtype=np.uint16,
        mode="r",
    )

    total_tokens = len(validation_data)

    if total_tokens <= SEQUENCE_LENGTH + 1:
        raise ValueError("Validation dataset is too small.")

    rng = np.random.default_rng(SEED)

    starts = rng.integers(
        0,
        total_tokens - SEQUENCE_LENGTH - 1,
        size=EVALUATION_BATCHES,
    )

    return validation_data, starts


# ============================================================
# VALIDATION EVALUATION
# ============================================================

def evaluate_model(model, validation_data, starts, model_name):

    print("\n" + "=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    total_loss = 0.0
    total_positions = 0

    with torch.inference_mode():

        for index, start in enumerate(starts, start=1):

            chunk = np.array(
                validation_data[
                    start:start + SEQUENCE_LENGTH + 1
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

            if index % 32 == 0:
                print(
                    f"Evaluated batches: {index}/{EVALUATION_BATCHES}"
                )

    average_loss = total_loss / total_positions
    perplexity = math.exp(min(average_loss, 20))

    print(f"\nValidation loss: {average_loss:.6f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Evaluated token positions: {total_positions:,}")

    return average_loss, perplexity


# ============================================================
# TEXT GENERATION
# ============================================================

def generate_text(model, tokenizer, prompt, seed):

    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        return ""

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
    )

    eos_id = tokenizer.eos_id()

    # Same random seed for corresponding model comparisons.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    with torch.inference_mode():

        for _ in range(GENERATION_TOKENS):

            context = input_ids[:, -SEQUENCE_LENGTH:]

            logits = model(context)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if isinstance(logits, dict):
                logits = logits["logits"]

            next_logits = logits[:, -1, :] / TEMPERATURE

            values, indices = torch.topk(
                next_logits,
                k=TOP_K,
                dim=-1,
            )

            filtered_logits = torch.full_like(
                next_logits,
                float("-inf"),
            )

            filtered_logits.scatter_(
                -1,
                indices,
                values,
            )

            probabilities = torch.softmax(
                filtered_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

            if eos_id >= 0 and next_token.item() == eos_id:
                break

    return tokenizer.decode(input_ids[0].tolist())


def run_generation_tests(model, tokenizer, model_name):

    print("\n" + "=" * 70)
    print(f"GENERATION: {model_name}")
    print("=" * 70)

    for index, prompt in enumerate(PROMPTS, start=1):

        generated = generate_text(
            model,
            tokenizer,
            prompt,
            seed=SEED + index,
        )

        print(f"\nPrompt {index}: {prompt}")
        print(f"Output: {generated}")


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():

    print("=" * 70)
    print("RCT 100M — ORIGINAL VS CONTINUED MODEL COMPARISON")
    print("=" * 70)

    torch.set_num_threads(CPU_THREADS)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"PyTorch: {torch.__version__}")
    print("Device: CPU")
    print(f"Threads: {CPU_THREADS}")
    print(f"Validation batches: {EVALUATION_BATCHES}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")

    print("\n[INITIAL MEMORY]")
    report_memory()

    tokenizer = load_tokenizer()

    validation_data, starts = prepare_validation_starts()

    results = {}

    for model_name, checkpoint_path in CHECKPOINTS.items():

        print("\n" + "#" * 70)
        print(model_name)
        print("#" * 70)

        print("\n[BEFORE MODEL LOADING]")
        report_memory()

        model, step = load_checkpoint(checkpoint_path)

        print("\n[AFTER MODEL LOADING]")
        report_memory()

        loss, perplexity = evaluate_model(
            model,
            validation_data,
            starts,
            model_name,
        )

        results[model_name] = {
            "step": step,
            "loss": loss,
            "perplexity": perplexity,
        }

        run_generation_tests(
            model,
            tokenizer,
            model_name,
        )

        del model
        gc.collect()

        print("\n[AFTER MODEL UNLOADING]")
        report_memory()

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    original = results["Original step-500"]
    continued = results["Continued step-1500"]

    loss_improvement = (
        original["loss"] - continued["loss"]
    ) / original["loss"] * 100

    perplexity_improvement = (
        original["perplexity"] - continued["perplexity"]
    ) / original["perplexity"] * 100

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Metric':<25}"
        f"{'Original':>18}"
        f"{'Continued':>18}"
    )

    print("-" * 61)

    print(
        f"{'Training step':<25}"
        f"{str(original['step']):>18}"
        f"{str(continued['step']):>18}"
    )

    print(
        f"{'Validation loss':<25}"
        f"{original['loss']:>18.6f}"
        f"{continued['loss']:>18.6f}"
    )

    print(
        f"{'Perplexity':<25}"
        f"{original['perplexity']:>18.2f}"
        f"{continued['perplexity']:>18.2f}"
    )

    print("-" * 61)

    print(f"\nValidation loss improvement: {loss_improvement:.2f}%")
    print(f"Perplexity improvement: {perplexity_improvement:.2f}%")

    if continued["loss"] < original["loss"]:
        print("\nRESULT: Continued model has lower validation loss.")
    elif continued["loss"] > original["loss"]:
        print("\nRESULT: Original model has lower validation loss.")
    else:
        print("\nRESULT: Both models have equal validation loss.")

    del validation_data
    gc.collect()

    print("\n[FINAL MEMORY]")
    report_memory()

    print("\nCOMPARISON COMPLETED.")


if __name__ == "__main__":
    main()
