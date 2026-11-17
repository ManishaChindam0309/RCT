
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

CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "rct100m" / "best_model.pt"
VALIDATION_PATH = PROJECT_DIR / "data" / "rct30" / "validation_expanded_tokens.bin"

VOCAB_SIZE = 8192
HIDDEN_SIZE = 512
NUM_LAYERS = 30
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048
MAX_SEQ_LEN = 512

# The model was trained using 32-token sequences.
# Keep evaluation and generation within this trained context.
EVALUATION_SEQUENCE_LENGTH = 32

CPU_THREADS = 4
EVALUATION_BATCHES = 256
BATCH_SIZE = 1

GENERATION_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40

SEED = 42


# ============================================================
# SYSTEM INFORMATION
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

    model = RCT100M(config)

    return model


def load_model():

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("\nLoading best RCT 100M checkpoint...")

    model = create_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise KeyError(
            "Could not find model weights in checkpoint. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    print("Checkpoint loaded successfully.")

    if "step" in checkpoint:
        print(f"Checkpoint training step: {checkpoint['step']}")

    if "val_loss" in checkpoint:
        print(f"Saved validation loss: {checkpoint['val_loss']:.6f}")

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    del checkpoint
    gc.collect()

    return model


# ============================================================
# TOKENIZER
# ============================================================

def find_tokenizer():

    preferred_paths = [
        PROJECT_DIR / "data" / "rct30" / "rct30_8k.model",
        PROJECT_DIR / "data" / "rct30" / "tokenizer" / "rct30_8k.model",
        PROJECT_DIR / "data" / "rct30" / "tokenizer" / "spm.model",
    ]

    for path in preferred_paths:
        if path.exists():
            return path

    candidates = list(
        (PROJECT_DIR / "data" / "rct30").rglob("*.model")
    )

    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Could not find the SentencePiece tokenizer model "
        "inside data/rct30."
    )


def load_tokenizer():

    tokenizer_path = find_tokenizer()

    print(f"\nTokenizer: {tokenizer_path}")

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(tokenizer_path)
    )

    print(f"Tokenizer vocabulary: {tokenizer.get_piece_size()}")

    if tokenizer.get_piece_size() != VOCAB_SIZE:
        raise ValueError(
            "Tokenizer vocabulary does not match model vocabulary."
        )

    return tokenizer


# ============================================================
# VALIDATION
# ============================================================

def evaluate_model(model):

    print("\n" + "=" * 65)
    print("VALIDATION EVALUATION")
    print("=" * 65)

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_PATH}"
        )

    validation_data = np.memmap(
        VALIDATION_PATH,
        dtype=np.uint16,
        mode="r",
    )

    sequence_length = EVALUATION_SEQUENCE_LENGTH
    total_tokens = len(validation_data)

    if total_tokens <= sequence_length:
        raise ValueError("Validation dataset is too small.")

    rng = np.random.default_rng(SEED)

    # Use deterministic random positions.
    starts = rng.integers(
        0,
        total_tokens - sequence_length - 1,
        size=EVALUATION_BATCHES,
    )

    total_loss = 0.0
    total_positions = 0

    model.eval()

    with torch.inference_mode():

        for batch_index, start in enumerate(starts, start=1):

            chunk = np.array(
                validation_data[
                    start:start + sequence_length + 1
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

            if batch_index % 32 == 0:
                print(
                    f"Evaluated batches: "
                    f"{batch_index}/{EVALUATION_BATCHES}"
                )

    average_loss = total_loss / total_positions
    perplexity = math.exp(min(average_loss, 20))

    print("\nValidation results:")
    print(f"Evaluation batches: {EVALUATION_BATCHES}")
    print(f"Token positions: {total_positions:,}")
    print(f"Validation loss: {average_loss:.6f}")
    print(f"Validation perplexity: {perplexity:.2f}")

    del validation_data
    gc.collect()

    return average_loss, perplexity


# ============================================================
# TEXT GENERATION
# ============================================================

def generate_text(model, tokenizer, prompt):

    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        raise ValueError("Prompt produced no tokens.")

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
    )

    eos_id = tokenizer.eos_id()

    model.eval()

    with torch.inference_mode():

        for _ in range(GENERATION_TOKENS):

            # Only use the last 32 tokens because that is the
            # context length used during this training experiment.
            context = input_ids[:, -EVALUATION_SEQUENCE_LENGTH:]

            logits = model(context)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if isinstance(logits, dict):
                logits = logits["logits"]

            next_token_logits = logits[:, -1, :]

            next_token_logits = next_token_logits / TEMPERATURE

            # Top-k sampling.
            values, indices = torch.topk(
                next_token_logits,
                k=TOP_K,
                dim=-1,
            )

            filtered_logits = torch.full_like(
                next_token_logits,
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
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

            if eos_id >= 0 and next_token.item() == eos_id:
                break

    generated_text = tokenizer.decode(
        input_ids[0].tolist()
    )

    return generated_text


def run_generation_tests(model, tokenizer):

    print("\n" + "=" * 65)
    print("TEXT GENERATION TESTS")
    print("=" * 65)

    prompts = [
        "Alice was beginning to get",
        "It was a dark and stormy night",
        "The detective entered the room",
        "Once upon a time",
        "The old man looked at the"
    ]

    for index, prompt in enumerate(prompts, start=1):

        print(f"\nTest {index}")
        print(f"Prompt: {prompt}")

        generated = generate_text(
            model,
            tokenizer,
            prompt,
        )

        print("\nGenerated text:")
        print(generated)
        print("-" * 65)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("RCT 100M — MODEL EVALUATION")
    print("=" * 65)

    torch.set_num_threads(CPU_THREADS)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: CPU")
    print(f"Evaluation sequence length: {EVALUATION_SEQUENCE_LENGTH}")
    print(f"Validation batches: {EVALUATION_BATCHES}")

    print("\n[INITIAL MEMORY]")
    report_memory()

    model = load_model()
    tokenizer = load_tokenizer()

    print("\n[AFTER MODEL LOADING]")
    report_memory()

    validation_loss, perplexity = evaluate_model(model)

    run_generation_tests(model, tokenizer)

    print("\n" + "=" * 65)
    print("RCT 100M EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Validation loss: {validation_loss:.6f}")
    print(f"Validation perplexity: {perplexity:.2f}")

    print("\n[FINAL MEMORY]")
    report_memory()

    del model
    gc.collect()


if __name__ == "__main__":
    main()
