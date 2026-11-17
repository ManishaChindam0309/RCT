
from pathlib import Path

import math
import torch
import torch.nn.functional as F
import sentencepiece as spm

from model.rct_100m import RCT100M, RCT100MConfig

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "rct30"

CHECKPOINT = ROOT / "checkpoints" / "rct30" / "best.pt"
VALIDATION_FILE = DATA_DIR / "validation_tokens.bin"
TOKENIZER_FILE = DATA_DIR / "rct30_tokenizer.model"

BATCH_SIZE = 2
SEQUENCE_LENGTH = 64


def main():

    torch.set_num_threads(4)
    torch.manual_seed(30)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    saved_config = checkpoint["training_config"]

    model_config = RCT100MConfig(
        vocab_size=saved_config["vocab_size"],
        hidden_size=saved_config["hidden_size"],
        num_layers=saved_config["num_layers"],
        num_heads=saved_config["num_heads"],
        intermediate_size=saved_config["intermediate_size"],
        max_seq_len=saved_config["max_seq_len"],
        dropout=saved_config["dropout"],
        tie_embeddings=True,
    )

    model = RCT100M(model_config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("=" * 60)
    print("RCT 3.0 CHECKPOINT EVALUATION")
    print("=" * 60)

    print("Checkpoint step:", checkpoint["step"])
    print(
        "Saved best validation loss:",
        checkpoint["best_validation_loss"],
    )
    print("Parameters:", f"{model.parameter_count():,}")

    tokens = torch.from_numpy(
        __import__("numpy").fromfile(
            VALIDATION_FILE,
            dtype=__import__("numpy").uint16,
        ).astype("int64")
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():

        for start in range(
            0,
            len(tokens) - SEQUENCE_LENGTH,
            SEQUENCE_LENGTH,
        ):

            x = tokens[
                start:start + SEQUENCE_LENGTH
            ].unsqueeze(0)

            y = tokens[
                start + 1:start + SEQUENCE_LENGTH + 1
            ].unsqueeze(0)

            logits = model(x)

            loss_sum = F.cross_entropy(
                logits.reshape(-1, model_config.vocab_size),
                y.reshape(-1),
                reduction="sum",
            )

            total_loss += loss_sum.item()
            total_tokens += y.numel()

    validation_loss = total_loss / total_tokens
    perplexity = math.exp(validation_loss)

    print("\nFULL VALIDATION RESULTS")
    print("-" * 60)
    print("Evaluated tokens:", f"{total_tokens:,}")
    print("Validation loss:", round(validation_loss, 4))
    print("Perplexity:", round(perplexity, 2))

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_FILE)
    )

    prompt = "Alice was beginning to get"

    prompt_ids = tokenizer.encode(
        prompt,
        out_type=int,
    )

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
    )

    torch.manual_seed(30)

    generated = model.generate(
        input_ids,
        max_new_tokens=40,
        temperature=0.8,
    )

    generated_text = tokenizer.decode(
        generated[0].tolist()
    )

    print("\nSAMPLE GENERATION")
    print("-" * 60)
    print(generated_text)

    print("\nCheckpoint evaluation completed.")


if __name__ == "__main__":
    main()
