
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Model architecture for the initial CPU pilot.
    vocab_size: int = 2048
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 512
    max_seq_len: int = 128
    dropout: float = 0.0

    # Training settings.
    batch_size: int = 2
    sequence_length: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    max_steps: int = 200
    eval_interval: int = 50
    eval_batches: int = 10
    gradient_clip: float = 1.0

    # Reproducibility and resources.
    seed: int = 30
    cpu_threads: int = 4

    # Paths.
    train_data: str = "data/rct30/train_tokens.bin"
    validation_data: str = "data/rct30/validation_tokens.bin"
    checkpoint_dir: str = "checkpoints/rct30"

    def validate(self):
        assert self.vocab_size > 0
        assert self.hidden_size % self.num_heads == 0
        assert self.sequence_length <= self.max_seq_len
        assert self.batch_size > 0
        assert self.max_steps > 0
        assert self.gradient_accumulation_steps > 0
