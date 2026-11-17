
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# RCT 100M CONFIGURATION
# ============================================================

@dataclass
class RCT100MConfig:
    vocab_size: int = 8192
    hidden_size: int = 512
    num_layers: int = 30
    num_heads: int = 8
    intermediate_size: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.0
    tie_embeddings: bool = True

    def validate(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


# ============================================================
# CAUSAL SELF-ATTENTION
# ============================================================

class RCT100MAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = (
            config.hidden_size // config.num_heads
        )

        self.qkv = nn.Linear(
            config.hidden_size,
            config.hidden_size * 3,
            bias=True
        )

        self.output = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=True
        )

        self.attention_dropout = config.dropout

    def forward(self, x):

        batch_size, sequence_length, hidden_size = x.shape

        qkv = self.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        # PyTorch scaled dot-product causal attention.
        # This avoids explicitly constructing an S x S mask.

        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=(
                self.attention_dropout
                if self.training
                else 0.0
            ),
            is_causal=True
        )

        attention = attention.transpose(1, 2).contiguous()

        attention = attention.reshape(
            batch_size,
            sequence_length,
            hidden_size
        )

        return self.output(attention)


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================

class RCT100MFeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size
        )

        self.fc2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        x = self.fc1(x)

        x = F.gelu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class RCT100MBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            config.hidden_size
        )

        self.attention = RCT100MAttention(config)

        self.norm2 = nn.LayerNorm(
            config.hidden_size
        )

        self.ffn = RCT100MFeedForward(config)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        # Pre-normalized attention residual.

        attention_output = self.attention(
            self.norm1(x)
        )

        x = x + self.dropout(attention_output)

        # Pre-normalized feed-forward residual.

        ffn_output = self.ffn(
            self.norm2(x)
        )

        x = x + self.dropout(ffn_output)

        return x


# ============================================================
# RCT 100M LANGUAGE MODEL
# ============================================================

class RCT100M(nn.Module):

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = RCT100MConfig()

        config.validate()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size
        )

        self.position_embedding = nn.Embedding(
            config.max_seq_len,
            config.hidden_size
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        self.blocks = nn.ModuleList([
            RCT100MBlock(config)
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            config.hidden_size
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False
        )

        if config.tie_embeddings:
            self.lm_head.weight = (
                self.token_embedding.weight
            )

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

        elif isinstance(module, nn.LayerNorm):

            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def parameter_count(self, trainable_only=True):

        if trainable_only:

            return sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad
            )

        return sum(
            p.numel()
            for p in self.parameters()
        )

    def forward(
        self,
        tokens,
        prompt_mode="standard"
    ):

        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape [batch, sequence]."
            )

        if tokens.dtype != torch.long:
            raise TypeError(
                "tokens must have dtype torch.long."
            )

        batch_size, sequence_length = tokens.shape

        if sequence_length > self.config.max_seq_len:
            raise ValueError(
                "Input sequence exceeds max_seq_len."
            )

        if tokens.numel() > 0:
            if (
                tokens.min().item() < 0
                or tokens.max().item() >= self.config.vocab_size
            ):
                raise ValueError(
                    "Token ID outside configured vocabulary."
                )

        if prompt_mode not in ("standard", "adaptive"):
            raise ValueError(
                "prompt_mode must be standard or adaptive."
            )

        # Adaptive prompt deduplication is inference-only.
        # Training always processes the original batch.

        if prompt_mode == "adaptive":

            if self.training or torch.is_grad_enabled():
                raise RuntimeError(
                    "Adaptive prompt deduplication requires "
                    "evaluation mode and disabled gradients."
                )

            unique_tokens, inverse_indices = torch.unique(
                tokens,
                dim=0,
                return_inverse=True
            )

            if unique_tokens.shape[0] < batch_size:

                unique_logits = self._forward_standard(
                    unique_tokens
                )

                return unique_logits.index_select(
                    0,
                    inverse_indices
                )

        return self._forward_standard(tokens)

    def _forward_standard(self, tokens):

        batch_size, sequence_length = tokens.shape

        positions = torch.arange(
            sequence_length,
            device=tokens.device
        )

        token_vectors = self.token_embedding(tokens)

        position_vectors = self.position_embedding(
            positions
        )

        x = token_vectors + position_vectors.unsqueeze(0)

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        max_new_tokens=20,
        temperature=1.0
    ):

        if self.training:
            raise RuntimeError(
                "Call model.eval() before generation."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        generated = tokens

        for _ in range(max_new_tokens):

            context = generated[
                :, -self.config.max_seq_len:
            ]

            logits = self(
                context,
                prompt_mode="standard"
            )

            next_token_logits = (
                logits[:, -1, :] / temperature
            )

            probabilities = F.softmax(
                next_token_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

        return generated
