
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rct_100m import RCT100M, RCT100MConfig


class RCT100MKVCache:
    """Preallocated per-layer key/value cache."""

    def __init__(self, config, batch_size, device, dtype):

        self.batch_size = batch_size
        self.max_seq_len = config.max_seq_len
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        shape = (
            batch_size,
            self.num_heads,
            self.max_seq_len,
            self.head_dim
        )

        self.keys = [
            torch.empty(shape, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

        self.values = [
            torch.empty(shape, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]

        self.position = 0

    def clear(self):
        self.position = 0

    def memory_bytes(self):

        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.keys + self.values
        )


class RCT100MKVAttention(nn.Module):
    """Multi-head attention with incremental KV caching."""

    def __init__(self, original_attention):
        super().__init__()

        self.hidden_size = original_attention.hidden_size
        self.num_heads = original_attention.num_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.qkv = original_attention.qkv
        self.output = original_attention.output

    def forward(
        self,
        x,
        key_cache,
        value_cache,
        position
    ):

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)

        qkv = qkv.view(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim
        )

        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        end_position = position + seq_len

        if end_position > key_cache.size(2):
            raise ValueError("KV cache capacity exceeded")

        # Write new keys and values into the cache.
        key_cache[:, :, position:end_position, :].copy_(k)
        value_cache[:, :, position:end_position, :].copy_(v)

        all_keys = key_cache[:, :, :end_position, :]
        all_values = value_cache[:, :, :end_position, :]

        scores = torch.matmul(
            q,
            all_keys.transpose(-2, -1)
        )

        scores = scores / (self.head_dim ** 0.5)

        # Causal mask for both prompt prefill and single-token decoding.
        query_positions = torch.arange(
            position,
            end_position,
            device=x.device
        )

        key_positions = torch.arange(
            end_position,
            device=x.device
        )

        mask = (
            key_positions.unsqueeze(0)
            <= query_positions.unsqueeze(1)
        )

        scores = scores.masked_fill(
            ~mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )

        probabilities = F.softmax(scores, dim=-1)

        context = torch.matmul(probabilities, all_values)

        context = context.transpose(1, 2).contiguous()

        context = context.view(
            batch_size,
            seq_len,
            self.hidden_size
        )

        return self.output(context)


class RCT100MKVBlock(nn.Module):
    """Transformer block using cached attention."""

    def __init__(self, original_block):
        super().__init__()

        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2

        self.attention = RCT100MKVAttention(
            original_block.attention
        )

        self.ffn = original_block.ffn
        self.dropout = original_block.dropout

    def forward(
        self,
        x,
        key_cache,
        value_cache,
        position
    ):

        x = x + self.dropout(
            self.attention(
                self.norm1(x),
                key_cache,
                value_cache,
                position
            )
        )

        x = x + self.dropout(
            self.ffn(self.norm2(x))
        )

        return x


class RCT100MKV(nn.Module):
    """RCT 100M model with autoregressive KV caching."""

    def __init__(self, base_model):

        super().__init__()

        self.base_model = base_model
        self.config = base_model.config

        self.blocks = nn.ModuleList([
            RCT100MKVBlock(block)
            for block in base_model.blocks
        ])

    @classmethod
    def from_config(cls, config=None):

        if config is None:
            config = RCT100MConfig()

        return cls(RCT100M(config))

    def parameter_count(self):
        return self.base_model.parameter_count()

    def create_cache(self, batch_size):

        if batch_size < 1:
            raise ValueError("Batch size must be positive")

        embedding = self.base_model.token_embedding

        return RCT100MKVCache(
            self.config,
            batch_size,
            embedding.weight.device,
            embedding.weight.dtype
        )

    @torch.inference_mode()
    def decode_step(self, tokens, cache):

        if self.training:
            raise RuntimeError(
                "Call model.eval() before KV decoding."
            )

        if tokens.ndim != 2:
            raise ValueError(
                "Expected tokens with shape [batch, sequence]."
            )

        batch_size, seq_len = tokens.shape

        if batch_size != cache.batch_size:
            raise ValueError("Cache batch size mismatch")

        if seq_len < 1:
            raise ValueError("Sequence cannot be empty")

        if tokens.dtype != torch.long:
            raise TypeError("Tokens must be torch.long")

        embedding = self.base_model.token_embedding

        if tokens.device != embedding.weight.device:
            raise ValueError("Tokens and model must be on the same device")

        if tokens.min().item() < 0 or tokens.max().item() >= self.config.vocab_size:
            raise ValueError("Token ID outside vocabulary")

        position = cache.position
        end_position = position + seq_len

        if end_position > self.config.max_seq_len:
            raise ValueError("Maximum sequence length exceeded")

        positions = torch.arange(
            position,
            end_position,
            device=tokens.device
        )

        x = (
            embedding(tokens)
            + self.base_model.position_embedding(positions).unsqueeze(0)
        )

        for index, block in enumerate(self.blocks):

            x = block(
                x,
                cache.keys[index],
                cache.values[index],
                position
            )

        x = self.base_model.final_norm(x)

        logits = self.base_model.lm_head(x)

        cache.position = end_position

        return logits

    @torch.inference_mode()
    def generate_cached(
        self,
        prompt,
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

        if prompt.ndim != 2 or prompt.size(1) < 1:
            raise ValueError(
                "Prompt must have shape [batch, sequence] and be nonempty."
            )

        if prompt.dtype != torch.long:
            raise TypeError("Prompt must use torch.long")

        if prompt.size(1) + max_new_tokens > self.config.max_seq_len:
            raise ValueError(
                "Generation exceeds maximum sequence length."
            )

        cache = self.create_cache(prompt.size(0))

        # Process the prompt once.
        logits = self.decode_step(prompt, cache)

        generated = prompt

        for step in range(max_new_tokens):

            next_token_logits = (
                logits[:, -1, :] / temperature
            )

            probabilities = F.softmax(
                next_token_logits,
                dim=-1
            )

            # Match the original model's multinomial sampling.
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

            # Only process the newly generated token.
            if step + 1 < max_new_tokens:

                logits = self.decode_step(
                    next_token,
                    cache
                )

        return generated
