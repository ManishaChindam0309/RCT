
import hashlib
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCT100MPrefixCache(nn.Module):
    """
    Exact shared-prefix KV cache for RCT100M.

    All prompts in a batch must have the same prefix.
    Different suffixes are processed independently.
    """

    def __init__(self, base_model, cache_capacity=16):
        super().__init__()

        if cache_capacity < 1:
            raise ValueError("cache_capacity must be positive.")

        self.base_model = base_model
        self.config = base_model.config
        self.cache_capacity = cache_capacity

        self.prefix_cache = OrderedDict()

        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0

    def parameter_signature(self):
        return tuple(
            parameter._version
            for parameter in self.base_model.parameters()
        )

    def prefix_fingerprint(self, prefix):
        prefix = prefix.detach().contiguous().cpu()

        digest = hashlib.sha256()

        digest.update(str(tuple(prefix.shape)).encode())
        digest.update(str(prefix.dtype).encode())
        digest.update(prefix.numpy().tobytes())

        return digest.hexdigest()

    def _attention(self, attention, x, prefix_key=None,
                   prefix_value=None, prefix_len=0):

        batch_size, seq_len, hidden_size = x.shape

        qkv = attention.qkv(x)

        qkv = qkv.reshape(
            batch_size,
            seq_len,
            3,
            attention.num_heads,
            attention.head_dim
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(0)

        if prefix_key is None:
            context = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=True
            )

        else:
            all_keys = torch.cat(
                [prefix_key, key],
                dim=2
            )

            all_values = torch.cat(
                [prefix_value, value],
                dim=2
            )

            query_positions = torch.arange(
                prefix_len,
                prefix_len + seq_len,
                device=x.device
            )

            key_positions = torch.arange(
                prefix_len + seq_len,
                device=x.device
            )

            # True means that attention to this position is allowed.
            causal_mask = (
                key_positions.unsqueeze(0)
                <= query_positions.unsqueeze(1)
            )

            context = F.scaled_dot_product_attention(
                query,
                all_keys,
                all_values,
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=False
            )

        context = context.transpose(1, 2).contiguous()

        context = context.reshape(
            batch_size,
            seq_len,
            hidden_size
        )

        return attention.output(context), key, value

    @torch.inference_mode()
    def _build_prefix(self, prefix_tokens):

        model = self.base_model

        prefix_len = prefix_tokens.size(1)

        positions = torch.arange(
            prefix_len,
            device=prefix_tokens.device
        )

        x = (
            model.token_embedding(prefix_tokens)
            + model.position_embedding(positions).unsqueeze(0)
        )

        x = model.embedding_dropout(x)

        keys = []
        values = []

        for block in model.blocks:

            normalized = block.norm1(x)

            attention_output, key, value = self._attention(
                block.attention,
                normalized
            )

            keys.append(key.detach())
            values.append(value.detach())

            x = x + block.dropout(attention_output)

            ffn_output = block.ffn(block.norm2(x))

            x = x + block.dropout(ffn_output)

        return {
            "keys": keys,
            "values": values,
            "hidden": x.detach()
        }

    @torch.inference_mode()
    def forward_cached(self, tokens, prefix_len):

        model = self.base_model

        if self.training:
            raise RuntimeError(
                "Call model.eval() before prefix caching."
            )

        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape [batch, sequence]."
            )

        if tokens.dtype != torch.long:
            raise TypeError("tokens must use torch.long.")

        batch_size, sequence_length = tokens.shape

        if sequence_length == 0:
            raise ValueError("Input sequence cannot be empty.")

        if sequence_length > self.config.max_seq_len:
            raise ValueError("Sequence exceeds max_seq_len.")

        if not isinstance(prefix_len, int):
            raise TypeError("prefix_len must be an integer.")

        if not 0 < prefix_len <= sequence_length:
            raise ValueError("Invalid prefix_len.")

        if tokens.device != model.token_embedding.weight.device:
            raise ValueError("Tokens and model must be on the same device.")

        if tokens.min().item() < 0 or tokens.max().item() >= self.config.vocab_size:
            raise ValueError("Token ID outside vocabulary.")

        prefix_tokens = tokens[:, :prefix_len]

        # All batch entries must share exactly the same prefix.
        reference_prefix = prefix_tokens[:1].expand_as(prefix_tokens)

        if not torch.equal(prefix_tokens, reference_prefix):
            raise ValueError(
                "Every prompt in the batch must have the same prefix."
            )

        prefix = prefix_tokens[:1]

        fingerprint = self.prefix_fingerprint(prefix)

        cache_key = (
            fingerprint,
            prefix_len,
            str(tokens.device),
            str(model.token_embedding.weight.dtype),
            self.parameter_signature()
        )

        if cache_key in self.prefix_cache:

            self.cache_hits += 1

            entry = self.prefix_cache.pop(cache_key)
            self.prefix_cache[cache_key] = entry

        else:

            self.cache_misses += 1

            entry = self._build_prefix(prefix)

            self.prefix_cache[cache_key] = entry

            while len(self.prefix_cache) > self.cache_capacity:

                self.prefix_cache.popitem(last=False)
                self.cache_evictions += 1

        suffix_tokens = tokens[:, prefix_len:]

        if suffix_tokens.size(1) == 0:

            hidden = entry["hidden"].expand(
                batch_size, -1, -1
            )

        else:

            suffix_len = suffix_tokens.size(1)

            positions = torch.arange(
                prefix_len,
                sequence_length,
                device=tokens.device
            )

            x = (
                model.token_embedding(suffix_tokens)
                + model.position_embedding(positions).unsqueeze(0)
            )

            x = model.embedding_dropout(x)

            for index, block in enumerate(model.blocks):

                normalized = block.norm1(x)

                prefix_key = entry["keys"][index].expand(
                    batch_size, -1, -1, -1
                )

                prefix_value = entry["values"][index].expand(
                    batch_size, -1, -1, -1
                )

                attention_output, _, _ = self._attention(
                    block.attention,
                    normalized,
                    prefix_key,
                    prefix_value,
                    prefix_len
                )

                x = x + block.dropout(attention_output)

                ffn_output = block.ffn(block.norm2(x))

                x = x + block.dropout(ffn_output)

            prefix_hidden = entry["hidden"].expand(
                batch_size, -1, -1
            )

            hidden = torch.cat(
                [prefix_hidden, x],
                dim=1
            )

        hidden = model.final_norm(hidden)

        return model.lm_head(hidden)

    def cache_statistics(self):

        total = self.cache_hits + self.cache_misses

        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": (
                self.cache_hits / total if total else 0.0
            ),
            "entries": len(self.prefix_cache),
            "capacity": self.cache_capacity,
            "evictions": self.cache_evictions
        }

    def clear_cache(self):

        self.prefix_cache.clear()

        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
