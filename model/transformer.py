
import hashlib
from collections import OrderedDict

import torch
import torch.nn as nn

from model.reusable_block import ReusableFFN


# ============================================================
# CAUSAL SELF ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, hidden_size, num_heads):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.in_proj = nn.Linear(
            hidden_size,
            hidden_size * 3
        )

        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size
        )

    def split_heads(self, x):

        batch_size, seq_len, _ = x.shape

        return (
            x.view(
                batch_size,
                seq_len,
                self.num_heads,
                self.head_dim
            )
            .transpose(1, 2)
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        qkv = self.in_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        scores = scores / (self.head_dim ** 0.5)

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        context = torch.matmul(
            weights,
            v
        )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.hidden_size
            )
        )

        return self.out_proj(context)


# ============================================================
# RCT TRANSFORMER BLOCK
# ============================================================

class RCTBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        expansion=4,
        cache_capacity=256
    ):
        super().__init__()

        self.attention = CausalSelfAttention(
            hidden_size,
            num_heads
        )

        self.norm1 = nn.LayerNorm(hidden_size)

        self.ffn = ReusableFFN(
            hidden_size,
            expansion=expansion,
            cache_capacity=cache_capacity
        )

        self.norm2 = nn.LayerNorm(hidden_size)

        self.dedup_calls = 0
        self.dedup_total_inputs = 0
        self.dedup_unique_inputs = 0

    def forward(
        self,
        x,
        reuse_prefix=False,
        ffn_mode="standard"
    ):

        if ffn_mode not in (
            "standard",
            "deduplicated"
        ):
            raise ValueError(
                "ffn_mode must be 'standard' or 'deduplicated'."
            )

        if ffn_mode == "deduplicated":

            if self.training or torch.is_grad_enabled():
                raise RuntimeError(
                    "Deduplicated mode is inference-only. "
                    "Use model.eval() and torch.inference_mode()."
                )

            if reuse_prefix:
                raise ValueError(
                    "reuse_prefix and deduplicated mode "
                    "cannot be combined."
                )

        attention_output = self.attention(x)

        x = self.norm1(
            x + attention_output
        )

        if ffn_mode == "deduplicated":

            batch_size, seq_len, hidden_size = x.shape

            flattened = x.reshape(
                batch_size * seq_len,
                hidden_size
            )

            unique, inverse = torch.unique(
                flattened,
                dim=0,
                return_inverse=True
            )

            unique_outputs = self.ffn(unique)

            ffn_output = unique_outputs.index_select(
                0,
                inverse
            ).reshape(
                batch_size,
                seq_len,
                hidden_size
            )

            self.dedup_calls += 1

            self.dedup_total_inputs += flattened.size(0)

            self.dedup_unique_inputs += unique.size(0)

        elif reuse_prefix:

            ffn_outputs = []

            for position in range(x.size(1)):

                token = x[:, position:position + 1, :]

                ffn_outputs.append(
                    self.ffn(token)
                )

            ffn_output = torch.cat(
                ffn_outputs,
                dim=1
            )

        else:

            ffn_output = self.ffn(x)

        x = self.norm2(
            x + ffn_output
        )

        return x

    def dedup_statistics(self):

        reuse_ratio = (
            1 - self.dedup_unique_inputs / self.dedup_total_inputs
            if self.dedup_total_inputs > 0
            else 0.0
        )

        return {
            "calls": self.dedup_calls,
            "total_inputs": self.dedup_total_inputs,
            "unique_inputs": self.dedup_unique_inputs,
            "reuse_ratio": reuse_ratio
        }

    def reset_dedup_statistics(self):

        self.dedup_calls = 0
        self.dedup_total_inputs = 0
        self.dedup_unique_inputs = 0

    # --------------------------------------------------------
    # PREFIX CACHE SUFFIX PROCESSING
    # --------------------------------------------------------

    def forward_suffix(
        self,
        x,
        prefix_key,
        prefix_value
    ):

        batch_size, suffix_len, hidden_size = x.shape

        qkv = self.attention.in_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = self.attention.split_heads(q)
        k = self.attention.split_heads(k)
        v = self.attention.split_heads(v)

        all_keys = torch.cat(
            [prefix_key, k],
            dim=2
        )

        all_values = torch.cat(
            [prefix_value, v],
            dim=2
        )

        prefix_len = prefix_key.size(2)

        total_len = prefix_len + suffix_len

        scores = torch.matmul(
            q,
            all_keys.transpose(-2, -1)
        )

        scores = scores / (
            self.attention.head_dim ** 0.5
        )

        query_positions = (
            torch.arange(
                suffix_len,
                device=x.device
            ) + prefix_len
        )

        key_positions = torch.arange(
            total_len,
            device=x.device
        )

        causal_mask = (
            key_positions.unsqueeze(0)
            > query_positions.unsqueeze(1)
        )

        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0),
            float("-inf")
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        context = torch.matmul(
            weights,
            all_values
        )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                suffix_len,
                hidden_size
            )
        )

        attention_output = self.attention.out_proj(context)

        x = self.norm1(
            x + attention_output
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        return x

    # --------------------------------------------------------
    # PREALLOCATED AUTOREGRESSIVE KV DECODING
    # --------------------------------------------------------

    def forward_incremental(
        self,
        x,
        key_buffer,
        value_buffer,
        position
    ):

        batch_size, seq_len, hidden_size = x.shape

        if seq_len != 1:
            raise ValueError(
                "Incremental decoding expects exactly one token."
            )

        qkv = self.attention.in_proj(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = self.attention.split_heads(q)
        k = self.attention.split_heads(k)
        v = self.attention.split_heads(v)

        key_buffer[
            :, :, position:position + 1, :
        ].copy_(k)

        value_buffer[
            :, :, position:position + 1, :
        ].copy_(v)

        cached_keys = key_buffer[
            :, :, :position + 1, :
        ]

        cached_values = value_buffer[
            :, :, :position + 1, :
        ]

        scores = torch.matmul(
            q,
            cached_keys.transpose(-2, -1)
        )

        scores = scores / (
            self.attention.head_dim ** 0.5
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        context = torch.matmul(
            weights,
            cached_values
        )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                1,
                hidden_size
            )
        )

        attention_output = self.attention.out_proj(context)

        x = self.norm1(
            x + attention_output
        )

        x = self.norm2(
            x + self.ffn(x)
        )

        return x


# ============================================================
# REUSABLE COMPUTATION TRANSFORMER
# ============================================================

class RCTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_seq_len=512,
        expansion=4,
        cache_capacity=256,
        prefix_cache_capacity=8,
        layers=None,
        heads=None
    ):
        super().__init__()

        if layers is not None:
            num_layers = layers

        if heads is not None:
            num_heads = heads

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            hidden_size
        )

        self.blocks = nn.ModuleList([

            RCTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                expansion=expansion,
                cache_capacity=cache_capacity
            )

            for _ in range(num_layers)

        ])

        self.final_norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False
        )

        # ----------------------------------------------------
        # PREFIX CACHE
        # ----------------------------------------------------

        self._prefix_cache_capacity = prefix_cache_capacity

        self._prefix_cache = OrderedDict()

        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        self._prefix_cache_evictions = 0

        self.prefix_cache_capacity = prefix_cache_capacity

        self.prefix_cache = self._prefix_cache

        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_evictions = 0

        # ----------------------------------------------------
        # PARAMETER VERSION TRACKING
        # ----------------------------------------------------

        self._parameter_signature = (
            self._get_parameter_signature()
        )

        # ----------------------------------------------------
        # RCT 2.1 PROMPT DEDUPLICATION STATISTICS
        # ----------------------------------------------------

        self._prompt_dedup_calls = 0
        self._prompt_dedup_original = 0
        self._prompt_dedup_unique = 0
        self._prompt_dedup_bypasses = 0
        self._prompt_dedup_activations = 0

    # --------------------------------------------------------
    # CACHE COMPATIBILITY PROPERTIES
    # --------------------------------------------------------

    @property
    def _prefix_cache(self):
        return self.__dict__["_prefix_cache_storage"]

    @_prefix_cache.setter
    def _prefix_cache(self, value):
        self.__dict__["_prefix_cache_storage"] = value

    @property
    def prefix_cache(self):
        return self._prefix_cache

    @prefix_cache.setter
    def prefix_cache(self, value):
        self._prefix_cache = value

    @property
    def _prefix_cache_capacity(self):
        return self.__dict__["_prefix_cache_capacity_storage"]

    @_prefix_cache_capacity.setter
    def _prefix_cache_capacity(self, value):
        self.__dict__["_prefix_cache_capacity_storage"] = value

    # --------------------------------------------------------
    # PARAMETER VERSION TRACKING
    # --------------------------------------------------------

    def _get_parameter_signature(self):

        return tuple(
            parameter._version
            for parameter in self.parameters()
        )

    def _check_parameter_versions(self):

        current_signature = self._get_parameter_signature()

        if current_signature != self._parameter_signature:
            self.invalidate_caches()

    # --------------------------------------------------------
    # RCT 2.1 PROMPT DEDUPLICATION STATISTICS
    # --------------------------------------------------------

    def prompt_dedup_statistics(self):

        duplicate_count = (
            self._prompt_dedup_original
            - self._prompt_dedup_unique
        )

        duplicate_ratio = (
            duplicate_count / self._prompt_dedup_original
            if self._prompt_dedup_original > 0
            else 0.0
        )

        return {
            "calls": self._prompt_dedup_calls,
            "original_prompts": self._prompt_dedup_original,
            "unique_prompts": self._prompt_dedup_unique,
            "duplicate_prompts": duplicate_count,
            "duplicate_ratio": duplicate_ratio,
            "dedup_activations": self._prompt_dedup_activations,
            "bypasses": self._prompt_dedup_bypasses
        }

    def reset_prompt_dedup_statistics(self):

        self._prompt_dedup_calls = 0
        self._prompt_dedup_original = 0
        self._prompt_dedup_unique = 0
        self._prompt_dedup_bypasses = 0
        self._prompt_dedup_activations = 0

    # --------------------------------------------------------
    # NORMAL FORWARD — RCT 2.1
    # --------------------------------------------------------

    def forward(
        self,
        tokens,
        reuse_prefix=False,
        ffn_mode="standard",
        prompt_mode="standard"
    ):

        if tokens.dim() != 2:
            raise ValueError(
                "Input tokens must have shape [batch, sequence]."
            )

        batch_size, seq_len = tokens.shape

        if seq_len == 0:
            raise ValueError("Input sequence cannot be empty.")

        if seq_len > self.max_seq_len:
            raise ValueError(
                "Sequence exceeds max_seq_len."
            )

        if ffn_mode not in (
            "standard",
            "deduplicated"
        ):
            raise ValueError(
                "Invalid FFN execution mode."
            )

        if prompt_mode not in (
            "standard",
            "adaptive"
        ):
            raise ValueError(
                "prompt_mode must be 'standard' or 'adaptive'."
            )

        # ----------------------------------------------------
        # ADAPTIVE PROMPT DEDUPLICATION
        # ----------------------------------------------------

        if prompt_mode == "adaptive":

            if self.training or torch.is_grad_enabled():
                raise RuntimeError(
                    "Adaptive prompt deduplication is "
                    "inference-only. Use model.eval() and "
                    "torch.inference_mode()."
                )

            if reuse_prefix:
                raise ValueError(
                    "Adaptive prompt deduplication cannot "
                    "be combined with reuse_prefix."
                )

            if ffn_mode != "standard":
                raise ValueError(
                    "Adaptive prompt deduplication requires "
                    "ffn_mode='standard'."
                )

            self._prompt_dedup_calls += 1
            self._prompt_dedup_original += batch_size

            # Single prompt: no cross-prompt duplicates possible.
            if batch_size == 1:

                self._prompt_dedup_bypasses += 1
                self._prompt_dedup_unique += 1

            else:

                unique_tokens, inverse_indices = torch.unique(
                    tokens,
                    dim=0,
                    sorted=True,
                    return_inverse=True
                )

                unique_count = unique_tokens.size(0)

                self._prompt_dedup_unique += unique_count

                # Duplicates found.
                if unique_count < batch_size:

                    self._prompt_dedup_activations += 1

                    unique_outputs = self.forward(
                        unique_tokens,
                        ffn_mode="standard",
                        prompt_mode="standard"
                    )

                    return unique_outputs.index_select(
                        0,
                        inverse_indices.to(unique_outputs.device)
                    )

                # No duplicates.
                self._prompt_dedup_bypasses += 1

        # ----------------------------------------------------
        # STANDARD TRANSFORMER EXECUTION
        # ----------------------------------------------------

        positions = torch.arange(
            seq_len,
            device=tokens.device
        ).unsqueeze(0)

        x = self.token_embedding(tokens)

        x = x + self.position_embedding(positions)

        for block in self.blocks:

            x = block(
                x,
                reuse_prefix=reuse_prefix,
                ffn_mode=ffn_mode
            )

        x = self.final_norm(x)

        return self.lm_head(x)

    # --------------------------------------------------------
    # RCT 2.0 FFN DEDUPLICATION STATISTICS
    # --------------------------------------------------------

    def dedup_statistics(self):

        return {
            f"layer_{index}": block.dedup_statistics()
            for index, block in enumerate(self.blocks)
        }

    def reset_dedup_statistics(self):

        for block in self.blocks:
            block.reset_dedup_statistics()

    # --------------------------------------------------------
    # PREFIX FINGERPRINT
    # --------------------------------------------------------

    def _prefix_fingerprint(self, prefix_tokens):

        tensor = (
            prefix_tokens
            .detach()
            .contiguous()
            .cpu()
        )

        digest = hashlib.sha256()

        digest.update(
            str(tuple(tensor.shape)).encode()
        )

        digest.update(
            str(tensor.dtype).encode()
        )

        digest.update(
            tensor.numpy().tobytes()
        )

        return digest.hexdigest()

    # --------------------------------------------------------
    # BUILD PREFIX CACHE
    # --------------------------------------------------------

    def _build_prefix_cache(self, prefix_tokens):

        batch_size, prefix_len = prefix_tokens.shape

        positions = torch.arange(
            prefix_len,
            device=prefix_tokens.device
        ).unsqueeze(0)

        x = self.token_embedding(prefix_tokens)

        x = x + self.position_embedding(positions)

        keys = []
        values = []

        for block in self.blocks:

            qkv = block.attention.in_proj(x)

            q, k, v = qkv.chunk(3, dim=-1)

            k = block.attention.split_heads(k)
            v = block.attention.split_heads(v)

            keys.append(k.detach().clone())
            values.append(v.detach().clone())

            x = block(x)

        return {
            "keys": keys,
            "values": values,
            "hidden": x.detach().clone(),
            "batch_size": batch_size,
            "prefix_len": prefix_len
        }

    # --------------------------------------------------------
    # MULTI-PREFIX CACHE FORWARD
    # --------------------------------------------------------

    @torch.inference_mode()
    def forward_cached(self, tokens, prefix_len):

        if self.training:
            raise RuntimeError(
                "Model must be in evaluation mode."
            )

        if tokens.device.type != "cpu":
            raise RuntimeError(
                "Prefix cache currently supports CPU only."
            )

        if tokens.dim() != 2:
            raise ValueError(
                "Tokens must have shape [batch, sequence]."
            )

        batch_size, seq_len = tokens.shape

        if not isinstance(prefix_len, int):
            raise TypeError(
                "prefix_len must be an integer."
            )

        if not 0 < prefix_len <= seq_len:
            raise ValueError(
                "Invalid prefix_len."
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                "Sequence exceeds max_seq_len."
            )

        self._check_parameter_versions()

        prefix_tokens = tokens[:, :prefix_len]
        suffix_tokens = tokens[:, prefix_len:]

        fingerprint = self._prefix_fingerprint(
            prefix_tokens
        )

        cache_key = (
            fingerprint,
            batch_size,
            prefix_len,
            self._get_parameter_signature()
        )

        if cache_key in self._prefix_cache:

            self._prefix_cache_hits += 1

            entry = self._prefix_cache.pop(cache_key)

            self._prefix_cache[cache_key] = entry

        else:

            self._prefix_cache_misses += 1

            entry = self._build_prefix_cache(
                prefix_tokens
            )

            self._prefix_cache[cache_key] = entry

            while (
                len(self._prefix_cache)
                > self._prefix_cache_capacity
            ):

                self._prefix_cache.popitem(last=False)

                self._prefix_cache_evictions += 1

        prefix_hidden = entry["hidden"]

        if suffix_tokens.size(1) == 0:

            x = prefix_hidden

        else:

            suffix_positions = torch.arange(
                prefix_len,
                seq_len,
                device=tokens.device
            ).unsqueeze(0)

            x = self.token_embedding(suffix_tokens)

            x = x + self.position_embedding(
                suffix_positions
            )

            for index, block in enumerate(self.blocks):

                x = block.forward_suffix(
                    x,
                    entry["keys"][index],
                    entry["values"][index]
                )

            x = torch.cat(
                [prefix_hidden, x],
                dim=1
            )

        x = self.final_norm(x)

        return self.lm_head(x)

    # --------------------------------------------------------
    # PREFIX CACHE STATISTICS
    # --------------------------------------------------------

    def prefix_cache_statistics(self):

        total = (
            self._prefix_cache_hits
            + self._prefix_cache_misses
        )

        hit_rate = (
            self._prefix_cache_hits / total
            if total > 0 else 0.0
        )

        return {
            "hits": self._prefix_cache_hits,
            "misses": self._prefix_cache_misses,
            "entries": len(self._prefix_cache),
            "capacity": self._prefix_cache_capacity,
            "evictions": self._prefix_cache_evictions,
            "hit_rate": hit_rate
        }

    # --------------------------------------------------------
    # INVALIDATE ALL CACHES
    # --------------------------------------------------------

    def invalidate_caches(self):

        for block in self.blocks:
            block.ffn.invalidate_cache()

        self._prefix_cache.clear()

        self._prefix_cache_hits = 0
        self._prefix_cache_misses = 0
        self._prefix_cache_evictions = 0

        self._parameter_signature = (
            self._get_parameter_signature()
        )

    # --------------------------------------------------------
    # AUTOREGRESSIVE CACHE PREFILL
    # --------------------------------------------------------

    @torch.inference_mode()
    def create_generation_cache(
        self,
        tokens,
        max_cache_len=None
    ):

        if self.training:
            raise RuntimeError(
                "Model must be in evaluation mode."
            )

        if tokens.device.type != "cpu":
            raise RuntimeError(
                "Generation cache currently supports CPU only."
            )

        if tokens.dim() != 2:
            raise ValueError(
                "Tokens must have shape [batch, sequence]."
            )

        batch_size, seq_len = tokens.shape

        if seq_len == 0:
            raise ValueError(
                "Prompt cannot be empty."
            )

        if max_cache_len is None:
            max_cache_len = self.max_seq_len

        if not isinstance(max_cache_len, int):
            raise TypeError(
                "max_cache_len must be an integer."
            )

        if not seq_len <= max_cache_len <= self.max_seq_len:
            raise ValueError(
                "max_cache_len must be between prompt length "
                "and max_seq_len."
            )

        positions = torch.arange(
            seq_len,
            device=tokens.device
        ).unsqueeze(0)

        x = self.token_embedding(tokens)

        x = x + self.position_embedding(positions)

        keys = []
        values = []

        for block in self.blocks:

            qkv = block.attention.in_proj(x)

            q, k, v = qkv.chunk(3, dim=-1)

            k = block.attention.split_heads(k)
            v = block.attention.split_heads(v)

            key_buffer = torch.empty(
                batch_size,
                self.num_heads,
                max_cache_len,
                self.hidden_size // self.num_heads,
                dtype=k.dtype,
                device=k.device
            )

            value_buffer = torch.empty_like(key_buffer)

            key_buffer[:, :, :seq_len, :].copy_(k)
            value_buffer[:, :, :seq_len, :].copy_(v)

            x = block(x)

            keys.append(key_buffer)
            values.append(value_buffer)

        x = self.final_norm(x)

        logits = self.lm_head(x[:, -1, :])

        cache = {
            "keys": keys,
            "values": values,
            "length": seq_len,
            "capacity": max_cache_len,
            "batch_size": batch_size,
            "parameter_signature": self._get_parameter_signature()
        }

        return logits, cache

    # --------------------------------------------------------
    # AUTOREGRESSIVE DECODE STEP
    # --------------------------------------------------------

    @torch.inference_mode()
    def decode_step(self, next_token, cache):

        if self.training:
            raise RuntimeError(
                "Model must be in evaluation mode."
            )

        if next_token.device.type != "cpu":
            raise RuntimeError(
                "Generation cache currently supports CPU only."
            )

        if next_token.dim() == 1:
            next_token = next_token.unsqueeze(1)

        if next_token.dim() != 2 or next_token.size(1) != 1:
            raise ValueError(
                "next_token must have shape [batch] or [batch, 1]."
            )

        if next_token.size(0) != cache["batch_size"]:
            raise ValueError(
                "Batch size does not match the cache."
            )

        if (
            self._get_parameter_signature()
            != cache["parameter_signature"]
        ):
            raise RuntimeError(
                "Model parameters changed after cache creation. "
                "Create a new generation cache."
            )

        if len(cache["keys"]) != len(self.blocks):
            raise ValueError("Invalid key cache.")

        if len(cache["values"]) != len(self.blocks):
            raise ValueError("Invalid value cache.")

        position = cache["length"]

        if position >= cache["capacity"]:
            raise ValueError(
                "KV cache capacity exceeded."
            )

        if position >= self.max_seq_len:
            raise ValueError(
                "Maximum sequence length exceeded."
            )

        positions = torch.full(
            (next_token.size(0), 1),
            position,
            dtype=torch.long,
            device=next_token.device
        )

        x = self.token_embedding(next_token)

        x = x + self.position_embedding(positions)

        for index, block in enumerate(self.blocks):

            x = block.forward_incremental(
                x,
                cache["keys"][index],
                cache["values"][index],
                position
            )

        cache["length"] += 1

        x = self.final_norm(x)

        logits = self.lm_head(x[:, -1, :])

        return logits, cache

    # --------------------------------------------------------
    # TOKEN SELECTION
    # --------------------------------------------------------

    @staticmethod
    def _sample_next_token(
        logits,
        temperature=0.0,
        top_k=None
    ):

        if top_k is not None:

            if not isinstance(top_k, int):
                raise TypeError(
                    "top_k must be an integer or None."
                )

            if top_k < 0:
                raise ValueError(
                    "top_k cannot be negative."
                )

        if temperature < 0:
            raise ValueError(
                "temperature cannot be negative."
            )

        if top_k is not None and top_k > 0:

            k = min(top_k, logits.size(-1))

            top_values, _ = torch.topk(
                logits,
                k=k,
                dim=-1
            )

            threshold = top_values[:, -1].unsqueeze(-1)

            logits = logits.masked_fill(
                logits < threshold,
                float("-inf")
            )

        if temperature == 0:

            return torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        probabilities = torch.softmax(
            logits / temperature,
            dim=-1
        )

        return torch.multinomial(
            probabilities,
            num_samples=1
        )

    # --------------------------------------------------------
    # CACHED GENERATION
    # --------------------------------------------------------

    @torch.inference_mode()
    def generate_cached(
        self,
        tokens,
        max_new_tokens,
        temperature=0.0,
        top_k=None
    ):

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        if tokens.dim() != 2:
            raise ValueError(
                "Tokens must have shape [batch, sequence]."
            )

        if (
            tokens.size(1) + max_new_tokens
            > self.max_seq_len
        ):
            raise ValueError(
                "Prompt plus generated tokens exceeds max_seq_len."
            )

        generated = tokens.clone()

        if max_new_tokens == 0:
            return generated

        logits, cache = self.create_generation_cache(
            generated,
            max_cache_len=(
                generated.size(1) + max_new_tokens
            )
        )

        for step in range(max_new_tokens):

            next_token = self._sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

            if step + 1 < max_new_tokens:

                logits, cache = self.decode_step(
                    next_token,
                    cache
                )

        return generated

    # --------------------------------------------------------
    # STANDARD GENERATION BASELINE
    # --------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        tokens,
        max_new_tokens,
        temperature=0.0,
        top_k=None
    ):

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        generated = tokens.clone()

        for _ in range(max_new_tokens):

            if generated.size(1) >= self.max_seq_len:
                break

            logits = self.forward(generated)

            logits = logits[:, -1, :]

            next_token = self._sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

        return generated
