
import torch
import torch.nn as nn

from model.rct_100m_kv import RCT100MKV
from model.rct_100m_prefix import RCT100MPrefixCache


class RCT100MPrefixGeneration(nn.Module):
    """
    Greedy autoregressive generation using:

    1. Exact shared-prefix KV caching.
    2. Automatic longest-common-prefix detection.
    3. Per-request suffix processing.
    4. Incremental KV decoding for generated tokens.

    All prompts in a batch must share the selected prefix.
    """

    def __init__(self, base_model, cache_capacity=16):
        super().__init__()

        self.base_model = base_model
        self.config = base_model.config

        self.prefix_cache = RCT100MPrefixCache(
            base_model,
            cache_capacity=cache_capacity,
        )

        self.kv_model = RCT100MKV(base_model)

    @staticmethod
    def longest_common_prefix_length(tokens):
        """
        Find the longest prefix shared by every batch row.

        Returns:
            Number of shared tokens.
            Returns 0 when the first tokens differ.
        """

        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor.")

        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape [batch, sequence]."
            )

        if tokens.size(0) < 1 or tokens.size(1) < 1:
            raise ValueError("tokens cannot be empty.")

        if tokens.size(0) == 1:
            return tokens.size(1)

        matches = (
            tokens == tokens[:1]
        ).all(dim=0)

        mismatches = torch.nonzero(
            ~matches,
            as_tuple=False,
        )

        if mismatches.numel() == 0:
            return tokens.size(1)

        return int(mismatches[0, 0].item())

    @torch.inference_mode()
    def _get_prefix_entry(self, prefix_tokens):

        prefix = prefix_tokens[:1]

        fingerprint = self.prefix_cache.prefix_fingerprint(
            prefix
        )

        prefix_len = prefix.size(1)

        model = self.base_model

        cache_key = (
            fingerprint,
            prefix_len,
            str(prefix.device),
            str(model.token_embedding.weight.dtype),
            self.prefix_cache.parameter_signature(),
        )

        memory = self.prefix_cache.prefix_cache

        if cache_key in memory:

            self.prefix_cache.cache_hits += 1

            entry = memory.pop(cache_key)
            memory[cache_key] = entry

        else:

            self.prefix_cache.cache_misses += 1

            entry = self.prefix_cache._build_prefix(prefix)

            memory[cache_key] = entry

            while len(memory) > self.prefix_cache.cache_capacity:

                memory.popitem(last=False)
                self.prefix_cache.cache_evictions += 1

        return entry

    @torch.inference_mode()
    def prefill(self, prompt, prefix_len=None):

        if self.training or self.base_model.training:
            raise RuntimeError(
                "Call eval() before prefix-assisted generation."
            )

        if not isinstance(prompt, torch.Tensor):
            raise TypeError("prompt must be a torch.Tensor.")

        if prompt.ndim != 2 or prompt.size(1) < 1:
            raise ValueError(
                "Prompt must have shape [batch, sequence]."
            )

        if prompt.size(0) < 1:
            raise ValueError("Prompt batch cannot be empty.")

        if prompt.dtype != torch.long:
            raise TypeError("Prompt must use torch.long.")

        batch_size, sequence_length = prompt.shape

        if sequence_length > self.config.max_seq_len:
            raise ValueError("Prompt exceeds maximum sequence length.")

        embedding = self.base_model.token_embedding

        if prompt.device != embedding.weight.device:
            raise ValueError(
                "Prompt and model must be on the same device."
            )

        if prompt.min().item() < 0 or (
            prompt.max().item() >= self.config.vocab_size
        ):
            raise ValueError("Token ID outside vocabulary.")

        # Automatically discover the shared prefix when omitted.
        if prefix_len is None:

            prefix_len = self.longest_common_prefix_length(
                prompt
            )

            if prefix_len == 0:
                raise ValueError(
                    "No shared prefix exists across the batch."
                )

        if isinstance(prefix_len, bool) or not isinstance(
            prefix_len, int
        ):
            raise TypeError("prefix_len must be an integer.")

        if not 1 <= prefix_len <= sequence_length:
            raise ValueError("Invalid prefix length.")

        prefix_tokens = prompt[:, :prefix_len]

        expected_prefix = prefix_tokens[:1].expand_as(
            prefix_tokens
        )

        if not torch.equal(prefix_tokens, expected_prefix):
            raise ValueError(
                "All prompts must share the selected prefix."
            )

        entry = self._get_prefix_entry(prefix_tokens)

        # Allocate the ordinary incremental KV cache.
        cache = self.kv_model.create_cache(batch_size)

        suffix_tokens = prompt[:, prefix_len:]

        # Copy shared-prefix KV tensors directly into the
        # writable batch cache.
        #
        # copy_ supports broadcasting from batch dimension 1
        # to batch_size. The destination remains independently
        # writable for each batch item during decoding.
        #
        # These copies are required even when the suffix is empty.
        for index in range(len(self.base_model.blocks)):

            cache.keys[index][
                :, :, :prefix_len, :
            ].copy_(
                entry["keys"][index]
            )

            cache.values[index][
                :, :, :prefix_len, :
            ].copy_(
                entry["values"][index]
            )

        if suffix_tokens.size(1) > 0:

            positions = torch.arange(
                prefix_len,
                sequence_length,
                device=prompt.device,
            )

            x = (
                embedding(suffix_tokens)
                + self.base_model.position_embedding(
                    positions
                ).unsqueeze(0)
            )

            x = self.base_model.embedding_dropout(x)

            for index, block in enumerate(self.base_model.blocks):

                normalized = block.norm1(x)

                prefix_key = entry["keys"][index].expand(
                    batch_size, -1, -1, -1
                )

                prefix_value = entry["values"][index].expand(
                    batch_size, -1, -1, -1
                )

                attention_output, suffix_key, suffix_value = (
                    self.prefix_cache._attention(
                        block.attention,
                        normalized,
                        prefix_key,
                        prefix_value,
                        prefix_len,
                    )
                )

                cache.keys[index][
                    :, :, prefix_len:sequence_length, :
                ].copy_(suffix_key)

                cache.values[index][
                    :, :, prefix_len:sequence_length, :
                ].copy_(suffix_value)

                x = x + block.dropout(attention_output)

                ffn_output = block.ffn(block.norm2(x))

                x = x + block.dropout(ffn_output)

            prefix_hidden = entry["hidden"].expand(
                batch_size, -1, -1
            )

            hidden = torch.cat(
                [prefix_hidden, x],
                dim=1,
            )

        else:

            hidden = entry["hidden"].expand(
                batch_size, -1, -1
            )

        cache.position = sequence_length

        hidden = self.base_model.final_norm(hidden)

        logits = self.base_model.lm_head(hidden)

        return logits, cache

    @torch.inference_mode()
    def generate_greedy(
        self,
        prompt,
        prefix_len=None,
        max_new_tokens=8,
    ):

        if isinstance(max_new_tokens, bool) or not isinstance(
            max_new_tokens, int
        ):
            raise TypeError(
                "max_new_tokens must be an integer."
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        if (
            prompt.size(1) + max_new_tokens
            > self.config.max_seq_len
        ):
            raise ValueError(
                "Generation exceeds maximum sequence length."
            )

        logits, cache = self.prefill(
            prompt,
            prefix_len,
        )

        generated = prompt.clone()

        for step in range(max_new_tokens):

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

            if step + 1 < max_new_tokens:

                logits = self.kv_model.decode_step(
                    next_token,
                    cache,
                )

        return generated
