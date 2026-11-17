
import torch

from model.rct_100m import RCT100M, RCT100MConfig
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_prefix_generation import RCT100MPrefixGeneration


class RCTInference:
    """
    Unified RCT inference interface.

    Modes:
      standard       : Full-context generation.
      kv             : Incremental generation with KV caching.
      prefix         : Shared-prefix generation, manual or automatic.
      adaptive       : Exact complete-prompt deduplication.
      auto           : Benchmark-guided adaptive prefix gate with
                       adaptive KV fallback.

    Adaptive prefix policy:
      - Batch 4, prefix 16/24: use prefix generation.
      - Batch 8, prefix 24: use prefix generation.
      - Other tested or untested combinations: use adaptive KV.

    The initial gate is deliberately restricted to:
      - Prompt length: 32 tokens.
      - New tokens: 8.

    These restrictions reflect the benchmark conditions. They are
    not universal performance guarantees.
    """

    AUTO_PREFIX_PROMPT_LENGTH = 32
    AUTO_PREFIX_NEW_TOKENS = 8

    AUTO_PREFIX_POLICY = {
    4: {24},
    8: {24},
    }

    def __init__(
        self,
        model,
        prefix_cache_capacity=16,
        dedup_threshold=0.75,
    ):

        if not isinstance(model, RCT100M):
            raise TypeError("model must be an RCT100M instance.")

        self._validate_dedup_threshold(dedup_threshold)

        self.model = model
        self.model.eval()

        self.dedup_threshold = float(dedup_threshold)

        self.kv_model = RCT100MKV(self.model)
        self.kv_model.eval()

        self.prefix_model = RCT100MPrefixGeneration(
            self.model,
            cache_capacity=prefix_cache_capacity,
        )
        self.prefix_model.eval()

        self.reset_adaptive_statistics()
        self.reset_prefix_statistics()
        self.reset_prefix_gate_statistics()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    @staticmethod
    def _validate_dedup_threshold(value):

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(
                "dedup_threshold must be a number between 0 and 1."
            )

        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                "dedup_threshold must be between 0 and 1."
            )

    @staticmethod
    def _validate_prompt(prompt):

        if not isinstance(prompt, torch.Tensor):
            raise TypeError("prompt must be a torch.Tensor.")

        if prompt.ndim != 2:
            raise ValueError(
                "prompt must have shape [batch_size, sequence_length]."
            )

        if prompt.size(0) < 1 or prompt.size(1) < 1:
            raise ValueError("prompt cannot be empty.")

        if prompt.dtype != torch.long:
            raise TypeError("prompt must use torch.long token IDs.")

    def _validate_generation_length(self, prompt, max_new_tokens):

        if isinstance(max_new_tokens, bool) or not isinstance(
            max_new_tokens, int
        ):
            raise TypeError("max_new_tokens must be an integer.")

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        max_seq_len = self.model.config.max_seq_len

        if prompt.size(1) + max_new_tokens > max_seq_len:
            raise ValueError(
                f"Prompt length plus generated tokens exceeds "
                f"max_seq_len={max_seq_len}."
            )

    @staticmethod
    def _validate_fallback(fallback):

        if not isinstance(fallback, bool):
            raise TypeError("fallback must be a boolean.")

    def _validate_prefix(self, prompt, prefix_len):

        if isinstance(prefix_len, bool) or not isinstance(
            prefix_len, int
        ):
            raise TypeError("prefix_len must be an integer.")

        if prefix_len < 1 or prefix_len > prompt.size(1):
            raise ValueError(
                "prefix_len must be between 1 and prompt length."
            )

        if prompt.size(0) > 1:

            shared_prefix = prompt[:1, :prefix_len].expand(
                prompt.size(0),
                -1,
            )

            if not torch.equal(
                prompt[:, :prefix_len],
                shared_prefix,
            ):
                raise ValueError(
                    "Prefix mode requires every prompt to share "
                    "the exact same prefix."
                )

    # --------------------------------------------------
    # Shared-prefix detection
    # --------------------------------------------------

    @torch.inference_mode()
    def detect_shared_prefix(self, prompt):
        """
        Return the longest exact prefix shared by every prompt.

        Returns zero if the first token differs across prompts.
        """

        self._validate_prompt(prompt)

        return self.prefix_model.longest_common_prefix_length(
            prompt
        )

    # --------------------------------------------------
    # Prefix statistics
    # --------------------------------------------------

    def reset_prefix_statistics(self):

        self._prefix_calls = 0
        self._prefix_automatic_calls = 0
        self._prefix_manual_calls = 0
        self._prefix_bypasses = 0
        self._prefix_last_detected_length = 0

    def prefix_statistics(self):

        return {
            "prefix_calls": self._prefix_calls,
            "automatic_prefix_calls": self._prefix_automatic_calls,
            "manual_prefix_calls": self._prefix_manual_calls,
            "prefix_bypasses": self._prefix_bypasses,
            "last_detected_prefix_length": (
                self._prefix_last_detected_length
            ),
            "cache": self.prefix_model.prefix_cache.cache_statistics(),
        }

    # --------------------------------------------------
    # Adaptive prefix gate statistics
    # --------------------------------------------------

    def reset_prefix_gate_statistics(self):

        self._prefix_gate_calls = 0
        self._prefix_gate_activations = 0
        self._prefix_gate_bypasses = 0
        self._prefix_gate_fallbacks = 0

        self._prefix_gate_last_batch_size = 0
        self._prefix_gate_last_prefix_length = 0
        self._prefix_gate_last_decision = "not_run"
        self._prefix_gate_last_reason = "not_run"

    def prefix_gate_statistics(self):

        return {
            "gate_calls": self._prefix_gate_calls,
            "gate_activations": self._prefix_gate_activations,
            "gate_bypasses": self._prefix_gate_bypasses,
            "gate_fallbacks": self._prefix_gate_fallbacks,
            "last_batch_size": self._prefix_gate_last_batch_size,
            "last_prefix_length": (
                self._prefix_gate_last_prefix_length
            ),
            "last_decision": self._prefix_gate_last_decision,
            "last_reason": self._prefix_gate_last_reason,
            "supported_prompt_length": (
                self.AUTO_PREFIX_PROMPT_LENGTH
            ),
            "supported_new_tokens": self.AUTO_PREFIX_NEW_TOKENS,
            "policy": {
                batch: sorted(prefixes)
                for batch, prefixes in self.AUTO_PREFIX_POLICY.items()
            },
        }

    def _prefix_gate_decision(
        self,
        prompt,
        max_new_tokens,
        prefix_len,
    ):
        """
        Decide whether the benchmark supports prefix reuse
        for this workload.

        Returns:
            (True, reason) if prefix generation is selected.
            (False, reason) otherwise.
        """

        batch_size = prompt.size(0)
        prompt_length = prompt.size(1)

        if prefix_len <= 0:
            return False, "no_shared_prefix"

        if prompt_length != self.AUTO_PREFIX_PROMPT_LENGTH:
            return False, "untested_prompt_length"

        if max_new_tokens != self.AUTO_PREFIX_NEW_TOKENS:
            return False, "untested_generation_length"

        supported_prefixes = self.AUTO_PREFIX_POLICY.get(
            batch_size
        )

        if supported_prefixes is None:
            return False, "untested_batch_size"

        if prefix_len not in supported_prefixes:
            return False, "unprofitable_or_untested_prefix_length"

        return True, "benchmark_supported_workload"

    # --------------------------------------------------
    # Adaptive deduplication statistics
    # --------------------------------------------------

    def reset_adaptive_statistics(self):

        self._adaptive_calls = 0
        self._adaptive_checks = 0
        self._adaptive_bypasses = 0
        self._adaptive_activations = 0

        self._adaptive_total_prompts = 0
        self._adaptive_unique_processed = 0

        self._adaptive_duplicates_detected = 0
        self._adaptive_duplicates_saved = 0

        self._last_duplicate_rate = 0.0

    def adaptive_statistics(self):

        total = self._adaptive_total_prompts

        if total > 0:
            overall_duplicate_rate = (
                self._adaptive_duplicates_detected / total
            )
        else:
            overall_duplicate_rate = 0.0

        return {
            "dedup_threshold": self.dedup_threshold,
            "adaptive_calls": self._adaptive_calls,
            "dedup_checks": self._adaptive_checks,
            "dedup_bypasses": self._adaptive_bypasses,
            "dedup_activations": self._adaptive_activations,
            "total_prompts_received": total,
            "unique_prompts_processed": (
                self._adaptive_unique_processed
            ),
            "duplicate_prompts_detected": (
                self._adaptive_duplicates_detected
            ),
            "duplicate_prompts_saved": (
                self._adaptive_duplicates_saved
            ),
            "overall_duplicate_rate": overall_duplicate_rate,
            "last_batch_duplicate_rate": self._last_duplicate_rate,
        }

    # --------------------------------------------------
    # Stable prompt deduplication
    # --------------------------------------------------

    def _stable_deduplicate(self, prompts):
        """
        Deduplicate complete prompts while preserving the
        order of their first occurrence.
        """

        unique_prompts, inverse_indices = torch.unique(
            prompts,
            dim=0,
            return_inverse=True,
        )

        unique_count = unique_prompts.size(0)
        batch_size = prompts.size(0)

        first_positions = torch.full(
            (unique_count,),
            batch_size,
            dtype=torch.long,
            device=prompts.device,
        )

        first_positions.scatter_reduce_(
            0,
            inverse_indices,
            torch.arange(
                batch_size,
                device=prompts.device,
            ),
            reduce="amin",
            include_self=True,
        )

        order = torch.argsort(first_positions)

        unique_prompts = unique_prompts.index_select(
            0,
            order,
        )

        old_to_new = torch.empty_like(order)

        old_to_new[order] = torch.arange(
            unique_count,
            device=prompts.device,
        )

        inverse_indices = old_to_new[inverse_indices]

        return unique_prompts, inverse_indices

    # --------------------------------------------------
    # Prefix generation
    # --------------------------------------------------

    @torch.inference_mode()
    def _prefix_generate(
        self,
        prompt,
        max_new_tokens,
        prefix_len=None,
        automatic=False,
    ):
        """
        Run prefix-assisted generation.

        automatic=True records that the prefix length was
        selected by automatic detection or the adaptive gate.
        """

        self._prefix_calls += 1

        if prefix_len is None:

            automatic = True

            prefix_len = self.detect_shared_prefix(prompt)

            self._prefix_last_detected_length = prefix_len

            if prefix_len == 0:

                self._prefix_bypasses += 1

                raise ValueError(
                    "No shared prefix exists across the batch."
                )

        if automatic:
            self._prefix_automatic_calls += 1
        else:
            self._prefix_manual_calls += 1

        self._validate_prefix(
            prompt,
            prefix_len,
        )

        self._prefix_last_detected_length = prefix_len

        return self.prefix_model.generate_greedy(
            prompt,
            prefix_len=prefix_len,
            max_new_tokens=max_new_tokens,
        )

    # --------------------------------------------------
    # Adaptive prefix generation
    # --------------------------------------------------

    @torch.inference_mode()
    def _adaptive_prefix_generate(
        self,
        prompt,
        max_new_tokens,
        prefix_len=None,
        fallback=True,
    ):
        """
        Apply the measured prefix gate.

        If the workload is not supported by the benchmark,
        use adaptive KV generation instead.
        """

        self._prefix_gate_calls += 1

        batch_size = prompt.size(0)

        self._prefix_gate_last_batch_size = batch_size

        # Detect the common prefix unless the caller explicitly
        # supplied a prefix length.
        if prefix_len is None:

            prefix_len = self.detect_shared_prefix(prompt)

        else:

            self._validate_prefix(
                prompt,
                prefix_len,
            )

        self._prefix_gate_last_prefix_length = prefix_len

        should_use_prefix, reason = self._prefix_gate_decision(
            prompt,
            max_new_tokens,
            prefix_len,
        )

        if not should_use_prefix:

            self._prefix_gate_bypasses += 1
            self._prefix_gate_last_decision = "kv"
            self._prefix_gate_last_reason = reason

            if not fallback:
                raise ValueError(
                    f"Adaptive prefix gate selected KV fallback: {reason}."
                )

            return self._adaptive_kv_generate(
                prompt,
                max_new_tokens,
            )

        self._prefix_gate_activations += 1
        self._prefix_gate_last_decision = "prefix"
        self._prefix_gate_last_reason = reason

        try:

            return self._prefix_generate(
                prompt,
                max_new_tokens,
                prefix_len=prefix_len,
                automatic=True,
            )

        except (ValueError, RuntimeError):

            if not fallback:
                raise

            self._prefix_gate_fallbacks += 1
            self._prefix_gate_last_decision = "kv"
            self._prefix_gate_last_reason = (
                "prefix_generation_failed"
            )

            return self._adaptive_kv_generate(
                prompt,
                max_new_tokens,
            )

    # --------------------------------------------------
    # Public generation interface
    # --------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        max_new_tokens=8,
        mode="auto",
        prefix_len=None,
        fallback=True,
    ):

        self._validate_prompt(prompt)

        self._validate_generation_length(
            prompt,
            max_new_tokens,
        )

        valid_modes = (
            "standard",
            "kv",
            "prefix",
            "adaptive",
            "auto",
        )

        if mode not in valid_modes:
            raise ValueError(
                "mode must be standard, kv, prefix, adaptive, or auto."
            )

        self._validate_fallback(fallback)

        if mode == "standard":

            return self._standard_generate(
                prompt,
                max_new_tokens,
            )

        if mode == "kv":

            return self._kv_generate(
                prompt,
                max_new_tokens,
            )

        if mode == "adaptive":

            return self._adaptive_kv_generate(
                prompt,
                max_new_tokens,
            )

        if mode == "prefix":

            try:

                return self._prefix_generate(
                    prompt,
                    max_new_tokens,
                    prefix_len=prefix_len,
                )

            except (ValueError, RuntimeError):

                if not fallback:
                    raise

                return self._adaptive_kv_generate(
                    prompt,
                    max_new_tokens,
                )

        # AUTO MODE:
        # Use the benchmark-guided prefix gate.
        # Unprofitable or untested workloads use adaptive KV.

        return self._adaptive_prefix_generate(
            prompt,
            max_new_tokens,
            prefix_len=prefix_len,
            fallback=fallback,
        )

    # --------------------------------------------------
    # Standard full-context generation
    # --------------------------------------------------

    @torch.inference_mode()
    def _standard_generate(self, prompt, max_new_tokens):

        generated = prompt.clone()

        for _ in range(max_new_tokens):

            logits = self.model(generated)

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated = torch.cat(
                [generated, next_token],
                dim=1,
            )

        return generated

    # --------------------------------------------------
    # KV-cache generation
    # --------------------------------------------------

    @torch.inference_mode()
    def _kv_generate(self, prompt, max_new_tokens):

        cache = self.kv_model.create_cache(
            prompt.size(0)
        )

        logits = self.kv_model.decode_step(
            prompt,
            cache,
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

    # --------------------------------------------------
    # Adaptive exact-prompt deduplication
    # --------------------------------------------------

    @torch.inference_mode()
    def _adaptive_kv_generate(self, prompt, max_new_tokens):
        """
        Deduplicate exact complete prompts only when their
        duplicate rate reaches dedup_threshold.
        """

        batch_size = prompt.size(0)

        self._adaptive_calls += 1
        self._adaptive_total_prompts += batch_size

        if batch_size == 1:

            self._adaptive_bypasses += 1
            self._adaptive_unique_processed += 1
            self._last_duplicate_rate = 0.0

            return self._kv_generate(
                prompt,
                max_new_tokens,
            )

        self._adaptive_checks += 1

        unique_prompts, inverse_indices = (
            self._stable_deduplicate(prompt)
        )

        unique_count = unique_prompts.size(0)
        duplicate_count = batch_size - unique_count

        duplicate_rate = duplicate_count / batch_size

        self._adaptive_duplicates_detected += duplicate_count
        self._last_duplicate_rate = duplicate_rate

        if duplicate_rate < self.dedup_threshold:

            self._adaptive_bypasses += 1
            self._adaptive_unique_processed += batch_size

            return self._kv_generate(
                prompt,
                max_new_tokens,
            )

        self._adaptive_activations += 1
        self._adaptive_unique_processed += unique_count
        self._adaptive_duplicates_saved += duplicate_count

        unique_outputs = self._kv_generate(
            unique_prompts,
            max_new_tokens,
        )

        return unique_outputs.index_select(
            0,
            inverse_indices,
        )


# ------------------------------------------------------
# Checkpoint loader
# ------------------------------------------------------

def load_rct_inference(
    checkpoint_path,
    device=None,
    prefix_cache_capacity=16,
    dedup_threshold=0.75,
):

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = RCT100MConfig(
        **checkpoint["model_config"]
    )

    model = RCT100M(config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    inference = RCTInference(
        model,
        prefix_cache_capacity=prefix_cache_capacity,
        dedup_threshold=dedup_threshold,
    )

    return inference
