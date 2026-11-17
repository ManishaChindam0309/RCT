
import math

import torch
import torch.nn as nn

from model.rct_100m_kv import RCT100MKV


class RCT100MAdaptive(nn.Module):
    """
    Adaptive RCT inference.

    Uses a small exact-comparison sample to estimate
    prompt duplication before choosing an inference path.

    High estimated duplication:
        Stable deduplication + KV cache.

    Low estimated duplication:
        KV cache without full deduplication.
    """

    def __init__(
        self,
        base_model,
        duplicate_threshold=0.25,
        probe_size=8
    ):
        super().__init__()

        self._validate_threshold(duplicate_threshold)

        if isinstance(probe_size, bool) or not isinstance(
            probe_size, int
        ):
            raise TypeError("probe_size must be an integer")

        if probe_size < 2:
            raise ValueError("probe_size must be at least 2")

        self.kv_model = RCT100MKV(base_model)

        self.duplicate_threshold = float(duplicate_threshold)
        self.probe_size = probe_size

        self.reset_statistics()
        self.eval()

    @staticmethod
    def _validate_threshold(value):

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(
                "duplicate_threshold must be numeric"
            )

        if (
            not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                "duplicate_threshold must be between 0 and 1"
            )

    def train(self, mode=True):

        super().train(False)
        self.kv_model.eval()

        return self

    def parameter_count(self):
        return self.kv_model.parameter_count()

    def reset_statistics(self):

        self._statistics = {
            "total_calls": 0,
            "kv_only_calls": 0,
            "deduplicated_calls": 0,
            "total_prompts": 0,
            "total_duplicate_prompts_saved": 0,
            "estimated_duplicate_rate_sum": 0.0,
            "last_estimated_duplicate_rate": 0.0,
            "last_decision": "none"
        }

    def statistics(self):

        result = dict(self._statistics)

        calls = max(result["total_calls"], 1)

        result["mean_estimated_duplicate_rate"] = (
            result["estimated_duplicate_rate_sum"] / calls
        )

        result["duplicate_threshold"] = (
            self.duplicate_threshold
        )

        result["probe_size"] = self.probe_size

        return result

    @torch.inference_mode()
    def estimate_duplicate_rate(self, prompts):
        """
        Estimate duplicate rate using at most probe_size rows.

        The sample is evenly spaced across the input batch.

        This is an estimate, not an exact count of all
        duplicates in the original batch.
        """

        batch_size = prompts.size(0)

        if batch_size < 2:
            return 0.0

        sample_size = min(batch_size, self.probe_size)

        indices = torch.linspace(
            0,
            batch_size - 1,
            steps=sample_size,
            device=prompts.device
        ).round().long()

        sample = prompts.index_select(0, indices)

        equality = (
            sample.unsqueeze(1) == sample.unsqueeze(0)
        ).all(dim=-1)

        previous_matches = torch.tril(
            equality,
            diagonal=-1
        )

        duplicate_rows = previous_matches.any(dim=1)

        return duplicate_rows.float().mean().item()

    @staticmethod
    def stable_deduplicate(prompts):

        unique_prompts, inverse_indices = torch.unique(
            prompts,
            dim=0,
            return_inverse=True
        )

        unique_count = unique_prompts.size(0)
        batch_size = prompts.size(0)

        first_positions = torch.full(
            (unique_count,),
            batch_size,
            dtype=torch.long,
            device=prompts.device
        )

        first_positions.scatter_reduce_(
            0,
            inverse_indices,
            torch.arange(
                batch_size,
                device=prompts.device
            ),
            reduce="amin",
            include_self=True
        )

        order = torch.argsort(first_positions)

        unique_prompts = unique_prompts.index_select(
            0,
            order
        )

        old_to_new = torch.empty_like(order)

        old_to_new[order] = torch.arange(
            unique_count,
            device=prompts.device
        )

        inverse_indices = old_to_new[inverse_indices]

        return unique_prompts, inverse_indices

    @torch.inference_mode()
    def generate(
        self,
        prompts,
        max_new_tokens=20,
        temperature=1.0
    ):

        if self.training:
            raise RuntimeError(
                "Call model.eval() before generation."
            )

        if not isinstance(prompts, torch.Tensor):
            raise TypeError("prompts must be a tensor")

        if prompts.ndim != 2:
            raise ValueError(
                "Prompts must have shape [batch, sequence]"
            )

        batch_size, sequence_length = prompts.shape

        if batch_size < 1 or sequence_length < 1:
            raise ValueError("Prompts cannot be empty")

        if prompts.dtype != torch.long:
            raise TypeError("Prompts must use torch.long")

        if isinstance(max_new_tokens, bool) or not isinstance(
            max_new_tokens, int
        ):
            raise TypeError(
                "max_new_tokens must be an integer"
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative"
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero"
            )

        if (
            sequence_length + max_new_tokens
            > self.kv_model.config.max_seq_len
        ):
            raise ValueError(
                "Generation exceeds maximum sequence length"
            )

        self._statistics["total_calls"] += 1
        self._statistics["total_prompts"] += batch_size

        # No need to detect duplicates for these cases.
        if batch_size == 1 or max_new_tokens == 0:

            estimated_rate = 0.0
            use_deduplication = False

        else:

            estimated_rate = self.estimate_duplicate_rate(
                prompts
            )

            use_deduplication = (
                estimated_rate >= self.duplicate_threshold
                and estimated_rate > 0.0
            )

        self._statistics[
            "last_estimated_duplicate_rate"
        ] = estimated_rate

        self._statistics[
            "estimated_duplicate_rate_sum"
        ] += estimated_rate

        # Deduplication path.
        if use_deduplication:

            unique_prompts, inverse_indices = (
                self.stable_deduplicate(prompts)
            )

            duplicate_count = (
                batch_size - unique_prompts.size(0)
            )

            self._statistics["deduplicated_calls"] += 1

            self._statistics[
                "total_duplicate_prompts_saved"
            ] += duplicate_count

            self._statistics["last_decision"] = (
                "deduplicated"
            )

            unique_outputs = self.kv_model.generate_cached(
                unique_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            return unique_outputs.index_select(
                0,
                inverse_indices
            )

        # KV-only path.
        self._statistics["kv_only_calls"] += 1

        self._statistics["last_decision"] = "kv_only"

        return self.kv_model.generate_cached(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
