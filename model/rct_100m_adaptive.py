
import math

import torch
import torch.nn as nn

from model.rct_100m_kv import RCT100MKV


class RCT100MAdaptive(nn.Module):
    """
    Adaptive RCT using fingerprint-based duplicate detection.

    Fingerprint collisions can cause unnecessary exact deduplication,
    but exact duplicate prompts cannot be missed by this prefilter.
    """

    def __init__(
        self,
        base_model,
        duplicate_threshold=0.25,
        fingerprint_columns=4
    ):
        super().__init__()

        self._validate_threshold(duplicate_threshold)

        if isinstance(fingerprint_columns, bool) or not isinstance(
            fingerprint_columns, int
        ):
            raise TypeError(
                "fingerprint_columns must be an integer"
            )

        if fingerprint_columns < 1:
            raise ValueError(
                "fingerprint_columns must be positive"
            )

        self.kv_model = RCT100MKV(base_model)
        self.duplicate_threshold = float(duplicate_threshold)
        self.fingerprint_columns = fingerprint_columns

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

        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
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
            "total_unique_prompts_processed": 0,
            "total_duplicate_prompts_saved": 0,
            "fingerprint_duplicate_rate_sum": 0.0,
            "actual_duplicate_rate_sum": 0.0,
            "exact_duplicate_rate_observations": 0,
            "last_fingerprint_duplicate_rate": 0.0,
            "last_actual_duplicate_rate": None,
            "last_decision": "none"
        }

    def statistics(self):
        result = dict(self._statistics)

        calls = max(result["total_calls"], 1)
        exact_observations = max(
            result["exact_duplicate_rate_observations"], 1
        )

        result["mean_fingerprint_duplicate_rate"] = (
            result["fingerprint_duplicate_rate_sum"] / calls
        )

        result["mean_actual_duplicate_rate_when_deduplicated"] = (
            result["actual_duplicate_rate_sum"]
            / exact_observations
        )

        result["duplicate_threshold"] = self.duplicate_threshold
        result["fingerprint_columns"] = self.fingerprint_columns

        return result

    @torch.inference_mode()
    def estimate_duplicate_rate(self, prompts):
        """
        Estimate duplicate rate using selected token positions.

        Identical full prompts necessarily have identical fingerprints.
        Fingerprint collisions can increase the estimate but cannot
        decrease it below the actual duplicate rate.
        """

        batch_size, sequence_length = prompts.shape

        if batch_size < 2:
            return 0.0

        positions = sorted(set(
            round(
                i * (sequence_length - 1)
                / max(self.fingerprint_columns - 1, 1)
            )
            for i in range(self.fingerprint_columns)
        ))

        positions = torch.tensor(
            positions,
            dtype=torch.long,
            device=prompts.device
        )

        fingerprints = prompts.index_select(1, positions)

        unique_fingerprints = torch.unique(
            fingerprints,
            dim=0
        )

        candidate_duplicates = (
            batch_size - unique_fingerprints.size(0)
        )

        return candidate_duplicates / batch_size

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
            0, order
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
                "Prompts must have shape [batch, sequence]."
            )

        batch_size, sequence_length = prompts.shape

        if batch_size < 1 or sequence_length < 1:
            raise ValueError("Prompts cannot be empty.")

        if prompts.dtype != torch.long:
            raise TypeError("Prompts must use torch.long.")

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
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError(
                "temperature must be a finite positive number."
            )

        if (
            sequence_length + max_new_tokens
            > self.kv_model.config.max_seq_len
        ):
            raise ValueError(
                "Generation exceeds maximum sequence length."
            )

        self._statistics["total_calls"] += 1
        self._statistics["total_prompts"] += batch_size

        # Stage 1: Fingerprint detection.
        if batch_size < 2 or max_new_tokens == 0:
            fingerprint_rate = 0.0
        else:
            fingerprint_rate = self.estimate_duplicate_rate(
                prompts
            )

        self._statistics[
            "last_fingerprint_duplicate_rate"
        ] = fingerprint_rate

        self._statistics[
            "fingerprint_duplicate_rate_sum"
        ] += fingerprint_rate

        use_deduplication = (
            fingerprint_rate >= self.duplicate_threshold
            and fingerprint_rate > 0.0
        )

        # Stage 2: Exact deduplication.
        if use_deduplication:
            unique_prompts, inverse_indices = (
                self.stable_deduplicate(prompts)
            )

            unique_count = unique_prompts.size(0)
            duplicate_count = batch_size - unique_count

            actual_rate = duplicate_count / batch_size

            self._statistics["deduplicated_calls"] += 1

            self._statistics[
                "total_unique_prompts_processed"
            ] += unique_count

            self._statistics[
                "total_duplicate_prompts_saved"
            ] += duplicate_count

            self._statistics[
                "actual_duplicate_rate_sum"
            ] += actual_rate

            self._statistics[
                "exact_duplicate_rate_observations"
            ] += 1

            self._statistics[
                "last_actual_duplicate_rate"
            ] = actual_rate

            self._statistics["last_decision"] = "deduplicated"

            unique_outputs = self.kv_model.generate_cached(
                unique_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

            return unique_outputs.index_select(
                0, inverse_indices
            )

        # Stage 3: KV-only inference.
        self._statistics["kv_only_calls"] += 1

        self._statistics[
            "total_unique_prompts_processed"
        ] += batch_size

        self._statistics["last_actual_duplicate_rate"] = None
        self._statistics["last_decision"] = "kv_only"

        return self.kv_model.generate_cached(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
