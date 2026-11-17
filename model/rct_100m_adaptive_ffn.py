import torch
import time


class RCT100MAdaptiveFFNExecutor:
    """
    Experimental inference-time FFN execution controller.

    Modes:
        standard:
            Execute the original FFN.

        exact_dedup:
            Compute identical activation vectors once.

        adaptive:
            Estimate duplication using a sample, then decide
            whether exact deduplication is worthwhile.

    The original FFN module and its trained parameters are
    preserved without modification.
    """

    VALID_MODES = (
        "standard",
        "exact_dedup",
        "adaptive",
    )

    def __init__(
        self,
        ffn,
        duplicate_threshold=0.25,
        sample_size=32,
    ):

        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError(
                "duplicate_threshold must be between 0 and 1."
            )

        if not isinstance(sample_size, int) or sample_size < 1:
            raise ValueError(
                "sample_size must be a positive integer."
            )

        self.ffn = ffn
        self.duplicate_threshold = duplicate_threshold
        self.sample_size = sample_size

        self.reset_statistics()

    def reset_statistics(self):

        self.calls = 0

        self.standard_executions = 0
        self.deduplicated_executions = 0

        self.adaptive_calls = 0
        self.adaptive_bypasses = 0
        self.adaptive_acceptances = 0

        self.total_vectors_processed = 0
        self.unique_vectors_computed = 0

        self.estimated_duplicate_ratios = []

        self.standard_execution_seconds = 0.0
        self.dedup_execution_seconds = 0.0
        self.adaptive_decision_seconds = 0.0

    def _validate_input(self, x):

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [batch, sequence, hidden]."
            )

        if x.shape[0] < 1 or x.shape[1] < 1:
            raise ValueError(
                "Batch and sequence dimensions must be positive."
            )

    def _require_inference(self):

        if self.ffn.training:
            raise RuntimeError(
                "FFN must be in evaluation mode."
            )

        if torch.is_grad_enabled():
            raise RuntimeError(
                "Deduplication requires disabled gradients. "
                "Use torch.inference_mode()."
            )

    def _standard(self, x):

        start = time.perf_counter()

        result = self.ffn(x)

        elapsed = time.perf_counter() - start

        self.standard_executions += 1
        self.standard_execution_seconds += elapsed

        self.total_vectors_processed += (
            x.shape[0] * x.shape[1]
        )

        return result

    def _estimate_duplicate_ratio(self, flattened):

        total_vectors = flattened.shape[0]

        count = min(
            self.sample_size,
            total_vectors
        )

        if count == total_vectors:
            sample = flattened
        else:
            indices = torch.linspace(
                0,
                total_vectors - 1,
                steps=count,
                device=flattened.device
            ).long()

            sample = flattened.index_select(
                0,
                indices
            )

        unique_sample = torch.unique(
            sample,
            dim=0
        )

        duplicate_ratio = (
            1.0 - unique_sample.shape[0] / count
        )

        return duplicate_ratio

    def _exact_deduplicate(self, x):

        self._require_inference()

        batch_size, sequence_length, hidden_size = x.shape

        start = time.perf_counter()

        flattened = x.reshape(
            batch_size * sequence_length,
            hidden_size
        )

        unique_vectors, inverse_indices = torch.unique(
            flattened,
            dim=0,
            return_inverse=True
        )

        unique_outputs = self.ffn(unique_vectors)

        reconstructed = unique_outputs.index_select(
            0,
            inverse_indices
        )

        result = reconstructed.reshape(
            batch_size,
            sequence_length,
            hidden_size
        )

        elapsed = time.perf_counter() - start

        total_vectors = flattened.shape[0]
        unique_count = unique_vectors.shape[0]

        self.deduplicated_executions += 1

        self.total_vectors_processed += total_vectors

        self.unique_vectors_computed += unique_count

        self.dedup_execution_seconds += elapsed

        return result

    def execute(self, x, mode="standard"):

        self._validate_input(x)

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode}. "
                f"Choose from {self.VALID_MODES}."
            )

        self.calls += 1

        if mode == "standard":
            return self._standard(x)

        self._require_inference()

        if mode == "exact_dedup":
            return self._exact_deduplicate(x)

        # --------------------------------------------
        # ADAPTIVE EXECUTION
        # --------------------------------------------

        self.adaptive_calls += 1

        flattened = x.reshape(
            -1,
            x.shape[-1]
        )

        decision_start = time.perf_counter()

        estimated_ratio = self._estimate_duplicate_ratio(
            flattened
        )

        self.estimated_duplicate_ratios.append(
            estimated_ratio
        )

        decision_time = (
            time.perf_counter() - decision_start
        )

        self.adaptive_decision_seconds += decision_time

        if estimated_ratio < self.duplicate_threshold:

            self.adaptive_bypasses += 1

            return self._standard(x)

        self.adaptive_acceptances += 1

        return self._exact_deduplicate(x)

    def statistics(self):

        average_estimated_ratio = (
            sum(self.estimated_duplicate_ratios)
            / len(self.estimated_duplicate_ratios)
            if self.estimated_duplicate_ratios
            else 0.0
        )

        reuse_ratio = (
            1.0
            - self.unique_vectors_computed
            / self.total_vectors_processed
            if self.total_vectors_processed > 0
            else 0.0
        )

        return {
            "calls": self.calls,

            "standard_executions": (
                self.standard_executions
            ),

            "deduplicated_executions": (
                self.deduplicated_executions
            ),

            "adaptive_calls": self.adaptive_calls,

            "adaptive_bypasses": self.adaptive_bypasses,

            "adaptive_acceptances": (
                self.adaptive_acceptances
            ),

            "total_vectors_processed": (
                self.total_vectors_processed
            ),

            "unique_vectors_computed": (
                self.unique_vectors_computed
            ),

            "reuse_ratio": reuse_ratio,

            "average_estimated_duplicate_ratio": (
                average_estimated_ratio
            ),

            "standard_execution_seconds": (
                self.standard_execution_seconds
            ),

            "dedup_execution_seconds": (
                self.dedup_execution_seconds
            ),

            "adaptive_decision_seconds": (
                self.adaptive_decision_seconds
            ),

            "duplicate_threshold": (
                self.duplicate_threshold
            ),

            "sample_size": self.sample_size,
        }
