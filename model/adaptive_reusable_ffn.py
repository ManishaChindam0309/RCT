
import torch
import torch.nn as nn

from model.adaptive_reuse import AdaptiveReusePolicy


class AdaptiveReusableFFN(nn.Module):

    def __init__(
        self,
        hidden_size,
        expansion=4,
        sample_size=8
    ):
        super().__init__()

        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.GELU(),
            nn.Linear(hidden_size * expansion, hidden_size)
        )

        self.policy = AdaptiveReusePolicy()
        self.sample_size = sample_size

        self.reset_statistics()

    def reset_statistics(self):

        self.standard_executions = 0
        self.deduplicated_executions = 0

        self.sample_rejections = 0
        self.sample_acceptances = 0

        self.total_tokens_processed = 0
        self.unique_tokens_processed = 0

        self.estimated_duplicate_ratios = []

    def _estimate_duplicate_ratio(self, x):

        sequence_length = x.shape[1]

        count = min(
            self.sample_size,
            sequence_length
        )

        indices = (
            torch.arange(
                count,
                device=x.device
            ) * sequence_length
        ) // count

        sample = x[0].index_select(0, indices)

        # Count exact unique sampled token vectors.
        unique_sample = torch.unique(
            sample,
            dim=0
        )

        estimated_unique_ratio = (
            unique_sample.shape[0] / count
        )

        estimated_duplicate_ratio = (
            1.0 - estimated_unique_ratio
        )

        return estimated_duplicate_ratio

    def _record_standard(self, x):

        self.standard_executions += 1

        self.total_tokens_processed += (
            x.shape[0] * x.shape[1]
        )

    def forward(self, x, force_mode=None):

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape [batch, sequence, hidden]"
            )

        batch_size, sequence_length, hidden_size = x.shape

        if sequence_length <= 0:
            raise ValueError(
                "Sequence length must be positive"
            )

        if force_mode not in (
            None,
            "standard",
            "deduplicated"
        ):
            raise ValueError("Invalid force_mode")

        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # This implementation currently requires matching
        # token patterns across batches.
        if batch_size > 1:

            first_pattern = x[0]

            for batch_index in range(1, batch_size):

                if not torch.equal(
                    first_pattern,
                    x[batch_index]
                ):
                    raise ValueError(
                        "Adaptive deduplication currently requires "
                        "identical token patterns across batches"
                    )

        # Forced standard execution.
        if force_mode == "standard":

            self._record_standard(x)

            return self.network(x)

        # Adaptive sample-based decision.
        if force_mode is None:

            estimated_duplicate_ratio = (
                self._estimate_duplicate_ratio(x)
            )

            self.estimated_duplicate_ratios.append(
                estimated_duplicate_ratio
            )

            threshold = (
                self.policy.minimum_duplicate_ratio(
                    sequence_length
                )
            )

            if estimated_duplicate_ratio < threshold:

                self.sample_rejections += 1

                self._record_standard(x)

                return self.network(x)

            self.sample_acceptances += 1

        # Full deduplication.
        unique_batches = []
        inverse_batches = []
        unique_counts = []

        for batch_index in range(batch_size):

            unique_tokens, inverse_indices = torch.unique(
                x[batch_index],
                dim=0,
                return_inverse=True
            )

            unique_batches.append(unique_tokens)
            inverse_batches.append(inverse_indices)

            unique_counts.append(
                unique_tokens.shape[0]
            )

        unique_count = unique_counts[0]

        # The forced deduplicated mode bypasses the policy.
        should_deduplicate = (
            force_mode == "deduplicated"
            or (
                force_mode is None
                and self.policy.should_reuse(
                    sequence_length,
                    unique_count
                )
            )
        )

        # The sample predicted reuse, but full deduplication
        # found insufficient duplication.
        if not should_deduplicate:

            self._record_standard(x)

            return self.network(x)

        outputs = []

        for batch_index in range(batch_size):

            unique_output = self.network(
                unique_batches[batch_index]
            )

            reconstructed = unique_output.index_select(
                0,
                inverse_batches[batch_index]
            )

            outputs.append(reconstructed)

        result = torch.stack(
            outputs,
            dim=0
        )

        self.deduplicated_executions += 1

        self.total_tokens_processed += (
            batch_size * sequence_length
        )

        self.unique_tokens_processed += sum(
            unique_counts
        )

        return result

    def reuse_statistics(self):

        average_estimated_duplicate_ratio = (
            sum(self.estimated_duplicate_ratios)
            / len(self.estimated_duplicate_ratios)
            if self.estimated_duplicate_ratios
            else 0.0
        )

        return {
            "standard_executions": self.standard_executions,
            "deduplicated_executions": self.deduplicated_executions,
            "sample_rejections": self.sample_rejections,
            "sample_acceptances": self.sample_acceptances,
            "average_estimated_duplicate_ratio": (
                average_estimated_duplicate_ratio
            ),
            "total_tokens_processed": self.total_tokens_processed,
            "unique_tokens_processed": self.unique_tokens_processed
        }
