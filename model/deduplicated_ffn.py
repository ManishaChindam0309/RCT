
import torch
import torch.nn as nn


class DeduplicatedFFN(nn.Module):
    """
    Experimental vectorized FFN deduplication.

    Identical token representations are processed once.
    Outputs are reconstructed in their original positions.
    """

    def __init__(self, original_ffn):
        super().__init__()

        self.network = original_ffn

        self.calls = 0
        self.unique_inputs = 0
        self.total_inputs = 0

    def forward(self, x):

        if x.dim() != 3:
            raise ValueError(
                "Expected input shape [batch, sequence, hidden]."
            )

        batch, seq_len, hidden = x.shape

        flattened = x.reshape(
            batch * seq_len,
            hidden
        )

        unique, inverse = torch.unique(
            flattened,
            dim=0,
            return_inverse=True
        )

        unique_outputs = self.network(unique)

        output = unique_outputs.index_select(
            0,
            inverse
        )

        self.calls += 1
        self.unique_inputs += unique.size(0)
        self.total_inputs += flattened.size(0)

        return output.reshape(
            batch,
            seq_len,
            hidden
        )

    def statistics(self):

        reuse_ratio = (
            1 - self.unique_inputs / self.total_inputs
            if self.total_inputs else 0.0
        )

        return {
            "calls": self.calls,
            "total_inputs": self.total_inputs,
            "unique_inputs": self.unique_inputs,
            "reuse_ratio": reuse_ratio
        }

    def reset_statistics(self):

        self.calls = 0
        self.unique_inputs = 0
        self.total_inputs = 0
