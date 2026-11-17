
import torch


class PromptDeduplicator:
    """
    RCT 2.0.4

    Adaptive input-level prompt deduplication.

    - Batch size 1: bypass detection.
    - All prompts unique: standard inference.
    - Duplicate prompts: process only unique prompts.
    """

    def __init__(self):
        self.last_statistics = {}

    @torch.no_grad()
    def deduplicate(self, tokens):

        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")

        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape [batch, sequence_length]"
            )

        if tokens.shape[0] < 1:
            raise ValueError("Batch cannot be empty")

        unique_tokens, inverse_indices = torch.unique(
            tokens,
            dim=0,
            sorted=True,
            return_inverse=True
        )

        original_count = tokens.shape[0]
        unique_count = unique_tokens.shape[0]

        self.last_statistics = {
            "original_prompts": original_count,
            "unique_prompts": unique_count,
            "duplicate_prompts": original_count - unique_count,
            "duplicate_ratio": (
                original_count - unique_count
            ) / original_count,
        }

        return unique_tokens, inverse_indices

    def restore(self, unique_outputs, inverse_indices):

        if not isinstance(unique_outputs, torch.Tensor):
            raise TypeError("unique_outputs must be a tensor")

        if not isinstance(inverse_indices, torch.Tensor):
            raise TypeError("inverse_indices must be a tensor")

        if inverse_indices.ndim != 1 or inverse_indices.numel() == 0:
            raise ValueError("Invalid inverse_indices")

        if unique_outputs.ndim < 1 or unique_outputs.shape[0] == 0:
            raise ValueError("Invalid unique_outputs")

        if inverse_indices.min().item() < 0:
            raise ValueError("Negative inverse index")

        if inverse_indices.max().item() >= unique_outputs.shape[0]:
            raise ValueError("Inverse index out of range")

        return unique_outputs.index_select(
            0,
            inverse_indices.to(unique_outputs.device)
        )

    @torch.inference_mode()
    def forward(self, model, tokens):
        """
        Adaptive inference wrapper.

        Returns:
            output, statistics
        """

        if model.training:
            raise RuntimeError(
                "Prompt deduplication requires model.eval()"
            )

        if tokens.ndim != 2 or tokens.shape[0] < 1:
            raise ValueError(
                "tokens must have shape [batch, sequence_length]"
            )

        batch_size = tokens.shape[0]

        # Batch size one cannot contain cross-prompt duplicates.
        if batch_size == 1:

            output = model(
                tokens,
                ffn_mode="standard"
            )

            self.last_statistics = {
                "original_prompts": 1,
                "unique_prompts": 1,
                "duplicate_prompts": 0,
                "duplicate_ratio": 0.0,
                "mode": "bypass_single_prompt",
            }

            return output, self.statistics()

        # Detect repeated prompts.
        unique_tokens, inverse_indices = self.deduplicate(tokens)

        unique_count = unique_tokens.shape[0]

        # No duplicates: preserve original batch order
        # and avoid unnecessary output reconstruction.
        if unique_count == batch_size:

            output = model(
                tokens,
                ffn_mode="standard"
            )

            self.last_statistics["mode"] = "standard_unique"

            return output, self.statistics()

        # Duplicates found: process only unique prompts.
        unique_outputs = model(
            unique_tokens,
            ffn_mode="standard"
        )

        output = self.restore(
            unique_outputs,
            inverse_indices
        )

        self.last_statistics["mode"] = "deduplicated"

        return output, self.statistics()

    def statistics(self):
        return dict(self.last_statistics)

    def reset_statistics(self):
        self.last_statistics = {}
