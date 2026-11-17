
import torch
import torch.nn as nn

from model.rct_100m_kv import RCT100MKV


class RCT100MCombined(nn.Module):
    """
    RCT 100M combining stable adaptive prompt deduplication
    with autoregressive KV caching.
    """

    def __init__(self, base_model):

        super().__init__()

        self.kv_model = RCT100MKV(base_model)
        self.config = base_model.config

        self.duplicate_prompts_saved = 0
        self.unique_prompts_processed = 0
        self.total_prompts_received = 0

    def parameter_count(self):
        return self.kv_model.parameter_count()

    def reset_statistics(self):

        self.duplicate_prompts_saved = 0
        self.unique_prompts_processed = 0
        self.total_prompts_received = 0

    def statistics(self):

        return {
            "total_prompts_received": self.total_prompts_received,
            "unique_prompts_processed": self.unique_prompts_processed,
            "duplicate_prompts_saved": self.duplicate_prompts_saved,
        }

    def train(self, mode=True):

        super().train(mode)
        self.kv_model.eval()

        return self

    def _stable_deduplicate(self, prompts):
        """
        Deduplicate prompts while preserving the order
        of their first occurrence.

        Returns:
            unique_prompts: unique prompts in stable order
            inverse_indices: maps original rows to stable
                             unique prompt indices
        """

        unique_prompts, inverse_indices = torch.unique(
            prompts,
            dim=0,
            return_inverse=True
        )

        unique_count = unique_prompts.size(0)
        batch_size = prompts.size(0)

        # Find the first original position of each unique prompt.
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

        # Sort unique prompts by first occurrence.
        order = torch.argsort(first_positions)

        unique_prompts = unique_prompts.index_select(
            0,
            order
        )

        # Convert old sorted indices into stable indices.
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
        temperature=1.0,
        deduplicate=True
    ):

        if self.training:
            raise RuntimeError(
                "Call model.eval() before generation."
            )

        if prompts.ndim != 2 or prompts.size(1) < 1:
            raise ValueError(
                "Prompts must have shape [batch, sequence]."
            )

        if prompts.dtype != torch.long:
            raise TypeError(
                "Prompts must use torch.long."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than zero."
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens cannot be negative."
            )

        batch_size = prompts.size(0)

        if batch_size < 1:
            raise ValueError(
                "Batch cannot be empty."
            )

        self.total_prompts_received += batch_size

        # --------------------------------------------
        # Stable adaptive prompt deduplication
        # --------------------------------------------

        if deduplicate and batch_size > 1:

            unique_prompts, inverse_indices = (
                self._stable_deduplicate(prompts)
            )

        else:

            unique_prompts = prompts

            inverse_indices = torch.arange(
                batch_size,
                device=prompts.device
            )

        unique_count = unique_prompts.size(0)

        self.unique_prompts_processed += unique_count

        self.duplicate_prompts_saved += (
            batch_size - unique_count
        )

        # --------------------------------------------
        # KV-cache generation on unique prompts
        # --------------------------------------------

        unique_outputs = self.kv_model.generate_cached(
            unique_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        # --------------------------------------------
        # Restore original batch ordering
        # --------------------------------------------

        outputs = unique_outputs.index_select(
            0,
            inverse_indices
        )

        return outputs
