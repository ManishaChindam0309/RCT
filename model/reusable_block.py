import torch
import torch.nn as nn

from model.computation_memory import ComputationMemory
from model.fingerprint import computation_fingerprint


class ReusableFFN(nn.Module):
    def __init__(
        self,
        hidden_size,
        expansion=4,
        cache_capacity=256
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.GELU(),
            nn.Linear(hidden_size * expansion, hidden_size)
        )

        self.memory = ComputationMemory(cache_capacity)
        self.module_version = 0

        # RCT 1.7 execution counters.
        self.network_executions = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _execute_network(self, x):
        """Run the actual FFN and record the execution."""
        self.network_executions += 1
        return self.network(x)

    def forward(self, x):

        # Never cache training or gradient-enabled computations.
        if self.training or torch.is_grad_enabled():
            return self._execute_network(x)

        parameter_versions = tuple(
            parameter._version
            for parameter in self.parameters()
        )

        version = (
            self.module_version,
            parameter_versions
        )

        key = computation_fingerprint(x, version)

        cached = self.memory.get(key)

        if cached is not None:
            self.cache_hits += 1

            return cached.to(
                device=x.device,
                dtype=x.dtype
            )

        self.cache_misses += 1

        result = self._execute_network(x)

        self.memory.put(key, result)

        return result

    def invalidate_cache(self):
        self.module_version += 1
        self.memory.clear()

    def reset_execution_statistics(self):
        self.network_executions = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_statistics(self):
        stats = self.memory.statistics()

        stats.update({
            "network_executions": self.network_executions,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses
        })

        return stats