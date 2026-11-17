
from collections import OrderedDict
from threading import RLock

import torch


class ComputationMemory:
    """
    Bounded LRU cache for exact neural computation results.
    """

    def __init__(self, capacity=256):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self.memory = OrderedDict()
        self.lock = RLock()

        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key):
        with self.lock:
            if key not in self.memory:
                self.misses += 1
                return None

            self.hits += 1
            self.memory.move_to_end(key)

            return self.memory[key].clone()

    def put(self, key, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError("Cache values must be tensors")

        stored_value = value.detach().clone().cpu()

        with self.lock:
            self.memory[key] = stored_value
            self.memory.move_to_end(key)

            while len(self.memory) > self.capacity:
                self.memory.popitem(last=False)
                self.evictions += 1

    def clear(self):
        with self.lock:
            self.memory.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0

    def statistics(self):
        with self.lock:
            total = self.hits + self.misses

            return {
                "hits": self.hits,
                "misses": self.misses,
                "entries": len(self.memory),
                "evictions": self.evictions,
                "hit_rate": (
                    self.hits / total if total else 0.0
                )
            }
