
class AdaptiveReusePolicy:
    """
    RCT 1.9.9 experimental adaptive reuse policy.

    Thresholds are based on synthetic CPU FFN benchmarks.
    They are not universal performance guarantees.
    """

    def __init__(self):
        self.thresholds = {
            64: 0.70,
            128: 0.50,
            256: 0.45,
        }

        self.default_threshold = 0.75

    def threshold_for(self, sequence_length):
        if sequence_length in self.thresholds:
            return self.thresholds[sequence_length]

        # Use the nearest tested sequence length.
        nearest = min(
            self.thresholds,
            key=lambda length: abs(
                length - sequence_length
            )
        )

        if sequence_length < 64:
            return self.default_threshold

        return self.thresholds[nearest]

    def should_reuse(
        self,
        sequence_length,
        unique_count
    ):
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if unique_count < 1 or unique_count > sequence_length:
            raise ValueError(
                "unique_count must be between 1 and sequence_length"
            )

        duplicate_ratio = (
            sequence_length - unique_count
        ) / sequence_length

        threshold = self.threshold_for(
            sequence_length
        )

        return duplicate_ratio >= threshold

    def statistics(self):
        return {
            "thresholds": dict(self.thresholds),
            "default_threshold": self.default_threshold,
        }
