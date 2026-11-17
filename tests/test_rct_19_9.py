
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.adaptive_reuse import AdaptiveReusePolicy


class TestAdaptiveReusePolicy(unittest.TestCase):

    def setUp(self):
        self.policy = AdaptiveReusePolicy()

    def test_sequence_64_threshold(self):
        self.assertTrue(
            self.policy.should_reuse(64, 19)
        )

        self.assertFalse(
            self.policy.should_reuse(64, 20)
        )

    def test_sequence_128_threshold(self):
        self.assertTrue(
            self.policy.should_reuse(128, 64)
        )

        self.assertFalse(
            self.policy.should_reuse(128, 65)
        )

    def test_sequence_256_threshold(self):
        self.assertTrue(
            self.policy.should_reuse(256, 140)
        )

        self.assertFalse(
            self.policy.should_reuse(256, 141)
        )

    def test_no_duplicates(self):
        self.assertFalse(
            self.policy.should_reuse(128, 128)
        )

    def test_all_duplicates(self):
        self.assertTrue(
            self.policy.should_reuse(128, 1)
        )

    def test_invalid_sequence(self):
        with self.assertRaises(ValueError):
            self.policy.should_reuse(0, 0)

    def test_invalid_unique_count(self):
        with self.assertRaises(ValueError):
            self.policy.should_reuse(64, 65)

    def test_statistics(self):
        stats = self.policy.statistics()

        self.assertIn(64, stats["thresholds"])
        self.assertIn(128, stats["thresholds"])
        self.assertIn(256, stats["thresholds"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
