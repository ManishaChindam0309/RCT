
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

    def test_short_sequence(self):

        self.assertTrue(
            self.policy.should_reuse(16, 4)
        )

        self.assertFalse(
            self.policy.should_reuse(16, 8)
        )

    def test_sequence_32(self):

        # Sequence lengths below 64 use 75%.
        self.assertTrue(
            self.policy.should_reuse(32, 8)
        )

        self.assertFalse(
            self.policy.should_reuse(32, 12)
        )

    def test_sequence_64(self):

        # RCT 1.9.9 threshold: 70%.
        self.assertTrue(
            self.policy.should_reuse(64, 19)
        )

        self.assertFalse(
            self.policy.should_reuse(64, 20)
        )

    def test_sequence_128(self):

        # RCT 1.9.9 threshold: 50%.
        self.assertTrue(
            self.policy.should_reuse(128, 64)
        )

        self.assertFalse(
            self.policy.should_reuse(128, 65)
        )

    def test_sequence_256(self):

        # RCT 1.9.9 threshold: 45%.
        self.assertTrue(
            self.policy.should_reuse(256, 140)
        )

        self.assertFalse(
            self.policy.should_reuse(256, 141)
        )

    def test_very_long_sequence(self):

        # Nearest tested threshold for 512 is 256's 45%.
        self.assertTrue(
            self.policy.should_reuse(512, 281)
        )

        self.assertFalse(
            self.policy.should_reuse(512, 282)
        )

    def test_invalid_inputs(self):

        with self.assertRaises(ValueError):
            self.policy.should_reuse(0, 0)

        with self.assertRaises(ValueError):
            self.policy.should_reuse(32, 0)

        with self.assertRaises(ValueError):
            self.policy.should_reuse(32, 33)

    def test_statistics(self):

        result = self.policy.statistics()

        self.assertEqual(
            result["thresholds"],
            {
                64: 0.70,
                128: 0.50,
                256: 0.45,
            }
        )

        self.assertEqual(
            result["default_threshold"],
            0.75
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
