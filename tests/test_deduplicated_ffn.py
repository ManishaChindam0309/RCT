
import unittest

import torch
import torch.nn as nn

from model.deduplicated_ffn import DeduplicatedFFN


class TestDeduplicatedFFN(unittest.TestCase):

    def test_output_correctness(self):

        torch.manual_seed(42)

        original = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, 32)
        ).eval()

        deduplicated = DeduplicatedFFN(
            original
        ).eval()

        x = torch.randn(1, 16, 32)

        x[:, 4] = x[:, 0]
        x[:, 8] = x[:, 0]
        x[:, 12] = x[:, 0]

        with torch.inference_mode():

            expected = original(x)

            actual = deduplicated(x)

        self.assertTrue(
            torch.allclose(
                expected,
                actual,
                rtol=1e-4,
                atol=1e-5
            )
        )

    def test_unique_count(self):

        original = nn.Linear(16, 16).eval()

        deduplicated = DeduplicatedFFN(
            original
        ).eval()

        x = torch.randn(1, 8, 16)

        x[:, 1] = x[:, 0]
        x[:, 2] = x[:, 0]
        x[:, 3] = x[:, 0]

        with torch.inference_mode():
            deduplicated(x)

        stats = deduplicated.statistics()

        self.assertEqual(
            stats["total_inputs"],
            8
        )

        self.assertEqual(
            stats["unique_inputs"],
            5
        )

    def test_all_identical(self):

        original = nn.Linear(16, 16).eval()

        deduplicated = DeduplicatedFFN(
            original
        ).eval()

        token = torch.randn(1, 1, 16)

        x = token.repeat(1, 8, 1)

        with torch.inference_mode():
            output = deduplicated(x)

        self.assertEqual(
            output.shape,
            (1, 8, 16)
        )

        self.assertEqual(
            deduplicated.statistics()["unique_inputs"],
            1
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
