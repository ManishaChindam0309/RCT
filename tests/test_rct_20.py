
import unittest
import torch

from model.transformer import RCTModel


class TestRCT20(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(2020)

        self.model = RCTModel(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=64
        )

        self.model.eval()

        self.tokens = torch.randint(
            0,
            1000,
            (1, 16)
        )

    def test_numerical_correctness(self):

        with torch.inference_mode():

            standard = self.model(
                self.tokens,
                ffn_mode="standard"
            )

            deduplicated = self.model(
                self.tokens,
                ffn_mode="deduplicated"
            )

        self.assertTrue(
            torch.allclose(
                standard,
                deduplicated,
                rtol=1e-4,
                atol=1e-5
            )
        )

    def test_output_shape(self):

        with torch.inference_mode():

            output = self.model(
                self.tokens,
                ffn_mode="deduplicated"
            )

        self.assertEqual(
            output.shape,
            (1, 16, 1000)
        )

    def test_statistics(self):

        self.model.reset_dedup_statistics()

        with torch.inference_mode():

            self.model(
                self.tokens,
                ffn_mode="deduplicated"
            )

        stats = self.model.dedup_statistics()

        for layer_stats in stats.values():

            self.assertEqual(
                layer_stats["calls"],
                1
            )

            self.assertEqual(
                layer_stats["total_inputs"],
                16
            )

            self.assertGreaterEqual(
                layer_stats["unique_inputs"],
                1
            )

            self.assertLessEqual(
                layer_stats["unique_inputs"],
                16
            )

    def test_invalid_mode(self):

        with self.assertRaises(ValueError):

            self.model(
                self.tokens,
                ffn_mode="invalid"
            )

    def test_inference_only(self):

        with self.assertRaises(RuntimeError):

            self.model(
                self.tokens,
                ffn_mode="deduplicated"
            )

    def test_default_standard_mode(self):

        with torch.inference_mode():

            output = self.model(self.tokens)

        self.assertEqual(
            output.shape,
            (1, 16, 1000)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
