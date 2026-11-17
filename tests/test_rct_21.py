
import unittest
import torch

from model.transformer import RCTModel


class TestRCT21(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(21)

        torch.set_num_threads(2)

        self.model = RCTModel(
            vocab_size=500,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=128
        )

        self.model.eval()

    def test_identical_prompts(self):

        tokens = torch.randint(0, 500, (1, 32))
        tokens = tokens.repeat(4, 1)

        with torch.inference_mode():

            standard = self.model(tokens)

            adaptive = self.model(
                tokens,
                prompt_mode="adaptive"
            )

        self.assertTrue(
            torch.allclose(
                standard,
                adaptive,
                rtol=1e-4,
                atol=1e-5
            )
        )

        stats = self.model.prompt_dedup_statistics()

        self.assertEqual(stats["unique_prompts"], 1)
        self.assertEqual(stats["duplicate_prompts"], 3)
        self.assertEqual(stats["dedup_activations"], 1)

    def test_unique_prompts(self):

        tokens = torch.randint(0, 500, (4, 32))

        with torch.inference_mode():

            standard = self.model(tokens)

            adaptive = self.model(
                tokens,
                prompt_mode="adaptive"
            )

        self.assertTrue(
            torch.allclose(
                standard,
                adaptive,
                rtol=1e-4,
                atol=1e-5
            )
        )

        stats = self.model.prompt_dedup_statistics()

        self.assertEqual(stats["dedup_activations"], 0)

    def test_paired_prompts(self):

        prompts = torch.randint(0, 500, (2, 32))

        tokens = prompts.repeat_interleave(2, dim=0)

        with torch.inference_mode():

            standard = self.model(tokens)

            adaptive = self.model(
                tokens,
                prompt_mode="adaptive"
            )

        self.assertTrue(
            torch.allclose(
                standard,
                adaptive,
                rtol=1e-4,
                atol=1e-5
            )
        )

    def test_single_prompt_bypass(self):

        tokens = torch.randint(0, 500, (1, 32))

        with torch.inference_mode():

            standard = self.model(tokens)

            adaptive = self.model(
                tokens,
                prompt_mode="adaptive"
            )

        self.assertTrue(
            torch.allclose(
                standard,
                adaptive,
                rtol=1e-4,
                atol=1e-5
            )
        )

        stats = self.model.prompt_dedup_statistics()

        self.assertEqual(stats["bypasses"], 1)

    def test_training_mode_rejected(self):

        self.model.train()

        tokens = torch.randint(0, 500, (2, 32))

        with torch.inference_mode():

            with self.assertRaises(RuntimeError):

                self.model(
                    tokens,
                    prompt_mode="adaptive"
                )

    def test_grad_enabled_rejected(self):

        self.model.eval()

        tokens = torch.randint(0, 500, (2, 32))

        with self.assertRaises(RuntimeError):

            self.model(
                tokens,
                prompt_mode="adaptive"
            )

    def test_invalid_prompt_mode(self):

        tokens = torch.randint(0, 500, (2, 32))

        with self.assertRaises(ValueError):

            self.model(
                tokens,
                prompt_mode="invalid"
            )

    def test_statistics_reset(self):

        tokens = torch.randint(0, 500, (2, 32))

        with torch.inference_mode():

            self.model(
                tokens,
                prompt_mode="adaptive"
            )

        self.model.reset_prompt_dedup_statistics()

        stats = self.model.prompt_dedup_statistics()

        self.assertEqual(stats["calls"], 0)
        self.assertEqual(stats["original_prompts"], 0)
        self.assertEqual(stats["unique_prompts"], 0)


if __name__ == "__main__":
    unittest.main()
