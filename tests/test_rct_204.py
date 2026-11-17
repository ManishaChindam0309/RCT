
import unittest
import torch

from model.transformer import RCTModel
from model.prompt_deduplication import PromptDeduplicator


class TestRCT204(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        torch.manual_seed(204)
        torch.set_num_threads(2)

        cls.model = RCTModel(
            vocab_size=500,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            max_seq_len=64
        )

        cls.model.eval()

        cls.deduplicator = PromptDeduplicator()

    def test_single_prompt_bypass(self):

        tokens = torch.randint(0, 500, (1, 16))

        with torch.inference_mode():

            expected = self.model(
                tokens,
                ffn_mode="standard"
            )

            actual, stats = self.deduplicator.forward(
                self.model,
                tokens
            )

        self.assertTrue(torch.allclose(expected, actual))
        self.assertEqual(stats["mode"], "bypass_single_prompt")

    def test_all_unique_prompts(self):

        tokens = torch.randint(0, 500, (4, 16))

        with torch.inference_mode():

            expected = self.model(
                tokens,
                ffn_mode="standard"
            )

            actual, stats = self.deduplicator.forward(
                self.model,
                tokens
            )

        self.assertTrue(torch.allclose(expected, actual))
        self.assertEqual(stats["mode"], "standard_unique")
        self.assertEqual(stats["unique_prompts"], 4)

    def test_identical_prompts(self):

        prompt = torch.randint(0, 500, (1, 16))
        tokens = prompt.repeat(4, 1)

        with torch.inference_mode():

            expected = self.model(
                tokens,
                ffn_mode="standard"
            )

            actual, stats = self.deduplicator.forward(
                self.model,
                tokens
            )

        self.assertTrue(torch.allclose(
            expected, actual, rtol=1e-4, atol=1e-5
        ))

        self.assertEqual(stats["unique_prompts"], 1)
        self.assertEqual(stats["mode"], "deduplicated")

    def test_paired_prompts_restore_order(self):

        prompts = torch.randint(0, 500, (2, 16))

        tokens = torch.stack([
            prompts[1],
            prompts[0],
            prompts[1],
            prompts[0],
        ])

        with torch.inference_mode():

            expected = self.model(
                tokens,
                ffn_mode="standard"
            )

            actual, stats = self.deduplicator.forward(
                self.model,
                tokens
            )

        self.assertTrue(torch.allclose(
            expected, actual, rtol=1e-4, atol=1e-5
        ))

        self.assertEqual(stats["unique_prompts"], 2)
        self.assertEqual(stats["duplicate_prompts"], 2)

    def test_training_mode_rejected(self):

        self.model.train()

        tokens = torch.randint(0, 500, (2, 16))

        with self.assertRaises(RuntimeError):
            self.deduplicator.forward(
                self.model,
                tokens
            )

        self.model.eval()


if __name__ == "__main__":
    unittest.main()
