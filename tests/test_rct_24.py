
import unittest

import torch

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV
from model.rct_100m_combined import RCT100MCombined


class TestRCT24Combined(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        torch.manual_seed(24)
        torch.set_num_threads(4)

        cls.base_model = RCT100M()
        cls.base_model.eval()

        cls.kv_model = RCT100MKV(cls.base_model)
        cls.kv_model.eval()

        cls.combined_model = RCT100MCombined(cls.base_model)
        cls.combined_model.eval()

    def test_parameter_count_unchanged(self):

        self.assertEqual(
            self.base_model.parameter_count(),
            self.combined_model.parameter_count()
        )

    def test_duplicate_prompts_produce_identical_outputs(self):

        prompt = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 8)
        )

        prompts = prompt.repeat(4, 1)

        with torch.inference_mode():

            torch.manual_seed(1234)

            outputs = self.combined_model.generate(
                prompts,
                max_new_tokens=4,
                deduplicate=True
            )

        self.assertEqual(outputs.shape, (4, 12))

        for index in range(1, 4):

            self.assertTrue(
                torch.equal(outputs[0], outputs[index])
            )

    def test_deduplication_statistics(self):

        self.combined_model.reset_statistics()

        prompt = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 8)
        )

        prompts = prompt.repeat(4, 1)

        with torch.inference_mode():

            self.combined_model.generate(
                prompts,
                max_new_tokens=2,
                deduplicate=True
            )

        stats = self.combined_model.statistics()

        self.assertEqual(stats["total_prompts_received"], 4)
        self.assertEqual(stats["unique_prompts_processed"], 1)
        self.assertEqual(stats["duplicate_prompts_saved"], 3)

    def test_unique_prompts(self):

        prompts = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (3, 8)
        )

        self.combined_model.reset_statistics()

        with torch.inference_mode():

            outputs = self.combined_model.generate(
                prompts,
                max_new_tokens=2,
                deduplicate=True
            )

        self.assertEqual(outputs.shape, (3, 10))

        stats = self.combined_model.statistics()

        self.assertEqual(stats["unique_prompts_processed"], 3)
        self.assertEqual(stats["duplicate_prompts_saved"], 0)

    def test_combined_matches_kv_generation(self):

        prompt = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 8)
        )

        prompts = prompt.repeat(3, 1)

        with torch.inference_mode():

            torch.manual_seed(5678)

            expected = self.kv_model.generate_cached(
                prompt,
                max_new_tokens=4
            )

            torch.manual_seed(5678)

            actual = self.combined_model.generate(
                prompts,
                max_new_tokens=4,
                deduplicate=True
            )

        for index in range(3):

            self.assertTrue(
                torch.equal(expected[0], actual[index])
            )

    def test_no_deduplication_mode(self):

        prompts = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (2, 8)
        )

        self.combined_model.reset_statistics()

        with torch.inference_mode():

            outputs = self.combined_model.generate(
                prompts,
                max_new_tokens=2,
                deduplicate=False
            )

        self.assertEqual(outputs.shape, (2, 10))

        stats = self.combined_model.statistics()

        self.assertEqual(stats["unique_prompts_processed"], 2)
        self.assertEqual(stats["duplicate_prompts_saved"], 0)


if __name__ == "__main__":
    unittest.main()
