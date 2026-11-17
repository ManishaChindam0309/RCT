
import unittest

import torch

from model.rct_100m import RCT100M
from model.rct_100m_adaptive import RCT100MAdaptive
from model.rct_100m_kv import RCT100MKV


class TestRCT243(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(243)

        cls.base_model = RCT100M()
        cls.base_model.eval()

        cls.kv_model = RCT100MKV(cls.base_model)
        cls.kv_model.eval()

        cls.adaptive_model = RCT100MAdaptive(
            cls.base_model,
            duplicate_threshold=0.50
        )

        cls.adaptive_model.eval()

    def test_unique_prompts_use_kv_only(self):
        prompts = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (4, 16)
        )

        self.adaptive_model.reset_statistics()

        torch.manual_seed(1234)

        output = self.adaptive_model.generate(
            prompts,
            max_new_tokens=4
        )

        stats = self.adaptive_model.statistics()

        self.assertEqual(stats["last_decision"], "kv_only")
        self.assertEqual(stats["kv_only_calls"], 1)
        self.assertEqual(stats["deduplicated_calls"], 0)

        self.assertEqual(output.shape, (4, 20))
        self.assertTrue(torch.equal(output[:, :16], prompts))

    def test_duplicate_prompts_use_deduplication(self):
        prompt_a = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 16)
        )

        prompt_b = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 16)
        )

        prompts = torch.cat(
            [prompt_a, prompt_a, prompt_a, prompt_b],
            dim=0
        )

        self.adaptive_model.reset_statistics()

        torch.manual_seed(5678)

        output = self.adaptive_model.generate(
            prompts,
            max_new_tokens=4
        )

        stats = self.adaptive_model.statistics()

        self.assertEqual(
            stats["last_decision"],
            "deduplicated"
        )

        self.assertEqual(
            stats["deduplicated_calls"],
            1
        )

        self.assertTrue(
            torch.equal(output[0], output[1])
        )

        self.assertTrue(
            torch.equal(output[1], output[2])
        )

        self.assertTrue(
            torch.equal(output[:, :16], prompts)
        )

    def test_adaptive_matches_kv_on_unique_prompts(self):
        prompts = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (3, 16)
        )

        self.adaptive_model.reset_statistics()

        torch.manual_seed(999)

        expected = self.kv_model.generate_cached(
            prompts,
            max_new_tokens=4
        )

        torch.manual_seed(999)

        actual = self.adaptive_model.generate(
            prompts,
            max_new_tokens=4
        )

        self.assertTrue(
            torch.equal(expected, actual)
        )

    def test_deduplicated_output_matches_unique_kv(self):
        prompt_a = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 16)
        )

        prompt_b = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 16)
        )

        prompts = torch.cat(
            [prompt_a, prompt_b, prompt_a, prompt_a],
            dim=0
        )

        unique_prompts, inverse = (
            self.adaptive_model.stable_deduplicate(prompts)
        )

        torch.manual_seed(4321)

        unique_expected = self.kv_model.generate_cached(
            unique_prompts,
            max_new_tokens=4
        )

        expected = unique_expected.index_select(
            0,
            inverse
        )

        torch.manual_seed(4321)

        actual = self.adaptive_model.generate(
            prompts,
            max_new_tokens=4
        )

        self.assertTrue(
            torch.equal(expected, actual)
        )

    def test_parameter_count_unchanged(self):
        self.assertEqual(
            self.adaptive_model.parameter_count(),
            self.base_model.parameter_count()
        )

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            RCT100MAdaptive(
                self.base_model,
                duplicate_threshold=1.5
            )


if __name__ == "__main__":
    unittest.main()
