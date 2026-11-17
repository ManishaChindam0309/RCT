
import unittest

import torch

from model.rct_100m import RCT100M
from model.rct_100m_kv import RCT100MKV


class TestRCT23KVCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        torch.manual_seed(23)
        torch.set_num_threads(4)

        cls.base_model = RCT100M()
        cls.base_model.eval()

        cls.kv_model = RCT100MKV(
            cls.base_model
        )

        cls.kv_model.eval()

    def test_parameter_count_unchanged(self):

        self.assertEqual(
            self.base_model.parameter_count(),
            self.kv_model.parameter_count()
        )

    def test_incremental_logits_match(self):

        tokens = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 8)
        )

        with torch.inference_mode():

            standard_logits = self.base_model(tokens)

            cache = self.kv_model.create_cache(1)

            incremental_logits = []

            for position in range(tokens.size(1)):

                step_logits = self.kv_model.decode_step(
                    tokens[:, position:position + 1],
                    cache
                )

                incremental_logits.append(step_logits)

            incremental_logits = torch.cat(
                incremental_logits,
                dim=1
            )

        max_difference = (
            standard_logits - incremental_logits
        ).abs().max().item()

        print(
            "\nIncremental maximum difference:",
            max_difference
        )

        self.assertTrue(
            torch.allclose(
                standard_logits,
                incremental_logits,
                atol=1e-4,
                rtol=1e-4
            )
        )

    def test_cached_generation_shape(self):

        prompt = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 8)
        )

        with torch.inference_mode():

            generated = self.kv_model.generate_cached(
                prompt,
                max_new_tokens=4
            )

        self.assertEqual(
            generated.shape,
            (1, 12)
        )

    def test_cached_generation_matches_sampling(self):

        prompt = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 8)
        )

        with torch.inference_mode():

            # Same random seed for both implementations.
            torch.manual_seed(1234)

            standard_generated = self.base_model.generate(
                prompt,
                max_new_tokens=4,
                temperature=1.0
            )

            torch.manual_seed(1234)

            cached_generated = self.kv_model.generate_cached(
                prompt,
                max_new_tokens=4,
                temperature=1.0
            )

        self.assertTrue(
            torch.equal(
                standard_generated,
                cached_generated
            ),
            "Cached generation differs from standard generation."
        )

    def test_cache_position(self):

        cache = self.kv_model.create_cache(1)

        tokens = torch.randint(
            0,
            self.base_model.config.vocab_size,
            (1, 3)
        )

        with torch.inference_mode():

            self.kv_model.decode_step(
                tokens,
                cache
            )

        self.assertEqual(
            cache.position,
            3
        )

    def test_cache_memory(self):

        cache = self.kv_model.create_cache(1)

        expected_bytes = (
            2
            * self.base_model.config.num_layers
            * self.base_model.config.max_seq_len
            * self.base_model.config.hidden_size
            * 4
        )

        self.assertEqual(
            cache.memory_bytes(),
            expected_bytes
        )


if __name__ == "__main__":
    unittest.main()
