
import unittest
import sys

from pathlib import Path

import torch

# ==========================================
# PROJECT IMPORT PATH FIX
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import RCTModel


# ==========================================
# RCT 1.2 CACHE SAFETY TESTS
# ==========================================

class RCT12CacheTests(unittest.TestCase):

    def setUp(self):

        torch.manual_seed(123)

        self.vocab_size = 256
        self.prefix_len = 8

        self.baseline = RCTModel(
            vocab_size=self.vocab_size,
            hidden_size=64,
            layers=2,
            heads=4,
            max_seq_len=64
        )

        self.rct = RCTModel(
            vocab_size=self.vocab_size,
            hidden_size=64,
            layers=2,
            heads=4,
            max_seq_len=64,
            prefix_cache_capacity=2
        )

        self.rct.load_state_dict(
            self.baseline.state_dict()
        )

        self.baseline.eval()
        self.rct.eval()

        # Disable FFN cache in both models.
        # We are testing prefix/KV caching only.
        for block in self.baseline.blocks:
            block.ffn.train()

        for block in self.rct.blocks:
            block.ffn.train()

        self.prefix_a = torch.tensor(
            [[10, 20, 30, 40, 50, 60, 70, 80]],
            dtype=torch.long
        )

        self.prefix_b = torch.tensor(
            [[11, 21, 31, 41, 51, 61, 71, 81]],
            dtype=torch.long
        )

        self.prefix_c = torch.tensor(
            [[12, 22, 32, 42, 52, 62, 72, 82]],
            dtype=torch.long
        )

        self.suffix_a = torch.tensor(
            [[90, 91, 92, 93]],
            dtype=torch.long
        )

        self.suffix_b = torch.tensor(
            [[100, 101, 102, 103]],
            dtype=torch.long
        )

    # ======================================
    # HELPER FUNCTIONS
    # ======================================

    def make_prompt(self, prefix, suffix):

        return torch.cat(
            [prefix, suffix],
            dim=1
        )

    def assert_model_outputs_match(
        self,
        prompt,
        prefix_len
    ):

        with torch.inference_mode():

            expected = self.baseline(prompt)

            actual = self.rct.forward_cached(
                prompt,
                prefix_len
            )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4
        )

        return actual

    # ======================================
    # TEST 1
    # SAME PREFIX, DIFFERENT SUFFIX
    # ======================================

    def test_01_same_prefix_different_suffix(self):

        prompt_a = self.make_prompt(
            self.prefix_a,
            self.suffix_a
        )

        prompt_b = self.make_prompt(
            self.prefix_a,
            self.suffix_b
        )

        self.assert_model_outputs_match(
            prompt_a,
            self.prefix_len
        )

        self.assert_model_outputs_match(
            prompt_b,
            self.prefix_len
        )

        self.assertEqual(
            len(self.rct._prefix_cache),
            1
        )

    # ======================================
    # TEST 2
    # DIFFERENT PREFIX, SAME LENGTH
    # ======================================

    def test_02_different_prefix_same_length(self):

        prompt_a = self.make_prompt(
            self.prefix_a,
            self.suffix_a
        )

        prompt_b = self.make_prompt(
            self.prefix_b,
            self.suffix_a
        )

        self.assert_model_outputs_match(
            prompt_a,
            self.prefix_len
        )

        self.assert_model_outputs_match(
            prompt_b,
            self.prefix_len
        )

        self.assertEqual(
            len(self.rct._prefix_cache),
            2
        )

    # ======================================
    # TEST 3
    # PARAMETER CHANGE INVALIDATION
    # ======================================

    def test_03_parameter_change_invalidates_cache(self):

        prompt = self.make_prompt(
            self.prefix_a,
            self.suffix_a
        )

        with torch.inference_mode():

            self.rct.forward_cached(
                prompt,
                self.prefix_len
            )

            self.assertEqual(
                len(self.rct._prefix_cache),
                1
            )

            # Change the same parameter in both models.
            with torch.no_grad():

                baseline_parameter = next(
                    self.baseline.parameters()
                )

                rct_parameter = next(
                    self.rct.parameters()
                )

                baseline_parameter.add_(0.01)
                rct_parameter.add_(0.01)

            expected = self.baseline(prompt)

            actual = self.rct.forward_cached(
                prompt,
                self.prefix_len
            )

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-4,
            atol=1e-4
        )

        self.assertEqual(
            len(self.rct._prefix_cache),
            1
        )

    # ======================================
    # TEST 4
    # EXPLICIT CACHE INVALIDATION
    # ======================================

    def test_04_explicit_cache_invalidation(self):

        prompt_a = self.make_prompt(
            self.prefix_a,
            self.suffix_a
        )

        prompt_b = self.make_prompt(
            self.prefix_b,
            self.suffix_b
        )

        with torch.inference_mode():

            self.rct.forward_cached(
                prompt_a,
                self.prefix_len
            )

            self.rct.forward_cached(
                prompt_b,
                self.prefix_len
            )

        self.assertEqual(
            len(self.rct._prefix_cache),
            2
        )

        self.rct.invalidate_caches()

        self.assertEqual(
            len(self.rct._prefix_cache),
            0
        )

        self.assert_model_outputs_match(
            prompt_a,
            self.prefix_len
        )

        self.assertEqual(
            len(self.rct._prefix_cache),
            1
        )

    # ======================================
    # TEST 5
    # LRU EVICTION
    # ======================================

    def test_05_lru_eviction(self):

        prompt_a = self.make_prompt(
            self.prefix_a,
            self.suffix_a
        )

        prompt_b = self.make_prompt(
            self.prefix_b,
            self.suffix_a
        )

        prompt_c = self.make_prompt(
            self.prefix_c,
            self.suffix_a
        )

        with torch.inference_mode():

            # Insert A.
            self.rct.forward_cached(
                prompt_a,
                self.prefix_len
            )

            # Insert B.
            self.rct.forward_cached(
                prompt_b,
                self.prefix_len
            )

            keys_a_b = list(
                self.rct._prefix_cache.keys()
            )

            self.assertEqual(
                len(keys_a_b),
                2
            )

            # Access A again.
            self.rct.forward_cached(
                prompt_a,
                self.prefix_len
            )

            # Insert C. B should be evicted.
            self.rct.forward_cached(
                prompt_c,
                self.prefix_len
            )

        final_keys = list(
            self.rct._prefix_cache.keys()
        )

        self.assertEqual(
            len(final_keys),
            2
        )

        self.assertIn(
            keys_a_b[0],
            final_keys
        )

        self.assertNotIn(
            keys_a_b[1],
            final_keys
        )

    # ======================================
    # TEST 6
    # REPEATED REQUESTS
    # ======================================

    def test_06_repeated_requests_remain_correct(self):

        prompt = self.make_prompt(
            self.prefix_a,
            self.suffix_a
        )

        with torch.inference_mode():

            expected = self.baseline(prompt)

            for _ in range(10):

                actual = self.rct.forward_cached(
                    prompt,
                    self.prefix_len
                )

                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=1e-4,
                    atol=1e-4
                )

    # ======================================
    # TEST 7
    # CACHE CAPACITY
    # ======================================

    def test_07_cache_capacity_is_respected(self):

        prefixes = [
            self.prefix_a,
            self.prefix_b,
            self.prefix_c
        ]

        with torch.inference_mode():

            for prefix in prefixes:

                prompt = self.make_prompt(
                    prefix,
                    self.suffix_a
                )

                self.rct.forward_cached(
                    prompt,
                    self.prefix_len
                )

                self.assertLessEqual(
                    len(self.rct._prefix_cache),
                    2
                )


# ==========================================
# RUN TESTS
# ==========================================

if __name__ == "__main__":

    print("\n===== RCT 1.2 CACHE SAFETY TESTS =====\n")

    unittest.main(verbosity=2)
