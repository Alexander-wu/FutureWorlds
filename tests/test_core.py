import unittest

import torch

from check_grouped import tests as grouped_checks
from check_ordinary import run_tests as ordinary_checks
from futureworlds.decoding import independent_check
from futureworlds.text_conditioning import TextResidual


class CoreChecks(unittest.TestCase):
    def test_grouped_search_and_rewards(self):
        self.assertTrue(grouped_checks()["grouped_search_uncached_exact"])

    def test_ordinary_search_and_gradient_direction(self):
        result = ordinary_checks()
        self.assertGreater(result["positive_advantage_update_direction"], 0)

    def test_evaluation_beam(self):
        self.assertTrue(independent_check()["complete"])

    def test_text_adapter_starts_as_identity(self):
        adapter = TextResidual(hidden=32, text_width=24, inner=16, heads=4)
        hidden = torch.randn(2, 5, 32)
        text = torch.randn(1, 3, 24)
        mask = torch.ones(1, 3, dtype=torch.bool)
        self.assertTrue(torch.equal(adapter(hidden, text, mask), hidden))


if __name__ == "__main__":
    unittest.main()
