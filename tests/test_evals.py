from __future__ import annotations

import unittest

from utils import extract_boxed_answer


class EvalTest(unittest.TestCase):
    def test_extracts_last_balanced_box(self) -> None:
        text = r"First \boxed{1}, then \boxed{\frac{6}{2}}."
        self.assertEqual(extract_boxed_answer(text), r"\frac{6}{2}")

    def test_returns_none_without_boxed_answer(self) -> None:
        self.assertIsNone(extract_boxed_answer("The answer is 6."))


if __name__ == "__main__":
    unittest.main()
