from __future__ import annotations

import unittest

from opd.datasets import (
    _is_selected_level,
    format_math_example,
    tokenize_math_example,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        tokens = [10 + index for index, _ in enumerate(text.split())]
        return [1] + tokens if add_special_tokens else tokens


class DatasetTest(unittest.TestCase):
    def test_formats_math_problem_and_solution(self) -> None:
        result = format_math_example(
            {"problem": "What is 1 + 1?", "solution": r"\boxed{2}"}
        )
        self.assertEqual(result["prompt"], "Problem:\nWhat is 1 + 1?\n\nSolution:\n")
        self.assertEqual(result["solution"], r"\boxed{2}")

    def test_tokenization_masks_prompt_tokens(self) -> None:
        result = tokenize_math_example(
            {"problem": "1 + 1", "solution": "2"},
            tokenizer=FakeTokenizer(),
            max_length=32,
            prompt_template="Problem: {problem} Answer: ",
        )
        self.assertEqual(result["input_ids"][-1], 99)
        self.assertEqual(result["labels"][-2:], [10, 99])
        self.assertTrue(all(label == -100 for label in result["labels"][:-2]))

    def test_selects_configured_math_level(self) -> None:
        self.assertTrue(_is_selected_level({"level": "Level 4"}, (3, 4, 5)))
        self.assertFalse(_is_selected_level({"level": "Level 2"}, (3, 4, 5)))


if __name__ == "__main__":
    unittest.main()
