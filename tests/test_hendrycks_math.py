from __future__ import annotations

import unittest

from hendrycks_math import _reference, _wandb_metrics


class HendrycksMathTest(unittest.TestCase):
    def test_uses_answer_when_present(self) -> None:
        self.assertEqual(_reference({"answer": "2", "solution": "work"}), "2")

    def test_uses_solution_when_answer_is_empty(self) -> None:
        self.assertEqual(_reference({"answer": "", "solution": "work"}), "work")

    def test_maps_summary_to_wandb_metrics(self) -> None:
        metrics = _wandb_metrics(
            {
                "accuracy": 0.5,
                "correct": 2,
                "examples": 4,
                "unparsed": 1,
                "reached_token_limit": 0,
            }
        )
        self.assertEqual(metrics["eval/math/score"], 0.5)
        self.assertEqual(metrics["eval/math/examples"], 4)


if __name__ == "__main__":
    unittest.main()
