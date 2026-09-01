from __future__ import annotations

import unittest
from collections import Counter
from types import ModuleType
from unittest.mock import patch

import torch
from datasets import Dataset

from utils import (
    GeneratedTokenPresencePenaltyProcessor,
    MATH500_LEVELS,
    MATH500_SUBJECTS,
    extract_boxed_answer,
    grade_math_response,
    select_balanced_math500_subset,
    split_reasoning_and_final_response,
)


class EvalTest(unittest.TestCase):
    def test_extracts_last_balanced_box(self) -> None:
        text = r"First \boxed{1}, then \boxed{\frac{6}{2}}."
        self.assertEqual(extract_boxed_answer(text), r"\frac{6}{2}")

    def test_returns_none_without_boxed_answer(self) -> None:
        self.assertIsNone(extract_boxed_answer("The answer is 6."))

    def test_grader_requires_boxed_prediction(self) -> None:
        correct, prediction, _ = grade_math_response("The answer is 6.", "6")
        self.assertFalse(correct)
        self.assertIsNone(prediction)

    def test_grader_verifies_only_boxed_prediction(self) -> None:
        parsed_inputs = []
        fake_math_verify = ModuleType("math_verify")

        def parse(value: str) -> list[str]:
            parsed_inputs.append(value)
            return [value]

        fake_math_verify.parse = parse
        fake_math_verify.verify = lambda gold, prediction: gold == prediction
        with patch.dict("sys.modules", {"math_verify": fake_math_verify}):
            correct, prediction, _ = grade_math_response(
                r"A wrong draft says 7. Final: \boxed{6}",
                "6",
            )
        self.assertTrue(correct)
        self.assertEqual(prediction, "6")
        self.assertEqual(parsed_inputs, ["$6$", "$6$"])

    def test_splits_completed_thinking_response(self) -> None:
        reasoning, response = split_reasoning_and_final_response(
            r"<think>work and \boxed{3}</think>Final: \boxed{4}<|im_end|>",
            expect_reasoning=True,
            special_tokens=("<|im_end|>",),
        )
        self.assertEqual(reasoning, r"work and \boxed{3}")
        self.assertEqual(response, r"Final: \boxed{4}")

    def test_incomplete_thinking_has_no_final_response(self) -> None:
        reasoning, response = split_reasoning_and_final_response(
            r"work that mentions \boxed{4}",
            expect_reasoning=True,
        )
        self.assertEqual(reasoning, r"work that mentions \boxed{4}")
        self.assertEqual(response, "")

    def test_presence_penalty_applies_only_to_generated_tokens(self) -> None:
        processor = GeneratedTokenPresencePenaltyProcessor(1.5, prompt_length=2)
        token_ids = torch.tensor([[1, 2, 3, 3]])
        scores = torch.zeros((1, 5))
        adjusted = processor(token_ids, scores)
        self.assertEqual(adjusted[0, 3].item(), -1.5)
        self.assertEqual(adjusted[0, 1].item(), 0.0)

    def test_selects_balanced_math500_subset(self) -> None:
        rows = []
        for subject in MATH500_SUBJECTS:
            for level in MATH500_LEVELS:
                for copy in range(3):
                    rows.append(
                        {
                            "subject": subject,
                            "level": level,
                            "unique_id": f"{subject}-{level}-{copy}",
                        }
                    )
        selected = select_balanced_math500_subset(Dataset.from_list(rows), seed=42)
        subject_counts = Counter(selected["subject"])
        level_counts = Counter(selected["level"])
        joint_counts = Counter(zip(selected["subject"], selected["level"]))

        self.assertEqual(len(selected), 50)
        self.assertEqual(sorted(subject_counts.values()), [7, 7, 7, 7, 7, 7, 8])
        self.assertEqual(set(level_counts.values()), {10})
        self.assertTrue(all(count in {1, 2} for count in joint_counts.values()))


if __name__ == "__main__":
    unittest.main()
