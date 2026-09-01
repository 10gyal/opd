from __future__ import annotations

import argparse
import unittest

from hendrycks_math import (
    DEFAULT_MODEL,
    DEFAULT_DATASET,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WANDB_RUN_NAME,
    _chat_prompt,
    _reference,
    _resolve_sampling_parameters,
    _wandb_metrics,
)


class FakeChatTokenizer:
    def __init__(self) -> None:
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return messages[0]["content"]


class HendrycksMathTest(unittest.TestCase):
    def test_uses_qwen_base_defaults(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(DEFAULT_DATASET, "HuggingFaceH4/MATH-500")
        self.assertIn("qwen3.5-0.8b-base", DEFAULT_OUTPUT_DIR)
        self.assertIn("qwen3.5-0.8b-base", DEFAULT_WANDB_RUN_NAME)
        self.assertEqual(DEFAULT_MAX_NEW_TOKENS, 2048)
        self.assertEqual(DEFAULT_MAX_MODEL_LEN, 8192)

    def test_disables_thinking_by_default(self) -> None:
        tokenizer = FakeChatTokenizer()
        _chat_prompt(tokenizer, "1 + 1")
        self.assertFalse(tokenizer.kwargs["enable_thinking"])

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

    def test_thinking_mode_uses_qwen_sampling_defaults(self) -> None:
        args = argparse.Namespace(
            enable_thinking=True,
            temperature=None,
            top_p=None,
            top_k=None,
            min_p=None,
            presence_penalty=None,
        )
        _resolve_sampling_parameters(args)
        self.assertEqual(args.temperature, 1.0)
        self.assertEqual(args.top_p, 0.95)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.presence_penalty, 1.5)


if __name__ == "__main__":
    unittest.main()
