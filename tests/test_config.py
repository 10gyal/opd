from __future__ import annotations

import unittest
from pathlib import Path

from opd.config import load_settings

PROJECT_DIR = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_experiment_schedule(self) -> None:
        settings = load_settings(PROJECT_DIR / "config.yaml")
        self.assertEqual(settings.model.name, "Qwen/Qwen3.5-2B-Base")
        self.assertEqual(settings.teacher.name, "Qwen/Qwen3.5-9B-Base")
        self.assertEqual(settings.teacher.quantization, "int8")
        self.assertEqual(settings.dataset.max_examples, 512)
        self.assertEqual(settings.dataset.levels, (3, 4, 5))
        self.assertEqual(settings.training.effective_batch_size, 32)
        self.assertEqual(settings.training.max_steps, 16)
        self.assertEqual(settings.eval.steps, (8, 16))
        self.assertEqual(settings.eval.max_new_tokens, 1024)
        self.assertEqual(settings.eval.max_examples, 128)
        self.assertEqual(settings.training.save_steps, 8)
        self.assertEqual(settings.distillation.top_k, 20)


if __name__ == "__main__":
    unittest.main()
