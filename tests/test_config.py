from __future__ import annotations

import unittest
from pathlib import Path

from config import load_settings

PROJECT_DIR = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_experiment_schedule(self) -> None:
        settings = load_settings(PROJECT_DIR / "config.yaml")
        self.assertEqual(settings.model.name, "Qwen/Qwen3.5-0.8B-Base")
        self.assertEqual(settings.teacher.name, "Qwen/Qwen3.5-4B")
        self.assertIsNone(settings.teacher.quantization)
        self.assertEqual(settings.teacher_targets.batch_size, 4)
        self.assertEqual(settings.teacher_targets.shard_size, 32)
        self.assertEqual(settings.teacher_targets.storage_dtype, "float16")
        self.assertEqual(settings.dataset.max_examples, 7500)
        self.assertEqual(len(settings.dataset.config_names), 7)
        self.assertEqual(settings.dataset.levels, ())
        self.assertEqual(settings.training.per_device_train_batch_size, 2)
        self.assertEqual(settings.training.gradient_accumulation_steps, 10)
        self.assertEqual(settings.training.effective_batch_size, 20)
        self.assertEqual(settings.training.expected_examples, 7500)
        self.assertEqual(settings.training.max_steps, 375)
        self.assertEqual(
            settings.eval.steps,
            (50, 100, 150, 200, 250, 300, 350, 375),
        )
        self.assertEqual(settings.eval.max_new_tokens, 1024)
        self.assertEqual(settings.eval.max_examples, 50)
        self.assertTrue(settings.eval.balanced)
        self.assertEqual(settings.eval.min_p, 0.0)
        self.assertEqual(settings.eval.presence_penalty, 0.0)
        self.assertEqual(settings.training.save_steps, 50)
        self.assertEqual(settings.distillation.top_k, 20)


if __name__ == "__main__":
    unittest.main()
