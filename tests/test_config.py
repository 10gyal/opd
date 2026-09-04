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
        self.assertEqual(settings.training.max_length, 2048)
        self.assertEqual(settings.training.expected_examples, 30000)
        self.assertEqual(settings.training.max_steps, 1500)
        self.assertEqual(
            settings.eval.steps,
            tuple(range(50, 1501, 50)),
        )
        self.assertEqual(
            settings.eval.full_steps,
            (250, 500, 750, 1000, 1250, 1500),
        )
        self.assertEqual(settings.eval.max_length, 2048)
        self.assertEqual(settings.eval.max_length, settings.training.max_length)
        self.assertEqual(settings.eval.max_examples, 50)
        self.assertEqual(settings.eval.full_max_examples, 500)
        self.assertTrue(settings.eval.balanced)
        self.assertEqual(settings.eval.min_p, 0.0)
        self.assertEqual(settings.eval.presence_penalty, 0.0)
        self.assertEqual(settings.training.save_steps, 50)
        self.assertEqual(settings.distillation.top_k, 20)


if __name__ == "__main__":
    unittest.main()
