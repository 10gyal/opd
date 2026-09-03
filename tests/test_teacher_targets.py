from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from precompute_teacher_targets import extract_teacher_targets
from teacher_targets import (
    CachedTeacherTargetDataset,
    CachedTeacherTargetsCollator,
    TeacherTargetCacheWriter,
)


def _record(
    example_index: int,
    input_ids: list[int],
    completion_start: int,
) -> dict[str, torch.Tensor | int]:
    completion_length = len(input_ids) - completion_start
    return {
        "example_index": example_index,
        "input_ids": torch.tensor(input_ids),
        "completion_start": completion_start,
        "teacher_topk_token_ids": torch.arange(
            completion_length * 2
        ).reshape(completion_length, 2),
        "teacher_topk_logprobs": torch.full(
            (completion_length, 2), -0.5
        ),
    }


class FakeTeacher(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.logits = logits

    def forward(self, **_: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.logits)


class TeacherTargetTest(unittest.TestCase):
    def test_resumes_writes_reads_and_collates_cache(self) -> None:
        identity = {"experiment": "test"}
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            writer = TeacherTargetCacheWriter(
                cache_dir,
                identity=identity,
                expected_examples=2,
                storage_dtype="float16",
            )
            writer.write_shard([_record(0, [1, 2, 3, 4], 2)])

            resumed = TeacherTargetCacheWriter(
                cache_dir,
                identity=identity,
                expected_examples=2,
                storage_dtype="float16",
            )
            self.assertEqual(resumed.examples_written, 1)
            resumed.write_shard([_record(1, [5, 6, 7], 1)])
            resumed.finalize()

            dataset = CachedTeacherTargetDataset(
                cache_dir,
                identity=identity,
                expected_examples=2,
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["completion_start"], 2)
            collated = CachedTeacherTargetsCollator(pad_token_id=0)(
                [dataset[0], dataset[1]]
            )
            self.assertEqual(collated["input_ids"].shape, (2, 4))
            self.assertEqual(collated["teacher_topk_token_ids"].shape, (2, 4, 2))
            self.assertEqual(collated["labels"].tolist()[0], [-100, -100, 3, 4])
            self.assertEqual(collated["labels"].tolist()[1], [-100, 6, 7, -100])

    def test_extracts_targets_for_completion_positions(self) -> None:
        logits = torch.tensor(
            [
                [
                    [0.0, 1.0, 2.0],
                    [3.0, 2.0, 1.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )
        teacher = FakeTeacher(logits)
        records = extract_teacher_targets(
            teacher,
            [
                {
                    "input_ids": [10, 11, 12, 13],
                    "attention_mask": [1, 1, 1, 1],
                    "labels": [-100, -100, 12, 13],
                }
            ],
            pad_token_id=0,
            top_k=2,
            first_example_index=7,
        )
        self.assertEqual(records[0]["example_index"], 7)
        self.assertEqual(records[0]["completion_start"], 2)
        self.assertEqual(
            torch.as_tensor(records[0]["teacher_topk_token_ids"]).tolist(),
            [[0, 1], [1, 2]],
        )


if __name__ == "__main__":
    unittest.main()
