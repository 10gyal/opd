from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from distillation import topk_soft_cross_entropy


class DistillationLossTest(unittest.TestCase):
    def test_top_one_matches_teacher_argmax_cross_entropy(self) -> None:
        student = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 0.0, 1.0]]])
        teacher = torch.tensor([[[0.0, 4.0, 1.0], [5.0, 0.0, 1.0], [0.0, 0.0, 0.0]]])
        labels = torch.tensor([[-100, 2, 0]])

        loss = topk_soft_cross_entropy(student, teacher, labels, top_k=1)
        expected = torch.stack(
            [
                F.cross_entropy(student[:, 0, :], torch.tensor([1])),
                F.cross_entropy(student[:, 1, :], torch.tensor([0])),
            ]
        ).mean()
        torch.testing.assert_close(loss, expected)

    def test_prompt_positions_do_not_add_loss(self) -> None:
        student = torch.zeros((1, 3, 2))
        teacher = torch.zeros((1, 3, 2))
        labels = torch.tensor([[-100, -100, 1]])
        loss = topk_soft_cross_entropy(student, teacher, labels, top_k=2)
        torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))


if __name__ == "__main__":
    unittest.main()
