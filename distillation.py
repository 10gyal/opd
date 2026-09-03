from __future__ import annotations

import torch


def cached_topk_soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_topk_token_ids: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute cross-entropy from cached top-K teacher log probabilities."""
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("Labels must match the batch and sequence dimensions")
    if teacher_topk_token_ids.shape[:2] != labels.shape:
        raise ValueError(
            "Teacher token IDs must match the batch and sequence dimensions"
        )
    if teacher_topk_logprobs.shape != teacher_topk_token_ids.shape:
        raise ValueError("Teacher log probabilities must match teacher token IDs")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    shifted_student = student_logits[:, :-1, :]
    shifted_teacher_ids = teacher_topk_token_ids[:, 1:, :]
    shifted_teacher_logprobs = teacher_topk_logprobs[:, 1:, :]
    completion_mask = labels[:, 1:].ne(-100)
    token_count = completion_mask.sum()
    if token_count.item() == 0:
        raise ValueError("The batch does not contain completion tokens")
    if (
        shifted_teacher_ids.min() < 0
        or shifted_teacher_ids.max() >= student_logits.shape[-1]
    ):
        raise ValueError("A cached teacher token ID is outside the student vocabulary")

    teacher_weights = torch.softmax(
        shifted_teacher_logprobs.float() / temperature,
        dim=-1,
    )
    selected_student_logits = torch.gather(
        shifted_student,
        dim=-1,
        index=shifted_teacher_ids,
    ).float()
    student_log_normalizer = torch.logsumexp(
        shifted_student,
        dim=-1,
    ).float()
    selected_student_log_probs = (
        selected_student_logits - student_log_normalizer.unsqueeze(-1)
    )
    token_losses = -(teacher_weights * selected_student_log_probs).sum(dim=-1)
    return token_losses.masked_select(completion_mask).mean()


def topk_soft_cross_entropy(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute top-K teacher cross-entropy on completion tokens.

    The logits at sequence position ``t`` predict the label at position ``t + 1``.
    Values of -100 in ``labels`` mark prompt or padding tokens and do not add loss.
    Teacher probabilities are normalized again over the selected top-K tokens. This
    matches the Tinker off-policy distillation implementation.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Student and teacher logits must have the same shape")
    if labels.shape != student_logits.shape[:2]:
        raise ValueError("Labels must match the batch and sequence dimensions")
    if top_k < 1 or top_k > student_logits.shape[-1]:
        raise ValueError("top_k must be in the model vocabulary range")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    shifted_student = student_logits[:, :-1, :]
    shifted_teacher = teacher_logits[:, :-1, :]
    completion_mask = labels[:, 1:].ne(-100)
    token_count = completion_mask.sum()
    if token_count.item() == 0:
        raise ValueError("The batch does not contain completion tokens")

    with torch.no_grad():
        teacher_topk_logits, teacher_topk_tokens = torch.topk(
            shifted_teacher,
            k=top_k,
            dim=-1,
        )
        teacher_weights = torch.softmax(
            teacher_topk_logits.float() / temperature,
            dim=-1,
        )

    selected_student_logits = torch.gather(
        shifted_student,
        dim=-1,
        index=teacher_topk_tokens,
    ).float()
    student_log_normalizer = torch.logsumexp(
        shifted_student,
        dim=-1,
    ).float()
    selected_student_log_probs = (
        selected_student_logits - student_log_normalizer.unsqueeze(-1)
    )
    token_losses = -(teacher_weights * selected_student_log_probs).sum(dim=-1)
    return token_losses.masked_select(completion_mask).mean()
