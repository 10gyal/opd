from __future__ import annotations

import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import LogitsProcessor, TrainerCallback

from config import EvalSettings

MATH500_SUBJECTS = (
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
)
MATH500_LEVELS = (1, 2, 3, 4, 5)
GRADING_VERSION = 2


class GeneratedTokenPresencePenaltyProcessor(LogitsProcessor):
    """Subtract a fixed penalty from tokens already present in the generation."""

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        generated_ids = input_ids[:, self.prompt_length :]
        if self.penalty == 0 or generated_ids.shape[1] == 0:
            return scores
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, generated_ids, True)
        return scores - seen.to(scores.dtype) * self.penalty


def split_reasoning_and_final_response(
    raw_response: str,
    *,
    expect_reasoning: bool,
    special_tokens: tuple[str, ...] = (),
) -> tuple[str | None, str]:
    """Separate Qwen thinking content from its final answer."""

    def clean(text: str) -> str:
        for token in special_tokens:
            if token not in {"<think>", "</think>"}:
                text = text.replace(token, "")
        return text.strip()

    text = raw_response.strip()
    close_marker = "</think>"
    open_marker = "<think>"
    if close_marker in text:
        reasoning_text, final_text = text.split(close_marker, 1)
        if open_marker in reasoning_text:
            reasoning_text = reasoning_text.split(open_marker, 1)[1]
        return clean(reasoning_text), clean(final_text)
    if expect_reasoning:
        if open_marker in text:
            text = text.split(open_marker, 1)[1]
        return clean(text), ""
    return None, clean(text)


def select_balanced_math500_subset(dataset: Any, seed: int) -> Any:
    """Select 50 examples balanced across all MATH subjects and levels."""
    groups: dict[tuple[str, int], list[int]] = {
        (subject, level): []
        for subject in MATH500_SUBJECTS
        for level in MATH500_LEVELS
    }
    for index, example in enumerate(dataset):
        key = (str(example.get("subject")), int(example.get("level", 0)))
        if key in groups:
            groups[key].append(index)

    missing = [key for key, indices in groups.items() if len(indices) < 2]
    if missing:
        raise ValueError(
            "MATH-500 needs at least two examples in each subject-level group; "
            f"insufficient groups: {missing}"
        )

    rng = random.Random(seed)
    subjects = list(MATH500_SUBJECTS)
    rng.shuffle(subjects)
    subject_targets = {subject: 7 for subject in MATH500_SUBJECTS}
    subject_targets[subjects[0]] = 8

    row_extras = {
        subject: subject_targets[subject] - len(MATH500_LEVELS)
        for subject in MATH500_SUBJECTS
    }
    column_extras = {level: 3 for level in MATH500_LEVELS}
    extra_cells: set[tuple[str, int]] = set()

    def assign_extras(row: int) -> bool:
        if row == len(subjects):
            return all(remaining == 0 for remaining in column_extras.values())
        subject = subjects[row]
        options = list(combinations(MATH500_LEVELS, row_extras[subject]))
        rng.shuffle(options)
        for levels in options:
            if any(column_extras[level] == 0 for level in levels):
                continue
            for level in levels:
                column_extras[level] -= 1
                extra_cells.add((subject, level))
            if assign_extras(row + 1):
                return True
            for level in levels:
                column_extras[level] += 1
                extra_cells.remove((subject, level))
        return False

    if not assign_extras(0):
        raise RuntimeError("Could not construct balanced MATH-500 quotas")

    selected_indices: list[int] = []
    for key, indices in groups.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected_indices.extend(shuffled[: 2 if key in extra_cells else 1])

    selected = dataset.select(sorted(selected_indices))
    subject_counts = Counter(str(example["subject"]) for example in selected)
    level_counts = Counter(int(example["level"]) for example in selected)
    if sorted(subject_counts.values()) != [7, 7, 7, 7, 7, 7, 8]:
        raise RuntimeError("Balanced MATH-500 subject quotas were not satisfied")
    if set(level_counts.values()) != {10}:
        raise RuntimeError("Balanced MATH-500 level quotas were not satisfied")
    return selected


def extract_boxed_answer(text: str) -> str | None:
    marker = "\\boxed{"
    answers: list[str] = []
    start = 0
    while True:
        marker_index = text.find(marker, start)
        if marker_index < 0:
            break
        content_start = marker_index + len(marker)
        depth = 1
        cursor = content_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            answers.append(text[content_start : cursor - 1])
        start = content_start
    return answers[-1] if answers else None


def grade_math_response(response: str, reference: Any) -> tuple[bool, str | None, str]:
    reference_text = str(reference)
    expected = extract_boxed_answer(reference_text) or reference_text.strip()
    prediction = extract_boxed_answer(response)
    if prediction is None:
        return False, None, expected
    from math_verify import parse, verify

    gold_parsed = parse(f"${expected}$")
    prediction_parsed = parse(f"${prediction}$")
    correct = bool(
        gold_parsed and prediction_parsed and verify(gold_parsed, prediction_parsed)
    )
    return correct, prediction, expected


def _question_and_answer(example: dict[str, Any]) -> tuple[str, Any]:
    question = example.get("problem", example.get("question"))
    answer = example.get("answer", example.get("solution"))
    if not isinstance(question, str):
        raise TypeError("A math example does not have a text question")
    return question, answer


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def run_math_evaluation(
    model: torch.nn.Module,
    tokenizer: Any,
    settings: EvalSettings,
    output_dir: Path,
    step: int,
    seed: int,
    prompts_seen: int,
) -> dict[str, float]:
    dataset = load_dataset(
        settings.dataset_name,
        settings.config_name,
        split=settings.split,
    )
    if settings.balanced:
        dataset = select_balanced_math500_subset(dataset, settings.seed)
    else:
        limit = min(settings.max_examples, len(dataset))
        dataset = dataset.select(range(limit))
    limit = len(dataset)

    records: list[dict[str, Any]] = []
    was_training = model.training
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    torch.manual_seed(seed + step)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + step)

    try:
        for offset in range(0, limit, settings.batch_size):
            batch = dataset.select(
                range(offset, min(offset + settings.batch_size, limit))
            )
            questions_and_answers = [_question_and_answer(example) for example in batch]
            prompt_texts = [
                settings.prompt_template.format(problem=question)
                for question, _ in questions_and_answers
            ]
            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
            inputs = {
                name: value.to(_model_device(model)) for name, value in inputs.items()
            }
            generation_args: dict[str, Any] = {
                "max_new_tokens": settings.max_new_tokens,
                "do_sample": settings.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if settings.temperature > 0:
                generation_args["temperature"] = settings.temperature
                generation_args["top_p"] = settings.top_p
                generation_args["top_k"] = settings.top_k
                if settings.min_p > 0:
                    generation_args["min_p"] = settings.min_p
                if settings.presence_penalty > 0:
                    generation_args["logits_processor"] = [
                        GeneratedTokenPresencePenaltyProcessor(
                            settings.presence_penalty,
                            inputs["input_ids"].shape[1],
                        )
                    ]

            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_args)
            prompt_width = inputs["input_ids"].shape[1]
            generated_only = generated[:, prompt_width:]
            raw_responses = tokenizer.batch_decode(
                generated_only,
                skip_special_tokens=False,
            )

            for index, ((question, reference), raw_response) in enumerate(
                zip(questions_and_answers, raw_responses, strict=True)
            ):
                reasoning, response = split_reasoning_and_final_response(
                    raw_response,
                    expect_reasoning=False,
                    special_tokens=tuple(tokenizer.all_special_tokens),
                )
                correct, predicted, expected = grade_math_response(response, reference)
                token_ids = generated_only[index]
                reached_limit = token_ids.shape[0] >= settings.max_new_tokens
                records.append(
                    {
                        "step": step,
                        "grading_version": GRADING_VERSION,
                        "question": question,
                        "subject": batch[index].get("subject"),
                        "level": batch[index].get("level"),
                        "reference": reference,
                        "expected": expected,
                        "prediction": predicted,
                        "correct": correct,
                        "reached_token_limit": reached_limit,
                        "reasoning": reasoning,
                        "response": response,
                    }
                )
    finally:
        tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()

    eval_dir = output_dir / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    result_path = eval_dir / f"math_step_{step:06d}.jsonl"
    with result_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    correct_count = sum(int(record["correct"]) for record in records)
    unparsed_count = sum(record["prediction"] is None for record in records)
    limit_count = sum(int(record["reached_token_limit"]) for record in records)
    total = len(records)
    return {
        "eval/math/score": correct_count / total if total else 0.0,
        "eval/math/correct": float(correct_count),
        "eval/math/examples": float(total),
        "eval/math/unparsed": float(unparsed_count),
        "eval/math/reached_token_limit": float(limit_count),
        "eval/prompts_seen": float(prompts_seen),
    }


class MathCallback(TrainerCallback):
    def __init__(
        self,
        settings: EvalSettings,
        output_dir: Path,
        seed: int,
        effective_batch_size: int,
    ) -> None:
        self.settings = settings
        self.output_dir = output_dir
        self.seed = seed
        self.effective_batch_size = effective_batch_size
        self.completed_steps: set[int] = set()
        self.trainer: Any = None

    def bind(self, trainer: Any) -> None:
        self.trainer = trainer

    def _evaluate(self, state: Any) -> None:
        step = int(state.global_step)
        if (
            self.trainer is None
            or step in self.completed_steps
            or step not in self.settings.steps
            or not state.is_world_process_zero
        ):
            return
        metrics = run_math_evaluation(
            model=self.trainer.model,
            tokenizer=self.trainer.processing_class,
            settings=self.settings,
            output_dir=self.output_dir,
            step=step,
            seed=self.seed,
            prompts_seen=step * self.effective_batch_size,
        )
        self.completed_steps.add(step)
        self.trainer.log(metrics)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._evaluate(state)
        return control

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        self._evaluate(state)
        return control
