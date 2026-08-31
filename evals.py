from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import TrainerCallback

from config import EvalSettings


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
    from math_verify import parse, verify

    reference_text = str(reference)
    expected = extract_boxed_answer(reference_text) or reference_text.strip()
    prediction = extract_boxed_answer(response)
    gold_parsed = parse(f"${expected}$")
    prediction_parsed = parse(response)
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
    limit = min(settings.max_examples, len(dataset))
    dataset = dataset.select(range(limit))

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

            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_args)
            prompt_width = inputs["input_ids"].shape[1]
            generated_only = generated[:, prompt_width:]
            responses = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

            for index, ((question, reference), response) in enumerate(
                zip(questions_and_answers, responses, strict=True)
            ):
                correct, predicted, expected = grade_math_response(response, reference)
                token_ids = generated_only[index]
                reached_limit = token_ids.shape[0] >= settings.max_new_tokens
                records.append(
                    {
                        "step": step,
                        "question": question,
                        "reference": reference,
                        "expected": expected,
                        "prediction": predicted,
                        "correct": correct,
                        "reached_token_limit": reached_limit,
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
