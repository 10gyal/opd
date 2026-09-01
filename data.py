from __future__ import annotations

from typing import Any

from datasets import IterableDataset, interleave_datasets, load_dataset

from config import DatasetSettings


def format_math_example(
    example: dict[str, Any],
    prompt_template: str = "Problem:\n{problem}\n\nSolution:\n",
) -> dict[str, str]:
    problem = example.get("problem", example.get("question"))
    solution = example.get("solution", example.get("answer"))
    if not isinstance(problem, str) or not isinstance(solution, str):
        raise TypeError("Each example must contain text problem and solution fields")
    return {
        "prompt": prompt_template.format(problem=problem),
        "solution": solution,
    }


def tokenize_math_example(
    example: dict[str, Any],
    tokenizer: Any,
    max_length: int,
    prompt_template: str,
) -> dict[str, list[int]]:
    formatted = format_math_example(example, prompt_template)
    prompt_ids = tokenizer.encode(formatted["prompt"], add_special_tokens=True)
    solution_ids = tokenizer.encode(formatted["solution"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None and (
        not solution_ids or solution_ids[-1] != tokenizer.eos_token_id
    ):
        solution_ids.append(tokenizer.eos_token_id)

    input_ids = (prompt_ids + solution_ids)[:max_length]
    prompt_length = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_length + input_ids[prompt_length:]
    if not any(label != -100 for label in labels):
        raise ValueError("A tokenized example does not contain solution tokens")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _is_selected_level(example: dict[str, Any], levels: tuple[int, ...]) -> bool:
    raw_level = str(example.get("level", ""))
    digits = "".join(character for character in raw_level if character.isdigit())
    return bool(digits) and int(digits) in levels


def load_training_dataset(
    settings: DatasetSettings,
    tokenizer: Any,
    max_length: int,
) -> IterableDataset:
    config_names = settings.config_names or (settings.config_name,)
    sources = [
        load_dataset(
            settings.name,
            config_name,
            split=settings.split,
            streaming=True,
        )
        for config_name in config_names
    ]
    dataset = (
        sources[0]
        if len(sources) == 1
        else interleave_datasets(sources, seed=settings.seed)
    )
    if settings.levels:
        dataset = dataset.filter(
            _is_selected_level,
            fn_kwargs={"levels": settings.levels},
        )
    dataset = dataset.shuffle(
        seed=settings.seed,
        buffer_size=settings.shuffle_buffer_size,
    )
    dataset = dataset.take(settings.max_examples)
    columns = list(dataset.column_names or [])
    return dataset.map(
        tokenize_math_example,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "prompt_template": settings.prompt_template,
        },
        remove_columns=columns,
    )
