from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForMultimodalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import Settings, load_settings
from data import load_training_dataset
from teacher_targets import TeacherTargetCacheWriter, build_cache_identity

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute teacher top-K targets for off-policy distillation."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override teacher_targets.output_dir from the configuration",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate the configuration and stop before downloads",
    )
    return parser.parse_args()


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _check_tokenizers(student: Any, teacher: Any) -> None:
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError(
            "Teacher and student tokenizers must use the same token-to-ID mapping"
        )


def _load_teacher(settings: Settings) -> torch.nn.Module:
    teacher_kwargs: dict[str, Any] = {
        "dtype": _torch_dtype(settings.teacher.dtype),
        "attn_implementation": settings.teacher.attn_implementation,
        "trust_remote_code": settings.teacher.trust_remote_code,
        "use_kernels": settings.teacher.use_kernels,
        "device_map": {"": torch.cuda.current_device()},
    }
    if settings.teacher.quantization == "int8":
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    teacher = AutoModelForMultimodalLM.from_pretrained(
        settings.teacher.name,
        **teacher_kwargs,
    )
    teacher.eval()
    teacher.requires_grad_(False)
    teacher.config.use_cache = False
    return teacher


def _batch_examples(
    examples: Iterable[dict[str, list[int]]],
    batch_size: int,
) -> Iterable[list[dict[str, list[int]]]]:
    batch: list[dict[str, list[int]]] = []
    for example in examples:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _teacher_batch(
    examples: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_length = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full(
        (len(examples), max_length), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
    for row, example in enumerate(examples):
        ids = torch.tensor(example["input_ids"], dtype=torch.long)
        input_ids[row, : ids.shape[0]] = ids
        attention_mask[row, : ids.shape[0]] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def extract_teacher_targets(
    teacher: torch.nn.Module,
    examples: list[dict[str, list[int]]],
    *,
    pad_token_id: int,
    top_k: int,
    first_example_index: int,
) -> list[dict[str, torch.Tensor | int]]:
    """Run one teacher batch and return top-K log probabilities."""
    batch = _teacher_batch(examples, pad_token_id)
    device = next(teacher.parameters()).device
    model_inputs = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        outputs = teacher(**model_inputs, use_cache=False)
    logits = outputs.logits
    if top_k > logits.shape[-1]:
        raise ValueError("distillation.top_k exceeds the teacher vocabulary size")

    records: list[dict[str, torch.Tensor | int]] = []
    for row, example in enumerate(examples):
        input_ids = torch.tensor(example["input_ids"], dtype=torch.int32)
        labels = torch.tensor(example["labels"], dtype=torch.long)
        completion_positions = labels.ne(-100).nonzero(as_tuple=False).flatten()
        if completion_positions.numel() == 0:
            raise ValueError("A training example does not contain completion tokens")
        completion_start = int(completion_positions[0])
        if not torch.equal(
            completion_positions,
            torch.arange(completion_start, labels.shape[0]),
        ):
            raise ValueError("Completion labels must be one continuous suffix")

        sequence_length = input_ids.shape[0]
        prediction_logits = logits[
            row,
            completion_start - 1 : sequence_length - 1,
            :,
        ]
        topk_logits, topk_token_ids = torch.topk(
            prediction_logits,
            k=top_k,
            dim=-1,
        )
        log_normalizer = torch.logsumexp(
            prediction_logits.float(), dim=-1, keepdim=True
        )
        topk_logprobs = topk_logits.float() - log_normalizer
        records.append(
            {
                "example_index": first_example_index + row,
                "input_ids": input_ids,
                "completion_start": completion_start,
                "teacher_topk_token_ids": topk_token_ids.to(
                    device="cpu", dtype=torch.int32
                ),
                "teacher_topk_logprobs": topk_logprobs.cpu(),
            }
        )
    return records


def run(settings: Settings, output_dir: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher-target precomputation needs one NVIDIA GPU")

    student_tokenizer = AutoTokenizer.from_pretrained(
        settings.model.name,
        trust_remote_code=settings.model.trust_remote_code,
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        settings.teacher.name,
        trust_remote_code=settings.teacher.trust_remote_code,
    )
    _check_tokenizers(student_tokenizer, teacher_tokenizer)
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    student_tokenizer.padding_side = "right"

    identity = build_cache_identity(settings, student_tokenizer)
    writer = TeacherTargetCacheWriter(
        output_dir,
        identity=identity,
        expected_examples=settings.dataset.max_examples,
        storage_dtype=settings.teacher_targets.storage_dtype,
    )
    if writer.is_complete:
        print(
            f"Teacher-target cache is already complete: {output_dir} "
            f"({writer.examples_written} examples)"
        )
        return

    dataset = load_training_dataset(
        settings.dataset,
        tokenizer=student_tokenizer,
        max_length=settings.training.max_length,
    )
    completed_examples = writer.examples_written
    remaining = (
        example
        for index, example in enumerate(dataset)
        if index >= completed_examples
    )
    teacher = _load_teacher(settings)
    buffered_records: list[dict[str, torch.Tensor | int]] = []
    next_example_index = completed_examples

    for examples in _batch_examples(
        remaining,
        settings.teacher_targets.batch_size,
    ):
        records = extract_teacher_targets(
            teacher,
            examples,
            pad_token_id=student_tokenizer.pad_token_id,
            top_k=settings.distillation.top_k,
            first_example_index=next_example_index,
        )
        next_example_index += len(records)
        buffered_records.extend(records)
        while len(buffered_records) >= settings.teacher_targets.shard_size:
            shard_records = buffered_records[: settings.teacher_targets.shard_size]
            buffered_records = buffered_records[settings.teacher_targets.shard_size :]
            path = writer.write_shard(shard_records)
            print(
                f"Saved {path.name}: {writer.examples_written}/"
                f"{settings.dataset.max_examples} examples"
            )

    if buffered_records:
        path = writer.write_shard(buffered_records)
        print(
            f"Saved {path.name}: {writer.examples_written}/"
            f"{settings.dataset.max_examples} examples"
        )
    writer.finalize()
    print(f"Teacher-target cache is complete: {output_dir}")


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    args = parse_args()
    settings = load_settings(_resolve_project_path(args.config))
    output_dir = _resolve_project_path(
        args.output_dir or settings.teacher_targets.output_dir
    )
    print(
        "Configuration is valid: "
        f"teacher {settings.teacher.name}, {settings.dataset.max_examples} examples, "
        f"top-{settings.distillation.top_k} targets, cache {output_dir}."
    )
    if args.check_config:
        return
    run(settings, output_dir)


if __name__ == "__main__":
    main()
