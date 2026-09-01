from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSettings:
    name: str
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    use_kernels: bool


@dataclass(frozen=True)
class TeacherSettings:
    name: str
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    use_kernels: bool
    quantization: str | None


@dataclass(frozen=True)
class LoraSettings:
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSettings:
    name: str
    config_name: str | None
    config_names: tuple[str, ...]
    split: str
    levels: tuple[int, ...]
    prompt_template: str
    max_examples: int
    shuffle_buffer_size: int
    seed: int


@dataclass(frozen=True)
class TrainingSettings:
    max_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    logging_steps: int
    save_steps: int
    save_total_limit: int
    gradient_checkpointing: bool
    bf16: bool
    tf32: bool
    seed: int
    resume_from_checkpoint: str | None

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def expected_examples(self) -> int:
        return self.effective_batch_size * self.max_steps


@dataclass(frozen=True)
class EvalSettings:
    enabled: bool
    dataset_name: str
    config_name: str | None
    split: str
    steps: tuple[int, ...]
    max_examples: int
    balanced: bool
    seed: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    batch_size: int
    prompt_template: str


@dataclass(frozen=True)
class DistillationSettings:
    output_dir: str
    top_k: int
    temperature: float
    wandb_run_name: str | None


@dataclass(frozen=True)
class LoggingSettings:
    wandb_project: str | None
    wandb_entity: str | None


@dataclass(frozen=True)
class Settings:
    model: ModelSettings
    teacher: TeacherSettings
    lora: LoraSettings
    dataset: DatasetSettings
    training: TrainingSettings
    eval: EvalSettings
    distillation: DistillationSettings
    logging: LoggingSettings


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise TypeError(f"Missing or invalid '{name}' section")
    return value


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def validate_settings(settings: Settings) -> None:
    if settings.model.dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("model.dtype must be bfloat16, float16, or float32")
    if settings.teacher.dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("teacher.dtype must be bfloat16, float16, or float32")
    if settings.teacher.quantization not in {None, "int8"}:
        raise ValueError("teacher.quantization must be int8 or null")

    _positive(settings.lora.rank, "lora.rank")
    _positive(settings.lora.alpha, "lora.alpha")
    if settings.lora.dropout < 0:
        raise ValueError("lora.dropout must be zero or greater")
    if not settings.lora.target_modules:
        raise ValueError("lora.target_modules must not be empty")

    _positive(settings.dataset.max_examples, "dataset.max_examples")
    _positive(settings.dataset.shuffle_buffer_size, "dataset.shuffle_buffer_size")
    if any(level < 1 or level > 5 for level in settings.dataset.levels):
        raise ValueError("dataset.levels values must be between 1 and 5")
    if settings.dataset.config_name and settings.dataset.config_names:
        raise ValueError("Use dataset.config_name or dataset.config_names, not both")
    _positive(settings.training.max_length, "training.max_length")
    _positive(
        settings.training.per_device_train_batch_size,
        "training.per_device_train_batch_size",
    )
    _positive(
        settings.training.gradient_accumulation_steps,
        "training.gradient_accumulation_steps",
    )
    _positive(settings.training.max_steps, "training.max_steps")
    _positive(settings.training.learning_rate, "training.learning_rate")
    _positive(settings.training.logging_steps, "training.logging_steps")
    _positive(settings.training.save_steps, "training.save_steps")
    _positive(settings.training.save_total_limit, "training.save_total_limit")
    if settings.training.max_grad_norm < 0:
        raise ValueError("training.max_grad_norm must be zero or greater")
    if not 0 <= settings.training.warmup_ratio <= 1:
        raise ValueError("training.warmup_ratio must be between zero and one")
    if settings.training.weight_decay < 0:
        raise ValueError("training.weight_decay must be zero or greater")

    if settings.dataset.max_examples != settings.training.expected_examples:
        raise ValueError(
            "dataset.max_examples must equal per_device_train_batch_size * "
            "gradient_accumulation_steps * max_steps for a one-GPU run"
        )

    _positive(settings.distillation.top_k, "distillation.top_k")
    _positive(settings.distillation.temperature, "distillation.temperature")

    if settings.eval.enabled:
        _positive(settings.eval.max_examples, "eval.max_examples")
        _positive(settings.eval.max_new_tokens, "eval.max_new_tokens")
        _positive(settings.eval.batch_size, "eval.batch_size")
        if settings.eval.balanced and settings.eval.max_examples != 50:
            raise ValueError("A balanced MATH-500 evaluation must use 50 examples")
        if settings.eval.temperature < 0:
            raise ValueError("eval.temperature must be zero or greater")
        if not 0 < settings.eval.top_p <= 1:
            raise ValueError(
                "eval.top_p must be greater than zero and not greater than one"
            )
        if settings.eval.top_k < 0:
            raise ValueError("eval.top_k must be zero or greater")
        if not 0 <= settings.eval.min_p <= 1:
            raise ValueError("eval.min_p must be between zero and one")
        if settings.eval.presence_penalty < 0:
            raise ValueError("eval.presence_penalty must be zero or greater")
        if tuple(sorted(set(settings.eval.steps))) != settings.eval.steps:
            raise ValueError("eval.steps must be sorted and contain no duplicates")
        if any(
            step < 1 or step > settings.training.max_steps
            for step in settings.eval.steps
        ):
            raise ValueError("Each eval step must be in the training step range")
        if settings.training.max_steps not in settings.eval.steps:
            raise ValueError("eval.steps must contain the final training step")


def load_settings(path: str | Path) -> Settings:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise TypeError("The configuration root must be a mapping")

    model = _section(raw, "model")
    teacher = _section(raw, "teacher")
    lora = _section(raw, "lora")
    dataset = _section(raw, "dataset")
    training = _section(raw, "training")
    evaluation = _section(raw, "eval")
    distillation = _section(raw, "distillation")
    logging = _section(raw, "logging")

    resume_value = training.get("resume_from_checkpoint")
    if resume_value is not None and not isinstance(resume_value, str):
        raise ValueError(
            "training.resume_from_checkpoint must be a path, auto, or null"
        )

    settings = Settings(
        model=ModelSettings(
            name=str(model["name"]),
            dtype=str(model.get("dtype", "bfloat16")),
            attn_implementation=str(model.get("attn_implementation", "sdpa")),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            use_kernels=bool(model.get("use_kernels", False)),
        ),
        teacher=TeacherSettings(
            name=str(teacher["name"]),
            dtype=str(teacher.get("dtype", "bfloat16")),
            attn_implementation=str(teacher.get("attn_implementation", "sdpa")),
            trust_remote_code=bool(teacher.get("trust_remote_code", False)),
            use_kernels=bool(teacher.get("use_kernels", False)),
            quantization=(
                str(teacher["quantization"])
                if teacher.get("quantization") is not None
                else None
            ),
        ),
        lora=LoraSettings(
            rank=int(lora["rank"]),
            alpha=int(lora["alpha"]),
            dropout=float(lora.get("dropout", 0.0)),
            target_modules=tuple(str(item) for item in lora["target_modules"]),
        ),
        dataset=DatasetSettings(
            name=str(dataset["name"]),
            config_name=(
                str(dataset["config_name"])
                if dataset.get("config_name") is not None
                else None
            ),
            config_names=tuple(
                str(config_name) for config_name in dataset.get("config_names", [])
            ),
            split=str(dataset.get("split", "train")),
            levels=tuple(int(level) for level in dataset.get("levels", [])),
            prompt_template=str(
                dataset.get("prompt_template", "Problem:\n{problem}\n\nSolution:\n")
            ),
            max_examples=int(dataset["max_examples"]),
            shuffle_buffer_size=int(dataset["shuffle_buffer_size"]),
            seed=int(dataset.get("seed", 42)),
        ),
        training=TrainingSettings(
            max_length=int(training["max_length"]),
            per_device_train_batch_size=int(training["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
            max_steps=int(training["max_steps"]),
            learning_rate=float(training["learning_rate"]),
            lr_scheduler_type=str(training.get("lr_scheduler_type", "linear")),
            warmup_ratio=float(training.get("warmup_ratio", 0.0)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
            logging_steps=int(training.get("logging_steps", 1)),
            save_steps=int(training["save_steps"]),
            save_total_limit=int(training.get("save_total_limit", 2)),
            gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
            bf16=bool(training.get("bf16", True)),
            tf32=bool(training.get("tf32", True)),
            seed=int(training.get("seed", 42)),
            resume_from_checkpoint=resume_value,
        ),
        eval=EvalSettings(
            enabled=bool(evaluation.get("enabled", True)),
            dataset_name=str(evaluation["dataset_name"]),
            config_name=(
                str(evaluation["config_name"])
                if evaluation.get("config_name") is not None
                else None
            ),
            split=str(evaluation.get("split", "train")),
            steps=tuple(int(step) for step in evaluation.get("steps", [])),
            max_examples=int(evaluation["max_examples"]),
            balanced=bool(evaluation.get("balanced", False)),
            seed=int(evaluation.get("seed", 42)),
            max_new_tokens=int(evaluation["max_new_tokens"]),
            temperature=float(evaluation.get("temperature", 0.0)),
            top_p=float(evaluation.get("top_p", 1.0)),
            top_k=int(evaluation.get("top_k", 0)),
            min_p=float(evaluation.get("min_p", 0.0)),
            presence_penalty=float(evaluation.get("presence_penalty", 0.0)),
            batch_size=int(evaluation.get("batch_size", 1)),
            prompt_template=str(
                evaluation.get("prompt_template", "Problem:\n{problem}\n\nSolution:\n")
            ),
        ),
        distillation=DistillationSettings(
            output_dir=str(distillation["output_dir"]),
            top_k=int(distillation.get("top_k", 20)),
            temperature=float(distillation.get("temperature", 1.0)),
            wandb_run_name=distillation.get("wandb_run_name"),
        ),
        logging=LoggingSettings(
            wandb_project=logging.get("wandb_project"),
            wandb_entity=logging.get("wandb_entity"),
        ),
    )
    validate_settings(settings)
    return settings
