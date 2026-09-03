from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForMultimodalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from config import Settings, load_settings
from distillation import cached_topk_soft_cross_entropy
from teacher_targets import (
    CachedTeacherTargetDataset,
    CachedTeacherTargetsCollator,
    build_cache_identity,
)
from utils import MathCallback

PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-K off-policy distillation on fixed math solutions."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML")
    parser.add_argument(
        "--teacher-targets",
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


def _set_logging_environment(settings: Settings, output_dir: Path) -> list[str]:
    if not settings.logging.wandb_project:
        return []
    os.environ.setdefault("WANDB_PROJECT", settings.logging.wandb_project)
    if settings.logging.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", settings.logging.wandb_entity)
    run_id_path = output_dir / ".wandb_run_id"
    run_id = os.environ.get("WANDB_RUN_ID")
    if run_id is None and run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        run_id = uuid.uuid4().hex
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ.setdefault("WANDB_RESUME", "allow")
    return ["wandb"]


def _resume_checkpoint(value: str | None, output_dir: Path) -> str | None:
    if value is None:
        return None
    if value.casefold() == "auto":
        return get_last_checkpoint(str(output_dir)) if output_dir.exists() else None
    checkpoint = _resolve_project_path(value)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    return str(checkpoint)


def _load_student(settings: Settings) -> torch.nn.Module:
    student = AutoModelForMultimodalLM.from_pretrained(
        settings.model.name,
        dtype=_torch_dtype(settings.model.dtype),
        attn_implementation=settings.model.attn_implementation,
        trust_remote_code=settings.model.trust_remote_code,
        use_kernels=settings.model.use_kernels,
    )
    student.config.use_cache = False
    return student


class OffPolicyDistillationTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        distillation_temperature: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.distillation_temperature = distillation_temperature
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> Any:
        del num_items_in_batch
        labels = inputs["labels"]
        teacher_topk_token_ids = inputs["teacher_topk_token_ids"]
        teacher_topk_logprobs = inputs["teacher_topk_logprobs"]
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        student_outputs = model(**model_inputs, use_cache=False)
        loss = cached_topk_soft_cross_entropy(
            student_logits=student_outputs.logits,
            teacher_topk_token_ids=teacher_topk_token_ids,
            teacher_topk_logprobs=teacher_topk_logprobs,
            labels=labels,
            temperature=self.distillation_temperature,
        )
        if return_outputs:
            return loss, student_outputs
        return loss


def run(settings: Settings, teacher_targets_dir: Path) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 1:
        raise RuntimeError("This configuration uses one GPU. WORLD_SIZE must be 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("Off-policy distillation needs one NVIDIA GPU.")

    output_dir = _resolve_project_path(settings.distillation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_to = _set_logging_environment(settings, output_dir)

    student_tokenizer = AutoTokenizer.from_pretrained(
        settings.model.name,
        trust_remote_code=settings.model.trust_remote_code,
    )
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    student_tokenizer.padding_side = "right"

    cache_identity = build_cache_identity(settings, student_tokenizer)
    train_dataset = CachedTeacherTargetDataset(
        teacher_targets_dir,
        identity=cache_identity,
        expected_examples=settings.dataset.max_examples,
    )
    data_collator = CachedTeacherTargetsCollator(
        pad_token_id=student_tokenizer.pad_token_id,
    )

    student = _load_student(settings)
    peft_config = LoraConfig(
        r=settings.lora.rank,
        lora_alpha=settings.lora.alpha,
        lora_dropout=settings.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(settings.lora.target_modules),
    )
    student = get_peft_model(student, peft_config)
    if settings.training.gradient_checkpointing:
        student.enable_input_require_grads()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=settings.training.per_device_train_batch_size,
        gradient_accumulation_steps=settings.training.gradient_accumulation_steps,
        max_steps=settings.training.max_steps,
        learning_rate=settings.training.learning_rate,
        lr_scheduler_type=settings.training.lr_scheduler_type,
        warmup_ratio=settings.training.warmup_ratio,
        weight_decay=settings.training.weight_decay,
        max_grad_norm=settings.training.max_grad_norm,
        logging_strategy="steps",
        logging_steps=settings.training.logging_steps,
        save_strategy="steps",
        save_steps=settings.training.save_steps,
        save_total_limit=settings.training.save_total_limit,
        eval_strategy="no",
        report_to=report_to,
        run_name=settings.distillation.wandb_run_name,
        bf16=settings.training.bf16,
        fp16=settings.model.dtype == "float16",
        tf32=settings.training.tf32,
        gradient_checkpointing=settings.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        seed=settings.training.seed,
        data_seed=settings.dataset.seed,
        remove_unused_columns=False,
    )

    callback = MathCallback(
        settings=settings.eval,
        output_dir=output_dir,
        seed=settings.training.seed,
        effective_batch_size=settings.training.effective_batch_size,
    )
    callbacks = [callback] if settings.eval.enabled else []
    trainer = OffPolicyDistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=student_tokenizer,
        callbacks=callbacks,
        distillation_temperature=settings.distillation.temperature,
    )
    callback.bind(trainer)
    trainer.model.config.use_cache = False

    checkpoint = _resume_checkpoint(
        settings.training.resume_from_checkpoint,
        output_dir,
    )
    trainer.train(resume_from_checkpoint=checkpoint)

    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    student_tokenizer.save_pretrained(str(final_dir))


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    args = parse_args()
    settings = load_settings(_resolve_project_path(args.config))
    teacher_targets_dir = _resolve_project_path(
        args.teacher_targets or settings.teacher_targets.output_dir
    )
    print(
        "Configuration is valid: "
        f"student {settings.model.name}, cached teacher {settings.teacher.name}, "
        f"{settings.dataset.max_examples} fixed examples, "
        f"top-{settings.distillation.top_k} targets, cache {teacher_targets_dir}."
    )
    if args.check_config:
        return
    run(settings, teacher_targets_dir)


if __name__ == "__main__":
    main()
