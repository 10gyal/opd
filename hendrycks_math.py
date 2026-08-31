from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

from utils import grade_math_response


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = """Problem:
{problem}

Solve the problem. Put the final answer in \\boxed{{}}."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a model on MATH.")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--dataset", default="nlile/hendrycks-MATH-benchmark"
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="runs/qwen3.5-0.8b-hendrycks-math")
    parser.add_argument("--max-examples", type=int, default=500)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "opd")
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-run-name", default="qwen3.5-0.8b-hendrycks-math"
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    return parser.parse_args()


def _reference(example: dict[str, Any]) -> Any:
    answer = example.get("answer")
    if answer is not None and str(answer).strip():
        return answer
    return example.get("solution", "")


def _chat_prompt(tokenizer: Any, problem: str) -> str:
    content = DEFAULT_PROMPT.format(problem=problem)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            completed[str(record["unique_id"])] = record
    return completed


def _write_summary(
    path: Path,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    total = len(records)
    correct = sum(int(record["correct"]) for record in records)
    unparsed = sum(record["prediction"] is None for record in records)
    reached_limit = sum(int(record["reached_token_limit"]) for record in records)
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "unparsed": unparsed,
        "reached_token_limit": reached_limit,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _wandb_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "eval/math/score": summary["accuracy"],
        "eval/math/correct": summary["correct"],
        "eval/math/examples": summary["examples"],
        "eval/math/unparsed": summary["unparsed"],
        "eval/math/reached_token_limit": summary["reached_token_limit"],
    }


def _init_wandb(args: argparse.Namespace, output_dir: Path) -> Any:
    import wandb

    run_id_path = output_dir / ".wandb_run_id"
    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = uuid.uuid4().hex
        run_id_path.write_text(run_id + "\n", encoding="utf-8")
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        id=run_id,
        resume="allow",
        mode=args.wandb_mode,
        config=vars(args),
        dir=str(output_dir),
    )


def _log_artifact(
    wandb_run: Any,
    args: argparse.Namespace,
    records_path: Path,
    summary_path: Path,
) -> None:
    import wandb

    artifact = wandb.Artifact(
        name=f"{args.wandb_run_name}-results",
        type="evaluation",
        metadata={
            "model": args.model,
            "dataset": args.dataset,
            "split": args.split,
        },
    )
    if records_path.exists():
        artifact.add_file(str(records_path))
    artifact.add_file(str(summary_path))
    wandb_run.log_artifact(artifact)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    wandb_run = _init_wandb(args, output_dir)

    try:
        dataset = load_dataset(
            args.dataset,
            args.dataset_config,
            split=args.split,
        )
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
        completed = _load_completed(records_path)

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = LLM(
            model=args.model,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=args.seed,
            language_model_only=True,
        )
        sampling = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )

        pending = [
            example
            for example in dataset
            if str(example.get("unique_id", "")) not in completed
        ]
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            prompts = [
                _chat_prompt(tokenizer, str(example["problem"])) for example in batch
            ]
            outputs = model.generate(prompts, sampling, use_tqdm=True)
            with records_path.open("a", encoding="utf-8") as file:
                for example, output in zip(batch, outputs, strict=True):
                    response = output.outputs[0].text
                    expected_reference = _reference(example)
                    correct, prediction, expected = grade_math_response(
                        response, expected_reference
                    )
                    record = {
                        "unique_id": str(example.get("unique_id", "")),
                        "problem": example["problem"],
                        "subject": example.get("subject"),
                        "level": example.get("level"),
                        "reference": expected_reference,
                        "expected": expected,
                        "prediction": prediction,
                        "correct": correct,
                        "reached_token_limit": len(output.outputs[0].token_ids)
                        >= args.max_new_tokens,
                        "response": response,
                    }
                    completed[record["unique_id"]] = record
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
            summary = _write_summary(summary_path, list(completed.values()), args)
            wandb_run.log(_wandb_metrics(summary), step=summary["examples"])
            print(json.dumps(summary, sort_keys=True), flush=True)

        summary = _write_summary(summary_path, list(completed.values()), args)
        wandb_run.summary.update(summary)
        _log_artifact(wandb_run, args, records_path, summary_path)
        return summary
    finally:
        wandb_run.finish()


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    summary = run(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
