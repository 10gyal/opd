from __future__ import annotations

import hashlib
import json
import uuid
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from config import Settings

CACHE_FORMAT_VERSION = 1
_STORAGE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def tokenizer_vocab_hash(tokenizer: Any) -> str:
    """Return a stable hash of a tokenizer token-to-ID mapping."""
    digest = hashlib.sha256()
    vocabulary = tokenizer.get_vocab()
    for token, token_id in sorted(vocabulary.items()):
        encoded = token.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def build_cache_identity(settings: Settings, tokenizer: Any) -> dict[str, Any]:
    """Describe all inputs that change the saved teacher targets."""
    return {
        "teacher": {
            "model": settings.teacher.name,
            "dtype": settings.teacher.dtype,
            "attn_implementation": settings.teacher.attn_implementation,
            "use_kernels": settings.teacher.use_kernels,
            "quantization": settings.teacher.quantization,
        },
        "tokenizer": {
            "model": settings.model.name,
            "vocabulary_sha256": tokenizer_vocab_hash(tokenizer),
        },
        "dataset": {
            "name": settings.dataset.name,
            "config_name": settings.dataset.config_name,
            "config_names": list(settings.dataset.config_names),
            "split": settings.dataset.split,
            "levels": list(settings.dataset.levels),
            "prompt_template": settings.dataset.prompt_template,
            "max_examples": settings.dataset.max_examples,
            "shuffle_buffer_size": settings.dataset.shuffle_buffer_size,
            "seed": settings.dataset.seed,
        },
        "max_length": settings.training.max_length,
        "top_k": settings.distillation.top_k,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise TypeError(f"Invalid teacher-target metadata: {path}")
    return metadata


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    identity: dict[str, Any],
    expected_examples: int,
    storage_dtype: str,
) -> None:
    if metadata.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError("The teacher-target cache format does not match this code")
    if metadata.get("identity") != identity:
        raise ValueError(
            "The teacher-target cache does not match the current model, "
            "tokenizer, dataset, or target settings"
        )
    if metadata.get("expected_examples") != expected_examples:
        raise ValueError("The teacher-target cache has a different example count")
    if metadata.get("storage_dtype") != storage_dtype:
        raise ValueError("The teacher-target cache has a different storage dtype")


def _pack_records(records: list[dict[str, torch.Tensor | int]]) -> dict[str, torch.Tensor]:
    input_offsets = [0]
    target_offsets = [0]
    input_parts: list[torch.Tensor] = []
    target_id_parts: list[torch.Tensor] = []
    target_logprob_parts: list[torch.Tensor] = []
    completion_starts: list[int] = []
    example_indices: list[int] = []

    for record in records:
        input_ids = torch.as_tensor(record["input_ids"], dtype=torch.int32).cpu()
        target_ids = torch.as_tensor(
            record["teacher_topk_token_ids"], dtype=torch.int32
        ).cpu()
        target_logprobs = torch.as_tensor(
            record["teacher_topk_logprobs"]
        ).cpu()
        completion_start = int(record["completion_start"])
        if input_ids.ndim != 1:
            raise ValueError("Cached input_ids must have one dimension")
        if target_ids.ndim != 2 or target_logprobs.shape != target_ids.shape:
            raise ValueError("Cached teacher targets must have shape (tokens, top_k)")
        if target_ids.shape[0] != input_ids.shape[0] - completion_start:
            raise ValueError("Teacher target count does not match completion length")
        if completion_start < 1 or completion_start >= input_ids.shape[0]:
            raise ValueError("completion_start must select a non-empty completion")

        input_parts.append(input_ids)
        target_id_parts.append(target_ids)
        target_logprob_parts.append(target_logprobs)
        completion_starts.append(completion_start)
        example_indices.append(int(record["example_index"]))
        input_offsets.append(input_offsets[-1] + input_ids.shape[0])
        target_offsets.append(target_offsets[-1] + target_ids.shape[0])

    return {
        "input_ids": torch.cat(input_parts),
        "input_offsets": torch.tensor(input_offsets, dtype=torch.int64),
        "completion_starts": torch.tensor(completion_starts, dtype=torch.int32),
        "example_indices": torch.tensor(example_indices, dtype=torch.int64),
        "target_offsets": torch.tensor(target_offsets, dtype=torch.int64),
        "teacher_topk_token_ids": torch.cat(target_id_parts),
        "teacher_topk_logprobs": torch.cat(target_logprob_parts),
    }


class TeacherTargetCacheWriter:
    """Write resumable, checksummed teacher-target shards."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        identity: dict[str, Any],
        expected_examples: int,
        storage_dtype: str,
    ) -> None:
        if storage_dtype not in _STORAGE_DTYPES:
            raise ValueError("Unsupported teacher-target storage dtype")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.cache_dir / "metadata.json"
        self.identity = identity
        self.expected_examples = expected_examples
        self.storage_dtype = storage_dtype

        if self.metadata_path.exists():
            self.metadata = _read_metadata(self.metadata_path)
            _validate_metadata(
                self.metadata,
                identity=identity,
                expected_examples=expected_examples,
                storage_dtype=storage_dtype,
            )
            self._validate_shards()
        else:
            existing_shards = list(self.cache_dir.glob("shard-*.safetensors"))
            if existing_shards:
                raise ValueError(
                    "Teacher-target shards exist without metadata; use a new directory"
                )
            self.metadata = {
                "format_version": CACHE_FORMAT_VERSION,
                "status": "in_progress",
                "identity": identity,
                "expected_examples": expected_examples,
                "storage_dtype": storage_dtype,
                "examples_written": 0,
                "shards": [],
            }
            _write_json_atomic(self.metadata_path, self.metadata)

    @property
    def examples_written(self) -> int:
        return int(self.metadata["examples_written"])

    @property
    def is_complete(self) -> bool:
        return self.metadata.get("status") == "complete"

    def _validate_shards(self) -> None:
        example_count = 0
        for index, shard in enumerate(self.metadata.get("shards", [])):
            expected_name = f"shard-{index:05d}.safetensors"
            if shard.get("file") != expected_name:
                raise ValueError("Teacher-target shard names are not sequential")
            path = self.cache_dir / expected_name
            if not path.is_file():
                raise FileNotFoundError(f"Teacher-target shard does not exist: {path}")
            if _file_sha256(path) != shard.get("sha256"):
                raise ValueError(f"Teacher-target shard checksum failed: {path}")
            example_count += int(shard["examples"])
        if example_count != self.examples_written:
            raise ValueError("Teacher-target metadata has an invalid example count")
        if example_count > self.expected_examples:
            raise ValueError("Teacher-target cache contains too many examples")
        if self.is_complete and example_count != self.expected_examples:
            raise ValueError("Completed teacher-target cache is incomplete")

    def write_shard(self, records: list[dict[str, torch.Tensor | int]]) -> Path:
        if self.is_complete:
            raise ValueError("The teacher-target cache is already complete")
        if not records:
            raise ValueError("A teacher-target shard must not be empty")
        if self.examples_written + len(records) > self.expected_examples:
            raise ValueError("The teacher-target shard exceeds the expected example count")

        expected_indices = list(
            range(self.examples_written, self.examples_written + len(records))
        )
        actual_indices = [int(record["example_index"]) for record in records]
        if actual_indices != expected_indices:
            raise ValueError("Teacher-target example indices are not sequential")

        tensors = _pack_records(records)
        tensors["teacher_topk_logprobs"] = tensors[
            "teacher_topk_logprobs"
        ].to(_STORAGE_DTYPES[self.storage_dtype])
        shard_index = len(self.metadata["shards"])
        filename = f"shard-{shard_index:05d}.safetensors"
        path = self.cache_dir / filename
        temporary = self.cache_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
        save_file(tensors, str(temporary))
        temporary.replace(path)
        shard_info = {
            "file": filename,
            "examples": len(records),
            "input_tokens": int(tensors["input_ids"].shape[0]),
            "target_tokens": int(tensors["teacher_topk_token_ids"].shape[0]),
            "sha256": _file_sha256(path),
        }
        self.metadata["shards"].append(shard_info)
        self.metadata["examples_written"] = self.examples_written + len(records)
        _write_json_atomic(self.metadata_path, self.metadata)
        return path

    def finalize(self) -> None:
        if self.examples_written != self.expected_examples:
            raise ValueError(
                "Cannot complete the teacher-target cache before all examples are saved"
            )
        self.metadata["status"] = "complete"
        _write_json_atomic(self.metadata_path, self.metadata)


class CachedTeacherTargetDataset(Dataset[dict[str, torch.Tensor | int]]):
    """Read a complete teacher-target cache as a map-style dataset."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        identity: dict[str, Any],
        expected_examples: int,
        verify_checksums: bool = True,
    ) -> None:
        self.cache_dir = cache_dir
        metadata_path = cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Teacher-target metadata does not exist: {metadata_path}"
            )
        self.metadata = _read_metadata(metadata_path)
        storage_dtype = str(self.metadata.get("storage_dtype"))
        if storage_dtype not in _STORAGE_DTYPES:
            raise ValueError("Teacher-target cache has an unsupported storage dtype")
        _validate_metadata(
            self.metadata,
            identity=identity,
            expected_examples=expected_examples,
            storage_dtype=storage_dtype,
        )
        if self.metadata.get("status") != "complete":
            raise ValueError(
                "Teacher-target precomputation is incomplete; run the precompute command"
            )

        self.shards: list[dict[str, torch.Tensor]] = []
        self.cumulative_examples: list[int] = []
        total = 0
        for shard_info in self.metadata.get("shards", []):
            path = cache_dir / str(shard_info["file"])
            if not path.is_file():
                raise FileNotFoundError(f"Teacher-target shard does not exist: {path}")
            if verify_checksums and _file_sha256(path) != shard_info.get("sha256"):
                raise ValueError(f"Teacher-target shard checksum failed: {path}")
            tensors = load_file(str(path), device="cpu")
            shard_examples = int(shard_info["examples"])
            expected_indices = torch.arange(total, total + shard_examples)
            if not torch.equal(tensors["example_indices"], expected_indices):
                raise ValueError("Teacher-target example indices are not sequential")
            self.shards.append(tensors)
            total += shard_examples
            self.cumulative_examples.append(total)
        if total != expected_examples:
            raise ValueError("Teacher-target cache has an invalid total example count")

    def __len__(self) -> int:
        return self.cumulative_examples[-1] if self.cumulative_examples else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self.cumulative_examples, index)
        previous = 0 if shard_index == 0 else self.cumulative_examples[shard_index - 1]
        local_index = index - previous
        shard = self.shards[shard_index]
        input_start = int(shard["input_offsets"][local_index])
        input_end = int(shard["input_offsets"][local_index + 1])
        target_start = int(shard["target_offsets"][local_index])
        target_end = int(shard["target_offsets"][local_index + 1])
        return {
            "input_ids": shard["input_ids"][input_start:input_end].long(),
            "completion_start": int(shard["completion_starts"][local_index]),
            "teacher_topk_token_ids": shard["teacher_topk_token_ids"][
                target_start:target_end
            ].long(),
            "teacher_topk_logprobs": shard["teacher_topk_logprobs"][
                target_start:target_end
            ],
        }


class CachedTeacherTargetsCollator:
    """Pad cached examples and align targets with their completion tokens."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        features: Iterable[dict[str, torch.Tensor | int]],
    ) -> dict[str, torch.Tensor]:
        items = list(features)
        if not items:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(
            int(torch.as_tensor(item["input_ids"]).shape[0]) for item in items
        )
        first_targets = torch.as_tensor(items[0]["teacher_topk_token_ids"])
        top_k = int(first_targets.shape[1])
        logprob_dtype = torch.as_tensor(
            items[0]["teacher_topk_logprobs"]
        ).dtype

        input_ids = torch.full(
            (len(items), max_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(items), max_length), dtype=torch.long)
        labels = torch.full((len(items), max_length), -100, dtype=torch.long)
        target_ids = torch.zeros(
            (len(items), max_length, top_k), dtype=torch.long
        )
        target_logprobs = torch.zeros(
            (len(items), max_length, top_k), dtype=logprob_dtype
        )

        for row, item in enumerate(items):
            ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
            completion_start = int(item["completion_start"])
            teacher_ids = torch.as_tensor(
                item["teacher_topk_token_ids"], dtype=torch.long
            )
            teacher_logprobs = torch.as_tensor(item["teacher_topk_logprobs"])
            length = ids.shape[0]
            completion_length = length - completion_start
            if teacher_ids.shape != (completion_length, top_k):
                raise ValueError("Cached teacher token IDs have an invalid shape")
            if teacher_logprobs.shape != teacher_ids.shape:
                raise ValueError(
                    "Cached teacher log probabilities have an invalid shape"
                )

            input_ids[row, :length] = ids
            attention_mask[row, :length] = 1
            labels[row, completion_start:length] = ids[completion_start:length]
            target_ids[row, completion_start:length] = teacher_ids
            target_logprobs[row, completion_start:length] = teacher_logprobs

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "teacher_topk_token_ids": target_ids,
            "teacher_topk_logprobs": target_logprobs,
        }
