"""Provenance-checked cache for frozen teacher velocity targets."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


SCHEMA = "onus.ternary-quality/v7-teacher-target-cache"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def input_identity(
    state_cache: Path,
    teacher_weights: Path,
    target_contract: dict[str, object],
) -> dict[str, object]:
    return {
        "state_cache": str(state_cache.resolve()),
        "states": file_identity(state_cache / "states.npz"),
        "conditions": file_identity(state_cache / "conditions.npz"),
        "state_cache_manifest": file_identity(state_cache / "manifest.json"),
        "teacher_weights": file_identity(teacher_weights),
        "target_contract": target_contract,
    }


def _target_digest(targets: np.ndarray) -> str:
    canonical = np.ascontiguousarray(targets.astype(np.float16, copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def save_teacher_targets(
    path: Path,
    targets: np.ndarray,
    state_cache: Path,
    teacher_weights: Path,
    target_contract: dict[str, object],
) -> dict[str, object]:
    """Save fp16 targets and provenance without replacing existing artifacts."""
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite teacher target cache: {path}")
    values = np.asarray(targets)
    if values.dtype != np.float16 or values.ndim < 2:
        raise ValueError("teacher targets must be an fp16 array with a state axis")
    if not np.isfinite(values).all():
        raise ValueError("teacher targets contain NaN or Inf")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_npz = path.with_name(f".{path.stem}.tmp.npz")
    temporary_json = metadata_path.with_name(f".{metadata_path.stem}.tmp.json")
    np.savez_compressed(temporary_npz, targets=values)
    metadata = {
        "schema": SCHEMA,
        "cache_file": str(path.resolve()),
        "inputs": input_identity(state_cache, teacher_weights, target_contract),
        "target_shape": list(values.shape),
        "target_dtype": str(values.dtype),
        "target_sha256": _target_digest(values),
    }
    temporary_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_npz, path)
    os.replace(temporary_json, metadata_path)
    return metadata


def load_teacher_targets(
    path: Path,
    state_cache: Path,
    teacher_weights: Path,
    expected_count: int,
    target_contract: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"unsupported teacher target cache schema: {metadata.get('schema')!r}")
    actual_inputs = input_identity(state_cache, teacher_weights, target_contract)
    if metadata.get("inputs") != actual_inputs:
        raise ValueError("teacher target cache provenance does not match current inputs")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"targets"}:
            raise ValueError("teacher target cache must contain only the targets array")
        targets = np.array(arrays["targets"])
    if targets.dtype != np.float16 or targets.shape[0] != expected_count:
        raise ValueError(
            f"teacher targets must have {expected_count} fp16 states; "
            f"got shape={targets.shape}, dtype={targets.dtype}"
        )
    if list(targets.shape) != metadata.get("target_shape"):
        raise ValueError("teacher target cache shape disagrees with its metadata")
    if metadata.get("target_dtype") != str(targets.dtype):
        raise ValueError("teacher target cache dtype disagrees with its metadata")
    if _target_digest(targets) != metadata.get("target_sha256"):
        raise ValueError("teacher target cache checksum mismatch")
    if not np.isfinite(targets).all():
        raise ValueError("teacher targets contain NaN or Inf")
    return targets, metadata
