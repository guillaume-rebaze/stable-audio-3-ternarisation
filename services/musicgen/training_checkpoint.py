"""Exact-enough MLX training checkpoint primitives for v3 block resumes."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
from pathlib import Path
import os
import random

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


def _flat_arrays(tree: object) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, value in tree_flatten(tree):
        safe_key = key or "__root__"
        result[safe_key] = np.asarray(value)
    return result


def _tree_from_arrays(arrays: dict[str, np.ndarray]) -> object:
    flat = [
        ("" if key == "__root__" else key, mx.array(value))
        for key, value in arrays.items()
    ]
    return tree_unflatten(flat)


def _pickle_text(value: object) -> str:
    return base64.b64encode(pickle.dumps(value, protocol=5)).decode("ascii")


def _unpickle_text(value: str) -> object:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_step_checkpoint(
    path: Path,
    model_state: object,
    optimizer_state: object,
    metadata: dict,
    python_rng_state: object,
    numpy_rng_state: object,
    mlx_rng_state: list[mx.array] | None = None,
) -> None:
    """Write model/optimizer/RNG state atomically; no implicit FP16 cast."""
    model_arrays = _flat_arrays(model_state)
    optimizer_arrays = _flat_arrays(optimizer_state)
    arrays: dict[str, np.ndarray] = {}
    for key, value in model_arrays.items():
        arrays[f"model::{key}"] = value
    for key, value in optimizer_arrays.items():
        arrays[f"optimizer::{key}"] = value
    if mlx_rng_state is not None:
        for index, value in enumerate(mlx_rng_state):
            arrays[f"mlx_rng::{index}"] = np.asarray(value)
    meta = dict(metadata)
    meta.update(
        {
            "schema": "onus.ternary-quality/v4-step-checkpoint",
            "model_keys": sorted(model_arrays),
            "optimizer_keys": sorted(optimizer_arrays),
            "mlx_rng_count": len(mlx_rng_state or []),
            "python_rng_pickle": _pickle_text(python_rng_state),
            "numpy_rng_pickle": _pickle_text(numpy_rng_state),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = path.with_suffix(".tmp.npz")
    tmp_json = path.with_suffix(".tmp.json")
    np.savez_compressed(str(tmp_npz), **arrays)
    meta["payload_sha256"] = _sha256(tmp_npz)
    meta["complete"] = True
    tmp_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(str(tmp_npz), str(path))
    os.replace(str(tmp_json), str(path.with_suffix(".json")))


def load_step_checkpoint(path: Path) -> dict:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    schema = metadata.get("schema")
    if schema not in {
        "onus.ternary-quality/v3-step-checkpoint",
        "onus.ternary-quality/v4-step-checkpoint",
    }:
        raise ValueError(f"unsupported step checkpoint schema: {metadata.get('schema')!r}")
    if schema == "onus.ternary-quality/v4-step-checkpoint":
        if metadata.get("complete") is not True:
            raise ValueError("step checkpoint is not marked complete")
        if _sha256(path) != metadata.get("payload_sha256"):
            raise ValueError("step checkpoint payload checksum mismatch")
    model_arrays: dict[str, np.ndarray] = {}
    optimizer_arrays: dict[str, np.ndarray] = {}
    mlx_rng: dict[int, mx.array] = {}
    with np.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            value = np.array(arrays[key])
            if key.startswith("model::"):
                model_arrays[key[7:]] = value
            elif key.startswith("optimizer::"):
                optimizer_arrays[key[11:]] = value
            elif key.startswith("mlx_rng::"):
                mlx_rng[int(key.split("::", 1)[1])] = mx.array(value)
    return {
        "metadata": metadata,
        "model_state": _tree_from_arrays(model_arrays),
        "optimizer_state": _tree_from_arrays(optimizer_arrays),
        "python_rng_state": _unpickle_text(metadata["python_rng_pickle"]),
        "numpy_rng_state": _unpickle_text(metadata["numpy_rng_pickle"]),
        "mlx_rng_state": [mlx_rng[i] for i in sorted(mlx_rng)],
    }


def restore_rngs(state: dict, rng: random.Random | None = None) -> random.Random:
    if rng is None:
        rng = random.Random()
    rng.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    if state.get("mlx_rng_state"):
        mx.random.state = state["mlx_rng_state"]
    return rng
