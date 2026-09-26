"""Profile block-0 input activations by production sigma for V8 calibration.

This is an inference-only pass.  It records the actual inputs seen by the
seven ternary candidate projections, grouped by sigma, instead of optimizing
from weight MSE or from a single timestep.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import train_ternary_quality as tq
from ternary_provenance_v8 import canonical_digest, sha256_file, validate_contract
from ternary_runtime_contract import timestep_tensor
from models.defs import dit_mlx_medium


TARGETS = (
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_kv",
    "cross_attn.to_out",
    "ff.ff.0.proj",
    "ff.ff.2",
)


class _Accumulator:
    def __init__(self, channels: int) -> None:
        self.count = 0
        self.calls = 0
        self.sum = np.zeros(channels, dtype=np.float64)
        self.sum_sq = np.zeros(channels, dtype=np.float64)
        self.minimum = np.full(channels, np.inf, dtype=np.float32)
        self.maximum = np.full(channels, -np.inf, dtype=np.float32)
        self.abs_maximum = np.zeros(channels, dtype=np.float32)
        self.p01_sum = np.zeros(channels, dtype=np.float64)
        self.p99_sum = np.zeros(channels, dtype=np.float64)

    def update(self, values: mx.array) -> None:
        matrix = np.asarray(values).astype(np.float32).reshape(-1, values.shape[-1])
        if matrix.size == 0:
            return
        self.calls += 1
        self.count += matrix.shape[0]
        self.sum += matrix.sum(axis=0, dtype=np.float64)
        self.sum_sq += np.square(matrix, dtype=np.float64).sum(axis=0)
        self.minimum = np.minimum(self.minimum, matrix.min(axis=0))
        self.maximum = np.maximum(self.maximum, matrix.max(axis=0))
        self.abs_maximum = np.maximum(self.abs_maximum, np.abs(matrix).max(axis=0))
        self.p01_sum += np.quantile(matrix, 0.01, axis=0)
        self.p99_sum += np.quantile(matrix, 0.99, axis=0)

    def arrays(self) -> dict[str, np.ndarray]:
        if self.count <= 0 or self.calls <= 0:
            raise ValueError("activation accumulator is empty")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "min": self.minimum,
            "max": self.maximum,
            "abs_max": self.abs_maximum,
            "p01": (self.p01_sum / self.calls).astype(np.float32),
            "p99": (self.p99_sum / self.calls).astype(np.float32),
        }


class _RecordingLinear(nn.Module):
    def __init__(self, source: nn.Linear, key: str, collector: "_ActivationCollector") -> None:
        super().__init__()
        self.weight = source.weight
        self.bias = getattr(source, "bias", None)
        self._key = key
        self._collector = collector

    def __call__(self, x: mx.array) -> mx.array:
        self._collector.record(self._key, self._sigma, x)
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output


class _ActivationCollector:
    def __init__(self) -> None:
        self.sigma: float | None = None
        self.accumulators: dict[str, dict[str, _Accumulator]] = {}

    @staticmethod
    def sigma_key(sigma: float) -> str:
        return f"{float(sigma):.8f}"

    def record(self, key: str, sigma: float, values: mx.array) -> None:
        sigma_key = self.sigma_key(sigma)
        by_sigma = self.accumulators.setdefault(key, {})
        accumulator = by_sigma.get(sigma_key)
        if accumulator is None:
            accumulator = _Accumulator(int(values.shape[-1]))
            by_sigma[sigma_key] = accumulator
        accumulator.update(values)


def _wrap_block_zero(teacher: nn.Module, collector: _ActivationCollector) -> None:
    block = teacher.transformer.layers[0]
    targets = {
        "self_attn.to_qkv": (block.self_attn, "to_qkv"),
        "self_attn.to_out": (block.self_attn, "to_out"),
        "cross_attn.to_q": (block.cross_attn, "to_q"),
        "cross_attn.to_kv": (block.cross_attn, "to_kv"),
        "cross_attn.to_out": (block.cross_attn, "to_out"),
        "ff.ff.0.proj": (block.ff.ff[0], "proj"),
        "ff.ff.2": (block.ff.ff[2], None),
    }
    for key, (parent, attribute) in targets.items():
        source = parent if attribute is None else getattr(parent, attribute)
        wrapped = _RecordingLinear(source, key, collector)
        if attribute is None:
            block.ff.ff[2] = wrapped
        else:
            setattr(parent, attribute, wrapped)


def _select_indices(
    prompt_indices: np.ndarray,
    sigmas: np.ndarray,
    max_states: int,
    replicas: int,
) -> list[int]:
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    prompt_values = sorted({int(value) for value in prompt_indices})
    sigma_values = sorted({float(value) for value in sigmas})
    selected: list[int] = []
    for prompt_index in prompt_values:
        for sigma in sigma_values:
            matches = np.flatnonzero(
                (prompt_indices == prompt_index) & np.isclose(sigmas, sigma, rtol=0.0, atol=1e-7)
            )
            selected.extend(int(index) for index in matches[:replicas])
    if len(selected) > max_states:
        if max_states < len(sigma_values):
            raise ValueError("max_states must preserve at least one sample per sigma")
        positions = np.linspace(0, len(selected) - 1, max_states, dtype=np.int64)
        selected = [selected[int(position)] for position in positions]
    if not selected:
        raise ValueError("state cache contains no selectable states")
    return selected


def _memory_snapshot() -> dict[str, float]:
    return tq.memory_snapshot()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, default=tq.MLX_RUNTIME_ROOT / "models/mlx/dit_medium_f16.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-states", type=int, default=128)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--max-metal-bytes", type=int, default=11_000_000_000)
    args = parser.parse_args()
    if args.max_states <= 0:
        raise ValueError("max_states must be positive")
    project_root = args.project_root.resolve()
    contract_report = validate_contract(args.dataset_contract, project_root)

    state_cache = args.state_cache.resolve()
    states_path = state_cache / "states.npz"
    conditions_path = state_cache / "conditions.npz"
    manifest_path = state_cache / "manifest.json"
    for path in (states_path, conditions_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    states_file = np.load(states_path, allow_pickle=False)
    conditions_file = np.load(conditions_path, allow_pickle=False)
    states = states_file["states"]
    sigmas = states_file["sigmas"].astype(np.float32)
    prompt_indices = states_file["prompt_indices"].astype(np.int32)
    selected_indices = _select_indices(
        prompt_indices, sigmas, args.max_states, args.replicas
    )
    selected_prompts = cache_manifest["dataset"]["selected_prompts"]
    if not selected_prompts:
        raise ValueError("state cache has no selected prompts")
    if any(int(index) >= len(selected_prompts) for index in prompt_indices[selected_indices]):
        raise ValueError("state cache prompt index exceeds selected prompt list")

    collector = _ActivationCollector()
    teacher = dit_mlx_medium.DiT(T_lat=int(states.shape[-1]))
    teacher.load_weights(str(args.teacher_weights), strict=False)
    teacher.freeze()
    _wrap_block_zero(teacher, collector)
    peak = 0
    for position, state_index in enumerate(selected_indices, start=1):
        sigma = float(sigmas[state_index])
        prompt_index = int(prompt_indices[state_index])
        for wrapped_key in TARGETS:
            parent = teacher.transformer.layers[0]
            if wrapped_key == "self_attn.to_qkv":
                module = parent.self_attn.to_qkv
            elif wrapped_key == "self_attn.to_out":
                module = parent.self_attn.to_out
            elif wrapped_key == "cross_attn.to_q":
                module = parent.cross_attn.to_q
            elif wrapped_key == "cross_attn.to_kv":
                module = parent.cross_attn.to_kv
            elif wrapped_key == "cross_attn.to_out":
                module = parent.cross_attn.to_out
            elif wrapped_key == "ff.ff.0.proj":
                module = parent.ff.ff[0].proj
            else:
                module = parent.ff.ff[2]
            module._sigma = sigma
        x = mx.array(states[state_index][None], dtype=mx.float16)
        cross = mx.array(conditions_file[f"cross_{prompt_index:04d}"], dtype=mx.float16)
        global_cond = mx.array(conditions_file["global_cond"], dtype=mx.float16)
        output = teacher(x, timestep_tensor(sigma), cross, global_cond)
        mx.eval(output)
        snapshot = _memory_snapshot()
        peak = max(peak, int(snapshot["metal_peak_gb"] * (1024**3)))
        if peak > args.max_metal_bytes:
            raise RuntimeError(f"activation profile exceeded Metal guard at state {state_index}")
        if position % 16 == 0 or position == len(selected_indices):
            print(json.dumps({"processed": position, "total": len(selected_indices), "peak_gb": peak / (1024**3)}), flush=True)

    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "schema": "onus.ternary-quality/v8-activation-profile",
        "status": "prepared",
        "state_cache": {
            "path": str(state_cache),
            "states_sha256": sha256_file(states_path),
            "conditions_sha256": sha256_file(conditions_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        "dataset_contract": contract_report,
        "teacher": {"path": str(args.teacher_weights), "sha256": sha256_file(args.teacher_weights)},
        "selection": {
            "state_count": len(selected_indices),
            "replicas_per_prompt_sigma": args.replicas,
            "indices": selected_indices,
            "prompt_count": len({int(prompt_indices[index]) for index in selected_indices}),
            "sigma_count": len({f"{float(sigmas[index]):.8f}" for index in selected_indices}),
            "prompts": selected_prompts,
        },
        "targets": list(TARGETS),
        "metrics": {},
        "memory": {"peak_metal_bytes": peak, "peak_metal_gb": peak / (1024**3)},
    }
    for target in TARGETS:
        target_summary: dict[str, Any] = {}
        for sigma_key, accumulator in collector.accumulators.get(target, {}).items():
            metric_arrays = accumulator.arrays()
            target_summary[sigma_key] = {
                "count": accumulator.count,
                "calls": accumulator.calls,
                "channels": int(metric_arrays["mean"].shape[0]),
            }
            for metric_name, values in metric_arrays.items():
                arrays[f"{target.replace('.', '__')}__sigma_{sigma_key}__{metric_name}"] = values
        summary["metrics"][target] = target_summary

    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "activation_profile.npz"
    np.savez_compressed(profile_path, **arrays)
    summary["profile"] = {
        "path": str(profile_path),
        "sha256": sha256_file(profile_path),
        "array_count": len(arrays),
    }
    summary["profile_digest"] = canonical_digest(summary["metrics"])
    (args.output_dir / "activation_profile.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir), "profile_digest": summary["profile_digest"], "peak_metal_gb": summary["memory"]["peak_metal_gb"]}, indent=2), flush=True)
    del teacher, states_file, conditions_file
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
