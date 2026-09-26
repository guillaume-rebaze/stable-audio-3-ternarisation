"""Validate v3 config and write a provenance-safe dataset/split manifest.

The historical bank is intentionally emitted as ``debug_train_seen``.  This
tool refuses to call it validation/test because those samples were already
used to inspect and select previous candidates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from prepare_ternary_v3 import array_headers, scope_from_headers, ternary_storage_bytes
from ternary_contract import scope_digest


TOP_KEYS = {
    "schema", "dataset", "teacher", "model", "quantizer", "training",
    "gates", "resources",
}
DATASET_KEYS = {
    "train_dir", "independent_validation_dir", "independent_test_dir",
    "min_train_parents", "min_validation_parents", "min_validation_prompts",
    "min_test_prompts",
}


def require_exact_keys(value: dict, expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys mismatch: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def git_value(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], check=False, capture_output=True, text=True
        ).stdout.strip() or None
    except OSError:
        return None


def parent_key(meta: dict, path: Path) -> str:
    source = str(meta.get("src_relpath") or meta.get("path") or path.name)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def scan_dataset(
    dataset_dir: Path,
    role: str = "debug_train_seen",
) -> tuple[list[dict], set[str], set[str]]:
    samples: list[dict] = []
    parents: set[str] = set()
    prompts: set[str] = set()
    for path in sorted(dataset_dir.glob("*.npy")):
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        shape = tuple(int(x) for x in np.load(path, mmap_mode="r").shape)
        if len(shape) != 2 or shape[0] != 256:
            raise ValueError(f"invalid latent shape {path}: {shape}")
        parent = parent_key(meta, path)
        prompt = str(meta.get("prompt", ""))
        parents.add(parent)
        prompts.add(prompt)
        samples.append({
            "file": str(path),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata": str(meta_path) if meta_path.exists() else None,
            "shape": list(shape),
            "parent_id": parent,
            "prompt": prompt,
            "genre": str(meta.get("genre", "unknown")),
            "source": meta.get("src_relpath", meta.get("path", "")),
            "role": role,
            "rights": meta.get("rights", "not_declared_in_latent_manifest"),
            "usage_authorization": (
                "user_authorized_personal_study" if role in {"validation", "test"} else None
            ),
        })
    return samples, parents, prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v3 provenance-safe dataset manifests")
    parser.add_argument("--config", type=Path, default=Path("configs/ternary_quality_v3.json"))
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=Path.home() / ".cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/sample-expertise-pilot/ternary-quality-v3/manifests"),
    )
    parser.add_argument(
        "--require-independent",
        action="store_true",
        help="return code 2 if independent validation/test data is unavailable",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "onus.ternary-quality/v3-config":
        raise ValueError(f"unsupported config schema: {config.get('schema')!r}")
    if set(config) != TOP_KEYS:
        raise ValueError(f"top-level config keys mismatch: {sorted(set(config) ^ TOP_KEYS)}")
    require_exact_keys(config["dataset"], DATASET_KEYS, "dataset")
    if config["quantizer"]["mode"] != "symmetric":
        raise ValueError("v3 dataset preparation requires the strict symmetric lane")
    if config["model"]["blocks"] != 24 or len(config["model"]["core_names"]) != 7:
        raise ValueError("unexpected DiT core contract")

    train_dir = Path(config["dataset"]["train_dir"])
    samples, parents, prompts = scan_dataset(train_dir, "debug_train_seen")
    validation_dir_value = config["dataset"]["independent_validation_dir"]
    test_dir_value = config["dataset"]["independent_test_dir"]
    validation_dir = Path(validation_dir_value) if validation_dir_value else None
    test_dir = Path(test_dir_value) if test_dir_value else None
    validation_samples: list[dict] = []
    validation_parents: set[str] = set()
    validation_prompts: set[str] = set()
    test_samples: list[dict] = []
    test_parents: set[str] = set()
    test_prompts: set[str] = set()
    blockers: list[str] = []
    if validation_dir is not None:
        validation_samples, validation_parents, validation_prompts = scan_dataset(
            validation_dir, "validation"
        )
    if test_dir is not None:
        test_samples, test_parents, test_prompts = scan_dataset(test_dir, "test")
    if validation_dir is None or test_dir is None:
        blockers.append("independent validation and test directories are both required")
    if validation_dir is not None and not validation_samples:
        blockers.append("independent validation directory has no latent samples")
    if test_dir is not None and not test_samples:
        blockers.append("independent test directory has no latent samples")
    if validation_dir is not None and test_dir is not None:
        if parents & validation_parents:
            raise ValueError("train/validation parent overlap detected")
        if parents & test_parents:
            raise ValueError("train/test parent overlap detected")
        if validation_parents & test_parents:
            raise ValueError("validation/test parent overlap detected")
        if len(validation_parents) < int(config["dataset"]["min_validation_parents"]):
            blockers.append(
                "validation parent count below minimum "
                f"({len(validation_parents)} < {config['dataset']['min_validation_parents']})"
            )
        if len(validation_prompts) < int(config["dataset"]["min_validation_prompts"]):
            blockers.append(
                "validation prompt count below minimum "
                f"({len(validation_prompts)} < {config['dataset']['min_validation_prompts']})"
            )
        if len(test_prompts) < int(config["dataset"]["min_test_prompts"]):
            blockers.append(
                "test prompt count below minimum "
                f"({len(test_prompts)} < {config['dataset']['min_test_prompts']})"
            )
    independent_available = not blockers
    teacher_headers = array_headers(args.teacher_weights)
    scope = scope_from_headers(teacher_headers)
    group_size = int(config["quantizer"]["group_size"])
    full_bytes = sum(int(np.prod(shape)) * dtype.itemsize for shape, dtype in teacher_headers.values())
    core_bytes = sum(int(np.prod(teacher_headers[key][0])) * teacher_headers[key][1].itemsize for key in scope)
    ternary_bytes = sum(ternary_storage_bytes(teacher_headers[key][0], group_size) for key in scope)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = {
        "schema": "onus.ternary-quality/v3/environment",
        "python": sys.version,
        "platform": platform.platform(),
        "git_revision": git_value(["rev-parse", "HEAD"]),
        "git_status_short": git_value(["status", "--short"]),
        "config": str(args.config),
        "teacher_weights": str(args.teacher_weights),
        "authorization": {
            "authorized_by_user": True,
            "scope": "personal_study",
            "redistribution": False,
            "source_metadata_preserved": True,
        },
    }
    dataset_manifest = {
        "schema": "onus.ternary-quality/v3/dataset",
        "status": "debug_only" if not independent_available else "prepared",
        "train_dir": str(train_dir),
        "sample_count": len(samples),
        "parent_count": len(parents),
        "prompt_count": len(prompts),
        "samples": samples,
        "validation_dir": str(validation_dir) if validation_dir else None,
        "validation_sample_count": len(validation_samples),
        "validation_parent_count": len(validation_parents),
        "validation_prompt_count": len(validation_prompts),
        "validation_samples": validation_samples,
        "test_dir": str(test_dir) if test_dir else None,
        "test_sample_count": len(test_samples),
        "test_parent_count": len(test_parents),
        "test_prompt_count": len(test_prompts),
        "test_samples": test_samples,
        "independent_validation_available": independent_available,
        "independent_test_available": independent_available,
        "authorization": {
            "authorized_by_user": True,
            "scope": "personal_study",
            "redistribution": False,
            "source_metadata_preserved": True,
        },
        "blocker": None if independent_available else "; ".join(blockers),
    }
    splits = {
        "schema": "onus.ternary-quality/v3/splits",
        "status": "debug_only" if not independent_available else "prepared",
        "roles": {
            "debug_train_seen": [s["file"] for s in samples],
            "validation": [s["file"] for s in validation_samples],
            "test": [s["file"] for s in test_samples],
        },
        "parent_disjoint": not (parents & validation_parents or parents & test_parents or validation_parents & test_parents),
        "heldout": bool(validation_samples and test_samples and not blockers),
        "validation": [s["file"] for s in validation_samples],
        "test": [s["file"] for s in test_samples],
        "validation_parent_count": len(validation_parents),
        "test_parent_count": len(test_parents),
        "validation_prompt_count": len(validation_prompts),
        "test_prompt_count": len(test_prompts),
        "blocker": dataset_manifest["blocker"],
    }
    scope_manifest = {
        "schema": "onus.ternary-quality/v3/scope",
        "scope_train": scope,
        "scope_export": scope,
        "scope_reload": scope,
        "scope_inference": scope,
        "count": len(scope),
        "digest": scope_digest(scope),
        "group_size": group_size,
    }
    size_budget = {
        "schema": "onus.ternary-quality/v3/size-budget",
        "teacher_dense_array_bytes": full_bytes,
        "core_dense_array_bytes": core_bytes,
        "core_ternary_array_bytes": ternary_bytes,
        "estimated_compact_export_array_bytes": full_bytes - core_bytes + ternary_bytes,
        "max_artifact_bytes": config["gates"]["max_artifact_bytes"],
        "target_pass_estimate": full_bytes - core_bytes + ternary_bytes <= config["gates"]["max_artifact_bytes"],
    }
    run_config = {
        "schema": "onus.ternary-quality/v3/run-config",
        "source_config": str(args.config),
        "config": config,
        "environment_file": str(args.output_dir / "environment.json"),
        "dataset_manifest_file": str(args.output_dir / "dataset_manifest.json"),
        "splits_file": str(args.output_dir / "splits.json"),
        "scope_file": str(args.output_dir / "scope.json"),
        "size_budget_file": str(args.output_dir / "size_budget.json"),
    }
    for name, payload in {
        "environment.json": environment,
        "dataset_manifest.json": dataset_manifest,
        "splits.json": splits,
        "scope.json": scope_manifest,
        "size_budget.json": size_budget,
        "run_config.json": run_config,
    }.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    summary = {
        "status": dataset_manifest["status"],
        "samples": len(samples),
        "parents": len(parents),
        "prompts": len(prompts),
        "scope_count": len(scope),
        "scope_digest": scope_digest(scope),
        "independent_available": independent_available,
        "validation_samples": len(validation_samples),
        "validation_parents": len(validation_parents),
        "validation_prompts": len(validation_prompts),
        "test_samples": len(test_samples),
        "test_parents": len(test_parents),
        "test_prompts": len(test_prompts),
        "blockers": blockers,
        "target_pass_estimate": size_budget["target_pass_estimate"],
    }
    print(json.dumps(summary, indent=2))
    if args.require_independent and not independent_available:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
