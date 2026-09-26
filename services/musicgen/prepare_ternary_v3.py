"""Prepare auditable G0/G2 inputs for the ternary quality v3 lane.

This command does not train and does not fabricate a held-out split.  The
current latent bank is explicitly labelled ``debug_train_seen`` until an
independent source bank is supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np

from ternary_contract import scope_digest


CORE_NAMES = (
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_kv",
    "cross_attn.to_out",
    "ff.ff.0.proj",
    "ff.ff.2",
)


def read_npy_header(info: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], np.dtype]:
    with info.open(name, "r") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported npy header version {version} for {name}")
    del fortran_order
    return tuple(int(x) for x in shape), np.dtype(dtype)


def array_headers(weights: Path) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    result: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    with zipfile.ZipFile(weights) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            result[name[:-4]] = read_npy_header(archive, name)
    return result


def is_core_weight(key: str) -> bool:
    return key.startswith("transformer.layers.") and any(
        key.endswith(f".{name}.weight") for name in CORE_NAMES
    )


def scope_from_headers(headers: dict[str, tuple[tuple[int, ...], np.dtype]]) -> list[str]:
    return sorted(key for key in headers if is_core_weight(key))


def ternary_storage_bytes(shape: tuple[int, ...], group_size: int) -> int:
    out_dim, in_dim = shape
    if in_dim % group_size:
        raise ValueError(f"{shape=} is not divisible by {group_size=}")
    packed_codes = out_dim * (in_dim // 16) * 4
    metadata = out_dim * (in_dim // group_size) * 2 * 2
    return packed_codes + metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare auditable ternary v3 manifests")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("output/sample-expertise-pilot/universal-dataset/latents-12s"),
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=Path.home() / ".cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/sample-expertise-pilot/ternary-quality-v3/g0-g2"),
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--crop-len", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=24)
    args = parser.parse_args()

    if args.group_size % 16:
        raise ValueError("group size must be divisible by 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.dataset_dir.glob("*.npy"))
    samples: list[dict] = []
    prompt_counter: Counter[str] = Counter()
    genre_counter: Counter[str] = Counter()
    for path in files:
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        shape = tuple(int(x) for x in np.load(path, mmap_mode="r").shape)
        if len(shape) != 2 or shape[0] != 256:
            raise ValueError(f"unexpected latent shape: {path} -> {shape}")
        prompt = str(meta.get("prompt", ""))
        genre = str(meta.get("genre", "unknown"))
        prompt_counter[prompt] += 1
        genre_counter[genre] += 1
        samples.append(
            {
                "file": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": list(shape),
                "prompt": prompt,
                "genre": genre,
                "source": meta.get("src_relpath", meta.get("path", "")),
                "role": "debug_train_seen",
            }
        )

    headers = array_headers(args.teacher_weights)
    scope = scope_from_headers(headers)
    expected_scope_count = args.blocks * len(CORE_NAMES)
    if len(scope) != expected_scope_count:
        raise ValueError(
            f"core scope mismatch: found={len(scope)} expected={expected_scope_count}"
        )

    full_dense_bytes = sum(
        int(np.prod(shape)) * dtype.itemsize for shape, dtype in headers.values()
    )
    core_dense_bytes = sum(
        int(np.prod(headers[key][0])) * headers[key][1].itemsize for key in scope
    )
    core_ternary_bytes = sum(
        ternary_storage_bytes(headers[key][0], args.group_size) for key in scope
    )
    estimated_export_bytes = full_dense_bytes - core_dense_bytes + core_ternary_bytes

    inventory = {
        "schema": "onus.ternary-quality-v3/inventory",
        "status": "prepared",
        "created_at": __import__("datetime").datetime.now().astimezone().isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "data": {
            "directory": str(args.dataset_dir),
            "sample_count": len(samples),
            "prompt_count": len(prompt_counter),
            "prompts": dict(prompt_counter),
            "genres": dict(genre_counter),
            "latent_shapes": sorted({tuple(s["shape"]) for s in samples}),
            "split_contract": {
                "all_roles": ["debug_train_seen"],
                "heldout_available": False,
                "reason": "No independent source bank is present in the configured dataset directory.",
            },
        },
        "teacher": {
            "weights": str(args.teacher_weights),
            "file_bytes": args.teacher_weights.stat().st_size,
            "array_count": len(headers),
        },
        "ternary_scope": {
            "group_size": args.group_size,
            "crop_len": args.crop_len,
            "paths": scope,
            "count": len(scope),
            "digest": scope_digest(scope),
            "quantizer": "s_q",
            "runtime_encoding": "MLX affine fields with scales=-s and biases=s; biases are derived metadata",
        },
        "size_budget": {
            "full_dense_array_bytes": full_dense_bytes,
            "core_dense_array_bytes": core_dense_bytes,
            "core_ternary_array_bytes": core_ternary_bytes,
            "estimated_export_array_bytes": estimated_export_bytes,
            "estimated_reduction_vs_dense": 1.0 - estimated_export_bytes / full_dense_bytes,
            "target_under_512_mib": estimated_export_bytes < 512 * 1024**2,
        },
        "samples": samples,
    }
    (args.output_dir / "inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "schema": "onus.ternary-quality-v3/split",
                "status": "debug_only",
                "heldout": False,
                "roles": {"debug_train_seen": [s["file"] for s in samples]},
                "blocked_release_claim": "Acquire independent source material before validation/test claims.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": inventory["status"],
        "samples": len(samples),
        "prompts": len(prompt_counter),
        "scope_count": len(scope),
        "scope_digest": scope_digest(scope),
        "estimated_export_bytes": estimated_export_bytes,
        "estimated_export_mib": estimated_export_bytes / 1024**2,
        "target_under_512_mib": inventory["size_budget"]["target_under_512_mib"],
        "heldout_available": False,
    }, indent=2))


if __name__ == "__main__":
    main()
