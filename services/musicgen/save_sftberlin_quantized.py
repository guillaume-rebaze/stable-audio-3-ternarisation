"""Persist sftBerlin DiT and compatible Stable Audio 3 components as MLX models.

This script intentionally saves one quantization variant per process.  It loads
the ARC Medium DiT, merges the full-range SFT LoRA, quantizes compatible Linear
weights, and writes the resulting MLX ``.npz`` weights plus a provenance
sidecar.  The same one-process-at-a-time path can save INT1/INT2 T5Gemma and
SAME-L component artifacts for educational testing.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
from pathlib import Path
import time
from typing import Any
import zipfile

from run_sftberlin_quantized import (
    DEFAULT_GROUP_SIZE,
    VALID_BITS,
    _load_runtime,
    quantize_dit,
    quantize_supported_layers,
    validate_quantization,
)


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent.parent
DEFAULT_RUNTIME = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
DEFAULT_ADAPTER = (
    REPOSITORY_ROOT
    / "output/sample-expertise-pilot/sftberlin/sft-runs"
    / "sftberlin-medium-lora-r8-spectral-body-safe-temporal-loss010-1e-4-20260910"
    / "4592172e/checkpoints/sftberlin-medium-lora-r8-spectral-body-safe-temporal-loss010-1e-4-20260910-step=50-epoch=0.safetensors"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "output/sample-expertise-pilot/sftberlin/quantized-models"
DEFAULT_T_LAT = 320
DEFAULT_DTYPE = "fp16"
COMPONENTS = ("dit", "t5", "same-l-decoder", "same-l-encoder")
COMPONENT_SPECS: dict[str, dict[str, Any]] = {
    "t5": {
        "source_rel": "models/mlx/t5gemma_f16.npz",
        "module": "models.defs.t5gemma_mlx",
        "scope": "t5gemma_encoder_linear_and_embedding_weights",
        "include_embeddings": True,
        "stem": "t5gemma_sftberlin",
        "notes": [
            "T5Gemma encoder Linear and Embedding weights are quantized.",
            "META and TOKENIZER_MODEL are copied into the archive for runtime reuse.",
            "The text encoder is not LoRA-adapted by this artifact.",
        ],
    },
    "same-l-decoder": {
        "source_rel": "models/mlx/same_l_decoder_f32.npz",
        "module": "models.defs.same_l_decoder",
        "scope": "same_l_decoder_linear_weights",
        "include_embeddings": False,
        "stem": "same_l_decoder_sftberlin",
        "notes": [
            "SAME-L decoder Transformer Linear weights are quantized.",
            "The final Conv1d mapping remains native.",
            "This codec component is shared by the Medium runtime and is not LoRA-adapted.",
        ],
    },
    "same-l-encoder": {
        "source_rel": "models/mlx/same_l_encoder_f32.npz",
        "module": "models.defs.same_l_encoder",
        "scope": "same_l_encoder_linear_weights",
        "include_embeddings": False,
        "stem": "same_l_encoder_sftberlin",
        "notes": [
            "SAME-L encoder Transformer and projection Linear weights are quantized.",
            "This input-audio component is only used with --init-audio.",
            "The encoder is not LoRA-adapted by this artifact.",
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_model_weights(model: Any, destination: Path, stem: str) -> Path:
    """Save an MLX archive atomically, keeping only one model in memory."""

    temporary = destination / f".{stem}.tmp.npz"
    if temporary.exists():
        temporary.unlink()
    model.save_weights(str(temporary))
    weights_path = destination / f"{stem}.npz"
    temporary.replace(weights_path)
    return weights_path


def _append_t5_runtime_blobs(weights_path: Path, source_path: Path) -> None:
    """Add T5Gemma config/tokenizer entries without copying model tensors."""

    with zipfile.ZipFile(weights_path, mode="a", allowZip64=True) as target:
        with zipfile.ZipFile(source_path, mode="r", allowZip64=True) as source:
            for name in ("META.npy", "TOKENIZER_MODEL.npy"):
                info = source.getinfo(name)
                target.writestr(info, source.read(name))


def _load_component_model(runtime: Any, component: str, source_path: Path):
    """Load exactly one non-DiT component at fp16 to cap save-process RAM."""

    import mlx.core as mx

    spec = COMPONENT_SPECS[component]
    if component == "t5":
        module = importlib.import_module(spec["module"])
        wrapper = module.T5Gemma.from_npz(str(source_path))
        return wrapper.encoder

    module = importlib.import_module(spec["module"])
    # The runtime decodes SAME-L in fp32, but quantization is applied to the
    # stored weights and the smaller fp16 load is enough for this save-only job.
    return module.load_model(weights_path=str(source_path), dtype=mx.float16, compile_=False)


def _load_merged_medium_dit(runtime: Any, *, adapter: Path, lora_strength: float, t_lat: int):
    """Load the base DiT and merge the adapter before quantization.

    Calling the Medium loader directly lets this process avoid loading T5Gemma,
    SAME-L, or any audio input.  That keeps the save-only peak substantially
    lower than an inference run and makes the LoRA merge unambiguous.
    """

    import mlx.core as mx

    config = runtime.DIT_CHOICES["medium"]
    checkpoint = runtime.ensure_local(config["ckpt"])
    module = importlib.import_module(config["loader"])
    model = module.load_dit(
        str(checkpoint),
        T_lat=int(t_lat),
        dtype=mx.float16,
        compile_=False,
        lora_paths=[str(adapter)],
        lora_strength=float(lora_strength),
        lora_log=lambda message: print(message),
    )
    return model, Path(checkpoint).resolve()


def save_quantized_model(
    *,
    bits: int,
    runtime_path: str | Path = DEFAULT_RUNTIME,
    adapter_path: str | Path = DEFAULT_ADAPTER,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    group_size: int = DEFAULT_GROUP_SIZE,
    lora_strength: float = 0.25,
    t_lat: int = DEFAULT_T_LAT,
    force: bool = False,
) -> dict[str, Any]:
    """Save one persistent quantized Medium DiT and return its manifest."""

    bits, group_size = validate_quantization(bits, group_size)
    if bits is None:
        raise ValueError("bits must be one of 1, 2, 4 or 8")
    runtime_root = Path(runtime_path).expanduser().resolve()
    adapter = Path(adapter_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if not adapter.is_file():
        raise FileNotFoundError(f"LoRA adapter does not exist: {adapter}")
    runtime_script = runtime_root / "scripts" / "sa3_mlx.py"
    python_path = runtime_root / ".venv" / "bin" / "python"
    if not runtime_script.is_file() or not python_path.is_file():
        raise FileNotFoundError(f"Stable Audio 3 MLX runtime is incomplete: {runtime_root}")
    if int(t_lat) <= 0:
        raise ValueError("t_lat must be positive")

    destination.mkdir(parents=True, exist_ok=True)
    stem = f"dit_medium_sftberlin_int{bits}_group{group_size}_affine"
    weights_path = destination / f"{stem}.npz"
    manifest_path = destination / f"{stem}.json"
    if weights_path.exists() and not force:
        raise FileExistsError(f"quantized weights already exist: {weights_path}; pass --force to replace")

    runtime = _load_runtime(runtime_script)
    started = time.monotonic()
    print(f"loading merged Medium DiT for INT{bits}…", flush=True)
    model, base_checkpoint = _load_merged_medium_dit(
        runtime,
        adapter=adapter,
        lora_strength=float(lora_strength),
        t_lat=int(t_lat),
    )
    quantization = quantize_dit(model, bits=bits, group_size=group_size)
    quantization["base_checkpoint"] = str(base_checkpoint)
    quantization["adapter"] = str(adapter)
    quantization["lora_strength"] = float(lora_strength)
    print(
        f"saving INT{bits} MLX weights ({quantization['quantized_layer_count']} Linear layers)…",
        flush=True,
    )

    temporary_weights = destination / f".{stem}.tmp.npz"
    if temporary_weights.exists():
        temporary_weights.unlink()
    model.save_weights(str(temporary_weights))
    temporary_weights.replace(weights_path)

    metadata: dict[str, Any] = {
        "schema": "onus.sftberlin.quantized-dit/v1",
        "artifact_type": "mlx_quantized_dit_weights",
        "created_at_unix": time.time(),
        "model": {
            "family": "stable-audio-3",
            "dit": "medium",
            "dtype": "float16",
            "t_lat_at_save": int(t_lat),
            "base_inference_weights": str(base_checkpoint),
            "base_inference_weights_sha256": _sha256(base_checkpoint),
            "adapter": str(adapter),
            "adapter_sha256": _sha256(adapter),
            "lora_strength": float(lora_strength),
            "decoder": "same-l (native runtime; not included)",
            "text_encoder": "T5Gemma (native runtime; not included)",
        },
        "quantization": quantization,
        "serialization": {
            "format": "MLX npz via nn.Module.save_weights",
            "weights": str(weights_path),
            "weights_sha256": _sha256(weights_path),
            "weights_size_bytes": weights_path.stat().st_size,
            "load_requires_same_quantization_layout": True,
        },
        "runtime": {
            "root": str(runtime_root),
            "script": str(runtime_script),
            "python": str(python_path),
            "reuse": (
                "services/musicgen/run_sftberlin_quantized.py "
                f"--quantized-dit {weights_path}"
            ),
        },
        "scope": [
            "LoRA is merged before quantization.",
            "Only compatible Medium DiT Linear weights are quantized.",
            "T5Gemma and SAME-L remain separate components in this DiT artifact.",
            "This artifact is local/private and inherits the adapter/source rights scope.",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(manifest_path, metadata)
    del model
    gc.collect()
    print(json.dumps({
        "bits": bits,
        "weights": str(weights_path),
        "manifest": str(manifest_path),
        "size_bytes": weights_path.stat().st_size,
        "sha256": metadata["serialization"]["weights_sha256"],
        "elapsed_seconds": metadata["elapsed_seconds"],
    }, ensure_ascii=False))
    return metadata


def save_quantized_component(
    *,
    component: str,
    bits: int,
    runtime_path: str | Path = DEFAULT_RUNTIME,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    group_size: int = DEFAULT_GROUP_SIZE,
    force: bool = False,
) -> dict[str, Any]:
    """Save one compatible T5Gemma or SAME-L component per process."""

    if component not in COMPONENT_SPECS:
        raise ValueError(f"unknown component {component!r}; expected {', '.join(COMPONENTS[1:])}")
    bits, group_size = validate_quantization(bits, group_size)
    if bits is None:
        raise ValueError("bits must be one of 1, 2, 4 or 8")

    runtime_root = Path(runtime_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    runtime_script = runtime_root / "scripts" / "sa3_mlx.py"
    python_path = runtime_root / ".venv" / "bin" / "python"
    if not runtime_script.is_file() or not python_path.is_file():
        raise FileNotFoundError(f"Stable Audio 3 MLX runtime is incomplete: {runtime_root}")

    runtime = _load_runtime(runtime_script)
    spec = COMPONENT_SPECS[component]
    # Keep the .npz link name: mlx.load uses the suffix to select its archive
    # loader, while hashing/opening still follows the cache symlink normally.
    source_path = Path(runtime.ensure_local(spec["source_rel"])).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    stem = f"{spec['stem']}_int{bits}_group{group_size}_affine"
    weights_path = destination / f"{stem}.npz"
    manifest_path = destination / f"{stem}.json"
    if weights_path.exists() and not force:
        raise FileExistsError(f"quantized weights already exist: {weights_path}; pass --force to replace")

    started = time.monotonic()
    print(f"loading {component} for INT{bits}…", flush=True)
    model = _load_component_model(runtime, component, source_path)
    import mlx.core as mx

    quantization = quantize_supported_layers(
        model,
        bits=bits,
        group_size=group_size,
        include_embeddings=bool(spec["include_embeddings"]),
        scope=str(spec["scope"]),
        notes=list(spec["notes"]),
    )
    quantization["source_component"] = component
    quantization["mlx_version"] = str(getattr(mx, "__version__", "unknown"))
    print(
        f"saving INT{bits} {component} weights "
        f"({quantization['quantized_layer_count']} compatible layers)…",
        flush=True,
    )
    _save_model_weights(model, destination, stem)
    if component == "t5":
        _append_t5_runtime_blobs(weights_path, source_path)

    flag = {
        "t5": "--quantized-t5",
        "same-l-decoder": "--quantized-decoder",
        "same-l-encoder": "--quantized-encoder",
    }[component]
    metadata: dict[str, Any] = {
        "schema": "onus.sftberlin.quantized-component/v1",
        "artifact_type": "mlx_quantized_component_weights",
        "created_at_unix": time.time(),
        "model": {
            "family": "stable-audio-3",
            "component": component,
            "source_weights": str(source_path),
            "source_weights_sha256": _sha256(source_path),
            "compute_load_dtype": "float16",
            "t5_runtime_blobs_included": component == "t5",
        },
        "quantization": quantization,
        "serialization": {
            "format": "MLX npz via nn.Module.save_weights",
            "weights": str(weights_path),
            "weights_sha256": _sha256(weights_path),
            "weights_size_bytes": weights_path.stat().st_size,
            "load_requires_same_quantization_layout": True,
        },
        "runtime": {
            "root": str(runtime_root),
            "script": str(runtime_script),
            "python": str(python_path),
            "wrapper_flag": flag,
            "reuse": f"services/musicgen/run_sftberlin_quantized.py {flag} {weights_path}",
        },
        "scope": [
            "This is an educational local artifact; the source component remains immutable.",
            "Only compatible Linear/Embedding layers are quantized.",
            "INT1/INT2 support depends on the local MLX build and may not be portable.",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(manifest_path, metadata)
    del model
    gc.collect()
    print(json.dumps({
        "component": component,
        "bits": bits,
        "weights": str(weights_path),
        "manifest": str(manifest_path),
        "size_bytes": weights_path.stat().st_size,
        "sha256": metadata["serialization"]["weights_sha256"],
        "elapsed_seconds": metadata["elapsed_seconds"],
    }, ensure_ascii=False))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=COMPONENTS, default="dit")
    parser.add_argument("--bits", required=True, type=int, choices=sorted(VALID_BITS))
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--lora-strength", type=float, default=0.25)
    parser.add_argument("--t-lat", type=int, default=DEFAULT_T_LAT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.component == "dit":
        save_quantized_model(
            bits=args.bits,
            runtime_path=args.runtime,
            adapter_path=args.adapter,
            output_dir=args.output_dir,
            group_size=args.group_size,
            lora_strength=args.lora_strength,
            t_lat=args.t_lat,
            force=args.force,
        )
    else:
        save_quantized_component(
            component=args.component,
            bits=args.bits,
            runtime_path=args.runtime,
            output_dir=args.output_dir,
            group_size=args.group_size,
            force=args.force,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
