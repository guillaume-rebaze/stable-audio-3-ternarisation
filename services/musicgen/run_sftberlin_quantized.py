"""Run the Stable Audio 3 MLX CLI with optional model quantization.

The upstream runtime loads the ARC DiT and merges the SFT LoRA before this
wrapper quantizes it.  That order is intentional: the experiment measures a
quantized *SFT model*, not a quantized base model with a separately dequantized
adapter.  The persistent saver can also quantize other MLX components
independently.

This is an experiment wrapper, not a replacement for the production runtime.
It keeps quantized weights in memory for the process and does not modify the
upstream runtime or any source/model files.  It can also load a persisted
quantized Medium DiT written by ``save_sftberlin_quantized.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json
import math
import sys
from typing import Any

import numpy as np
import soundfile as sf


VALID_BITS = frozenset({1, 2, 4, 8})
DEFAULT_GROUP_SIZE = 64


def validate_quantization(bits: int | None, group_size: int = DEFAULT_GROUP_SIZE) -> tuple[int | None, int]:
    """Validate the intentionally small experiment surface."""

    if bits is not None:
        bits = int(bits)
        if bits not in VALID_BITS:
            allowed = ", ".join(str(value) for value in sorted(VALID_BITS))
            raise ValueError(f"quantization bits must be one of {allowed}, got {bits}")
    group_size = int(group_size)
    if group_size <= 0 or group_size % 32 != 0:
        raise ValueError("quantization group_size must be a positive multiple of 32")
    return bits, group_size


def _load_runtime(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        "_stable_audio_3_mlx_sftberlin_quantized_runtime", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Stable Audio 3 MLX runtime: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INT1_CLASS_CACHE: tuple[Any, Any] | None = None


def _int1_classes() -> tuple[Any, Any]:
    """Return the packed-binary MLX layers used for the educational INT1 path."""

    global _INT1_CLASS_CACHE
    if _INT1_CLASS_CACHE is not None:
        return _INT1_CLASS_CACHE

    import mlx.core as mx
    import mlx.nn as nn

    shifts = mx.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=mx.uint8)
    pack_shifts = mx.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=mx.uint8)

    def pack_weight(weight: Any, group_size: int) -> tuple[Any, Any]:
        output_dims, input_dims = (int(weight.shape[0]), int(weight.shape[1]))
        grouped = weight.reshape(output_dims, input_dims // group_size, group_size)
        scales = mx.mean(mx.abs(grouped), axis=-1, keepdims=True).astype(weight.dtype)
        sign_bits = (grouped >= 0).astype(mx.uint8)
        sign_bits = sign_bits.reshape(output_dims, input_dims // group_size, group_size // 8, 8)
        packed = mx.sum(sign_bits * pack_shifts, axis=-1).astype(mx.uint8)
        return packed, scales

    def unpack_weight(packed: Any, scales: Any, group_size: int, dtype: Any) -> Any:
        bits = mx.bitwise_and(
            mx.right_shift(packed[..., None], shifts),
            mx.array(1, dtype=mx.uint8),
        )
        signs = bits.astype(dtype) * 2.0 - 1.0
        grouped = signs.reshape(signs.shape[0], signs.shape[1], -1) * scales.astype(dtype)
        return grouped.reshape(grouped.shape[0], -1)

    class PackedBinaryLinear(nn.Module):
        def __init__(self, input_dims: int, output_dims: int, bias: bool, group_size: int):
            super().__init__()
            if input_dims % group_size != 0:
                raise ValueError("INT1 Linear input dimension must be divisible by group_size")
            self.group_size = int(group_size)
            self.input_dims = int(input_dims)
            self.output_dims = int(output_dims)
            self.packed_weight = mx.zeros(
                (output_dims, input_dims // group_size, group_size // 8), dtype=mx.uint8
            )
            self.scales = mx.ones((output_dims, input_dims // group_size, 1), dtype=mx.float16)
            if bias:
                self.bias = mx.zeros((output_dims,), dtype=mx.float16)

        @classmethod
        def from_linear(cls, layer: Any, group_size: int):
            input_dims = int(layer.weight.shape[-1])
            output_dims = int(layer.weight.shape[0])
            result = cls(input_dims, output_dims, "bias" in layer, group_size)
            result.packed_weight, result.scales = pack_weight(layer.weight, group_size)
            if "bias" in layer:
                result.bias = layer.bias
            return result

        def __call__(self, x: Any) -> Any:
            weight = unpack_weight(self.packed_weight, self.scales, self.group_size, x.dtype)
            result = x @ weight.T
            if "bias" in self:
                result = result + self.bias.astype(result.dtype)
            return result

    class PackedBinaryEmbedding(nn.Module):
        def __init__(self, num_embeddings: int, dims: int, group_size: int):
            super().__init__()
            if dims % group_size != 0:
                raise ValueError("INT1 Embedding dimension must be divisible by group_size")
            self.group_size = int(group_size)
            self.num_embeddings = int(num_embeddings)
            self.dims = int(dims)
            self.packed_weight = mx.zeros(
                (num_embeddings, dims // group_size, group_size // 8), dtype=mx.uint8
            )
            self.scales = mx.ones((num_embeddings, dims // group_size, 1), dtype=mx.float16)

        @classmethod
        def from_embedding(cls, layer: Any, group_size: int):
            num_embeddings, dims = (int(layer.weight.shape[0]), int(layer.weight.shape[1]))
            result = cls(num_embeddings, dims, group_size)
            result.packed_weight, result.scales = pack_weight(layer.weight, group_size)
            return result

        def __call__(self, x: Any) -> Any:
            weight = unpack_weight(
                self.packed_weight, self.scales, self.group_size, self.scales.dtype
            )
            return weight[x]

    _INT1_CLASS_CACHE = (PackedBinaryLinear, PackedBinaryEmbedding)
    return _INT1_CLASS_CACHE


def _quantize_int1_model(
    model: Any,
    *,
    group_size: int,
    include_embeddings: bool,
    scope: str,
    notes: list[str] | None,
) -> dict[str, Any]:
    """Replace supported leaves with packed sign-weight educational INT1 layers."""

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_map_with_path

    binary_linear, binary_embedding = _int1_classes()
    quantized_layers: list[dict[str, Any]] = []
    skipped_layers: list[dict[str, Any]] = []

    def convert(path: str, layer: Any) -> Any:
        if isinstance(layer, nn.Linear):
            layer_kind = "Linear"
        elif include_embeddings and isinstance(layer, nn.Embedding):
            layer_kind = "Embedding"
        else:
            return layer
        shape = tuple(int(value) for value in layer.weight.shape)
        item = {
            "path": str(path),
            "kind": layer_kind,
            "shape": list(shape),
            "weight_parameters": int(np.prod(shape)),
            "implementation": "packed_binary_sign",
        }
        if shape[-1] % group_size != 0:
            item["reason"] = "input/hidden dimension is not divisible by group_size"
            skipped_layers.append(item)
            return layer
        quantized_layers.append(item)
        if layer_kind == "Linear":
            return binary_linear.from_linear(layer, group_size)
        return binary_embedding.from_embedding(layer, group_size)

    leaves = tree_map_with_path(convert, model.leaf_modules(), is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    mx.eval(model.parameters())
    quantized_parameters = sum(item["weight_parameters"] for item in quantized_layers)
    skipped_parameters = sum(item["weight_parameters"] for item in skipped_layers)
    total_supported_parameters = quantized_parameters + skipped_parameters
    return {
        "bits": 1,
        "group_size": group_size,
        "mode": "packed_binary_sign",
        "scope": scope,
        "quantized_layer_count": len(quantized_layers),
        "skipped_layer_count": len(skipped_layers),
        "quantized_weight_parameters": quantized_parameters,
        "skipped_weight_parameters": skipped_parameters,
        "quantized_weight_fraction": (
            quantized_parameters / total_supported_parameters
            if total_supported_parameters
            else 0.0
        ),
        "quantized_layers": quantized_layers,
        "skipped_layers": skipped_layers,
        "notes": list(notes or []) + [
            "INT1 uses packed sign weights with one fp16 scale per group; it is an educational custom path.",
            "INT1 is not the native MLX affine quantization kernel.",
        ],
    }


def quantize_supported_layers(
    model: Any,
    *,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    include_embeddings: bool = False,
    scope: str,
    notes: list[str] | None = None,
    allowed_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Quantize compatible MLX Linear/Embedding layers in-place.

    Affine weight quantization requires the input/hidden dimension to be
    divisible by the group size. Other layer classes (notably Conv1d) remain
    native. ``bits=1`` and ``bits=2`` are intentionally exposed for the local
    MLX build used by this project; portability should be checked separately.
    """

    bits, group_size = validate_quantization(bits, group_size)
    if bits is None:
        raise ValueError("quantization requires an explicit bit depth")

    if bits == 1:
        return _quantize_int1_model(
            model,
            group_size=group_size,
            include_embeddings=include_embeddings,
            scope=scope,
            notes=notes,
        )

    import mlx.core as mx
    import mlx.nn as nn

    quantized_layers: list[dict[str, Any]] = []
    skipped_layers: list[dict[str, Any]] = []

    def predicate(path: str, layer: Any) -> bool:
        if allowed_paths is not None and str(path) not in allowed_paths:
            return False
        if isinstance(layer, nn.Linear):
            layer_kind = "Linear"
        elif include_embeddings and isinstance(layer, nn.Embedding):
            layer_kind = "Embedding"
        else:
            return False
        shape = tuple(int(value) for value in layer.weight.shape)
        weight_parameters = int(np.prod(shape))
        item = {
            "path": str(path),
            "kind": layer_kind,
            "shape": list(shape),
            "weight_parameters": weight_parameters,
        }
        if shape[-1] % group_size != 0:
            skipped_layers.append(item)
            return False
        quantized_layers.append(item)
        return True

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode="affine",
        class_predicate=predicate,
    )
    mx.eval(model.parameters())

    quantized_parameters = sum(item["weight_parameters"] for item in quantized_layers)
    skipped_parameters = sum(item["weight_parameters"] for item in skipped_layers)
    total_supported_parameters = quantized_parameters + skipped_parameters
    return {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
        "scope": scope,
        "quantized_layer_count": len(quantized_layers),
        "skipped_layer_count": len(skipped_layers),
        "quantized_weight_parameters": quantized_parameters,
        "skipped_weight_parameters": skipped_parameters,
        "quantized_weight_fraction": (
            quantized_parameters / total_supported_parameters
            if total_supported_parameters
            else 0.0
        ),
        "quantized_layers": quantized_layers,
        "skipped_layers": skipped_layers,
        "notes": list(notes or []),
    }


def quantize_dit(
    model: Any,
    *,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    allowed_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Quantize compatible DiT Linear layers in-place and return an audit report."""

    return quantize_supported_layers(
        model,
        bits=bits,
        group_size=group_size,
        include_embeddings=False,
        scope="dit_linear_weights_after_lora_merge",
        allowed_paths=allowed_paths,
        notes=[
            "T5Gemma and SAME-L remain separate components in this DiT artifact.",
            "DiT Conv1d layers remain native.",
        ],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_quantized_metadata(weights_path: Path) -> dict[str, Any]:
    metadata_path = weights_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"persisted quantized weights require the sidecar manifest: {metadata_path}"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "onus.sftberlin.quantized-dit/v1":
        raise ValueError(f"unsupported quantized DiT manifest schema in {metadata_path}")
    return payload


def load_persisted_quantized_dit(
    runtime: Any,
    *,
    weights_path: Path,
    dit_name: str,
    T_lat: int,
    dtype: Any,
) -> tuple[Any, str, dict[str, Any]]:
    """Instantiate the matching quantized module and load a saved MLX archive."""

    if dit_name != "medium":
        raise ValueError("persisted sftBerlin quantized weights currently support --dit medium only")
    metadata = _read_quantized_metadata(weights_path)
    model_meta = metadata.get("model", {})
    if model_meta.get("dit") != dit_name:
        raise ValueError(
            f"quantized weights were saved for DiT {model_meta.get('dit')!r}, not {dit_name!r}"
        )
    quant_meta = metadata.get("quantization", {})
    bits, group_size = validate_quantization(
        quant_meta.get("bits"), quant_meta.get("group_size", DEFAULT_GROUP_SIZE)
    )
    if bits is None:
        raise ValueError(f"quantized manifest has no bit depth: {weights_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"persisted quantized weights do not exist: {weights_path}")

    import importlib
    import mlx.core as mx

    module_name = runtime.DIT_CHOICES[dit_name]["loader"]
    module = importlib.import_module(module_name)
    model = module.DiT(T_lat=int(T_lat))
    # Quantization must be applied before loading because the archive contains
    # QuantizedLinear q_weight/scales/bias parameters rather than Linear.weight.
    manifest_layers = quant_meta.get("quantized_layers")
    allowed_paths = {l["path"] for l in manifest_layers} if manifest_layers else None
    quantize_dit(model, bits=bits, group_size=group_size, allowed_paths=allowed_paths)
    model.load_weights(str(weights_path), strict=True)
    mx.eval(model.parameters())
    return model, str(weights_path), metadata


def _read_quantized_component_metadata(weights_path: Path, component: str) -> dict[str, Any]:
    metadata_path = weights_path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"persisted quantized weights require the sidecar manifest: {metadata_path}"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "onus.sftberlin.quantized-component/v1":
        raise ValueError(f"unsupported quantized component manifest schema in {metadata_path}")
    if payload.get("model", {}).get("component") != component:
        raise ValueError(
            f"quantized weights were saved for component "
            f"{payload.get('model', {}).get('component')!r}, not {component!r}"
        )
    if not weights_path.is_file():
        raise FileNotFoundError(f"persisted quantized weights do not exist: {weights_path}")
    return payload


def _component_quantization(metadata: dict[str, Any], weights_path: Path) -> tuple[int, int]:
    quant_meta = metadata.get("quantization", {})
    bits, group_size = validate_quantization(
        quant_meta.get("bits"), quant_meta.get("group_size", DEFAULT_GROUP_SIZE)
    )
    if bits is None:
        raise ValueError(f"quantized manifest has no bit depth: {weights_path}")
    return int(bits), int(group_size)


def load_persisted_quantized_t5(
    *,
    weights_path: Path,
) -> tuple[Any, dict[str, Any]]:
    """Load a complete quantized T5 archive, including its tokenizer blobs."""

    metadata = _read_quantized_component_metadata(weights_path, "t5")
    bits, group_size = _component_quantization(metadata, weights_path)

    import importlib
    import sentencepiece as spm
    import mlx.core as mx
    import numpy as np

    module = importlib.import_module("models.defs.t5gemma_mlx")
    with np.load(str(weights_path), allow_pickle=False) as arrs:
        if "META" not in arrs.files or "TOKENIZER_MODEL" not in arrs.files:
            raise ValueError(f"quantized T5 archive is missing META/TOKENIZER_MODEL: {weights_path}")
        cfg = module.T5GemmaConfig.from_json_bytes(np.asarray(arrs["META"]).tobytes())
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.LoadFromSerializedProto(np.asarray(arrs["TOKENIZER_MODEL"]).tobytes())

    encoder = module._Encoder(cfg)
    quantize_supported_layers(
        encoder,
        bits=bits,
        group_size=group_size,
        include_embeddings=True,
        scope="t5gemma_encoder_linear_and_embedding_weights",
    )
    raw = dict(mx.load(str(weights_path)))
    raw.pop("META", None)
    raw.pop("TOKENIZER_MODEL", None)
    if "rope_inv_freq" in raw:
        encoder.rope_inv_freq = raw["rope_inv_freq"]
    encoder.load_weights(list(raw.items()), strict=True)
    mx.eval(encoder.parameters())
    return module.T5Gemma(encoder, cfg, tokenizer), metadata


def load_persisted_quantized_codec(
    runtime: Any,
    *,
    component: str,
    weights_path: Path,
) -> tuple[Any, Any, tuple[int, int]]:
    """Load a persisted quantized SAME-L decoder or encoder skeleton."""

    if component not in {"same-l-decoder", "same-l-encoder"}:
        raise ValueError(f"unsupported quantized codec component: {component}")
    metadata = _read_quantized_component_metadata(weights_path, component)
    bits, group_size = _component_quantization(metadata, weights_path)
    import importlib
    import mlx.core as mx

    if component == "same-l-decoder":
        module_name = runtime.DECODER_CHOICES["same-l"][0]
        module = importlib.import_module(module_name)
        model = module.SAMELDecoder()
        chunk_fn = getattr(module, runtime.DECODER_CHOICES["same-l"][1])
        chunk_cfg = runtime.DECODER_CHOICES["same-l"][2]
    else:
        module_name = runtime.ENCODER_CHOICES["same-l"][0]
        module = importlib.import_module(module_name)
        model = module.SAMELEncoder()
        chunk_fn = None
        chunk_cfg = (runtime.ENCODER_CHOICES["same-l"][1],)

    quantize_supported_layers(
        model,
        bits=bits,
        group_size=group_size,
        include_embeddings=False,
        scope=metadata["quantization"].get("scope", f"{component}_linear_weights"),
    )
    model.load_weights(str(weights_path), strict=True)
    mx.eval(model.parameters())
    return model, (chunk_fn, chunk_cfg), metadata


def _write_lossless_float_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Keep the decoded render lossless until the caller's delivery stage."""

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 2 or not np.isfinite(samples).all():
        raise ValueError("decoded audio must be finite and shaped (channels, frames)")
    sf.write(str(path), samples.T, int(sample_rate), subtype="FLOAT", format="WAV")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--runtime-script", required=True, type=Path)
    parser.add_argument("--output-headroom-db", default=-4.0, type=float)
    parser.add_argument("--quant-bits", type=int, default=None, choices=sorted(VALID_BITS))
    parser.add_argument("--quant-group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--quantization-report", type=Path, default=None)
    parser.add_argument("--quantized-dit", type=Path, default=None,
                        help="Load a persisted quantized Medium DiT .npz with its JSON sidecar.")
    parser.add_argument("--quantized-t5", type=Path, default=None,
                        help="Load a persisted quantized T5Gemma archive with its JSON sidecar.")
    parser.add_argument("--quantized-decoder", type=Path, default=None,
                        help="Load a persisted quantized SAME-L decoder with its JSON sidecar.")
    parser.add_argument("--quantized-encoder", type=Path, default=None,
                        help="Load a persisted quantized SAME-L encoder for --init-audio.")
    wrapper_args, runtime_args = parser.parse_known_args()

    try:
        bits, group_size = validate_quantization(wrapper_args.quant_bits, wrapper_args.quant_group_size)
    except ValueError as exc:
        parser.error(str(exc))

    headroom_db = float(wrapper_args.output_headroom_db)
    if not math.isfinite(headroom_db) or headroom_db > 0.0:
        parser.error("--output-headroom-db must be a finite value <= 0 dB")
    runtime_script = wrapper_args.runtime_script.expanduser().resolve()
    if not runtime_script.is_file():
        parser.error(f"Stable Audio 3 MLX runtime script does not exist: {runtime_script}")
    if wrapper_args.quantized_dit is not None and bits is not None:
        parser.error("use either --quant-bits or --quantized-dit, not both")

    runtime = _load_runtime(runtime_script)
    quantization_report: dict[str, Any] | None = None
    persisted_quantized_dit = (
        wrapper_args.quantized_dit.expanduser().resolve()
        if wrapper_args.quantized_dit is not None
        else None
    )
    persisted_quantized_t5 = (
        wrapper_args.quantized_t5.expanduser().resolve()
        if wrapper_args.quantized_t5 is not None
        else None
    )
    persisted_quantized_decoder = (
        wrapper_args.quantized_decoder.expanduser().resolve()
        if wrapper_args.quantized_decoder is not None
        else None
    )
    persisted_quantized_encoder = (
        wrapper_args.quantized_encoder.expanduser().resolve()
        if wrapper_args.quantized_encoder is not None
        else None
    )

    if persisted_quantized_t5 is not None:
        def load_t5_persisted(cls, _path):
            model, metadata = load_persisted_quantized_t5(
                weights_path=persisted_quantized_t5,
            )
            quant_meta = metadata["quantization"]
            print(
                "quantization: loaded persisted T5Gemma "
                f"INT{quant_meta['bits']}, group_size={quant_meta['group_size']}, "
                f"{quant_meta['quantized_layer_count']} compatible layers",
                file=sys.stderr,
            )
            return model

        runtime.T5Gemma.from_npz = classmethod(load_t5_persisted)

    if persisted_quantized_decoder is not None:
        if "--decoder-weights" in runtime_args:
            parser.error("use either --decoder-weights or --quantized-decoder, not both")

        def load_decoder_persisted(decoder_name: str, dtype, weights_path=None):
            if decoder_name != "same-l":
                raise ValueError("persisted quantized decoder currently supports --decoder same-l only")
            model, (chunk_fn, chunk_cfg), metadata = load_persisted_quantized_codec(
                runtime,
                component="same-l-decoder",
                weights_path=persisted_quantized_decoder,
            )
            quant_meta = metadata["quantization"]
            print(
                "quantization: loaded persisted SAME-L decoder "
                f"INT{quant_meta['bits']}, group_size={quant_meta['group_size']}, "
                f"{quant_meta['quantized_layer_count']} compatible layers",
                file=sys.stderr,
            )
            return model, chunk_fn, chunk_cfg

        runtime.load_decoder = load_decoder_persisted

    if persisted_quantized_encoder is not None:
        def load_encoder_persisted(decoder_name: str, dtype):
            if decoder_name != "same-l":
                raise ValueError("persisted quantized encoder currently supports --decoder same-l only")
            model, _unused, metadata = load_persisted_quantized_codec(
                runtime,
                component="same-l-encoder",
                weights_path=persisted_quantized_encoder,
            )
            quant_meta = metadata["quantization"]
            print(
                "quantization: loaded persisted SAME-L encoder "
                f"INT{quant_meta['bits']}, group_size={quant_meta['group_size']}, "
                f"{quant_meta['quantized_layer_count']} compatible layers",
                file=sys.stderr,
            )
            return model, runtime.ENCODER_CHOICES["same-l"][1]

        runtime.load_encoder = load_encoder_persisted

    if persisted_quantized_dit is not None:
        if "--lora" in runtime_args:
            parser.error("--quantized-dit already contains the merged LoRA; omit --lora")

        def load_dit_persisted(dit_name: str, T_lat: int, dtype, lora_specs=None, num_steps=None):
            nonlocal quantization_report
            if lora_specs:
                raise ValueError("--quantized-dit already contains the merged LoRA; omit --lora")
            model, checkpoint, metadata = load_persisted_quantized_dit(
                runtime,
                weights_path=persisted_quantized_dit,
                dit_name=dit_name,
                T_lat=T_lat,
                dtype=dtype,
            )
            quantization_report = dict(metadata["quantization"])
            print(
                "quantization: loaded persisted "
                f"INT{quantization_report['bits']}, "
                f"group_size={quantization_report['group_size']}, "
                f"{quantization_report['quantized_layer_count']} Linear layers",
                file=sys.stderr,
            )
            return model, checkpoint

        runtime.load_dit = load_dit_persisted
    elif bits is not None:
        original_load_dit = runtime.load_dit

        def load_dit_quantized(dit_name: str, T_lat: int, dtype, lora_specs=None, num_steps=None):
            nonlocal quantization_report
            if lora_specs and any(spec.get("steps") is not None for spec in lora_specs):
                raise ValueError(
                    "step-gated LoRA is not supported by the in-memory quantization experiment; "
                    "use a full-range --lora adapter"
                )
            model, checkpoint = original_load_dit(
                dit_name,
                T_lat,
                dtype,
                lora_specs=lora_specs,
                num_steps=num_steps,
            )
            quantization_report = quantize_dit(model, bits=bits, group_size=group_size)
            quantization_report["dit"] = str(dit_name)
            quantization_report["checkpoint"] = str(checkpoint)
            print(
                "quantization: "
                f"INT{bits}, group_size={group_size}, "
                f"{quantization_report['quantized_layer_count']} Linear layers, "
                f"{quantization_report['quantized_weight_fraction']:.2%} of Linear weights",
                file=sys.stderr,
            )
            return model, checkpoint

        runtime.load_dit = load_dit_quantized

    def save_wav_with_headroom(path, audio, sample_rate=runtime.SAMPLE_RATE):
        audio = np.asarray(audio, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        peak_ceiling = 10.0 ** (headroom_db / 20.0)
        if peak > peak_ceiling:
            audio = audio * (peak_ceiling / peak)
        _write_lossless_float_wav(path, audio, sample_rate)

    runtime.save_wav = save_wav_with_headroom
    sys.argv = [str(runtime_script), *runtime_args]
    result = runtime.main()
    if quantization_report is not None and wrapper_args.quantization_report is not None:
        _write_json(wrapper_args.quantization_report.expanduser().resolve(), quantization_report)
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
