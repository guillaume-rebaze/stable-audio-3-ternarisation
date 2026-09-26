"""Run a matched FP16/INT8/INT4 canary for the private sftBerlin adapter.

The experiment quantizes only the Stable Audio 3 Medium DiT after the SFT
LoRA is merged.  T5Gemma, SAME-L and the source material remain on their
normal paths.  Each variant is an isolated subprocess so MLX can release its
memory before the next model is loaded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping

import numpy as np
import soundfile as sf

from generate_sftberlin_demo import (
    DEFAULT_APG,
    DEFAULT_CFG,
    DEFAULT_INIT_NOISE_LEVEL,
    DEFAULT_LORA_STRENGTH,
    DEFAULT_OUTPUT_HEADROOM_DB,
    DEFAULT_STEPS,
    _analyse_output,
    _convert_input,
    _master_output,
    _read_jsonl,
    _sha256,
    build_demo_jobs,
)
from run_sftberlin_quantized import DEFAULT_GROUP_SIZE, VALID_BITS, validate_quantization


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parent.parent
DEFAULT_RUNTIME = Path.home() / ".cache" / "onus" / "stable-audio-3-mlx" / "optimized" / "mlx"
DEFAULT_ADAPTER = (
    REPOSITORY_ROOT
    / "output/sample-expertise-pilot/sftberlin/sft-runs"
    / "sftberlin-medium-lora-r8-spectral-body-safe-temporal-loss010-1e-4-20260910"
    / "4592172e/checkpoints"
    / "sftberlin-medium-lora-r8-spectral-body-safe-temporal-loss010-1e-4-20260910-step=50-epoch=0.safetensors"
)
DEFAULT_ANNOTATIONS = REPOSITORY_ROOT / "sftberlin" / "annotations.jsonl"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "output/sample-expertise-pilot/sftberlin/quantization-int8-int4-canary"
DEFAULT_TRACKS = (1, 150)
VARIANT_BITS: dict[str, int | None] = {
    "fp16": None,
    "int8": 8,
    "int4": 4,
    "int2": 2,
    "int1": 1,
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rms(samples: np.ndarray) -> float:
    value = float(np.sqrt(np.mean(np.square(np.asarray(samples, dtype=np.float64)))))
    return value if math.isfinite(value) else 0.0


def compare_audio(reference: Path, candidate: Path) -> dict[str, float | None]:
    """Compare raw matched renders without hiding level or phase differences."""

    ref, ref_rate = sf.read(reference, dtype="float32", always_2d=True)
    actual, actual_rate = sf.read(candidate, dtype="float32", always_2d=True)
    if ref_rate != actual_rate or ref.shape[1] != actual.shape[1] or ref.shape[0] != actual.shape[0]:
        return {
            "reference_rate": float(ref_rate),
            "candidate_rate": float(actual_rate),
            "reference_frames": float(ref.shape[0]),
            "candidate_frames": float(actual.shape[0]),
            "diff_rms_db_relative": None,
            "correlation": None,
            "rms_delta_db": None,
        }
    ref_flat = ref.reshape(-1).astype(np.float64)
    actual_flat = actual.reshape(-1).astype(np.float64)
    reference_rms = max(_rms(ref_flat), 1e-12)
    diff_rms = _rms(actual_flat - ref_flat)
    correlation = float(np.corrcoef(ref_flat, actual_flat)[0, 1]) if ref_flat.size > 1 else None
    candidate_rms = max(_rms(actual_flat), 1e-12)
    return {
        "reference_rate": float(ref_rate),
        "candidate_rate": float(actual_rate),
        "reference_frames": float(ref.shape[0]),
        "candidate_frames": float(actual.shape[0]),
        "diff_rms_db_relative": 20.0 * math.log10(max(diff_rms, 1e-12) / reference_rms),
        "correlation": correlation,
        "rms_delta_db": 20.0 * math.log10(candidate_rms / reference_rms),
    }


def _runtime_observations(log_text: str) -> dict[str, Any]:
    sample_match = re.search(r"sample\s+([0-9.]+)\s+ms", log_text)
    step_match = re.search(r"sample\s+[0-9.]+\s+ms\s+\(([0-9.]+)\s+ms/step\)", log_text)
    memory_values = [float(value) for value in re.findall(r"\s([0-9]+(?:\.[0-9]+)?)\s+GB", log_text)]
    return {
        "sample_ms": float(sample_match.group(1)) if sample_match else None,
        "sample_ms_per_step": float(step_match.group(1)) if step_match else None,
        "peak_ram_gb_reported": max(memory_values) if memory_values else None,
    }


def _variant_name(value: str) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in VARIANT_BITS:
        raise ValueError(
            f"unknown quantization variant {value!r}; expected fp16, int8, int4, int2 or int1"
        )
    return normalized


def run_experiment(
    *,
    annotations_path: str | Path = DEFAULT_ANNOTATIONS,
    runtime: str | Path = DEFAULT_RUNTIME,
    adapter: str | Path = DEFAULT_ADAPTER,
    output: str | Path = DEFAULT_OUTPUT,
    track_numbers: Iterable[int] = DEFAULT_TRACKS,
    variants: Iterable[str] = ("fp16", "int8", "int4"),
    duration_seconds: int = 60,
    seed_base: int = 2026091000,
    steps: int = DEFAULT_STEPS,
    init_noise_level: float = DEFAULT_INIT_NOISE_LEVEL,
    lora_strength: float = DEFAULT_LORA_STRENGTH,
    group_size: int = DEFAULT_GROUP_SIZE,
    output_headroom_db: float = DEFAULT_OUTPUT_HEADROOM_DB,
    force: bool = False,
) -> dict[str, Any]:
    """Generate matched canaries and persist a complete experiment manifest."""

    annotations = Path(annotations_path).expanduser().resolve()
    runtime_path = Path(runtime).expanduser().resolve()
    adapter_path = Path(adapter).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    normalized_variants = tuple(dict.fromkeys(_variant_name(value) for value in variants))
    if not normalized_variants:
        raise ValueError("at least one quantization variant is required")
    for variant in normalized_variants:
        validate_quantization(VARIANT_BITS[variant], group_size)
    if not annotations.is_file():
        raise FileNotFoundError(f"annotations file does not exist: {annotations}")
    if not adapter_path.is_file():
        raise FileNotFoundError(f"LoRA adapter does not exist: {adapter_path}")
    python_path = runtime_path / ".venv" / "bin" / "python"
    runtime_script = runtime_path / "scripts" / "sa3_mlx.py"
    wrapper = HERE / "run_sftberlin_quantized.py"
    if not python_path.is_file() or not runtime_script.is_file() or not wrapper.is_file():
        raise FileNotFoundError(f"Stable Audio 3 MLX runtime or quantization wrapper is incomplete: {runtime_path}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to prepare the guarded input WAV")
    if int(duration_seconds) <= 0 or int(steps) <= 0:
        raise ValueError("duration_seconds and steps must be positive")
    if not math.isfinite(float(init_noise_level)) or not 0.0 <= float(init_noise_level) <= 1.0:
        raise ValueError("init_noise_level must be between 0 and 1")

    jobs = build_demo_jobs(
        _read_jsonl(annotations),
        track_numbers=tuple(int(value) for value in track_numbers),
        duration_seconds=int(duration_seconds),
        seed_base=int(seed_base),
        steps=int(steps),
    )
    input_dir = output_path / "inputs-44k"
    raw_dir = output_path / "raw-tracks"
    final_dir = output_path / "tracks"
    log_dir = output_path / "logs"
    report_dir = output_path / "quantization-reports"
    manifest_path = output_path / "quantization-manifest.json"
    output_path.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema": "onus.sample-expertise.sftberlin-quantization-experiment/v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_annotations": str(annotations),
        "source_rights_scope": "user-confirmed recordings for local project; underlying composition clearance not asserted",
        "experiment": {
            "variants": list(normalized_variants),
            "baseline": "fp16" if "fp16" in normalized_variants else normalized_variants[0],
            "bits": {variant: VARIANT_BITS[variant] for variant in normalized_variants},
            "group_size": int(group_size),
            "quantization_mode": "affine",
            "quantization_scope": "medium DiT Linear weights after LoRA merge",
            "unquantized_components": ["T5Gemma", "SAME-L encoder", "SAME-L decoder", "DiT Conv1d"],
        },
        "model": {
            "dit": "medium",
            "base_inference_weights": str(runtime_path / "models/mlx/dit_medium_f16.npz"),
            "adapter": str(adapter_path),
            "adapter_sha256": _sha256(adapter_path),
            "lora_strength": float(lora_strength),
            "decoder": "same-l",
            "runtime_root": str(runtime_path),
            "runtime_script": str(runtime_script),
            "wrapper": str(wrapper),
        },
        "sampling": {
            "duration_seconds": int(duration_seconds),
            "steps": int(steps),
            "init_noise_level": float(init_noise_level),
            "cfg": DEFAULT_CFG,
            "apg": DEFAULT_APG,
            "renoise_mode": "fresh",
            "output_headroom_db": float(output_headroom_db),
        },
        "tracks": [],
        "comparisons": [],
    }
    _write_json(manifest_path, payload)

    prepared_inputs: dict[str, dict[str, Any]] = {}
    for job in jobs:
        source = annotations.parent / str(job["source_audio"])
        input_wav = input_dir / f"{job['slug']}.wav"
        if not input_wav.is_file() or force:
            prep = _convert_input(source, input_wav, duration_seconds=int(duration_seconds), ffmpeg=ffmpeg)
        else:
            prep = {"mode": "reused_existing", "path": str(input_wav)}
        prepared_inputs[job["slug"]] = {
            "path": str(input_wav),
            "source": str(source),
            "source_sha256": _sha256(source),
            "preparation": prep,
        }

    for variant in normalized_variants:
        bits = VARIANT_BITS[variant]
        for job in jobs:
            slug = str(job["slug"])
            input_info = prepared_inputs[slug]
            raw_output = raw_dir / variant / f"{slug}.wav"
            final_output = final_dir / variant / f"{slug}.wav"
            log_path = log_dir / variant / f"{slug}.log"
            quant_report_path = report_dir / variant / f"{slug}.json"
            command = [
                str(python_path),
                str(wrapper),
                "--runtime-script",
                str(runtime_script),
                "--output-headroom-db",
                str(float(output_headroom_db)),
                "--prompt",
                str(job["prompt"]),
                "--negative-prompt",
                str(job["negative_prompt"]),
                "--dit",
                "medium",
                "--decoder",
                "same-l",
                "--seconds",
                str(int(duration_seconds)),
                "--steps",
                str(int(steps)),
                "--seed",
                str(int(job["seed"])),
                "--init-noise-level",
                str(float(init_noise_level)),
                "--renoise-mode",
                "fresh",
                "--cfg",
                str(DEFAULT_CFG),
                "--apg",
                str(DEFAULT_APG),
                "--init-audio",
                str(input_info["path"]),
                "--out",
                str(raw_output),
                "--free-models",
                "--lora",
                str(adapter_path),
                "--lora-strength",
                str(float(lora_strength)),
            ]
            if bits is not None:
                command.extend(
                    [
                        "--quant-bits",
                        str(bits),
                        "--quant-group-size",
                        str(int(group_size)),
                        "--quantization-report",
                        str(quant_report_path),
                    ]
                )
            record: dict[str, Any] = {
                "variant": variant,
                "quantization_bits": bits,
                "quantization_group_size": int(group_size) if bits is not None else None,
                **job,
                "source_audio_path": input_info["source"],
                "source_sha256": input_info["source_sha256"],
                "input_wav": input_info["path"],
                "raw_output_wav": str(raw_output),
                "output_wav": str(final_output),
                "quantization_report": str(quant_report_path) if bits is not None else None,
                "command": command,
                "status": "pending",
            }
            started = time.monotonic()
            if raw_output.is_file() and final_output.is_file() and not force:
                record["status"] = "reused_existing"
            else:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                completed = subprocess.run(
                    command,
                    cwd=runtime_path,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                log_path.write_text(
                    "STDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
                    encoding="utf-8",
                )
                record["exit_code"] = completed.returncode
                record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                if completed.returncode != 0 or not raw_output.is_file():
                    record["status"] = "failed"
                    record["error"] = (completed.stderr or completed.stdout)[-3000:]
            if raw_output.is_file() and record["status"] != "failed":
                record["raw_output_sha256"] = _sha256(raw_output)
                record["raw_technical"] = _analyse_output(raw_output)
                log_text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
                record["runtime_observations"] = _runtime_observations(log_text)
                if bits is not None and quant_report_path.is_file():
                    record["quantization"] = json.loads(quant_report_path.read_text(encoding="utf-8"))
                try:
                    final_output.parent.mkdir(parents=True, exist_ok=True)
                    record["mastering"] = _master_output(
                        raw_output,
                        final_output,
                        ffmpeg=ffmpeg,
                        bass_trim_db=0.0,
                        bass_presence_db=0.0,
                        master_profile="direct",
                    )
                    record["status"] = "generated" if record["status"] == "pending" else record["status"]
                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = str(exc)
            if final_output.is_file():
                record["output_sha256"] = _sha256(final_output)
                record["technical"] = _analyse_output(final_output)
            payload["tracks"].append(record)
            _write_json(manifest_path, payload)
            print(json.dumps({
                "variant": variant,
                "track": job["track_number"],
                "status": record["status"],
                "output": str(final_output),
                "technical_pass": (record.get("technical") or {}).get("technical_gate", {}).get("passed"),
            }, ensure_ascii=False))

    baseline = payload["experiment"]["baseline"]
    by_key = {(item["track_number"], item["variant"]): item for item in payload["tracks"]}
    for job in jobs:
        reference = by_key.get((job["track_number"], baseline))
        if not reference or not Path(reference["raw_output_wav"]).is_file():
            continue
        for variant in normalized_variants:
            if variant == baseline:
                continue
            candidate = by_key.get((job["track_number"], variant))
            if not candidate or not Path(candidate["raw_output_wav"]).is_file():
                continue
            payload["comparisons"].append({
                "track_number": job["track_number"],
                "baseline": baseline,
                "candidate": variant,
                "raw_audio": compare_audio(
                    Path(reference["raw_output_wav"]),
                    Path(candidate["raw_output_wav"]),
                ),
            })
    summary_by_variant: dict[str, Any] = {}
    for variant in normalized_variants:
        rows = [item for item in payload["tracks"] if item["variant"] == variant]
        summary_by_variant[variant] = {
            "requested": len(rows),
            "generated": sum(item["status"] in {"generated", "reused_existing"} for item in rows),
            "technical_passes": sum(
                bool((item.get("technical") or {}).get("technical_gate", {}).get("passed"))
                for item in rows
            ),
            "failed": sum(item["status"] == "failed" for item in rows),
        }
    payload["summary"] = {
        "by_variant": summary_by_variant,
        "manifest": str(manifest_path),
        "output": str(output_path),
    }
    _write_json(manifest_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--track", action="append", type=int, dest="tracks")
    parser.add_argument("--variant", action="append", choices=tuple(VARIANT_BITS), dest="variants")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--seed-base", type=int, default=2026091000)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--init-noise-level", type=float, default=DEFAULT_INIT_NOISE_LEVEL)
    parser.add_argument("--lora-strength", type=float, default=DEFAULT_LORA_STRENGTH)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--headroom-db", type=float, default=DEFAULT_OUTPUT_HEADROOM_DB)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    payload = run_experiment(
        annotations_path=args.annotations,
        runtime=args.runtime,
        adapter=args.adapter,
        output=args.output,
        track_numbers=tuple(args.tracks) if args.tracks else DEFAULT_TRACKS,
        variants=tuple(args.variants) if args.variants else tuple(VARIANT_BITS),
        duration_seconds=args.duration,
        seed_base=args.seed_base,
        steps=args.steps,
        init_noise_level=args.init_noise_level,
        lora_strength=args.lora_strength,
        group_size=args.group_size,
        output_headroom_db=args.headroom_db,
        force=args.force,
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0 if not any(item["failed"] for item in payload["summary"]["by_variant"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
