"""Build an activation-aware TTQ record from a dense teacher and V8 profile.

This is a bounded discrete proposal stage.  It keeps the source record as a
candidate, tries a small threshold grid per group, fits positive/negative
levels with activation RMS weights, and accepts only groups whose weighted
reconstruction improves.  It does not claim quality until the reloaded record
passes the full audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import train_ternary_quality as tq
from ternary_contract import TernaryWeights, pack_codes, validate_ternary_weights
from ternary_provenance_v8 import canonical_digest, sha256_file


TARGETS = (
    "self_attn.to_qkv",
    "self_attn.to_out",
    "cross_attn.to_q",
    "cross_attn.to_kv",
    "cross_attn.to_out",
    "ff.ff.0.proj",
    "ff.ff.2",
)


def _profile_key(target: str, sigma: str, metric: str) -> str:
    return f"{target.replace('.', '__')}__sigma_{sigma}__{metric}"


def load_importance(profile_dir: Path, target: str, mode: str) -> np.ndarray:
    metadata = json.loads((profile_dir / "activation_profile.json").read_text(encoding="utf-8"))
    profile = np.load(profile_dir / "activation_profile.npz", allow_pickle=False)
    sigma_values = sorted(metadata["metrics"][target], key=float)
    rms_squared = []
    for sigma in sigma_values:
        mean = profile[_profile_key(target, sigma, "mean")].astype(np.float32)
        std = profile[_profile_key(target, sigma, "std")].astype(np.float32)
        rms_squared.append(np.square(mean) + np.square(std))
    stacked = np.stack(rms_squared, axis=0)
    if mode == "mean":
        importance = np.mean(stacked, axis=0)
    elif mode == "max":
        importance = np.max(stacked, axis=0)
    elif mode == "late":
        late = [index for index, sigma in enumerate(sigma_values) if float(sigma) <= 0.74554658]
        importance = np.max(stacked[late], axis=0)
    else:
        raise ValueError(f"unknown importance mode: {mode}")
    return np.maximum(importance.astype(np.float32), 1e-8)


def _fit_candidate(
    centered: np.ndarray,
    importance: np.ndarray,
    threshold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = (np.sign(centered) * (np.abs(centered) >= threshold[..., None])).astype(np.int8)
    weighted = importance[None, :, :]
    positive_mask = q == 1
    negative_mask = q == -1
    positive_mass = np.sum(weighted * positive_mask, axis=-1)
    negative_mass = np.sum(weighted * negative_mask, axis=-1)
    positive = np.divide(
        np.sum(weighted * centered * positive_mask, axis=-1),
        np.maximum(positive_mass, 1e-8),
        out=np.zeros_like(positive_mass, dtype=np.float32),
        where=positive_mass > 0,
    )
    negative = np.divide(
        -np.sum(weighted * centered * negative_mask, axis=-1),
        np.maximum(negative_mass, 1e-8),
        out=np.zeros_like(negative_mass, dtype=np.float32),
        where=negative_mass > 0,
    )
    positive = np.maximum(positive, 0.0)
    negative = np.maximum(negative, 0.0)
    target = np.where(q == 1, positive[..., None], np.where(q == -1, -negative[..., None], 0.0))
    loss = np.sum(weighted * np.square(centered - target), axis=-1)
    return q, positive, negative, loss


def activation_aware_record(
    weight: np.ndarray,
    source: TernaryWeights,
    importance: np.ndarray,
    factors: tuple[float, ...],
) -> tuple[TernaryWeights, dict[str, Any]]:
    if source.mode not in {"ttq", "ttq_hadamard"}:
        raise ValueError(f"activation-aware V8 requires TTQ source, got {source.mode}")
    if source.positive_scales is None or source.negative_scales is None:
        raise ValueError("TTQ source is missing branch scales")
    out_dim, in_dim = weight.shape
    group_size = int(source.group_size)
    groups = np.asarray(weight, dtype=np.float32).reshape(source.q.shape)
    means = np.asarray(source.group_means, dtype=np.float32)
    centered = groups - means[..., None]
    channel_importance = np.asarray(importance, dtype=np.float32)
    if channel_importance.shape != (in_dim,):
        raise ValueError(f"importance shape {channel_importance.shape} != {(in_dim,)}")
    channel_importance = channel_importance.reshape(source.q.shape[1], group_size)

    current_q = np.asarray(source.q, dtype=np.int8)
    current_positive = np.asarray(source.positive_scales, dtype=np.float32)
    current_negative = np.asarray(source.negative_scales, dtype=np.float32)
    current_target = np.where(
        current_q == 1,
        current_positive[..., None],
        np.where(current_q == -1, -current_negative[..., None], 0.0),
    )
    weighted = channel_importance[None, :, :]
    best_loss = np.sum(weighted * np.square(centered - current_target), axis=-1)
    baseline_loss = best_loss.copy()
    best_q = current_q.copy()
    best_positive = current_positive.copy()
    best_negative = current_negative.copy()
    base_threshold = np.maximum(0.5 * (current_positive + current_negative), 1e-6)
    proposals = 0
    for factor in factors:
        q, positive, negative, loss = _fit_candidate(
            centered, channel_importance, base_threshold * float(factor)
        )
        improve = loss + 1e-8 < best_loss
        proposals += int(np.count_nonzero(improve))
        best_loss = np.where(improve, loss, best_loss)
        best_q = np.where(improve[..., None], q, best_q).astype(np.int8)
        best_positive = np.where(improve, positive, best_positive)
        best_negative = np.where(improve, negative, best_negative)

    record = TernaryWeights(
        packed_codes=pack_codes(best_q),
        scales=-best_positive.astype(np.float16),
        biases=(means.astype(np.float16) + best_positive.astype(np.float16)).astype(np.float16),
        q=best_q,
        group_means=means,
        group_size=group_size,
        mode=source.mode,
        linear_bias=None if source.linear_bias is None else np.asarray(source.linear_bias).copy(),
        positive_scales=best_positive.astype(np.float16),
        negative_scales=best_negative.astype(np.float16),
    )
    validate_ternary_weights(record)
    return record, {
        "groups": int(best_q.shape[0] * best_q.shape[1]),
        "code_flips": int(np.count_nonzero(best_q != current_q)),
        "groups_changed": int(np.count_nonzero(np.any(best_q != current_q, axis=-1))),
        "proposal_group_winners": proposals,
        "weighted_loss_before": float(np.sum(baseline_loss)),
        "weighted_loss_after": float(np.sum(best_loss)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-records", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--importance-mode", choices=("mean", "max", "late"), default="max")
    parser.add_argument(
        "--threshold-factors",
        type=float,
        nargs="+",
        default=(0.50, 0.75, 1.00, 1.25, 1.50),
    )
    args = parser.parse_args()
    records, metadata = tq.load_records_checkpoint(args.source_records)
    profile_metadata = json.loads((args.profile_dir / "activation_profile.json").read_text(encoding="utf-8"))
    output_records: dict[str, TernaryWeights] = {}
    report: dict[str, Any] = {
        "schema": "onus.ternary-quality/v8-activation-aware-records",
        "source_records": {"path": str(args.source_records), "sha256": sha256_file(args.source_records)},
        "teacher_weights": {"path": str(args.teacher_weights), "sha256": sha256_file(args.teacher_weights)},
        "profile": profile_metadata.get("profile"),
        "profile_digest": profile_metadata.get("profile_digest"),
        "importance_mode": args.importance_mode,
        "threshold_factors": [float(value) for value in args.threshold_factors],
        "per_module": {},
    }
    with np.load(args.teacher_weights, allow_pickle=False) as dense:
        for target in TARGETS:
            path = f"transformer.layers.0.{target}"
            if path not in records:
                raise ValueError(f"source records omit {path}")
            dense_key = f"{path}.weight"
            weight = np.asarray(dense[dense_key], dtype=np.float32)
            importance = load_importance(args.profile_dir, target, args.importance_mode)
            record, metrics = activation_aware_record(
                weight,
                records[path],
                importance,
                tuple(float(value) for value in args.threshold_factors),
            )
            output_records[path] = record
            report["per_module"][path] = metrics
    report["scope_digest"] = tq.scope_digest(output_records)
    report["dataset_digest"] = profile_metadata["dataset_contract"].get("dataset_digest")
    report["record_digest"] = canonical_digest(
        {
            path: {
                "q": record.q.tolist(),
                "positive": np.asarray(record.positive_scales).tolist(),
                "negative": np.asarray(record.negative_scales).tolist(),
            }
            for path, record in sorted(output_records.items())
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "records_checkpoint.npz"
    tq.save_records_checkpoint(
        checkpoint,
        output_records,
        next_block=1,
        group_size=int(metadata["group_size"]),
        crop_len=int(metadata.get("crop_len", 128)),
        quantizer_mode=str(metadata["quantizer_mode"]),
    )
    report["records_checkpoint"] = str(checkpoint)
    report["records_checkpoint_sha256"] = sha256_file(checkpoint)
    (args.output_dir / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "prepared", "output_dir": str(args.output_dir), "scope_digest": report["scope_digest"], "record_digest": report["record_digest"]}, indent=2))


if __name__ == "__main__":
    main()
