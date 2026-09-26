"""Strict dataset provenance contract for ternary V8 experiments.

The V7 state cache exposed a dangerous failure mode: its declared dataset
directory was the SFT Voices train split while selected sample paths pointed
to the universal latent corpus.  This module makes that mismatch impossible
to ignore before a cache or an artifact is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "onus.ternary-quality/v8-dataset-contract"
SPLITS = ("train", "validation", "test")


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"dataset path is outside project root: {path}") from exc


def _relative_lexical(root: Path, path: Path) -> str:
    """Return a path relative to root without dereferencing symlinks."""
    lexical = path if path.is_absolute() else root / path
    try:
        return lexical.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as exc:
        raise ValueError(f"dataset path is outside project root: {path}") from exc


def _file_entry(root: Path, path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing dataset file: {path}")
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"sha256 mismatch for {path}: expected {expected_sha256}, got {digest}"
        )
    return {
        "path": _relative_lexical(root, path),
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def _manifest_source_index(root: Path, selection: dict[str, Any]) -> dict[Path, dict[str, Any]]:
    index: dict[Path, dict[str, Any]] = {}
    sections = [selection.get("train_extension", {}), selection.get("validation", {}), selection.get("test", {})]
    for section in sections:
        for item in section.get("sources", []):
            staged = item.get("staged_latent")
            if not staged:
                continue
            path = _resolve(root, staged)
            if path in index:
                raise ValueError(f"duplicate staged latent in selection manifest: {path}")
            index[path] = item
    return index


def _sample_record(
    root: Path,
    latent_path: Path,
    metadata_path: Path,
    role: str,
    source_item: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid metadata JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object: {metadata_path}")

    source_item = source_item or {}
    prompt = source_item.get("prompt") or metadata.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"missing prompt for {latent_path}")
    metadata_prompt = metadata.get("prompt")
    if metadata_prompt is not None and metadata_prompt != prompt:
        raise ValueError(f"prompt mismatch between manifest and metadata: {latent_path}")

    parent_id = (
        source_item.get("parent_id")
        or metadata.get("parent_id")
        or metadata.get("src_relpath")
        or latent_path.name
    )
    source = source_item.get("source") or metadata.get("src_relpath") or latent_path.name
    expected_latent = source_item.get("latent_sha256")
    expected_metadata = source_item.get("metadata_sha256")
    latent = _file_entry(root, latent_path, expected_latent)
    metadata_file = _file_entry(root, metadata_path, expected_metadata)

    return {
        "role": role,
        "parent_id": str(parent_id),
        "source": str(source),
        "prompt": prompt,
        "genre": source_item.get("genre") or metadata.get("genre"),
        "latent": latent,
        "metadata": metadata_file,
    }


def _collect_split(
    root: Path,
    directory: Path,
    role: str,
    source_index: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    if not directory.is_dir():
        raise ValueError(f"missing {role} directory: {directory}")
    records: list[dict[str, Any]] = []
    for latent_path in sorted(directory.glob("*.npy")):
        metadata_path = latent_path.with_suffix(".json")
        records.append(
            _sample_record(
                root,
                latent_path,
                metadata_path,
                role,
                source_index.get(latent_path.resolve()),
            )
        )
    if not records:
        raise ValueError(f"empty {role} directory: {directory}")
    prompts = sorted({record["prompt"] for record in records})
    return {
        "directory": _relative(root, directory.resolve()),
        "sample_count": len(records),
        "parent_count": len({record["parent_id"] for record in records}),
        "prompt_count": len(prompts),
        "prompts": prompts,
        "prompt_set_digest": canonical_digest(prompts),
        "samples": records,
        "sample_digest": canonical_digest(records),
    }


def build_contract(
    selection_manifest_path: Path,
    output_path: Path,
    project_root: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    selection_path = selection_manifest_path.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_index = _manifest_source_index(root, selection)

    train_dir = _resolve(root, selection["staged_train_dir"])
    validation_dir = _resolve(root, selection["validation"]["dir"])
    test_dir = _resolve(root, selection["test"]["dir"])
    splits = {
        "train": _collect_split(root, train_dir, "train", source_index),
        "validation": _collect_split(root, validation_dir, "validation", source_index),
        "test": _collect_split(root, test_dir, "test", source_index),
    }

    latent_hashes: dict[str, str] = {}
    parent_ids: dict[str, str] = {}
    for role, split in splits.items():
        for sample in split["samples"]:
            latent_hash = sample["latent"]["sha256"]
            parent_id = sample["parent_id"]
            if latent_hash in latent_hashes and latent_hashes[latent_hash] != role:
                raise ValueError(f"latent overlap between {latent_hashes[latent_hash]} and {role}")
            if parent_id in parent_ids and parent_ids[parent_id] != role:
                raise ValueError(f"parent overlap between {parent_ids[parent_id]} and {role}: {parent_id}")
            latent_hashes[latent_hash] = role
            parent_ids[parent_id] = role

    all_prompts = sorted({prompt for split in splits.values() for prompt in split["prompts"]})
    contract: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "prepared",
        "project_root": ".",
        "selection_manifest": {
            "path": _relative(root, selection_path),
            "bytes": selection_path.stat().st_size,
            "sha256": sha256_file(selection_path),
        },
        "splits": splits,
        "prompt_set_digest": canonical_digest(all_prompts),
        "prompt_count": len(all_prompts),
        "dataset_digest": canonical_digest({role: split["sample_digest"] for role, split in splits.items()}),
        "independence": {
            "latent_overlap_count": 0,
            "parent_overlap_count": 0,
            "prompt_overlap_allowed": True,
        },
    }
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return contract


def validate_contract(contract_path: Path, project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != SCHEMA:
        raise ValueError(f"unsupported dataset contract schema: {contract.get('schema')!r}")
    splits = contract.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ValueError("dataset contract must contain exactly train/validation/test splits")

    seen_latents: dict[str, str] = {}
    seen_parents: dict[str, str] = {}
    checked = 0
    for role in SPLITS:
        split = splits[role]
        directory = _resolve(root, split["directory"])
        if not directory.is_dir():
            raise ValueError(f"missing contract directory: {directory}")
        samples = split.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"empty contract split: {role}")
        for sample in samples:
            latent = sample["latent"]
            metadata = sample["metadata"]
            latent_logical = root / latent["path"]
            metadata_logical = root / metadata["path"]
            directory_logical = root / split["directory"]
            if latent_logical.parent != directory_logical or metadata_logical.parent != directory_logical:
                raise ValueError(f"sample outside declared {role} directory: {latent_logical}")
            latent_path = _resolve(root, latent["path"])
            metadata_path = _resolve(root, metadata["path"])
            _file_entry(root, latent_path, latent["sha256"])
            _file_entry(root, metadata_path, metadata["sha256"])
            metadata_json = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_json.get("prompt") not in (None, sample["prompt"]):
                raise ValueError(f"contract prompt mismatch: {latent_path}")
            latent_hash = latent["sha256"]
            parent_id = sample["parent_id"]
            if latent_hash in seen_latents and seen_latents[latent_hash] != role:
                raise ValueError(f"latent overlap between {seen_latents[latent_hash]} and {role}")
            if parent_id in seen_parents and seen_parents[parent_id] != role:
                raise ValueError(f"parent overlap between {seen_parents[parent_id]} and {role}: {parent_id}")
            seen_latents[latent_hash] = role
            seen_parents[parent_id] = role
            checked += 1
        if split.get("sample_count") != len(samples):
            raise ValueError(f"sample_count mismatch in {role}")
        prompts = sorted({sample["prompt"] for sample in samples})
        if split.get("prompts") != prompts:
            raise ValueError(f"prompt list mismatch in {role}")
        if split.get("prompt_set_digest") != canonical_digest(prompts):
            raise ValueError(f"prompt digest mismatch in {role}")
        if split.get("sample_digest") != canonical_digest(samples):
            raise ValueError(f"sample digest mismatch in {role}")

    all_prompts = sorted({prompt for role in SPLITS for prompt in splits[role]["prompts"]})
    if contract.get("prompt_set_digest") != canonical_digest(all_prompts):
        raise ValueError("global prompt digest mismatch")
    expected_dataset_digest = canonical_digest(
        {role: splits[role]["sample_digest"] for role in SPLITS}
    )
    if contract.get("dataset_digest") != expected_dataset_digest:
        raise ValueError("dataset digest mismatch")
    return {
        "valid": True,
        "schema": SCHEMA,
        "contract": str(contract_path),
        "sample_count": checked,
        "prompt_count": len(all_prompts),
        "split_counts": {role: splits[role]["sample_count"] for role in SPLITS},
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--selection-manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--project-root", type=Path, default=Path.cwd())

    validate = subparsers.add_parser("validate")
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--project-root", type=Path, default=Path.cwd())

    args = parser.parse_args()
    if args.command == "build":
        contract = build_contract(args.selection_manifest, args.output, args.project_root)
        print(json.dumps({"built": True, "dataset_digest": contract["dataset_digest"]}, indent=2))
    else:
        print(json.dumps(validate_contract(args.contract, args.project_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
