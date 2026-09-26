"""Build a deterministic, source-disjoint validation/test pool for ternary v3.

The local corpora are authorized for personal study by the project owner. That
authorization is recorded in the derived manifest; source metadata is kept
unchanged. A parent/source can only belong to one split, and any source already
used by the v3 train bank is excluded from held-out data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def source_key(meta: dict, latent_path: Path) -> str:
    return str(
        meta.get("src_relpath")
        or meta.get("path")
        or meta.get("relpath")
        or latent_path.name
    )


def parent_key(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]


def prompt_key(prompt: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(input_dirs: Iterable[Path]) -> list[dict]:
    records: list[dict] = []
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            raise FileNotFoundError(f"corpus directory does not exist: {input_dir}")
        corpus = input_dir.parent.name
        for latent_path in sorted(input_dir.glob("*.npy")):
            metadata_path = latent_path.with_suffix(".json")
            if not metadata_path.is_file():
                raise ValueError(f"missing sidecar metadata: {metadata_path}")
            shape = tuple(int(value) for value in np.load(latent_path, mmap_mode="r").shape)
            if len(shape) != 2 or shape[0] != 256:
                raise ValueError(f"invalid latent shape {latent_path}: {shape}")
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = source_key(meta, latent_path)
            prompt = str(meta.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"missing prompt in {metadata_path}")
            records.append(
                {
                    "input_latent": str(latent_path),
                    "input_metadata": str(metadata_path),
                    "corpus": corpus,
                    "source": source,
                    "parent_id": parent_key(source),
                    "prompt": prompt,
                    "genre": str(meta.get("genre", "unknown")),
                    "shape": list(shape),
                    "latent_sha256": sha256_file(latent_path),
                    "metadata_sha256": sha256_file(metadata_path),
                }
            )
    return records


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = source.resolve()
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != target:
            raise FileExistsError(f"destination collision: {destination}")
        return
    destination.symlink_to(target)


def stage_split(records: list[dict], split_dir: Path, split: str) -> list[dict]:
    staged: list[dict] = []
    for index, record in enumerate(records):
        stem = f"{index:04d}__{record['corpus']}__{record['parent_id']}"
        destination_latent = split_dir / f"{stem}.npy"
        destination_metadata = split_dir / f"{stem}.json"
        link_file(Path(record["input_latent"]), destination_latent)
        link_file(Path(record["input_metadata"]), destination_metadata)
        staged_record = dict(record)
        staged_record.update(
            {
                "split": split,
                "staged_latent": str(destination_latent),
                "staged_metadata": str(destination_metadata),
                "role": split,
                "rights": "user_authorized_personal_study",
            }
        )
        staged.append(staged_record)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build source-disjoint authorized ternary validation/test corpora"
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("output/sample-expertise-pilot/universal-dataset/latents-12s"),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        required=True,
        help="external corpus directory; repeat for every authorized corpus",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/sample-expertise-pilot/ternary-quality-v3/g2-authorized-independent"),
    )
    parser.add_argument("--test-prompts", type=int, default=24)
    parser.add_argument(
        "--validation-prompts",
        type=int,
        default=0,
        help="validation prompt groups; zero assigns every non-test group to validation",
    )
    parser.add_argument(
        "--train-extra-prompts",
        type=int,
        default=0,
        help="authorized prompt groups reserved for an expanded training pool",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    train_records = load_records([args.train_dir])
    candidate_records = load_records(args.input_dir)
    train_parents = {record["parent_id"] for record in train_records}

    overlap = [record for record in candidate_records if record["parent_id"] in train_parents]
    eligible = [record for record in candidate_records if record["parent_id"] not in train_parents]

    by_parent: dict[str, list[dict]] = defaultdict(list)
    for record in eligible:
        by_parent[record["parent_id"]].append(record)
    cross_corpus_duplicates = {
        parent: records
        for parent, records in by_parent.items()
        if len({record["corpus"] for record in records}) > 1
    }
    if cross_corpus_duplicates:
        raise ValueError(
            "same source appears more than once across authorized corpora: "
            + ", ".join(sorted(cross_corpus_duplicates))
        )

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for record in eligible:
        by_prompt[record["prompt"]].append(record)
    prompts = sorted(by_prompt, key=lambda value: prompt_key(value, args.seed))
    required_prompts = args.test_prompts + max(args.validation_prompts, 12) + args.train_extra_prompts
    if len(prompts) < required_prompts:
        raise ValueError(
            f"need at least {required_prompts} prompts, found {len(prompts)}"
        )
    test_prompts = set(prompts[: args.test_prompts])
    remaining_prompts = prompts[args.test_prompts :]
    if args.train_extra_prompts:
        train_extra_prompts = set(remaining_prompts[-args.train_extra_prompts :])
        validation_prompts = set(remaining_prompts[: -args.train_extra_prompts])
    else:
        train_extra_prompts = set()
        validation_prompts = set(remaining_prompts)
    if test_prompts & validation_prompts:
        raise AssertionError("prompt split overlap")
    if test_prompts & train_extra_prompts or validation_prompts & train_extra_prompts:
        raise AssertionError("training extension split overlap")

    test_records = [record for record in eligible if record["prompt"] in test_prompts]
    validation_records = [
        record for record in eligible if record["prompt"] in validation_prompts
    ]
    train_extra_records = [
        record for record in eligible if record["prompt"] in train_extra_prompts
    ]
    test_parents = {record["parent_id"] for record in test_records}
    validation_parents = {record["parent_id"] for record in validation_records}
    train_extra_parents = {record["parent_id"] for record in train_extra_records}
    if test_parents & validation_parents:
        raise AssertionError("parent split overlap")
    if train_extra_parents & test_parents or train_extra_parents & validation_parents:
        raise AssertionError("training extension parent overlap")

    train_dir = args.output_root / "train"
    validation_dir = args.output_root / "validation"
    test_dir = args.output_root / "test"
    staged_train = stage_split(train_records, train_dir, "debug_train_seen")
    staged_train_extension = stage_split(
        overlap + train_extra_records, train_dir, "authorized_train_extension"
    )
    staged_validation = stage_split(validation_records, validation_dir, "validation")
    staged_test = stage_split(test_records, test_dir, "test")

    manifest = {
        "schema": "onus.ternary-quality/v3/authorized-independent-corpus",
        "status": "prepared",
        "authorization": {
            "authorized_by_user": True,
            "scope": "personal_study",
            "redistribution": False,
            "source_metadata_preserved": True,
        },
        "train_dir": str(args.train_dir),
        "staged_train_dir": str(train_dir),
        "input_dirs": [str(path) for path in args.input_dir],
        "seed": args.seed,
        "overlap_with_train_excluded": len(overlap),
        "overlap_reused_for_train_extension": len(overlap),
        "train_extension_records": len(staged_train_extension),
        "train_extension_parents": len(
            {record["parent_id"] for record in overlap + train_extra_records}
        ),
        "train_extension_prompts": len(
            {record["prompt"] for record in overlap + train_extra_records}
        ),
        "eligible_records": len(eligible),
        "eligible_parents": len({record["parent_id"] for record in eligible}),
        "eligible_prompts": len(prompts),
        "validation": {
            "dir": str(validation_dir),
            "records": len(staged_validation),
            "parents": len(validation_parents),
            "prompts": len(validation_prompts),
            "sources": staged_validation,
        },
        "train_extension": {
            "dir": str(train_dir),
            "base_records": len(staged_train),
            "records": len(staged_train_extension),
            "parents": len(
                {record["parent_id"] for record in overlap + train_extra_records}
            ),
            "prompts": len(
                {record["prompt"] for record in overlap + train_extra_records}
            ),
            "sources": staged_train_extension,
        },
        "test": {
            "dir": str(test_dir),
            "records": len(staged_test),
            "parents": len(test_parents),
            "prompts": len(test_prompts),
            "sources": staged_test,
        },
        "excluded_train_overlap": overlap,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = {
        "status": manifest["status"],
        "overlap_with_train_excluded": len(overlap),
        "eligible_records": len(eligible),
        "staged_train_records": len(staged_train),
        "train_extension_records": len(staged_train_extension),
        "train_extension_prompts": len(train_extra_prompts),
        "validation_records": len(staged_validation),
        "validation_parents": len(validation_parents),
        "validation_prompts": len(validation_prompts),
        "test_records": len(staged_test),
        "test_parents": len(test_parents),
        "test_prompts": len(test_prompts),
        "authorization": manifest["authorization"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
