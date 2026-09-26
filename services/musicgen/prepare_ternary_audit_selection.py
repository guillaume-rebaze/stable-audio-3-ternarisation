"""Create a provenance-checked audit selection from a V7 state cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache_manifest_path = args.state_cache / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    prompts = list(cache_manifest["cache"]["prompt_index"])
    if not prompts:
        raise ValueError("state cache has no prompt index")

    selected: list[str] = []
    selected_meta: list[dict[str, object]] = []
    for prompt in prompts:
        matches: list[Path] = []
        for latent_path in sorted(args.dataset_dir.glob("*.npy")):
            metadata_path = latent_path.with_suffix(".json")
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            if str(metadata.get("prompt", "")) == prompt:
                matches.append(latent_path)
        if not matches:
            raise FileNotFoundError(
                f"cache prompt has no matching latent in dataset: {prompt!r}"
            )
        latent_path = matches[0]
        metadata_path = latent_path.with_suffix(".json")
        selected.append(str(latent_path.resolve()))
        selected_meta.append(
            {
                "path": str(latent_path.resolve()),
                "latent": file_identity(latent_path),
                "metadata": file_identity(metadata_path)
                if metadata_path.is_file()
                else None,
                "prompt": prompt,
            }
        )

    canonical = json.dumps(selected_meta, sort_keys=True, separators=(",", ":"))
    prompt_digest = hashlib.sha256(
        "\n".join(prompts).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema": "onus.ternary-quality/v7-dataset-contract",
        "status": "prepared",
        "dataset": {
            "directory": str(args.dataset_dir.resolve()),
            "prompt_set_digest": prompt_digest,
            "prompt_count": len(prompts),
            "prompts": prompts,
            "selected": selected_meta,
            "selected_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        },
        "state_cache": {
            "directory": str(args.state_cache.resolve()),
            "manifest": file_identity(cache_manifest_path),
            "states": file_identity(args.state_cache / "states.npz"),
            "conditions": file_identity(args.state_cache / "conditions.npz"),
        },
        "roles": {
            "debug_train_seen": selected,
            "validation": [],
            "test": [],
        },
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "prompt_count": len(prompts),
                "prompt_set_digest": prompt_digest,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
