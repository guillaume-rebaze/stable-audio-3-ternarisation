from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from build_ternary_independent_corpus import load_records, parent_key, prompt_key  # noqa: E402


def test_parent_and_prompt_keys_are_deterministic() -> None:
    assert parent_key("same-source.mp3") == parent_key("same-source.mp3")
    assert parent_key("same-source.mp3") != parent_key("other-source.mp3")
    assert prompt_key("prompt", 7) == prompt_key("prompt", 7)
    assert prompt_key("prompt", 7) != prompt_key("prompt", 8)


def test_load_records_keeps_source_lineage_and_shape(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus" / "latents-12s"
    corpus.mkdir(parents=True)
    latent = corpus / "sample.npy"
    metadata = latent.with_suffix(".json")
    np.save(latent, np.zeros((256, 4), dtype=np.float16))
    metadata.write_text(
        json.dumps(
            {
                "src_relpath": "original/sample.mp3",
                "prompt": "test prompt",
                "genre": "test",
            }
        ),
        encoding="utf-8",
    )

    records = load_records([corpus])

    assert len(records) == 1
    assert records[0]["source"] == "original/sample.mp3"
    assert records[0]["shape"] == [256, 4]
    assert records[0]["parent_id"] == parent_key("original/sample.mp3")
    assert len(records[0]["latent_sha256"]) == 64
