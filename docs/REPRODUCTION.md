# Reproduction guide

This guide is intended for a clean Apple-Silicon checkout. It reproduces the
verification of the published package first; only then should anyone attempt a
new training run.

## 1. Prerequisites

You need:

- Python 3.12.x;
- MLX and the Stable Audio 3 MLX runtime used by the source project;
- the original dense `dit_medium_f16.npz` teacher;
- a compatible Stable Audio 3 latent dataset for the optional audio audit;
- enough disk for temporary caches and checkpoints;
- a Metal memory ceiling appropriate to the machine. The historical run used
  an 11,000,000,000-byte guard, not an assumption that every Mac has 12 GB
  free.

The source environment reported Python `3.12.6` and MLX `0.31.2`. Exact
dependency versions should be captured before a new run because MLX kernel and
dtype changes can invalidate numerical comparisons.

The dense teacher is not included in this repository. Obtain it through the
authorised Stable Audio 3 runtime and verify its SHA-256 against the method
document before training:

```bash
shasum -a 256 /path/to/dit_medium_f16.npz
# f9e5647ea3225818657d47d47ae4b34afa29c0568206ca89566c1a758944a38e
```

## 2. Verify the compact package

From the repository root:

```bash
python3 services/musicgen/verify_ternary_bonsai_package.py \
  --package-dir model \
  --teacher-weights /path/to/dit_medium_f16.npz \
  --dataset-dir /path/to/ternary-quality-v3/g2-authorized-independent/train \
  --sample-index 0 \
  --sigma 0.5 \
  --seed 20260925
```

The dataset is used for the seeded latent/canary check; the structural reload
does not require copying the source project's entire output directory.

Expected key fields:

```text
status=pass
core_count=168
support_count=357
parameter_reload.exact=true
forward.relative_l2=0.0
forward.cosine≈1.0
```

The exact historical output is preserved in
[`model/reload_report.json`](../model/reload_report.json).

## 3. Inspect the artifact without MLX

The package is an NPZ/ZIP container and the metadata is ordinary JSON:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for name in ("manifest.json", "reload_report.json", "acceptance.json"):
    print(f"\n--- {name} ---")
    print(json.dumps(json.loads(Path("model", name).read_text()), indent=2))
PY

shasum -a 256 model/payload.npz
du -h model/payload.npz
```

No loader should infer scope from filename patterns. Read `manifest.json` and
honour the recorded mode and member names.

## 4. Optional tests and static checks

The extracted test files target the original runtime layout. Run only after
installing that runtime:

```bash
python3 -m compileall -q services/musicgen
python3 -m pytest services/musicgen/tests/test_ternary*.py
```

If a test imports project modules outside this extraction, document the missing
dependency instead of adding a fake fallback. The goal is an auditable failure,
not a green test that exercises a different implementation.

## 5. Recreate a training run

Training is intentionally not the default command. Start with a one-block
pilot and an independent held-out split. Never launch all 24 blocks before a
single-block candidate has passed the full reload, velocity and audio gates.

The extracted components are:

| Purpose | Script |
|---|---|
| contract and packing | `services/musicgen/ternary_contract.py` |
| QAT/cascade block trainer | `services/musicgen/train_ternary_quality.py` |
| guarded cascade driver | `services/musicgen/run_ternary_bonsai_cascade.py` |
| velocity audit | `services/musicgen/audit_ternary_quality.py` |
| audio canary | `services/musicgen/render_ternary_bonsai_canary.py` |
| export | `services/musicgen/export_ternary_bonsai_package.py` |
| fresh verification | `services/musicgen/verify_ternary_bonsai_package.py` |

The historical config is preserved at
[`configs/ternary_quality_v6_learned_pilot.json`](../configs/ternary_quality_v6_learned_pilot.json).
It is evidence of an earlier pilot, not a promise that its small step budget
is sufficient for a new release.

For a new campaign, record at minimum:

- teacher hash and runtime versions;
- dataset and held-out split hashes;
- seed, crop, group size, quantiser mode and surrogate;
- prompt/sigma sampling and state-cache provenance;
- peak Metal memory and disk usage;
- every block's reload result and quality gate;
- raw teacher/student WAVs and unmodified metrics;
- final payload and manifest hashes.

The release rule is simple: one failed block stops promotion. The historical
`--promote-on-quality-fail` option exists for forensic completion only and must
not be used to label a new release as quality-passing.

## 6. Reproduce the documentation package

The repository intentionally excludes cumulative checkpoints, generated
training caches and the dense teacher. If you need the full source experiment,
use [`docs/SOURCE_MAP.md`](SOURCE_MAP.md) to reconstruct the paths from the
original `abelton` checkout, then copy only the specific evidence required for
the run. Keep large intermediates outside Git and record their hashes in a
manifest.

