# Extraction map

The source work lived in the dirty `abelton` repository. This repository is a
deliberate extraction: it keeps the ternarisation implementation, its tests,
the relevant plans and the final evidence, while leaving unrelated application
code and large working directories behind.

## Source and destination

| Original `abelton` path | This repository |
|---|---|
| `services/musicgen/*ternary*.py` | `services/musicgen/` |
| `services/musicgen/*bonsai*.py` | `services/musicgen/` |
| `services/musicgen/*distill*.py` | `services/musicgen/` |
| `services/musicgen/*quantiz*.py` | `services/musicgen/` |
| `services/musicgen/*audit*.py`, `*calibrate*.py`, `*profile*.py` | `services/musicgen/` |
| `services/musicgen/training_checkpoint.py` | `services/musicgen/training_checkpoint.py` |
| `services/musicgen/tests/test_ternary*.py` | `services/musicgen/tests/` |
| `services/musicgen/tests/test_run_sftberlin_quantized.py` | `services/musicgen/tests/` |
| `docs/TERNARY*.md` | `docs/history/` |
| `docs/archive/TERNARY*.md` | `docs/history/` |
| `configs/ternary*.json` | `configs/` |
| final cascade/audit JSON evidence | `evidence/` |
| `final-ternary-bonsai-g32-v10/*` | `model/` |

The source service README is retained as
[`services/musicgen/README.source.md`](../services/musicgen/README.source.md)
to distinguish it from this extraction's canonical documentation.

## Included implementation families

The extraction includes the progression from early experiments to the final
contract:

- contract, adapter and runtime modules;
- direct, Hadamard and learned symmetric quantisers;
- block-local and sequential cascade trainers;
- target/cache/dataset preparation;
- audits, calibration, profiling and resource measurement;
- compact export, reload and round-trip verification;
- audio canary rendering and raw comparisons;
- focused ternary tests.

The historical filenames are kept because they are evidence of how the method
changed. The canonical entry points are listed in
[`REPRODUCTION.md`](REPRODUCTION.md).

## Explicitly excluded

The following remain outside the repository by design:

- the dense `dit_medium_f16.npz` teacher (about 2.9 GB);
- generated teacher/state/target caches;
- cumulative checkpoints and temporary Metal dumps;
- the unrelated `abelton` web/application code;
- unrelated audio datasets, private prompts and raw working directories;
- user-specific absolute paths except where preserved inside historical JSON
  evidence for provenance.

Absolute paths in copied evidence are historical breadcrumbs, not required
repository paths. The canonical package paths are relative to `model/`.

## Recreating the extraction

Start from a clean source checkout, inspect the matching filename patterns, and
copy only the files needed for a declared run. Do not copy every `output/`
directory into Git. Keep large intermediates in external storage and place
their SHA-256 and role in a small evidence manifest.

