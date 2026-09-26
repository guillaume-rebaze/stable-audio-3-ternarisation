# Stable Audio 3 — Bonsai ternary G32

Compact model package for personal study. The full extraction and method are
documented in the [Git repository](https://github.com/guillaume-rebaze/stable-audio-3-ternarisation)
and the paired [Hugging Face model page](https://huggingface.co/guillaume-rebaze/stable-audio-3-ternarisation).

## Contenu

- `payload.npz` : 513 342 833 octets.
- `manifest.json` : scope, modes, provenance et hash.
- `reload_report.json` : vérification fraîche, paramètres exacts, forward exact.
- `acceptance.json` : acceptation locale après écoute humaine.
- `audio_canary/` : WAV teacher/student et mesures brutes.

Contrat : 24 blocs × 7 matrices ternaires, `q ∈ {-1,0,+1}`, G32, supports
natifs conservés. Les matrices indiquées `symmetric_hadamard` appliquent la
base Hadamard par groupe dans le runtime; elles restent ternaires dans cette
base déclarée.

## Vérifier

From the repository root:

```bash
rtk proxy python3 services/musicgen/verify_ternary_bonsai_package.py \
  --package-dir model \
  --teacher-weights /Users/guillaumegaillard/.cache/onus/stable-audio-3-mlx/optimized/mlx/models/mlx/dit_medium_f16.npz \
  --dataset-dir output/sample-expertise-pilot/ternary-quality-v3/g2-authorized-independent/train
```

The package is accepted locally for listening and personal study. The automatic
trajectory gates remain published in `audio_canary/audio_canary.json` and are
not replaced by that listening review.
