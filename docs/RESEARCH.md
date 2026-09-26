# Research notes and lessons learned

This file is the short, navigable synthesis of the historical plans. The
complete records remain under [`docs/history/`](history/). It exists so a new
experimenter does not have to infer the failure modes from a pile of logs.

## The original question

Can the Stable Audio 3 Medium DiT be reduced using the Bonsai compromise: a
strict ternary attention/FFN core, tiny per-group scales, native support
tensors, and a model package small enough for a constrained Apple-Silicon
runtime?

The answer splits into two parts:

1. **Yes for the engineering/storage contract.** The final G32 package is
   ternary in the declared core, compact, reloadable and structurally audited.
2. **Not yet as a general quality equivalence claim.** The final automatic
   velocity gates remain below threshold, although the local listener accepted
   the canary for personal study.

## What changed across the attempts

### Early attempts: format and measurement ambiguity

The first runs mixed different layer predicates, affine ternarisation,
Hadamard conventions and reload paths. Several attractive numbers measured a
local layer or a re-encoded representation rather than the final diffusion
trajectory. The knowledge base therefore made the scope, dtype and metric
contract explicit before another full run.

### V6–V9: better contract, still too much optimism in the cascade

The later plans added teacher caches, student-forced states, independent
records, exact scope digests and memory guards. They also showed why a good
block cosine does not imply good audio. Repeatedly launching a 24-block run
before a single-block recipe had passed held-out gates consumed disk and
produced ambiguous checkpoints.

### V10: structurally successful, quality-gated failure

V10 fixed the packaging path and completed 24 blocks under the Bonsai scope.
Blocks 0–5 were direct symmetric; blocks 6–23 used symmetric Hadamard input.
The package was exported in a fresh process and reloaded exactly. The final
trajectory still landed at velocity mean `0.876934` / minimum `0.646171`.

The most important conclusion is that the quality failure is not a packing
failure. The packed package reconstructs its own exported reference exactly.
The accumulated student trajectory is the part that needs a better training
objective, held-out gating and possibly a different basis schedule.

## Rules for the next experiment

1. Freeze one canonical list of 168 core matrices.
2. Keep `W = s*q` strict; no unreported affine centre or dense residual.
3. Use the same quantiser in QAT, export, reload and inference.
4. Train and evaluate with the real student prefix, not only teacher forcing.
5. Pass one-block held-out velocity/audio gates before expanding the cascade.
6. Keep dense masters and working caches outside the release payload.
7. Checkpoint only the active block and the accepted manifest; do not duplicate
   a complete cumulative package at every step.
8. Report raw audio, trajectory, memory, hashes and exact scope together.
9. Do not convert a local listening acceptance into a blind benchmark claim.

## Evidence index

| Topic | Document |
|---|---|
| V10 implementation plan | [`history/TERNARY_QUALITY_RECOVERY_PLAN_V10.md`](history/TERNARY_QUALITY_RECOVERY_PLAN_V10.md) |
| Consolidated knowledge base | [`history/TERNARY_DISTILLATION_KNOWLEDGE_BASE.md`](history/TERNARY_DISTILLATION_KNOWLEDGE_BASE.md) |
| V9 evidence | [`history/TERNARY_V9_EVIDENCE_2026-09-25.md`](history/TERNARY_V9_EVIDENCE_2026-09-25.md) |
| Final structural audit | [`../evidence/final_audit_summary.json`](../evidence/final_audit_summary.json) |
| Final audio canary | [`../evidence/final_audio_canary.json`](../evidence/final_audio_canary.json) |
| Export/reload manifest | [`../model/manifest.json`](../model/manifest.json) |

