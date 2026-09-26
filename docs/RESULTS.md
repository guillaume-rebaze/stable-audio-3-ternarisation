# Results and acceptance status

## Executive verdict

The **Bonsai package contract passed**. The **local personal-study acceptance
passed**. The **strict automatic release-quality gate did not pass**.

That three-way verdict is the correct reading of the work. A model can be
properly ternary, compact, reloadable and pleasant in one listening session
without being proven equivalent to the teacher across prompts and diffusion
trajectories.

## 1. Structural package

| Check | Result | Evidence |
|---|---:|---|
| Core matrices | 168 / 168 | `model/manifest.json` |
| Support tensors | 357 | `model/manifest.json` |
| Ternary code count | 1,358,954,496 | `model/reload_report.json` |
| Ternary alphabet | `{-1, 0, +1}`; code 3 unused | manifest + verifier |
| Payload size | 513,342,833 bytes | `model/reload_report.json` |
| Logical payload | 613,897,248 bytes | V10 export report |
| 650 MB envelope | pass | `model/acceptance.json` |
| Parameter inventory | 732 expected / 732 actual | `model/reload_report.json` |
| Reload mismatches | none | `model/reload_report.json` |
| Fresh forward relative L2 | 0.0 | `model/reload_report.json` |
| Fresh forward cosine | 1.0000000000000002 | `model/reload_report.json` |

The forward parity check is a package/reconstruction check on the controlled
verification input. It proves that export and reload implement the same
function as the exported reference under that test, not that ternarisation is
lossless relative to the dense teacher in every trajectory.

## 2. Quality measurements

The final cascade audit reported:

| Metric | Value | Gate |
|---|---:|---|
| Velocity mean cosine | **0.8769339072** | candidate `≥0.90`; release `≥0.93` |
| Velocity minimum cosine | **0.6461712122** | candidate `≥0.80`; release `≥0.85` |
| Audio cosine | **0.8520558362** | published as measurement, not a release gate |
| Audio relative L2 | 0.5413566938 | diagnostic |
| Latent relative L2 | 0.3963405412 | diagnostic |
| Audio finite | true | technical pass |
| Sample rate | 44,100 Hz | matched |
| Duration | 11.8886 s | matched |

The velocity candidate and release gates are therefore both false. The
automatic trajectory result is retained, not rounded into a pass.

## 3. Human listening acceptance

Guillaume reviewed the teacher/student canary locally and accepted the sound
for personal study. The acceptance record explicitly says:

```text
accepted_local_personal_study
```

This was a single A/B review, not a blind panel, not a broad prompt sweep and
not a claim of production readiness. The raw WAVs remain available under
[`model/audio_canary/`](../model/audio_canary/), and the machine-readable
measurement is [`model/audio_canary/audio_canary.json`](../model/audio_canary/audio_canary.json).

## 4. Why the scores fell

The source investigation identified accumulation across the sequential cascade,
especially at later diffusion states. Pointwise block similarity and exact
packing were not sufficient to protect the terminal velocity field. The mixed
direct/Hadamard schedule, local surrogate loss and the deliberate
`--promote-on-quality-fail` forensic continuation are all recorded in the
historical V10 plan.

The failures are useful evidence:

- the storage contract was not the cause of the audio drift;
- reloading the package did not introduce a hidden mismatch;
- the remaining problem is quality of the learned ternary trajectory, not
  whether the file can be decoded;
- future work must gate each block on held-out trajectory and audio checks.

## 5. What may be claimed

Safe claim:

> This repository contains a Stable Audio 3 Medium DiT Bonsai-style package
> with a strict ternary core, native support tensors, compact G32 packing and
> exact fresh reload verification. It was accepted locally for personal study.

Unsafe claim:

> This is a fully ternary, production-quality, teacher-equivalent Stable Audio
> 3 model.

The second statement overstates both the scope and the measured quality and is
not made here.

