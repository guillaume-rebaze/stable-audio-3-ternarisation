# Method — Stable Audio 3 Bonsai-style ternarisation

This document is the canonical technical description of the extracted work.
It separates the mathematical representation, the training procedure, the
export format and the validation gates so that a future run cannot silently
change one of them.

## 1. Target and boundaries

The target is the **Stable Audio 3 Medium DiT** as loaded by the MLX runtime
used in the source project. The latent crop used by the experiment is `128`.
The text encoder, codec, tokenizer, application wrappers and the dense teacher
are dependencies; they are not part of the compact model payload.

The Bonsai-style scope is deliberately narrower than “every tensor is ternary”:

```text
24 transformer blocks
  × 7 attention/FFN projections per block
  = 168 ternary core matrices
```

The seven exact suffixes are:

```text
self_attn.to_qkv.weight
self_attn.to_out.weight
cross_attn.to_q.weight
cross_attn.to_kv.weight
cross_attn.to_out.weight
ff.ff.0.proj.weight
ff.ff.2.weight
```

The remaining **357 support tensors** stay in their native runtime
representation. This includes continuous conditioning tensors, normalisation
parameters, non-core projections and retained vector biases. The package
manifest is authoritative for the exact names and counts.

## 2. Ternary representation

For each core matrix, weights are represented as:

```text
W_hat = s * q
q     ∈ {-1, 0, +1}
```

`s` is one positive FP16 scale for each group of 32 input values. There is no
affine centre, LoRA residual, dense escape matrix or hidden repair adapter in
the published payload. The zero fraction is measured per matrix and retained
in `manifest.json`.

### 2.1 Two-bit packing

The storage alphabet uses a fourth code as a guard value:

```text
q = +1  → code 0
q =  0  → code 1
q = -1  → code 2
code 3  → reserved and rejected
```

Sixteen two-bit codes are packed into one `uint32` word. The unpacker validates
the reserved code, reconstructs the ternary array and applies the recorded
scales. This is why the payload is compact while the logical dense-equivalent
size is still reported separately.

### 2.2 Runtime scale convention

The mathematical scale is positive. The MLX linear runtime stores the physical
scale with its historical sign convention, so export and reload explicitly
convert between positive manifest scales and the runtime representation. This
conversion is part of the contract; it must not be reproduced by an ad-hoc
loader.

### 2.3 Hadamard members

The first six blocks use direct symmetric members. Blocks 6–23 use the
declared `symmetric_hadamard` mode. A block-input Hadamard transform is applied
before the ternary core and its inverse is represented by the runtime contract.
The ternary codes remain ternary in the declared transformed basis; the
documentation does not claim that the dense matrix in the original basis has
only three values.

The manifest records `hadamard_input` for every core member. A loader must use
that field rather than infer the mode from the block number.

## 3. Training procedure extracted from the source run

The final package was produced from a sequential, student-forced cascade:

1. Build or load teacher targets and state caches from the dense FP16 teacher.
2. Select one DiT block and one quantiser mode.
3. Initialise the block's master weights from the dense teacher.
4. Run QAT with a straight-through ternary path and the declared runtime
   surrogate.
5. Evaluate the reloaded full prefix on the same prompt, sigma and state
   contract.
6. Promote only the accepted record to the next block in a quality-gated run.
7. Keep the dense teacher available for the next active block; do not use a
   dequantised cumulative package as the only master source.
8. Export once, in a fresh process, from the final accepted records checkpoint.

The source driver is [`run_ternary_bonsai_cascade.py`](../services/musicgen/run_ternary_bonsai_cascade.py).
The core trainer and its exact quantiser/reload logic are in
[`train_ternary_quality.py`](../services/musicgen/train_ternary_quality.py)
and [`ternary_contract.py`](../services/musicgen/ternary_contract.py).

### 3.1 Recorded operating envelope

The cascade driver was configured with:

| Setting | Value |
|---|---:|
| Group size | 32 |
| Latent crop | 128 |
| State-cache budget | 512 states |
| Prompt budget | 16 |
| Cascade update steps | 256 per active block |
| Gradient accumulation | 2 for the on-policy path |
| Metal memory guard | 11,000,000,000 bytes |
| Default quantiser | learned symmetric Hadamard |
| State sampling | without replacement |
| Seed | 20260925 |

The final structural cascade used a pointwise path and retained its per-block
audit/canary evidence. The historical fallback rollout was expensive and was
not allowed to become an implicit quality guarantee. In particular, the
structural run used a promotion override to finish the package and record the
quality failure; that is why its release gate is explicitly false.

### 3.2 What is frozen and what is learned

The shipped artifact contains only the final `q`, scales, support tensors,
biases and metadata. During training, dense master parameters, optimisers,
RNG state, teacher targets and caches are working state. They are not silently
needed to load the compact package and are not copied into this repository.

The training records keep the core mode, group shape, scale arrays, packed
codes, bias overrides and provenance. A record checkpoint is a training
intermediate, not the release format.

## 4. Export contract

[`export_ternary_bonsai_package.py`](../services/musicgen/export_ternary_bonsai_package.py)
creates the following package:

```text
model/
├── payload.npz
├── manifest.json
├── reload_report.json
├── acceptance.json
├── README.md
└── audio_canary/
    ├── audio_canary.json
    ├── student.wav
    └── teacher.wav
```

`payload.npz` contains:

- packed core codes under `__ternary_core__/…packed_codes`;
- positive FP16 group scales under `__ternary_core__/…scales`;
- native support tensors;
- explicit bias members when the core matrix has a retained bias;
- no dense duplicate of a core matrix.

`manifest.json` records the schema, model dimensions, exact core/support
scope, mode per matrix, shape, group count, zero fraction, provenance hashes,
payload format and acceptance pointer. The teacher provenance hash is:

```text
f9e5647ea3225818657d47d47ae4b34afa29c0568206ca89566c1a758944a38e
```

The teacher file itself is about 2.9 GB and is intentionally excluded.

## 5. Fresh verification

[`verify_ternary_bonsai_package.py`](../services/musicgen/verify_ternary_bonsai_package.py)
performs the following independent checks:

1. Load the manifest and inspect every declared member.
2. Decode all packed codes and reject code `3` or values outside the ternary
   alphabet.
3. Reconstruct the core and support parameters in a new MLX model instance.
4. Compare the exact parameter inventory with the reference teacher model.
5. Run the package and reference model on the same seeded latent, sigma and
   crop.
6. Check finite output, shape, relative L2 and cosine thresholds.
7. Report the payload SHA-256 and the complete reload inventory.

The recorded payload hash is:

```text
1e2206181875d74c6a21e51120b3366ca5b9cadd038df47e157ea54d1ce4fa04
```

The independent package check passed with 168 core matrices, 357 support
tensors, 1,358,954,496 ternary codes, 732 exact parameters, relative L2 `0.0`
and cosine `1.0000000000000002`.

## 6. Non-goals

This method does not claim that:

- the dense teacher is redistributed;
- all Stable Audio 3 components are ternary;
- one local audio canary proves general musical equivalence;
- a good pack/unpack round trip proves a good diffusion trajectory;
- the current automatic release gate passed.

Those distinctions are part of the method, not caveats added after the fact.

