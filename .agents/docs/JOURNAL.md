# Journal

## 2026-09-29: Initial persistent tensor-cache prototype

Created a standalone Rust crate with a path dependency on the sibling
`yesno-core`. No model-specific code or research instrument was added to yesno.

The shared contract is a finite row-major f32 tensor with named axes, a cache
address supplied by the producer, and an explicit quantization policy.
KIVI ( https://arxiv.org/abs/2402.02750 ) motivated token-axis grouping for keys,
channel-axis grouping for values, and an exact recent-token region. Adaptive
2/4/8-bit selection, sparse outlier exceptions and exact f32 fallback are
prototype choices, not a claim to implement or validate KIVI.

Payload and metadata are stored as two yesno sets in one atomic batch. This
avoids a separate sidecar publication protocol. The adapter's full address is
checked after the 63-bit database-key lookup. Tests force a collision to prove
that it cannot return another model's data or overwrite it.

The first reopen tests exposed floating metadata fidelity: checksumming a
reserialized JSON structure did not reproduce the original bytes. Checksums
now cover the original byte sequence before JSON decoding; serde_json's
float_roundtrip feature preserves f64 reconstruction parameters on input.
Binary group records preserve minimum and step values directly as IEEE bits.

The first measurement also exposed metadata cost. Full JSON group records
erased the payload savings. The persisted format now uses 29 bytes per group
( u64 ordinal offset, u8 bit width, f64 minimum, f64 step, u32 exception count ),
with a small JSON header for shape, axes, policy and identity. Local selection
counts standalone Roaring payload bytes plus those 29 bytes. Final measurement
serializes the combined sets; local costs are not a global optimality claim.

### Reproduction

From the project root:

```sh
cargo run --locked -- demo .agents-workspace/tmp/demo
```

The construction is deterministic. Model A uses shape `[64, 2, 64]` and axes
`[token, head, channel]`; model B uses `[3, 96, 32]` and
`[head, token, channel]`. For row-major index i, the value is 30 when i is
congruent to 0 modulo 251, otherwise `( ( i * 17 modulo 101 ) - 50 ) / 50`
in f32. Each model produces a key and a value tile. The absolute error bound
is 0.08, group size 32, candidate widths 2/4/8, outlier fraction 0.02 per tail,
and exact residual region 16 tokens. All records are read after closing and
reopening the database.

| Tile | f32 bytes | Payload bytes | Complete logical record bytes | Max absolute error |
| --- | ---: | ---: | ---: | ---: |
| A key | 32768 | 24620 | 32840 | 0.06667 |
| A value | 32768 | 16420 | 24640 | 0.06667 |
| B key | 36864 | 16420 | 26591 | 0.06400 |
| B value | 36864 | 16420 | 26144 | 0.06667 |

These are portable Roaring logical sizes including actual encoded metadata,
not allocated database bytes. A key keeps 128 of its 256 groups exact because
a 32-token group intersects the 16-token tail. The example deliberately keeps
that unfavorable short-tile case visible. Synthetic tensor reconstruction says
nothing about real-model accuracy or GPU throughput.

### Verification

All 15 shifou tests passed, including two property tests configured for 72
cases each, subprocess put/get, reopen/replace/delete, model and namespace
isolation, forced digest-bucket collisions, descriptor truncation, payload
corruption, and Roaring container-boundary reconstruction.

`cargo clippy --workspace --all-targets --all-features --offline -- -D warnings`
and `cargo fmt --check` passed in shifou. The dependency gate
`cargo test -p yesno-core --offline` passed in the sibling yesno checkout.
The exact README put/get/remove commands were also exercised; the 16-value
example reopened with maximum absolute error about 0.00000010.

## 2026-09-29: xinfer, TurboQuant, and real-model density

Added a generic lossless PackedSnapshot API and a separate pinned CUDA xinfer
experiment. It persists Turbo4/Turbo3 packed codes and scale tensors without
requantization, validates engine format, and restores native GPU buffers.
Qwen3-0.6B controls matched all native logits and greedy tokens after reopening.
SmolLM2 was rejected after a native 64-head-dimension prefill out-of-bounds read
was found with CUDA memcheck; Qwen3 128-channel controls passed all three modes.

Native codes were roughly half set bits. Bit-plane transposition provided no
Roaring code compression; actual packed payloads grew about 0.10%, with complete
records growing 0.16-0.37% over ordinary packed TurboQuant arrays. Scale planes
offer limited additional lossless opportunities. The 3.76x/4.26x savings against
BF16 come from native quantization, whose quality degraded considerably here.
Short-prefix reload was slower than native prefill. See xinfer-evaluation.md
for the construction and xinfer-results.json for durable measured rows.

Validation: shifou workspace clippy with all targets/features and denied
warnings, formatting, and all 19 tests passed. The standalone xinfer experiment
passed the same clippy/format checks. yesno-core regression tests and doctests
passed. The fetch script verified every pinned model/corpus checksum locally.
CUDA memcheck reported zero errors for Qwen3 auto, Turbo4 and Turbo3 smoke runs.

## 2026-09-29: calibrated affine KV precision

Added exact pre-write tile size estimates to shifou and a Qwen3 experiment
that calibrates each layer's K and V sensitivity to four error bounds on a
separate prefix, then selects within a uniform-error byte budget. Reopened
adaptive records matched pre-persistence inference exactly. Held-out
perplexity improved marginally, but KL and top-token agreement worsened at
both 128 and 512 tokens. The isolated sensitivity sum predicted the
opposite KL direction; layer interactions matter. The construction and
measurements are in adaptive-quantization.md and adaptive-results.json.

## 2026-09-29: joint calibration did not generalize

A follow-up selector accepted only whole-cache KL improvements on the
calibration prefix and searched 40 budget-safe policy changes in each of two
rounds. Calibration KL improved substantially, but held-out KL and top-token
agreement worsened for both prefix lengths. Perplexity improved only at the
512-token length. This is a negative result for one short calibration
continuation, not evidence against adaptive quantization generally. The
construction and measured choices are in adaptive-quantization.md and
adaptive-joint-results.json.
