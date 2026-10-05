# xinfer evaluation, 2026-09-29

## Construction

- Model: Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca.
- Weights SHA-256: f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b.
- xinfer: b88c15334fb607ff52cdc3fd875c3da796dcc020; Candle fork: 23a6f38;
  attention.rs: c0f19f2. Full resolved dependency revisions are in
  experiments/xinfer/Cargo.lock.
- Hardware: NVIDIA GB10, aarch64, CUDA 13.0, driver 580.159.03.
  Rust 1.97.1. Model activations, weights and the ordinary cache use BF16.
- Corpus: PyTorch examples WikiText-2 test.txt at revision
  d5678bc8ac0cdd79dbd5e44d4130271018bcec4e.
  SHA-256: d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0.
  This distributed text is already pretokenized and contains unknown-word
  markers. It is not raw WikiText-2.
- Tokenize the complete text with the pinned model tokenizer, without special
  tokens or a chat template. Cases are token offsets 0 and 4096, each with
  prefix lengths 128 and 512. The two prefix lengths at each offset overlap;
  this is not four independent documents.
- There are 28 layers, 8 KV heads, 128 channels per head, 16-token pages.
  Export exactly the valid prefix, excluding unused page padding. Restore to
  newly allocated pages and zero the unused slots.
- For each case, prefill the prefix, export it, then feed 32 actual suffix
  tokens one by one and score each following reference token. The first suffix
  token is an unscored seed; original prefill logits are never reused as a
  prediction after restoring. All variants use identical teacher-forced
  histories. A separate 16-token greedy continuation starts from the same seed.
- The ordinary cache stays BF16 during continuation. For affine persistence,
  export to f32, group K along tokens and V along channels, group size 64,
  candidate bits 2/4/8, tail 16, outlier fraction 0.02. Sweep maximum absolute
  error 0, 0.05, 0.20. Restored values are cast back to BF16; the codec's f32
  error guarantee does not include this final BF16 rounding.
- Native TurboQuant quantizes both the prefix and newly generated tokens.
  Turbo4 uses 4-bit K/V; Turbo3 uses 3-bit K and 4-bit V. Float32 scales and
  packed codes are preserved without any lossy conversion. Modes run in
  separate processes because the upstream cache mode is global.
- Hash weights, configuration, tokenizer and engine-format identity for the
  address, and hash the complete causal prefix token IDs. Local benchmark
  references additionally bind the corpus and model identity. This experiment
  has no adapters, multimodal inputs, tensor parallelism, or nontrivial page
  mapping. Those would require additional identity and layout handling.

## Controls and failure investigation

The first model attempted was SmolLM2-135M. Repeated native calls disagreed even
before shifou was used. CUDA compute-sanitizer memcheck located an out-of-bounds
global read in flash_prefill_paged_128 with 64-element heads. The pinned
attention source instantiates native prefill variants at 128, 256 and 512
channels. The experiment now rejects head dimensions other than the tested 128;
it does not silently reinterpret a 64-channel model. The failing model's results
were discarded. No dependency source was patched.

Qwen3 BF16, Turbo4 and Turbo3 native export/restore smoke controls each produced
identical logits and zero CUDA memcheck errors. Set SHIFOU_SMOKE=1 and run the
normal executable arguments under compute-sanitizer --tool memcheck
--error-exitcode 99 to repeat this check.

For every measured case, exact affine restoration preserved source f32 bits,
and packed restoration preserved every code and scale byte. The packed path
also re-exported the uploaded GPU buffers and checked byte equality. After
closing and reopening the database, all 32 full logit vectors and 16 greedy
tokens were identical to their native-mode controls for zero-error affine,
Turbo4 and Turbo3. These engine controls close/reopen storage within a process;
the core CLI tests separately exercise storage across process boundaries.

## Findings

README.md owns the result tables and reproduction commands. The full measured
rows, engine identities, quality metrics and density counters are preserved in
xinfer-results.json. Scratch caches and logs are under
.agents-workspace/tmp/eval-qwen-v1 and .agents-workspace/tmp/.

The generic affine codec gives a small storage win over BF16 at error 0.20
when aggregated across these cases, with little change in this slice's NLL.
The exact-f32 path expands to about 2.13 times native BF16 size. This is an
important baseline correction: the original synthetic demo compared with f32.

Native Turbo4/Turbo3 persistence gives about 3.76x/4.26x smaller logical records
than BF16 and adds no observed inference error. Native quality is poor on this
model/slice, especially Turbo3. The experiment does not distinguish expected
quantization sensitivity from numerical issues in the native quantized kernels.
Passing a memory checker and an exact persistence control does not establish
quantized attention accuracy. An independent engine or attention oracle would
be needed to diagnose the native quality loss. Do not present these measurements
as reproduction of the TurboQuant paper's results.

## Density measurement

experiments/xinfer/src/bin/density.rs reopens the measured databases without
running inference. It reports set-bit cardinality divided by logical span,
including zeros at the end of buffers. Sentinel bits used to encode byte
length are excluded. Counts are summed before division across all layers and
cases, so long prefixes receive proportionally greater weight.

For each native buffer, the instrument also constructs two optimized yesno
OrdSets: original bit order, and code bit planes. Plane j contains bit j of
every code in token/head/channel order. K uses 3 bits for Turbo3 and 4 for
Turbo4, V uses 4, and scale words use 32 little-endian bits. Reported plane
densities are little-endian, least-significant plane first. No padding is needed
for the measured 128-channel heads. The transposition experiment serializes
each buffer separately; it is distinct from the actual stored snapshot, which
concatenates all native buffers before encoding. Per-buffer container rounding
explains why scale-only Roaring records can exceed packed bytes.

Actual native code planes are dense and transposition gives no measured code
size reduction. V scales have zero low 16 bits in this BF16 experiment and offer
a small lossless optimization opportunity. Density alone is not entropy, and
these measurements do not rule out other compressors or orders. No speculative
compression policy was added based only on density.

The affine reports include code bits, exception masks, and exact f32 exceptions.
Aggregate densities are 27.448% at zero error, 37.006% at error 0.05, and 37.700%
at error 0.20. Payloads are slightly larger than densely packed equivalents:
293,892,480 versus 293,587,332 bytes; 111,297,334 versus 110,534,576;
82,096,451 versus 81,363,868. These figures exclude reconstruction metadata.

## Timing and allocation limits

Every write waits for visibility and checkpoints. The packed snapshot writes
all layers atomically; the affine experiment checkpoints each of 56 tiles.
Export, persist, reopen/read and GPU reconstruction are timed separately.
Model loading is excluded and kernels are warmed up. GPU boundaries synchronize.
Read timing includes validation and decoding; TurboQuant restore timing includes
a GPU re-export for verification. CPU staging constructs ordinal vectors and
makes several copies. Some measurements overlapped sanitizer or lint activity,
so these single observations are not throughput estimates.

At these short prefixes, native prefill was faster than persistent reload.
Do not claim a latency benefit. Native packed byte totals include scales but
exclude GPU page padding, temporary tensors, model weights and allocator state.
Tiny unused standard-cache placeholders remain for the xinfer API. GPU peak
memory was not measured. Filesystem allocated bytes and sparse-file apparent
bytes are reported separately from logical portable Roaring lengths.

## Remaining integration scope

This is a single-model, single-sequence Qwen3 experiment, not an installed
xinfer server plugin. The generic storage schema accepts other adapters.
Turbo8 requires preserving its standard FP8 state in addition to side buffers;
it is deliberately rejected here. MLA, multi-GPU shards, arbitrary block maps,
concurrent serving, calibrated TurboQuant mode selection and mixed per-layer
TurboQuant modes are not implemented. A separate affine-only per-layer
calibration experiment is documented in adaptive-quantization.md. The upstream mode is global, so the adapter must not
pretend to choose a different TurboQuant mode independently per stored layer.
