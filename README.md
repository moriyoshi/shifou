# shifou

An experimental model-independent tensor cache built on yesno bit planes.
It compresses finite f32 tensor tiles, persists the payload and reconstruction
metadata atomically, and restores them after reopening. It also preserves native
quantized buffers losslessly. A separate CUDA experiment connects xinfer's
Qwen3 and TurboQuant kernels to the storage layer.

## Run

Requires Rust 1.95 or later and the sibling checkout `../yesno/yesno-core`.
The lockfile records this prototype's dependencies. Build outputs and demo data
stay under `.agents-workspace/tmp/`.

```sh
cargo run -- demo .agents-workspace/tmp/demo
cargo run -- put .agents-workspace/tmp/example examples/request.json
cargo run -- get .agents-workspace/tmp/example examples/address.json
cargo run -- remove .agents-workspace/tmp/example examples/address.json
```

The demo writes four synthetic K/V tiles using two shapes and axis orders,
closes the database, reopens it, and checks every reconstructed value against
the configured error bound. `get` prints a tensor and storage report, or JSON
`null` on a cache miss. The example address contains placeholder fingerprints.

## Common model boundary

A `Tensor` is a contiguous row-major f32 array with named axes. Axis order,
head count, channel width and token count are data, not model-name branches.
Any producer that supplies this schema can use the library or JSON CLI.
An engine adapter must gather its tensors into this order and scatter restored
values back into the engine's layout. Each cache entry stores one tile; the
caller assigns distinct slots to multiple tiles of the same layer and prefix.

An `Address` contains namespace, model/computation fingerprint, full causal
input-prefix fingerprint, layer and slot. The adapter must include weights,
adapters, attention/position settings, non-text inputs and relevant numerical
compatibility in those identities. shifou does not derive or verify these
fingerprints. Shared infrastructure does not make tensors interchangeable
between models. There is exact-address lookup, not longest-prefix search.

The library exports `encode`, `decode`, `Tensor`, `Policy`, `Address`, and
`Cache::{open, put, get, remove, estimate, estimate_encoded}`. Model-specific imports belong outside the
codec. Tensor and policy validation limits tiles to 16M elements, eight axes,
and 262144 groups.

## Adaptive codec

1. Partition the selected named axis into groups while fixing all other axes.
2. Try the configured subset of 2-, 4-, and 8-bit affine quantizers.
3. Optionally clip a fraction from each end of the group's range. Preserve
   excluded values exactly using an exception-position bitmap and f32 words.
4. Reject candidates that exceed the per-element absolute error bound.
5. Choose the smallest standalone encoded candidate, counting its portable
   Roaring payload and 29-byte group descriptor. Exact f32 is always a candidate.
6. Store the selected codes as bit planes in one ordinal set. An exception
   mask and exact exception words follow a group's planes when needed.

This is local size selection under a numeric error constraint. It is not a
calibrated layer-sensitivity policy or a global byte-budget optimizer. Local
candidate sizes do not add up exactly to the merged representation because
Roaring container boundaries change. Final reports measure the merged payload.
A zero error bound requires exact f32 bit patterns, including signed zero.

`Policy::keys` groups across the `token` axis, preserving each channel's own
range. `Policy::values` groups across `channel`, preserving each token's own
range. Both default to groups of 32 and retain groups touching the final 16
tokens as exact f32. Thus a group crossing the tail boundary can retain more
than 16 tokens. Grouping and axis names remain configurable.

These defaults are inspired by [KIVI](https://arxiv.org/abs/2402.02750), which
motivates different grouping for K and V and retaining recent cache state at
higher precision. This prototype also uses local bit-width selection and optional
sparse exceptions. It does not reproduce KIVI's CUDA kernels, streaming cache
algorithm, FP16 storage, or accuracy results.

## Prior work and research direction

Research notes from 2026-09-29, expanded 2026-10-04. This is a selected
reading list for the prototype, not an exhaustive survey or a claim of novelty.

| Work | Relevant contribution | Relationship to shifou |
| --- | --- | --- |
| [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) ( ICML 2024 ) | Quantizes keys per channel and values per token, with a higher-precision residual cache for recent state. | Informs the grouping axes and exact tail. The streaming algorithm and inference kernels are not implemented. |
| [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://github.com/SqueezeAILab/KVQuant) ( NeurIPS 2024 ) | Combines per-channel and pre-RoPE key quantization, sensitivity-informed nonuniform codebooks, and dense-and-sparse representations that preserve outliers separately. | Prior art for the low-bit payload plus sparse exceptions. shifou uses affine quantization and does not implement its calibration, codebooks, or pre-RoPE integration. |
| [MiKV: No Token Left Behind](https://arxiv.org/abs/2402.18096) ( 2024 ) | Keeps important KV pairs at higher precision and retains less important pairs at lower precision instead of discarding their information. | A direction for an importance-aware retention policy. The current exact tail is positional and has no attention-importance estimator. |
| [KVTuner](https://arxiv.org/abs/2502.04420) ( ICML 2025 ) | Searches offline for layer-specific K/V precision pairs using quantization sensitivity and resource objectives, then uses the configuration during inference. | A precedent for model-calibrated precision allocation. The core codec uses each group's reconstruction error and encoded size; the xinfer paper comparison below tests a calibrated layerwise policy. |
| [AQUA-KV: Cache Me If You Must](https://proceedings.mlr.press/v267/shutova25a.html) ( ICML 2025 ) | Uses compact learned predictors to exploit dependencies between keys, values, and layers, then quantizes the unpredictable residual. | Suggests an optional predictive codec. No predictors or cross-layer reconstruction dependencies exist in shifou. |
| [RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory](https://arxiv.org/abs/2605.06675) ( 2026 preprint ) | Fits distortion curves for the chosen quantizer and allocates precision across attention heads under a budget. | The xinfer paper comparison below fits this codec's distortion slope and tests a forward-only head-sensitivity proxy under actual record-byte budgets. |

[Quantize What Counts: More for Keys, Less for Values](https://arxiv.org/abs/2502.15075),
initially titled *More for Keys, Less for Values: Adaptive KV Cache
Quantization*, also studies asymmetric K/V bit allocation. It supports
investigating separate precision budgets for K and V rather than assuming that
both should have the same width. shifou already accepts separate policies but
does not derive their budgets from model sensitivity.

The papers use several meanings of adaptive: choosing the quantization axis,
assigning bit widths by layer or token importance, and learning predictors or
codebooks. shifou's core codec implements local bit-width selection and optional
outlier preservation under a numeric error bound. The xinfer experiments
also test model-calibrated allocation policies. These are different from
calibrating the effect on attention or generated answers.

The following systems precede shifou's persistence and restore work:

| Work | Earlier approach | Relationship to shifou |
| --- | --- | --- |
| [Prompt Cache](https://arxiv.org/abs/2311.04934), [SGLang RadixAttention](https://arxiv.org/abs/2312.07104), and [vLLM automatic prefix caching](https://docs.vllm.ai/en/v0.17.0/design/prefix_caching/) | Reuse computed attention state for repeated prompt segments or token prefixes. | shifou stores a caller-addressed state bundle. It does not find the longest reusable prefix, schedule reuse, or assemble arbitrary prompt modules. |
| [CacheGen](https://arxiv.org/abs/2310.07240) | Compresses and streams KV tensors, adapting compression to transfer bandwidth. | shifou chooses each local codec candidate by encoded size under a reconstruction-error bound. It has no bandwidth-aware policy or streaming decoder. |
| [Mooncake](https://arxiv.org/abs/2407.00079), [LMCache](https://arxiv.org/abs/2510.09665), and [SGLANG-LSM](https://arxiv.org/abs/2511.16138) | Move or persist KV across memory and storage tiers; SGLANG-LSM applies a database-style layout to large cache objects. | shifou publishes exact-address pages atomically through yesno and checks bundle integrity. It has no serving-engine connector, tier scheduler, or prefill/decode orchestration comparable to these systems. |
| [SGLang hybrid prefix caching](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/) | Reuses attention KV and Mamba recurrent checkpoints at a matching prefix boundary. | shifou's Nemotron and Qwen bundles also keep both state types at one boundary and restore them into another model instance. shifou has no hybrid radix tree or engine-managed eviction. |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) and [Fiddler](https://arxiv.org/abs/2402.07033) | Cache and prefetch routed experts, or execute some experts on the CPU to avoid transfer costs. | shifou now stores individual expert snapshots and xinfer loads selected experts. Neither component predicts future routes, prefetches weights, or manages a host hot tier automatically. |
| [TurboQuant](https://arxiv.org/abs/2504.19874) | Applies online vector quantization and a one-bit residual transform for low-distortion KV representations. | shifou can persist xinfer's native packed TurboQuant buffers losslessly; its generic affine bit-plane codec does not implement the TurboQuant transform or kernel. |

A [2026 external KV-cache study](https://arxiv.org/abs/2609.11744) also finds
that loading can lose to recomputation for short prefixes or fast GPUs. That
break-even matters here, but must be measured in release mode: on one GB10,
Nemotron 9B's complete verified cache hit took 0.725 s at 8 tokens versus
0.484 s to prefill, tied fresh prefill at 20 tokens, and took 0.744-0.749 s
at 512 tokens versus 10.200-10.551 s to prefill. The earlier 14-second read
was a debug-build result. Qwen3-0.6B BF16 also favored recomputation for the
tested long prepared prompts, while its NVFP4 checkpoint favored restoration.
The [measurement construction](.agents/docs/prefill-cache.md) uses separate
model instances and exact continuation-logit checks.

The research question for this prototype is whether a persistent bit-plane
representation improves total storage size, selective loading, or cache
management compared with ordinary packed quantized arrays. Quantization and
sparse exceptions themselves already have precedent. Useful follow-up work is:

- Compare against packed arrays at matched precision and reconstruction error,
  counting scales, exception values, masks, descriptors, and database overhead.
- Evaluate generation while actually restoring compressed caches from at least
  two models; tensor error alone does not establish model quality.
- Explore calibrated precision allocation using measured encoded sizes.
  Dynamic refinement needs retained extra information or recomputation;
  discarded bits cannot be recovered by promoting a cache entry. Independently
  learned low- and high-precision codebooks need not share a nested bit encoding.
- Measure model-specific restore versus recomputation under warm, cold, and
  remote-cache conditions, then admit prefixes only where the expected reuse
  pays for publication and resident bytes. Nemotron crossed near 20 tokens in
  the local warm-cache probe; repeated prefill measurements put the conservative
  same-host Flight admission point near 64 tokens. Neither is a universal
  threshold.
- A bounded prepared-host tier now admits entries by expected reuse and
  resident bytes. On a seven-request Qwen3-0.6B peer trace, the 256 MiB and
  512 MiB limits reduced median request sums from 1,376 ms to 876 ms and
  524 ms, respectively, with exact continuation logits. The tier is currently
  a caller-managed primitive; route-aware prefetch for MoE experts remains to
  be tested. The real 30B file-backed path was far slower than host-preimported
  restore. See the [construction](.agents/docs/prefill-cache.md).

## Persistence and integrity

Each address hashes to a pair of numeric yesno keys. One holds tensor bits;
the other holds the descriptor encoded as a bit vector. Both are replaced or
deleted in one `WriteBatch`. Every mutation waits for visibility and checkpoints
before returning. `get` loads both keys from one snapshot.

Metadata has a version marker, a SHA-256 checksum over the exact stored bytes,
a JSON header, and fixed-width binary group descriptors. The descriptor also
contains a checksum of the payload's portable Roaring serialization. The full
address is checked on lookup: a collision in the 63-bit key bucket is a miss
on read and an explicit error on write, never a silent alias. Collision
resolution is outside this prototype.

The directory has an exclusive file lock. One process owns a cache at a time;
mutations require `&mut Cache`. There is no concurrent daemon or network API.
Persistence, WAL recovery, and atomic batches come from the yesno dependency.
The tests exercise reopening and process boundaries, not a new crash matrix.

## Measurements and limits

The report distinguishes:

- `raw_f32_bytes`: source values at four bytes each;
- `payload_bytes`: the portable Roaring serialization of the tensor bit planes;
- `payload_set_bits` and `payload_bit_len`: exact set-bit count and logical span;
- `packed_payload_bytes`: the same codes, masks and exceptions densely packed;
- `descriptor_json_bytes`: a diagnostic JSON rendering of codec metadata,
  not the actual persisted metadata size;
- `metadata_roaring_bytes`: the actual metadata bit vector's serialization;
- `total_roaring_bytes`: payload plus metadata serialization.

These are logical encoded sizes. They exclude database index pages, WALs,
preallocated slabs, allocator slack, and in-memory temporaries. They are not a
claim about filesystem usage or GPU memory. Tiny tiles can grow after encoding.
The real-model results below apply to one small model and a short corpus slice.
No speedup or universal compression ratio is claimed.

In the checked synthetic demo, three of four complete logical records are
smaller than f32; the short key tile with half its groups retained exactly is
slightly larger. See `.agents/docs/JOURNAL.md` for the construction and numbers.

The codec accepts finite f32 only. NaN/Inf are rejected. Quantization error is
not an attention-output or generation-quality bound. The xinfer experiment below supplies a BF16 adapter and restores native packed
TurboQuant buffers. Further engine integration, append/eviction policy,
global resource budgets, and a production serving connector remain follow-on
work. shifou itself supplies no GPU quantization kernel. The on-disk format
and public API are experimental.

## Native TurboQuant persistence and Qwen3 experiment

[TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874)
studies online vector quantization for distortion and inner-product preservation.
The runnable experiment now pins
[xinfer `17499e4`](https://github.com/moriyoshi/xinfer/commit/17499e450a174e25be333f88c654ff6743fd4465);
the original TurboQuant measurements below used
[`b88c153`](https://github.com/guoqingbao/xinfer/tree/b88c15334fb607ff52cdc3fd875c3da796dcc020).
Its codec details and quality must be evaluated separately from the paper.
In that implementation, `turbo4` stores 4-bit K and V codes;
`turbo3` stores 3-bit K and 4-bit V codes. Both retain float32 scale tensors.
`turbo8` is not supported by this adapter.

`PackedSnapshot` stores named, typed, shaped byte buffers in one atomic record
through `Cache::{put_packed, get_packed, remove_packed}`. It preserves codes and
scales exactly, including their byte order. There is no second lossy
quantization step. The codec/version identifier and computation fingerprint
prevent interpreting another engine format as this one. The adapter gathers
valid tokens in logical order and rebuilds contiguous native pages on restore;
physical GPU pointers and page IDs are not persisted. All layers are restored
before continuation. Limits are 64 MiB of payload and 4096 buffers per snapshot.
The generic storage API has no xinfer dependency.

The runnable experiment is in `experiments/xinfer/`, with pinned dependencies.
It requires an NVIDIA GPU and CUDA development tools; the recorded run used
GB10 and CUDA 13.0. Fetching the model downloads about 1.5 GB. Run from this
directory, choosing a fresh result directory:

```sh
python3 experiments/xinfer/fetch.py
CUDA_COMPUTE_CAP=121 cargo build --release --locked --manifest-path experiments/xinfer/Cargo.toml
for mode in auto turbo4 turbo3; do
  .agents-workspace/tmp/target/release/shifou-xinfer \
    .agents-workspace/tmp/models/Qwen3-0.6B \
    .agents-workspace/tmp/corpora/wikitext-2-test.txt \
    .agents-workspace/tmp/eval-qwen "$mode"
done
.agents-workspace/tmp/target/release/density \
  .agents-workspace/tmp/models/Qwen3-0.6B \
  .agents-workspace/tmp/corpora/wikitext-2-test.txt \
  .agents-workspace/tmp/eval-qwen \
  .agents-workspace/tmp/eval-qwen/density.json
```

Set `CUDA_COMPUTE_CAP` for your GPU. The experiment deliberately accepts Qwen3
with 128-element heads: a CUDA memory check found an out-of-bounds read in the
pinned native prefill kernel when initially trying SmolLM2 with 64-element heads.
Qwen3 BF16, Turbo4 and Turbo3 smoke controls all passed CUDA memory checking.
The adapter is single-sequence, single-GPU, with 16-token pages. xinfer owns
global TurboQuant state, so each mode runs in its own process.

Measured on **Qwen3-0.6B**, four WikiText-2 test slices: prefix lengths 128 and
512, token offsets 0 and 4096, 32 scored continuation tokens per slice. This
is 128 scored tokens, not full-corpus perplexity or a downstream task benchmark.
The tokenizer consumes the pinned PyTorch-distributed, pretokenized test text
without a chat template. The generic affine codec compresses the prefix once;
native TurboQuant also quantizes new tokens during decoding, so their quality
columns compare different runtime policies.

| Cache path | BF16 bytes / complete logical record | Subset perplexity | Top-1 agreement with BF16 |
| --- | ---: | ---: | ---: |
| BF16 reference, exact f32 persistence | 0.470x | 25.934 | 100% |
| Affine, absolute error 0.05 | 1.013x | 26.097 | 95.31% |
| Affine, absolute error 0.20 | 1.268x | 25.902 | 95.31% |
| Native Turbo4, lossless persistence | 3.757x | 56.864 | 57.81% |
| Native Turbo3, lossless persistence | 4.258x | 4340.362 | 11.72% |

Ratios aggregate source and stored bytes; values below 1 mean expansion.
Perplexity is exp of the mean token NLL. The exact affine control and both
TurboQuant modes produced **identical native/restored logits and greedy token
sequences** after closing and reopening storage, in every case. Persistence
therefore added no observed quality loss. Native TurboQuant degraded this model
substantially, especially Turbo3; these results do not justify a low-loss claim.

### Bitvector density

Measured set bits divided by the full logical bit span, weighted by element
count across all layers and cases:

| Buffer | Turbo4 density | Turbo3 density |
| --- | ---: | ---: |
| K codes | 50.003% | 46.202% |
| V codes | 50.026% | 50.010% |
| K float32 scales | 42.224% | 42.298% |
| V float32 scales | 21.130% | 21.045% |
| Whole packed payload | 48.936% | 47.265% |

Transposing codes into separate bit planes did not reduce their measured
Roaring size. Turbo3 K planes had densities 51.4%, 50.1%, and 37.1%;
Turbo4 K/V planes were all about 50%. Code density alone does not prove
incompressibility, but the tested Roaring layouts provided no code compression.

All lower 16 bits of V float32 scales were zero in this BF16 run. Transposing
these scales reduced the summed per-buffer Roaring sizes from 1,379,840 to
920,640 bytes per mode, versus 1,146,880 packed bytes. That suggests a separate,
lossless scale representation. The stored format currently preserves native
bytes unchanged; it does not implement that optimization.

The actual combined Roaring payload was about **0.10% larger** than ordinary
packed TurboQuant arrays; including the manifest made it **0.16–0.37% larger**.
The 3.76x/4.26x reduction is supplied by TurboQuant, while yesno supplies
persistence and integrity. These dense codes favor a dense representation.

CPU staging and checkpointing remain expensive. In these runs, reopening,
reading and uploading TurboQuant state took about 42–197 ms, while native
prefill took 13–32 ms. Persisting took 243–1021 ms. These are one-shot instrument
timings, include copies and correctness checks, and some runs overlapped other
checks; they establish no throughput benefit. Logical records exclude database
allocation. The measured TurboQuant databases allocated about 3.55–15.87 MB,
while sparse preallocated files had apparent size about 1 GiB.

Full construction, limitations, and measured rows are in
[the evaluation note](.agents/docs/xinfer-evaluation.md) and
[the numeric results](.agents/docs/xinfer-results.json).

## Adaptive quantization

The xinfer experiment now calibrates each layer's K and V precision using a
separate 128-token WikiText prefix. It measures the next-token KL change when
one tile at a time is quantized with a maximum f32 error of 0, 0.05, 0.20, or
0.50. For a new prefix, `Cache::estimate_encoded` gives each option's exact
logical bytes. A greedy selector spends no more than a uniform 0.20 policy,
buying the greatest calibrated KL reduction per extra byte. This model-aware
policy is outside the generic codec and does not switch xinfer's global
TurboQuant mode per layer.

| Held-out prefix | Uniform 0.20 bytes | Adaptive bytes | Uniform perplexity | Adaptive perplexity | Uniform KL | Adaptive KL | Top-1 agreement, uniform/adaptive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 tokens | 16,254,968 | 16,254,281 | 39.098 | 38.997 | 0.01442 | 0.01629 | 100% / 90.63% |
| 512 tokens | 41,572,621 | 41,520,261 | 64.732 | 64.552 | 0.01015 | 0.01088 | 93.75% / 90.63% |

The held-out prefixes start at token offset 4096; the eight-token calibration
continuation starts at offset 0. The two held-out prefixes overlap. All selected
tiles survived close and reopen with identical reconstructed bits and
continuation logits. Adaptive selection slightly improved likelihood on these
64 scored tokens, but KL and top-token agreement worsened. Individual tile KL
scores did not predict the quality of their joint configuration. This does not
establish a reliable model-quality gain.

Calibration took 25 seconds. Candidate encoding and exact byte costing took
15–68 seconds per prefix on the recorded machine, before storage. This is an
offline research path. Run it with the pinned model and corpus from the xinfer
experiment using a fresh output directory:

```sh
.agents-workspace/tmp/target/release/shifou-xinfer \
  .agents-workspace/tmp/models/Qwen3-0.6B \
  .agents-workspace/tmp/corpora/wikitext-2-test.txt \
  .agents-workspace/tmp/adaptive-qwen adaptive
```

A second mode accepts policy changes only when they improve the **combined**
cache on the calibration prefix, then tests up to 40 budget-safe single or
paired tile changes in each of two rounds. It still uses the same eight-token
calibration continuation. Joint calibration KL improved from 0.00897 to
0.00490 (128-token policy) or 0.00542 (512-token policy), but held-out KL
again worsened:

| Prefix | Uniform KL | Joint adaptive KL | Uniform perplexity | Joint adaptive perplexity |
| --- | ---: | ---: | ---: | ---: |
| 128 tokens | 0.01442 | 0.01784 | 39.098 | 39.335 |
| 512 tokens | 0.01015 | 0.01033 | 64.732 | 63.629 |

These policies stayed below the same uniform byte budgets. The 512-token
policy helped likelihood, while both reduced top-token agreement. Repeated
search on one short calibration continuation likely overfits; broader
calibration and a separate validation set are needed before using these
policies for serving. Reproduce it with the same command, replacing
`adaptive` with `adaptive-joint` and choosing another fresh output directory.

The construction and limitations are in
[the adaptive note](.agents/docs/adaptive-quantization.md), with full
[isolated-score results](.agents/docs/adaptive-results.json) and
[joint-search results](.agents/docs/adaptive-joint-results.json).

## Paper-inspired quantizer comparison

A separate xinfer mode requests fixed 2-, 4-, or 8-bit affine codes from
shifou's codec. It compares [KIVI](https://arxiv.org/abs/2402.02750)
style asymmetric K/V grouping at 2 bits, a
[KVTuner](https://arxiv.org/abs/2502.04420) style layerwise K/V precision
search, and a [RateQuant](https://arxiv.org/abs/2605.06675) style per-head
rate-distortion allocation. K4V2, uniform 4-bit, and a dispersed 2/4-bit
head allocation provide references. All use group size 32, a 32-token exact
tail, and no outlier exceptions. The codec may retain a group as f32 if that
is smaller than its requested low-bit encoding. This is a comparison of
paper-inspired policies on shifou's affine bitvector format, not a
reproduction of the authors' kernels or published accuracy.

On Qwen3-0.6B, calibration used a 128-token prefix at WikiText offset 0
and eight continuation tokens. Held-out prefixes started at offset 4096.
Each result scores 32 teacher-forced tokens; KL and top-1 agreement are
against the BF16 cache. The table reports **complete logical records**
including portable Roaring payloads and metadata, in decimal MB.

| 512-token prefix | MB | KL from BF16 | Subset perplexity | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| BF16 cache (raw bytes) | 58.72 | 0 | 63.616 | 100% |
| KIVI-style 2-bit | 40.59 | 0.08639 | 58.000 | 87.50% |
| Fixed K4V2 | 44.04 | 0.02120 | 64.457 | 81.25% |
| Uniform 4-bit, layer records | 47.48 | 0.00299 | 63.571 | 96.88% |
| KVTuner-style layer search | 47.11 | 0.00323 | 63.651 | 96.88% |
| Dispersed 2/4-bit heads, middle budget | 46.10 | 0.06375 | 56.434 | 81.25% |
| RateQuant-style 2/4-bit heads, same budget | 46.10 | 0.04300 | 61.875 | 87.50% |
| Uniform 4-bit, per-head records | 49.85 | 0.00299 | 63.571 | 96.88% |

The two middle-budget head policies each chose 224 2-bit and 224 4-bit
heads. Rate-based selection reduced KL by about 33% against dispersed
selection at nearly the same byte count, but had worse subset perplexity.
At the uniform 4-bit budget, the rate-based selector chose 4 bits for all
448 heads. The KVTuner-style search chose three 2-bit and 53 4-bit
layer/slot tiles at 512 tokens. Its whole-cache calibration KL improved
from 0.00271 to 0.00210, while held-out KL worsened slightly relative to
uniform 4-bit. This calibration slice is too small to establish a reliable
model-quality advantage.

At 128 tokens, the KIVI-style 2-bit records used 15.05 MB versus 14.68 MB
of raw BF16; their KL was 0.09766. Uniform layerwise 4-bit used 16.44 MB
and had KL 0.00456. The KVTuner-style policy used 16.44 MB and had KL
0.00598. Every separate per-head record cost the same at requested 2 and
4 bits, so the per-head allocator chose all 4-bit heads for 18.39 MB.
This shows how record metadata and Roaring container granularity can
erase a nominal bit-width saving on short prefixes.

KVTuner's original multi-objective search and RateQuant's gradient-based
head sensitivity are not implemented. Here, KVTuner-style calibration
uses isolated next-token KL and a small joint policy search; RateQuant-style
calibration fits separate K/V distortion slopes from this codec's MSE and
uses isolated 2-bit head KL as a forward-only sensitivity proxy. The
selected KIVI, KVTuner, and RateQuant policies were written, reopened,
and reproduced the same decoded bits, logits, and greedy tokens. The
generic affine path compresses the prefix only; new tokens use xinfer's
BF16 cache. These measurements do not describe native TurboQuant.

Run the pinned experiment with a fresh output directory:

```sh
.agents-workspace/tmp/target/release/shifou-xinfer \
  .agents-workspace/tmp/models/Qwen3-0.6B \
  .agents-workspace/tmp/corpora/wikitext-2-test.txt \
  .agents-workspace/tmp/paper-qwen paper-quantizers
```

See [the construction and limits](.agents/docs/paper-quantizers.md) and
[the full results](.agents/docs/paper-quantizers-results.json).

## Adaptive 1-bit compact KV experiment

A model-independent packed codec now supports dense 1-, 2-, and 4-bit
KV codes with shared group parameters and an exact 32-token tail. The
tail uses lossless BF16 storage when the exported f32 words allow it.
The xinfer Qwen3-0.6B adapter compares uniform widths, asymmetric K/V,
budgeted allocation, and sparse 1-bit tiles selected under a joint
calibration KL limit.

At a 512-token WikiText prefix, complete logical records used 58.72 MB
for raw BF16, 8.79 MB for uniform 1-bit, 18.92 MB for uniform 4-bit,
and 17.83 MB for six selected 1-bit tiles among otherwise 4-bit
tiles. Their respective KL from BF16 over 32 held-out tokens was 0,
1.23350, 0.01221, and 0.05259. The sparse policy saved 5.75% versus
uniform 4-bit but greedy agreement fell from 15/16 to 3/16. The
current scalar 1-bit representation therefore does not establish a
usable quality/storage tradeoff on this model.

This is motivated by [AsymKV](https://arxiv.org/abs/2410.13212),
[Coupled Quantization](https://arxiv.org/abs/2405.03917), and
[QJL](https://arxiv.org/abs/2406.03482), but implements their
ideas only as a simple calibration baseline, not their algorithms.
See [the construction and limits](.agents/docs/compact-onebit.md) and
[the full results](.agents/docs/compact-onebit-results.json). Run
`shifou-xinfer <model-dir> <corpus-file> <fresh-output-dir> compact-onebit`
with the pinned inputs described above.

## Exact tokenization cache

`Cache::put_token_ids` and `get_token_ids` store little-endian `u32` token
sequences under SHA-256 of the exact input bytes. The caller supplies a
fingerprint of tokenizer files and every option that changes token IDs,
including normalization, special tokens, chat template and multimodal
expansion. `CacheReader::get_token_ids` reads checkpoint-visible entries while
the writer remains open. On a 32,796-byte WikiText input, the real Qwen3.6
tokenizer produced 7,896 IDs; a release-mode warm hit took 0.123 ms median,
plus 0.056 ms to hash the input, versus 5.385 ms to tokenize with an already
loaded tokenizer. These are 21 local repetitions, with no network hop or
prompt assembly.

With the real Nemotron Nano 9B Japanese tokenizer and the same input, a warm
hit took 0.121 ms median and hashing took 0.056 ms median, versus 6.781 ms
to tokenize. The typed cache returned all 7,688 IDs exactly; this does not
exercise the model's chat template.

## Persistent prefill bundles

`PrefillBundle` groups exact token IDs, an opaque recurrent-state snapshot such
as xinfer's GDN bytes, and engine-owned attention snapshots at one token
boundary. `Cache::put_prefill_bundle` splits the recurrent state into 8 MiB
pages and publishes every record in one yesno batch. The final checkpoint
makes the bundle visible to `CacheReader::open` in another process. A read uses
one database snapshot and verifies the complete bundle before returning it;
a missing or damaged page is an error, never a partial cache hit. Reopen a
read-only handle to see a newer checkpoint. `remove_prefill_bundle` removes
all owned records in one batch.

The engine supplies a full model compatibility fingerprint in the address,
exports attention and recurrent state at the same token boundary, and checks
the recurrent-state format before importing it. The storage format does not
interpret the model's tensors or move them to a GPU. A Qwen3.6-27B test
persisted xinfer's exact GDN snapshot and attention KV, restored both into a
second model instance, and obtained identical continuation logits. The
[construction, measurements, and limitations](.agents/docs/prefill-cache.md)
include a release-mode full-size storage probe. For workers on one host, the
caller configures yesno's peer socket and engine GPU import.

With the optional `peer` feature, `PeerBundleReader` reads the same bundle
through yesno's Unix plugin socket. It holds one server-owned snapshot across
the bundle, batches packed keys as lanes, and writes state pages directly into
the final output buffer. Both the shared-memory arena and inline fallback are
supported. Packed records, tokens, state pages, and the whole state receive
full verification. The reader exposes state and total-bundle logical-byte limits.
Its synchronous connection is closed after a failed read; reconnect to retry.
The server must have its plugin channel enabled and ready.

```rust
let mut reader = shifou::PeerBundleReader::connect("/run/yesno/plugin.sock")?;
let bundle = reader.get_prefill_bundle(&address, "xinfer-gdn/v1")?;
```

The separate optional `flight` feature retains `FlightBundleReader` for
prefill-only interoperability; the prepared-prefix and session workflow uses
the peer socket. It fetches up to four independent
state pages concurrently, verifies every packed record and the aggregate
state, and uses one database version for the entire read. If a checkpoint
reclaims that version, it discards the partial bundle and retries once. The
reader exposes page-concurrency, state-size, and total-bundle-size limits;
the default total logical bundle limit is 2 GiB. It returns host bytes for the
engine to import and does not upload GPU pages itself.

On one GB10 host with loopback transport and a warm page cache, the production
reader fetched a fully verified 512-token Nemotron bundle in 0.877, 0.876,
and 0.902 seconds. A separate fresh 64-token run read the bundle in 0.845
seconds, imported it into another model in 0.182 seconds, and matched four
continuation steps exactly. These timings exclude server startup and model
loading. Cross-machine transport, cold storage, sustained arrival under load,
and a serving-engine integration remain unmeasured. The peer-socket and Flight
concurrent-reader measurements are in the linked note.

The Qwen3.6-27B cross-instance continuation also passed after repinning to
xinfer `78c237f`, with zero logit difference over four continuation tokens.
The 9B Nemotron Japanese checkpoint ran on the same GB10. With xinfer's
portable Mamba snapshot at `80555f8`, shifou stored a 512-token prefix as one
prefill bundle: 145,539,320 state bytes, 8,650,752 BF16 attention bytes, and
the exact token IDs. A second, independently loaded model imported both state
and attention KV and matched four continuation tokens with zero logit
difference. In one debug run, state export took 28.645 s, bundle write 19.112 s,
verified read 14.292 s, state import 23.347 s, and attention upload 0.240 s.
These are correctness measurements, not serving latency results. The exact
construction and the earlier attention-only run are in the linked note.

Reads verify payload SHA-256 by default. For a trusted local cache,
`get_prefill_bundle_with_verification(..., ReadVerification::StorageOnly)` and
`get_packed_with_verification` skip shifou's payload and bundle-content hashes.
Address, format, metadata digest, buffer lengths and yesno's stored-region
checks remain active. This trades end-to-end payload verification for lower
read latency; callers reading copied or untrusted cache data should use the
default. On one GB10 host, a 2,035-token Qwen3-0.6B prompt with tool
definitions had 222.6 MiB of BF16 KV. Three warm reads of the same prefill
bundle measured 496.0-497.8 ms with full verification and 89.7-91.3 ms with
`StorageOnly`; the returned bundles were equal. A 1,307-token policy-only
prompt measured 296.2-297.3 ms and 34.5-35.1 ms respectively. These are
same-host observations, without Flight transport or GPU upload.

The Qwen3 xinfer adapter can export and restore the exact BF16 KV pages without
an intermediate f32 tensor and can append a request suffix in one prefill call.
It uses
[xinfer revision `17499e4`](https://github.com/moriyoshi/xinfer/commit/17499e450a174e25be333f88c654ff6743fd4465),
whose public Qwen3 constructor normalizes compressed-tensors metadata for direct
NVFP4 loading. A real-model probe used
Qwen3-0.6B BF16 and the pinned `llmat/Qwen3-0.6B-NVFP4` revision `d83e6a5`
on one GB10. The NVFP4 checkpoint
quantizes model weights; its KV pages remain BF16. The table adds the median of
three warm `StorageOnly` reads to one GPU restore and one 38- or 41-token
suffix prefill, and compares that sum with a fresh full prefill. Times exclude
generation, bundle creation, and checkpoint publication.

| Weights | Prepared prefix | Fresh full prefill (ms) | Cache hit through suffix prefill (ms) |
| --- | ---: | ---: | ---: |
| BF16 | 1,307 tokens | 57.7-67.5 | 103.2-104.9 |
| BF16 | 2,035 tokens with tool definitions | 101.0-112.2 | 209.3-219.1 |
| NVFP4 | 1,307 tokens | 156.9-172.0 | 125.3-127.4 |
| NVFP4 | 2,035 tokens with tool definitions | 293.1-312.4 | 204.7-205.9 |

For all four NVFP4 requests, the restored batched path produced exactly the
same logits and 48 greedy tokens as a native batched prefix continuation.
It also matched the fresh full-prefill generation in this probe. The BF16
restored path matched native batched continuation exactly; fresh full prefill
occasionally produced different later tokens because the split and unsplit
prefill kernels differ numerically. These are single-run, warm-cache,
same-host observations; the relative NVFP4 gain does not establish a gain
for larger models or Flight transport.

An alternating six-sample NVFP4 restore probe compared the current per-tile
GPU upload with one contiguous upload sliced into 56 tensor views. The median
times were 49.2 versus 49.6 ms for 1,307 tokens, and 80.4 versus 81.7 ms for
2,035 tokens. The bulk path matched native continuation exactly but did not
improve restore latency, so the adapter keeps the per-tile path.

When the prepared prefix is already resident on the serving GPU, copying its
native KV cache and batching the request suffix took 35.1-36.9 ms for the
1,307-token NVFP4 case and 40.0-42.1 ms for the 2,035-token case. The prefix
occupies 149.9 or 233.4 MB of BF16 KV on that GPU, and an active request needs
its own copied cache. This is a useful hot tier for repeated prompts on one GPU;
a separate GPU still needs the persistent bundle read and upload above.

For repeated hits when the prefix cannot stay on the GPU, the Qwen3 adapter
also offers `prepare_packed_bf16` and `restore_prepared_bf16`. Preparation
validates and decodes the bundle once into padded BF16 pages in host RAM;
restoration uploads from those retained slices. On the NVFP4 probe, one-time
preparation took 47.5 ms for 1,307 tokens and 84.9 ms for 2,035 tokens. Six
warm restore samples had 3.2 and 4.6 ms medians respectively, and both cases
matched native batched continuation logits and 48 generated tokens exactly.
The prepared pages occupy roughly the logical KV byte count in host RAM, so
this tier helps repeated hits after a verified read, not the first cold hit.

Further measurements cover [Qwen hybrid KV quality](.agents/docs/qwen-hybrid-kv-20260930.md),
[token and prefill reuse](.agents/docs/token-and-prefill-cache-20260930.md), and
[attention page selection](.agents/docs/kv-attention-and-page-selection-20260930.md), and
[page residency planning](.agents/docs/page-residency-planner-experiment-20261002.md).

## Prepared prefixes and session checkpoints

`PrefixScope` binds a model compatibility fingerprint and a caller-supplied
causal-context fingerprint. The latter must cover chat-template and multimodal
inputs not captured by token IDs, plus the tenant's intended sharing scope.
`put_prepared_prefix` stores a complete `PrefillBundle` and updates a sorted
length index in one yesno batch. `find_longest_prepared_address` checks exact
token-chain addresses and returns a small hit descriptor before reading the
large state. The caller can compare measured restoration cost with recomputation,
then fetch the bundle and prefill the remaining request tokens. A convenience
`find_longest_prepared_prefix` performs lookup and full verified retrieval in
one call. `PeerBundleReader` exposes the same prepared lookup, including a
metadata-only first step, under one server-owned snapshot. No original
system-prompt text is needed to continue inference once
the exact token IDs and model state are restored.

`SessionKey` identifies one ongoing session. `put_session_checkpoint` publishes
a complete state bundle, an opaque application-workflow payload, and a new
generation pointer in one batch. `get_session_checkpoint` reads the pointer
and bundle under one yesno snapshot and validates their shared token boundary.
For token extensions whose attention buffers are exact byte prefixes,
`put_session_append_checkpoint` publishes a cumulative attention tail against
the latest full checkpoint, the current recurrent state, and the new head in
one batch. The full base remains required until a later full checkpoint;
the API rejects changed prior bytes and tails over 64 MiB. Peer-socket session
reads reconstruct and verify the full state within one server snapshot.
Older generations remain available if a later publication fails;
`prune_session_checkpoint` removes a superseded generation and refuses to
remove the current one or its live full base. `PeerBundleReader` resolves the current session pointer
and complete bundle within one snapshot. The caller
must provide conversation and tool state in the workflow payload, export model
state at the same token boundary, and restore that state through its engine
adapter. The workflow payload is limited to 1 MiB. Read-only handles must be
reopened to observe checkpoints published after they were opened.
An interrupted decode also needs its pending token or last logits and sampler
state; KV and token history alone do not reproduce the next sampling decision.

`AdmissionEstimate` chooses the lowest measured time to the first new token
among available GPU, prepared-host, local, peer, and Flight tiers, or chooses
recomputation. `should_publish` compares expected future savings against the
measured publication time. The caller supplies those measurements and enforces
memory, disk and tenant budgets. These APIs do not instrument the serving
engine, choose checkpoint boundaries, evict old generations, or restore model
weights. Append checkpoints reduce publication bytes; cold restores still
read and reconstruct the complete attention state.

For workers sharing one host, use `PeerBundleReader` through yesno's Unix peer
socket. It resolves both prepared prefixes and session heads without a second
text serialization, and keeps each multi-record read under one server snapshot.
It cannot connect across hosts. A cold peer read still copies and verifies the
complete bundle; keep a prepared host tier for repeated hits and use
`AdmissionEstimate` to compare a cold read with recomputation. A bounded
proactive Qwen probe overlapped cold peer reads with one model load: median
first prepared and session requests fell from 824/666 ms to 252/28 ms with
both host entries retained ( 466 MB declared ). The background fetch cost
1.3 s median; one run waited another 150 ms after model load.

A separate probe warmed entries while one loaded Qwen3-0.6B model served 64
serial one-token requests. Each entry became available when its own read and
decode finished; a request for an entry still pending used a verified peer
read immediately, which can duplicate the background read. Across three cold
database copies, median active-request p50 was 13.8 ms without warming and
13.9 ms with serial warming of both entries. The active-request loop plus
prepared and session requests took 2,496 ms without warming and 1,314 ms
with warming. The prepared request hit host memory in all three runs, but the
session request still used the peer in all three. Paired active-request p50
differences ranged from -4.0 to +4.9 ms, so this small shared-host probe does
not establish a stable inference-latency cost. Two parallel peer readers did
not consistently improve the total over one reader. The 512 MiB speculative
tier excludes the already-hot active session and transient buffers. A serving
scheduler still needs to coalesce duplicate reads and invalidate entries when
prepared indexes or session heads change. See the
[construction and measurements](.agents/docs/prefill-cache.md).

An earlier full-checkpoint run of the Qwen workflow in
`experiments/xinfer/src/bin/session_workflows.rs` tested both paths with
separate model instances and exact continuation logits. On one GB10, a warm
1,980-token NVFP4-weight prefix plus a 32-token suffix took 277.8 ms to
prefill, 271.0 ms through a trusted-local durable hit, and 40.2 ms from
xinfer's prepared-host BF16 pages. Resuming its later session generation with
one new input token took 274.0 ms by recomputation, 274.1 ms through yesno,
and 22.2 ms from prepared host pages. These are three-sample medians after
model load; preparation and retention of host pages are additional costs.
The [construction and caveats](.agents/docs/prefill-cache.md) also show that
shorter prefixes favor recomputation.

The same 1,980-token Qwen state was read from a separate `yesnod` process
through its peer socket on one GB10. A first verified peer hit took 758 ms for
prepared startup and 733 ms for session resume, including GPU restore and the
next model operation; fresh recomputation took 346 ms and 328 ms. After one
peer read and host preparation, repeat hits took 42 ms and 24 ms. Maximum
continuation-logit difference was 0 in both paths. These are single warm runs
with model load and peer-server startup excluded;
the [measurement note](.agents/docs/prefill-cache.md) records the byte counts
and 512-token comparison.

The retained workflow probe now publishes generation 2 as an append
checkpoint. With metadata digest validation, it took 442 ms, versus roughly
1.17 s for a full checkpoint. The attention tail added 114,688 bytes to a
230.8 MB base. A separate `yesnod` peer process restored the same append
format with zero logit difference;
its 602 ms peer read and 726 ms first-hit continuation remained slower than
330 ms recomputation. These are individual warm runs, not a throughput claim.

For a repeatable end-to-end run on the local Qwen3-0.6B NVFP4 model, use:

```sh
python3 experiments/xinfer/bench_session_e2e.py \
  /path/to/Qwen3-0.6B-NVFP4 /path/to/corpus.txt \
  .agents-workspace/tmp/qwen-session-e2e \
  --yesnod /path/to/yesnod --prefix-tokens 1980 --samples 3
```

The output directory must be new. The runner builds the pinned xinfer probes,
publishes a fresh prepared prefix and two session generations, starts a
separate `yesnod`, and restores both paths through its Unix peer socket. It
records raw samples and a `summary.json` with publication, peer read, GPU
restore, next-token, and accuracy results. Model load and server startup have
separate process-wall timings; they are excluded from first-hit latency. In
one three-sample GB10 run, median first-hit startup and session resume were
751 and 724 ms through the peer socket, versus 366 and 329 ms by fresh
computation. Prepared-host repeat hits took 42 and 24 ms; every continuation
had zero maximum logit difference. The [measurement note](.agents/docs/prefill-cache.md)
has the construction and raw-result path.

The Qwen experiment adapter also has an opt-in four-worker CPU decode for its
56 independent BF16 attention tiles. Set `SHIFOU_PARALLEL_RESTORE=1` and
`RAYON_NUM_THREADS=4` when running the peer probe or E2E runner. A paired
experiment found lower restore time with exact logits, but it ran under heavy
host contention; the default remains the serial path. `SHIFOU_BENCH_DECODE=1` makes the peer probe
also time serial and parallel CPU tile decoding on the same verified bundle,
independently of peer transfer and GPU upload.

`PeerBundleReader::with_parallel_verification(true)` verifies independent
packed records on Rayon workers after each peer scan. The peer probe selects
it with `SHIFOU_PARALLEL_VERIFY=1`; reads remain fully checksummed and
ordered. On the same 1,980-token Qwen checkpoint and separate `yesnod`, four
workers cut the median session peer read from 577 to 300 ms across three
paired runs. Enabling parallel BF16 restore too brought session first-hit
latency to 349 ms versus 334 ms for fresh recomputation, with zero logit
difference. Prepared startup was 411 ms versus 339 ms fresh. The
[measurement note](.agents/docs/prefill-cache.md) has the construction, raw
samples, and contention caveat.

`PeerBundleReader::with_pipelined_verification(true)` also overlaps checksum
verification of one completed batch with fetching the next batch on the same
peer connection. The probe selects it with `SHIFOU_PIPELINE_VERIFY=1`; it
enables parallel verification and needs no server protocol change. With 16
peer lanes and four workers, a three-run Qwen comparison reduced the session
first hit from 338 to 264 ms and prepared startup from 384 to 301 ms. Fresh
computation took 332 and 337 ms in the pipelined runs. Continuation logits
remained exact. The pipeline is opt-in while its memory and concurrency
behavior is evaluated in a serving engine.

A peer-only control with no GPU work restored the same 231 MB session for four
simultaneous readers in 291 ms group wall time with 16 shared verification
workers, versus 523 ms with four workers. Three isolated storage-cold copies
took 781 ms median on their first read, against 187 ms after warming. The
[experiment note](.agents/docs/prefill-cache.md) records the construction and
major-fault evidence; concurrent model inference remains unmeasured. A
follow-up page-residency experiment found that 220.5 MiB of the touched
payload occupied one contiguous file span. Prewarming that span in the
background cost 255 ms median and cut a storage-cold foreground peer read
from 535 to 179 ms, with six rather than 953 server major faults. The span
was identified from a prior read, so a deployable targeted prefetch still
needs a yesno-owned physical access plan. In real Qwen continuation runs,
background warming of both previously observed bundle footprints cost 463 ms
median; subsequent prepared startup was 273 ms and session restore plus the
next token was 228 ms, versus 334 and 325 ms for fresh computation. Both
restored continuations matched fresh logits exactly. The warm-up cost is
separate from those foreground hit times.

## Routed expert snapshots

`ExpertSnapshotKey` identifies one routed expert by the model fingerprint,
engine snapshot format, layer, and expert number. It has no causal-prefix
component: the same model weights can serve many prompts. `Cache::put_expert_snapshot`
stores opaque bytes losslessly; `put_expert_snapshots` publishes a bounded group
atomically with one commit and checkpoint. A caller can choose groups of roughly
8-16 experts to avoid staging the whole model in memory. `CacheReader` loads
only the requested expert through `get_expert_snapshot`, and removal is per
expert. Each snapshot is limited by the packed-record 64 MiB limit.

The caller must supply a fingerprint of actual checkpoint weights, adapters,
and numerical settings. Shifou verifies its own address and payload digest by
default, but does not interpret or authenticate an engine's expert envelope.
Xinfer's `NemotronExpertSnapshot::from_bytes` validates the envelope, layout,
and payload digest before GPU admission. A xinfer application can implement
`NemotronExpertRestoreSource::load_expert(layer, expert)` by constructing an
`ExpertSnapshotKey` with the same fingerprint and format
`xinfer-nemotron-expert/v1`, then calling
`CacheReader::get_expert_snapshot_with_verification`. For that specific callback,
`ReadVerification::StorageOnly` avoids a duplicate shifou payload hash because
xinfer immediately validates the returned bytes; use the default full check
for other consumers. Return `None` on a missing key so xinfer can fall back to
its checkpoint. Reopen `CacheReader` after later checkpoints become visible.

The application-side adapter can be this small (the application depends on
xinfer, Candle, and shifou):

```rust
struct ShifouExperts {
    reader: shifou::CacheReader,
    fingerprint: [u8; 32],
}

impl xinfer::models::nemotron_h::NemotronExpertRestoreSource for ShifouExperts {
    fn load_expert(&self, layer: usize, expert: usize)
        -> candle_core::Result<Option<Vec<u8>>>
    {
        let key = shifou::ExpertSnapshotKey {
            model_fingerprint: self.fingerprint,
            format: "xinfer-nemotron-expert/v1".into(),
            layer: layer.try_into().map_err(|_| candle_core::Error::Msg("layer overflow".into()))?,
            expert: expert.try_into().map_err(|_| candle_core::Error::Msg("expert overflow".into()))?,
        };
        self.reader.get_expert_snapshot_with_verification(
            &key, shifou::ReadVerification::StorageOnly,
        ).map_err(|error| candle_core::Error::Msg(error.to_string()))
    }
}
```

Export with xinfer's `export_expert_snapshot_bytes` and store the returned bytes
under the same key. Install `ShifouExperts` with
`set_expert_restore_source(fingerprint, std::sync::Arc::new(source))` on the
serving model. The external adapter was compiled and exercised against the
local xinfer branch; a missing expert returned `None`, and xinfer parsed the
selected snapshot after a shifou read.

A release-mode storage probe used eight synthetic, half-dense, 5,612,544-byte
expert payloads (44,900,352 logical bytes total), each filled with `0x55` and
one expert-specific leading byte. One atomic publication took 0.376 s and the
portable Roaring records totaled 45,032,150 bytes. Seven warm reads of one
selected expert took 10.988-11.845 ms with full shifou verification and
1.189-1.210 ms with `StorageOnly`. Construction and source are under
`.agents-workspace/tmp/expert-store-bench`. These times exclude xinfer's
snapshot-envelope validation, network transfer, GPU upload, and inference;
the synthetic byte pattern is not a Nemotron checkpoint.

These records contain **expert weights**, not prompt-dependent Mamba state or
attention KV. They require xinfer's per-expert snapshot API, currently on the
local `epic/xinfer-session-enhancements` branch at `bc2b77d` or later. Shifou
has no xinfer dependency; the small trait adapter belongs with the inference
application. The agent's real 30B test found file-backed per-expert reads much
slower than checkpoint misses, while preimporting snapshots into host RAM
reduced token latency. This storage API supplies durable, selective retrieval;
an application still needs a host hot tier or prefetch policy for low-latency
serving.

## Verify

```sh
cargo test --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo fmt --check
```

Tests cover randomized shapes, grouping along either axis, numeric bounds,
exact finite-f32 reconstruction, sparse exceptions, residual tails, mixed
precision, container boundaries, process reopening, replacement, deletion,
namespace/model isolation, corrupted payload/metadata, and forced address
collisions. Native packed-buffer tests also cover arbitrary bytes, trailing zeros,
format rejection, checksum corruption, replacement and removal. Prefill tests cover
large state pages, a read-only handle, incomplete bundles and atomic removal;
the optional Flight tests cover a three-page read, a missing page, and a
checkpoint that forces a whole-bundle retry;
token tests cover input and tokenizer identity. Test databases remain under
`.agents-workspace/tmp/tests/`.
