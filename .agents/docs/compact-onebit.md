# Adaptive 1-bit KV cache prototype, 2026-09-29

## Research basis and implementation

[AsymKV](https://arxiv.org/abs/2410.13212) reports that key and value
quantization have different effects on attention and explores layerwise
asymmetric configurations with 1-bit tiles. [Coupled
Quantization](https://arxiv.org/abs/2405.03917) exploits dependencies
between channels at very low rates. [QJL](https://arxiv.org/abs/2406.03482)
uses a Johnson-Lindenstrauss transform followed by sign bits and an
inner-product estimator; its sign sketch is not a direct reconstruction
codec. These results motivated a small, independently measured baseline,
not a reproduction of any of the three methods.

The new generic compact snapshot stores 1-, 2-, or 4-bit densely packed
codes in a self-describing PackedSnapshot. At 1 bit, each quantized group
has two f32 centroids fitted by eight Lloyd updates; at 2 and 4 bits,
it stores an affine minimum and step. There are eight parameter bytes
per nonempty group. Keys group a head/channel over the whole old prefix;
values group a token/head over all channels. The newest 32 tokens are
exact. When all tail f32 words are BF16-representable, the tail is stored
as BF16 words and decoded bit-exactly; otherwise it stays f32. This is
a format-level codec independent of xinfer or a particular transformer.
The codec does not implement channel coupling, QJL projection, or
low-bit attention.

The xinfer adapter exported all 56 K/V tiles from Qwen3-0.6B. It measured
isolated mean KL(BF16 || one changed tile) for 1, 2, and 4 bits on two
128-token calibration prefixes at WikiText-2 test offsets 0 and 1024,
using eight teacher-forced continuation tokens per prefix. Uniform
1/2/4, K2V1, K4V1, dispersed, and several allocation policies were
then evaluated on held-out prefixes at offset 4096, lengths 128 and 512.
The two held-out prefixes overlap. Each policy scored 32 teacher-forced
tokens and generated 16 greedy tokens.

The sparse policy starts with 4 bits everywhere. For at most six rounds,
it ranks 1-bit tile substitutions by isolated KL increase per byte saved,
then jointly checks the 20 highest-ranked candidates on both calibration
prefixes. It chooses the largest saving whose whole-cache calibration KL
is at most 0.01 above the uniform 4-bit calibration KL. Exact record bytes
are evaluated at the held-out prefix length, while held-out logits do not
participate in selection. This is an offline, length-specific policy
search; it is not an online adaptive cache.

## Results

Complete logical bytes count the portable Roaring payload and metadata
for every tile address. They exclude database pages, allocator overhead,
and temporary BF16 tensors in xinfer. BF16 subset perplexity was 40.316
at 128 tokens and 63.616 at 512 tokens.

| Prefix | Policy | Logical bytes | Effective stored bits/value | KL from BF16 | Subset perplexity | Top-1 | Greedy matches / 16 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | Raw BF16 bytes | 14,680,064 | 16.000 | 0 | 40.316 | 100% | 16 |
| 128 | Uniform 1-bit | 5,342,294 | 5.823 | 0.77596 | 85.569 | 37.50% | 1 |
| 128 | Uniform 2-bit | 5,828,476 | 6.353 | 0.32033 | 46.868 | 65.63% | 2 |
| 128 | Uniform 4-bit | 7,205,558 | 7.853 | 0.00543 | 41.259 | 100% | 3 |
| 128 | Sparse 1-bit (6 K/V tiles) | 7,004,152 | 7.634 | 0.02614 | 42.286 | 93.75% | 3 |
| 512 | Raw BF16 bytes | 58,720,256 | 16.000 | 0 | 63.616 | 100% | 16 |
| 512 | Uniform 1-bit | 8,785,990 | 2.394 | 1.23350 | 170.007 | 46.88% | 0 |
| 512 | Uniform 2-bit | 12,027,560 | 3.277 | 0.79035 | 120.657 | 56.25% | 4 |
| 512 | Uniform 4-bit | 18,916,485 | 5.154 | 0.01221 | 64.246 | 87.50% | 15 |
| 512 | Sparse 1-bit (6 K/V tiles) | 17,829,039 | 4.858 | 0.05259 | 73.274 | 87.50% | 3 |

The bit label is the code width on quantized values, not the complete
stored density. The recent exact tail, group parameters, metadata, and
Roaring framing dominate the 128-token result. BF16 tail packing saves
about 3.67 MB on either prefix versus the same codec with f32 tail and
changes no logits. At 512 tokens, six sparse 1-bit tiles save 1,087,446
bytes (5.75%) against uniform 4-bit, but held-out KL increases by 0.04039
and greedy agreement falls from 15/16 to 3/16. Whole-cache calibration
KL rises from 0.00431 to 0.01284, within the prespecified +0.01 cap.
The 128-token sparse result has the same pattern: 201,406 bytes saved,
KL from 0.00543 to 0.02614. This narrow calibration does not predict
held-out quality well enough to recommend the policy.

K2V1 at 512 tokens used 10,420,122 bytes and had KL 0.72882. K4V1
used 13,864,637 bytes and had KL 0.38322. The adaptive budgeted mix
under the uniform 2-bit budget improved KL to 0.30396, still far from
the uniform 4-bit result. The complete policy results, selected tile
bits, calibration scores, and timings are in compact-onebit-results.json.

Selected K2V1, adaptive-mid, adaptive-u2, and sparse-1bit caches were
written to shifou, closed, reopened, and checked for identical decoded
f32 bits, all scored logits, and greedy tokens. Cache::estimate_packed
matched the stored logical byte count for each record. The best quality
among these compact policies comes from 4-bit; the current scalar
two-centroid 1-bit scheme is useful as a storage lower bound, not a
quality-preserving model cache on this test.

## Limits and next experiment

The grouping deliberately spans a complete axis to amortize parameter
overhead. That may lose local structure, especially for keys over long
prefixes. The experiment covers one small model and 32 held-out tokens
on one corpus. The 128- and 512-token samples overlap. Calibration
is only 16 next-token distributions, so six joint substitutions can
overfit despite a fixed KL cap. The code decodes to xinfer's BF16 cache
before attention and therefore does not measure low-bit attention speed.
For the next test, a useful controlled comparison is channel-coupled
or transformed 1-bit groups against this scalar baseline, with a larger
separate calibration set and independent documents. That would test
the core mechanism of Coupled Quantization or QJL instead of assuming
all 1-bit codes have equivalent quality.

## Reproduction

The executable mode is `compact-onebit` in the xinfer experiment.
Build its release binary and run it with the pinned model and corpus
in xinfer-evaluation.md, choosing a fresh output directory. The recorded
run used `.agents-workspace/tmp/compact-onebit-qwen-v2`. Calibration took
27 seconds before candidate encoding, policy search, inference, and
persistence. The JSON report is copied to compact-onebit-results.json.
