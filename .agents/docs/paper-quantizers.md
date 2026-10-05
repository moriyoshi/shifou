# Paper-inspired KV quantizer comparison, 2026-09-29

## Scope and provenance

This experiment compares shifou's affine bitvector representation under
three published policy ideas. [KIVI](https://arxiv.org/abs/2402.02750)
motivates per-channel keys, per-token values, and an exact recent tail.
[KVTuner](https://arxiv.org/abs/2502.04420) motivates model-calibrated
layerwise K/V precision pairs and a full-model accuracy objective.
[RateQuant](https://arxiv.org/abs/2605.06675) motivates separate K/V
distortion fits and per-head rate-distortion allocation.

The implementations are approximations on one common codec, not paper
reproductions. The codec uses affine min/max quantization with requested
2, 4, or 8 bits and may choose raw f32 for a group when its measured
standalone Roaring cost is lower. It uses group size 32, a 32-token exact
tail, and no outlier exceptions. It persists the quantized prefix. New
tokens use BF16. There are no KIVI CUDA kernels, KVTuner MOEA/D search,
RateQuant gradients, or mixed-layer native TurboQuant kernels.

## Construction

Model, xinfer revision, corpus, hardware, and tokenizer are pinned in
xinfer-evaluation.md. The experiment uses Qwen3-0.6B with 28 layers,
eight KV heads, and 128 channels per head. Each layer/slot tile has axes
[token, head, channel]. Per-head policy records split a tile into eight
head-specific records and merge them before xinfer restore.

Calibrate on WikiText token offset 0, prefix length 128, and eight
continuation tokens. KVTuner-style scoring measures mean
KL(BF16 || candidate) after changing one layer's key or value tile
to 2-, 4-, or 8-bit reconstruction. It greedily allocates exact complete
record bytes up to a uniform 4-bit budget, then compares the proposed
complete cache with uniform 4-bit on the calibration continuation.
Two rounds each evaluate up to 40 budget-safe single or paired policy
changes, shortlisted by isolated KL. The winner minimizes whole-cache
calibration KL. This is a small proxy search, not the paper's
multi-objective optimizer.

RateQuant-style scoring splits each tile by KV head. It fits
D(b) = alpha * beta^(-b) separately for K and V from mean tensor MSE
at requested b in {2, 4, 8}. Fitted beta was 4.466 for K and 4.361 for V.
Each head's importance proxy is its isolated next-token KL when requesting
2 bits; it replaces RateQuant's gradient-based sensitivity. Greedy
marginal predicted distortion reduction per actual complete record byte
allocates precision under two budgets: uniform 4-bit head records, and
the midpoint in bytes between uniform 2- and 4-bit head records.
A deterministic dispersed policy upgrades 2-bit heads to 4-bit in
index order permuted by (index * 73) modulo 448, until the same midpoint
budget is exhausted.

Held-out text begins at token offset 4096 with 128- and 512-token
prefixes. Each policy scores 32 teacher-forced next tokens and generates
16 greedy tokens. The 128 and 512 cases overlap and are not independent
documents. The held-out suffix is never used to select precision.
Byte budgets count exact portable Roaring payload and metadata for each
address via Cache::estimate_encoded; selected record sums are checked
against Cache::put. They exclude database pages and GPU padding.
Selected KIVI, KVTuner, and RateQuant records are closed and reopened;
decoded f32 bits, all scored logits, and greedy tokens match the
pre-persistence selections exactly.

## Results

Raw BF16 cache size is 14,680,064 bytes at 128 tokens and 58,720,256
bytes at 512 tokens. BF16 subset perplexity is 40.316 and 63.616.
All quality metrics below compare against that BF16 cache.

| Prefix | Policy | Logical bytes | KL | Subset perplexity | Top-1 | Greedy matches / 16 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | KIVI-style 2-bit | 15,050,170 | 0.09766 | 40.537 | 78.13% | 4 |
| 128 | Fixed K4V2 | 15,738,970 | 0.02896 | 40.401 | 81.25% | 14 |
| 128 | Uniform 4-bit layer records | 16,438,528 | 0.00456 | 40.015 | 96.88% | 14 |
| 128 | KVTuner-style | 16,438,526 | 0.00598 | 39.659 | 96.88% | 3 |
| 128 | RateQuant-style and uniform 4-bit head records | 18,385,920 | 0.00456 | 40.015 | 96.88% | 14 |
| 512 | KIVI-style 2-bit | 40,590,488 | 0.08639 | 58.000 | 87.50% | 6 |
| 512 | Fixed K4V2 | 44,042,724 | 0.02120 | 64.457 | 81.25% | 16 |
| 512 | Uniform 4-bit layer records | 47,477,252 | 0.00299 | 63.571 | 96.88% | 16 |
| 512 | KVTuner-style | 47,108,864 | 0.00323 | 63.651 | 96.88% | 16 |
| 512 | Dispersed 2/4-bit heads | 46,095,728 | 0.06375 | 56.434 | 81.25% | 5 |
| 512 | RateQuant-style 2/4-bit heads | 46,096,784 | 0.04300 | 61.875 | 87.50% | 6 |
| 512 | RateQuant-style and uniform 4-bit head records | 49,853,368 | 0.00299 | 63.571 | 96.88% | 16 |

At 128 tokens, requested 2- and 4-bit per-head records cost exactly the
same for all 448 heads: 18,385,920 bytes in aggregate. The allocator
takes the free 4-bit quality upgrade. At 512 tokens, uniform per-head
2- and 4-bit storage costs 42,340,588 and 49,853,368 bytes. The midpoint
budget is 46,096,978 bytes. Both rate-based and dispersed selections
used 224 2-bit and 224 4-bit heads, within 1,056 bytes of each other.
Rate-based placement reduced held-out KL from 0.06375 to 0.04300 and
raised top-1 agreement from 81.25% to 87.50%, but worsened subset
perplexity. All three 4-bit head policies decode to the same values,
yet head-specific storage is 2,376,116 bytes larger than layer records
at 512 tokens.

KVTuner-style joint calibration reduced KL from 0.002706 for uniform
4-bit to 0.001988 at the 128-token budget and 0.002099 at the 512-token
budget, after 82 joint trials each. On held-out text it was slightly
worse than uniform 4-bit in KL at both lengths. The 512-token policy
selected three 2-bit and 53 4-bit layer/slot tiles, saving 368,388
bytes versus uniform 4-bit. The 128-token policy selected two 2-bit,
53 4-bit, and one 8-bit tile, saving just two bytes. This single short
calibration continuation appears insufficient for robust policy selection.

KIVI-style 2-bit storage was larger than raw BF16 at 128 tokens and
1.447 times smaller at 512 tokens. It also caused much larger
distribution shifts than 4-bit. Its 512-token subset perplexity
happened to be lower than BF16 despite high KL and low greedy agreement;
32 tokens are too few to treat this as a quality gain.

## Reproduction and artifacts

The executable is experiments/xinfer/src/paper_quantizers.rs, reached
through the paper-quantizers mode in experiments/xinfer/src/main.rs.
The complete measured scores, selected bits, byte counts, and
persistence checks are in paper-quantizers-results.json. Scratch cache
directories and the run log are under .agents-workspace/tmp/paper-qwen-v4
and .agents-workspace/tmp/paper-qwen-v4.log.

Build and run with the pinned model and corpus described in
xinfer-evaluation.md, choosing a fresh output directory. On the
measured machine, isolated calibration took 67.4 seconds. Candidate
encoding and byte costing took 13.1 seconds at 128 tokens and
57.2 seconds at 512 tokens, before persistence. These are offline
costs, not serving latency estimates.
