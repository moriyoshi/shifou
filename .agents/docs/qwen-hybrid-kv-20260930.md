# Qwen hybrid attention KV experiment, 2026-09-30

## Scope and construction

The standalone scratch crate at `../yesno/.agents-workspace/tmp/qwen35-kv-probe` uses
pinned xinfer `b88c15334fb607ff52cdc3fd875c3da796dcc020`, Candle
`23a6f38`, attention-rs `c0f19f2`, and the local shifou compact4 codec. It
loads the text submodel of `Qwen/Qwen3.5-0.8B` from the downloaded multimodal
checkpoint, using the official `text_config` and native xinfer Qwen3.5
Gated DeltaNet and paged attention layers. The checkpoint has 24 layers:
18 Gated DeltaNet and six full attention, with two KV heads of dimension
256. The model file SHA-256 is
`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`;
config SHA-256 is
`b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204`.
The WikiText-2 test text is
`.agents-workspace/tmp/corpora/wikitext-2-test.txt`, SHA-256
`d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0`.

The probe tokenizes the corpus without special tokens, pre-fills one sequence,
exports all six attention K/V cache pairs, and round-trips each through
shifou's compact4 policy ( keys and values use their respective policies ).
The Gated DeltaNet prefix state is captured once and restored before each
teacher-forced continuation arm. Both BF16 and compact4 arms use
xinfer's native forward path. Reconstructing the BF16 KV from exported f32
values changed no logit in either reported run ( maximum absolute difference
0.0 ). The compact4 byte count includes code, parameters, metadata, and the
codec's exact 32-token tail. The same continuation tokens are scored against
the same BF16 reference; this is a narrow quality check, not a benchmark of
serving throughput or broad model quality.

| Prefix and corpus offset | BF16 KV | Complete compact4 buffers | Code bytes | Key / value code one fraction | Mean KL from BF16 | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 / 4,096 | 25,165,824 B | 6,806,304 B | 6,193,152 B | 50.01% / 49.76% | 0.000450 | 8/8 |
| 8,192 / 16,384 | 100,663,296 B | 26,270,496 B | 25,067,520 B | 49.98% / 49.69% | 0.008783 | 8/8 |

The complete buffer ratio is 3.70 times at 2K and 3.83 times at 8K. The code
bits are approximately half one, so a Roaring set is unlikely to compress
these dense payloads. This agrees with the earlier Qwen3-0.6B observation.

A follow-up scored 64 teacher-forced tokens at the same two prefixes, with identical
BF16 reconstruction and packed bytes. At 2K, compact4 mean KL was 0.002826
and top-1 agreed on 63 of 64 tokens. At 8K, compact4 mean KL was 0.005650
and top-1 agreed on 61 of 64. The first eight logits are included in those
64-token samples. The output files are `result-2048-64.json` and
`result-8192-64-v2.json` in the scratch crate. These samples are too small
to infer a safe compression policy. The single prefill and encode timings in
the result JSON are cold-run observations and should not be compared with a
warm serving path. No yesno database I/O,
selective attention, or end-to-end latency was measured in this run.

Two more 8K windows used offsets 4,096 and 8,192, again with 64 tokens each.
Their compact4 mean KL values were 0.003584 and 0.007686, and top-1 matched
on 62/64 in each. Across all three 8K windows ( 192 scored tokens ), mean KL
was 0.005640 and top-1 matched on 185/192. The windows overlap in corpus
content and their continuations differ; this is a sensitivity check rather
than an independent benchmark sample. Their result files are
`result-8192-offset4096-64.json` and `result-8192-offset8192-64.json`.

Reproduce with the release binary and the model and corpus files above:

```sh
cargo build --release --manifest-path ../yesno/.agents-workspace/tmp/qwen35-kv-probe/Cargo.toml
../yesno/.agents-workspace/tmp/qwen35-kv-probe/target/release/qwen35-kv-probe \
  ../yesno/.agents-workspace/tmp/models/Qwen3.5-0.8B \
  .agents-workspace/tmp/corpora/wikitext-2-test.txt \
  ../yesno/.agents-workspace/tmp/qwen35-kv-probe/result-2048-v3.json 2048 4096
```

The 8K eight-token run uses output `result-8192-v3.json`, prefix `8192`, and
offset `16384`. Append `64` to either command to reproduce the longer
continuation. Each output path must be new because the probe refuses to
overwrite one. The JSON files hold the full numbers and buffer breakdown.

## Implications for yesno page layout

For four-bit codes, one 16-token K/V page pair for one KV head with dimension
256 occupies `16 * 256 = 4,096` bytes. Two such page pairs fill one 8,192-byte
bitmap chunk. The earlier Qwen3-0.6B geometry had dimension 128 and four
page pairs per chunk. A page-major shifou format can select the group size
from `8,192 / ( page_tokens * head_dim )` when K and V share a chunk; the
container is then filled through the existing public
`BitmapContainer::from_words` and `OrdSet::from_chunks` APIs. This is an
inference from dimensions and previous storage measurements, not a measured
Qwen3.5 selected-read speedup. A true serving adapter must account for the
Gated DeltaNet state independently of attention KV selection.

## Larger Qwen checkpoint feasibility

The official `Qwen/Qwen3.8-Flash-Next-FP8` repository is about 186 GB and
contains a 125B-parameter hybrid model with a 51B n-gram embedding. It does
not fit this single GB10 host's 121 GiB unified memory for normal in-memory
inference. See the [official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
and [file listing](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/main).

The official `Qwen/Qwen3.6-27B-FP8` repository is about 30.9 GB and uses the
Qwen3.5 architecture. Its `text_config` specifies 64 layers, including 16
full attention layers, four KV heads, and dimension 256. BF16 attention KV is
65,536 bytes per token: 512 MiB at 8K and 16 GiB at 262,144 tokens. Four-bit
code bytes alone would be 4 GiB at that native maximum, before scales,
tails, and Gated DeltaNet state. The transfer pins revision
`e89b16ebf1988b3d6befa7de50abc2d76f26eb09`; the 65 shards needed by the
text path total 30,389,664,704 bytes, excluding the separate MTP shard. All
65 text-serving shards were checked against the pinned Hugging Face LFS SHA-256
values with `verify_qwen36.py` in the scratch crate. The full checkpoint then
loaded and ran through xinfer on the single GB10. This confirms functional
single-device fit, but does not measure production serving throughput.

The same scratch probe used BF16 attention KV replay and compact4 round-trip
with exact Gated DeltaNet prefix-state restoration. Replaying exported BF16
attention KV changed no logits in any run. Each row scores 64 teacher-forced
tokens from WikiText-2; the 256-token smoke row scores eight. The two 8K
prefix spans are disjoint, though the first continuation falls inside the
second prefix span.

| Prefix / corpus offset | BF16 attention KV | Complete compact4 buffers | Key / value code one fraction | Mean KL from BF16 | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 / 4,096 | 16,777,216 B | 6,017,760 B | 49.98% / 50.45% | 0.002299 | 8/8 |
| 2,048 / 4,096 | 134,217,728 B | 36,295,424 B | 50.03% / 50.42% | 0.001438 | 63/64 |
| 8,192 / 8,192 | 536,870,912 B | 140,104,448 B | 50.00% / 50.28% | 0.004006 | 62/64 |
| 8,192 / 16,384 | 536,870,912 B | 140,104,448 B | 50.01% / 50.28% | 0.005367 | 62/64 |

Complete compact4 buffers are 3.70 times smaller at 2K and 3.83 times
smaller at 8K. The two 8K windows together have mean KL 0.004686 and 124/128
top-1 agreement. Their code bits are again about half one, so set compression
does not help the dense code stream. At 8K, the cold prefill took about 71
seconds and compact4 encode plus decode took about 5 seconds. Those single-run
timings exclude database I/O and are not a serving benchmark. A 64-token
teacher-forced continuation is still too small to establish general model
quality or an acceptable deployment policy.

The result files are `result-qwen36-256.json`, `result-qwen36-2048-64.json`,
`result-qwen36-8192-offset8192-64.json`, and `result-qwen36-8192-64.json` in
`../yesno/.agents-workspace/tmp/qwen35-kv-probe`. To reproduce the latter:

```sh
../yesno/.agents-workspace/tmp/qwen35-kv-probe/target/release/qwen35-kv-probe \
  ../yesno/.agents-workspace/tmp/models/Qwen3.6-27B-FP8 \
  .agents-workspace/tmp/corpora/wikitext-2-test.txt \
  ../yesno/.agents-workspace/tmp/qwen35-kv-probe/new-qwen36-8192.json 8192 16384 64
```

The scratch `fp8_smoke` binary separately loaded one real `F8_E4M3`
block-scaled `in_proj_z` weight from `layers-0.safetensors` and ran its
`5120 -> 6144` xinfer projection; all 6,144 outputs were finite. The full
model experiment above is the stronger feasibility check.
See the [official model card](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
and [file listing](https://huggingface.co/Qwen/Qwen3.6-27B-FP8/tree/main).
