# Token and disaggregated prefill cache, 2026-09-30

## Exact tokenization cache experiment

The standalone `token_cache_probe` binary in
`../yesno/.agents-workspace/tmp/qwen35-kv-probe` uses shifou's generic
`PackedSnapshot` path to persist a Qwen3.6 token sequence in yesno. The
address hashes the 32,796-byte UTF-8 input under a SHA-256 fingerprint of the
checkpoint's `tokenizer.json`. The value is the exact little-endian `u32`
token sequence. The buffer is returned byte-for-byte after a warm lookup and
after closing and reopening the cache. This is exact-input tokenization reuse,
not a prefix KV lookup.

The source is the first 32,768 Unicode characters of the pinned WikiText-2
test text described in `qwen-hybrid-kv-20260930.md`. It produces 7,896
tokens. On the single GB10 host, the release binary measured 21 repetitions
of tokenization, input hashing, and a warm cache lookup, reporting each
median. The result is in
`../yesno/.agents-workspace/tmp/qwen35-kv-probe/token-cache-result.json`.

| Operation or size | Result |
| --- | ---: |
| Tokenization, already loaded tokenizer | 5.699 ms median |
| SHA-256 of input bytes | 0.057 ms median |
| Warm yesno token hit, including shifou payload verification | 0.168 ms median |
| One put, including commit, visibility and explicit checkpoint | 28.905 ms |
| Open database and get once | 4.752 ms |
| Raw `u32` IDs | 31,584 B |
| Roaring payload | 32,820 B |
| Roaring payload plus metadata | 36,666 B |

The warm hit plus input hash is about 25 times faster than tokenization for
this text. The insert cost is repaid after roughly six subsequent warm hits,
under the same loaded-tokenizer conditions. A fresh open plus one hit is only
slightly faster than tokenization; the useful configuration keeps a cache
service open. These timings are local, sequential, and omit network transport,
lock contention, and an engine's full prompt assembly. The token buffer used
shifou's current bit-by-bit staging path, so this is a baseline rather than
the fastest dense constructor.

The cache identity must include the tokenizer files and options that affect
IDs: normalization, special-token behavior, chat template, input byte
encoding, and multimodal expansion. Model weights need not be part of a pure
tokenization key when two models truly share all those semantics. The stored
full digest and exact buffer validation prevent a truncated yesno key
collision from silently returning another sequence.

Reproduce with a new database and result path:

```sh
../yesno/.agents-workspace/tmp/qwen35-kv-probe/target/release/token_cache_probe \
  ../yesno/.agents-workspace/tmp/models/Qwen3.6-27B-FP8 \
  .agents-workspace/tmp/corpora/wikitext-2-test.txt \
  ../yesno/.agents-workspace/tmp/qwen35-kv-probe/new-token-db \
  ../yesno/.agents-workspace/tmp/qwen35-kv-probe/new-token-result.json
```

## Typed token cache follow-up

Shifou now exposes `put_token_ids` and `get_token_ids`, with a read-only
`CacheReader` variant. `token_address` hashes exact input bytes, while the
caller supplies a tokenizer fingerprint covering files and tokenization
options. A new release probe at
`../yesno/.agents-workspace/tmp/qwen35-kv-probe/src/bin/token_cache_typed.rs`
uses the same 32,796-byte WikiText input and Qwen3.6 tokenizer as the
initial experiment. It measures 21 repetitions of each operation after the
tokenizer is loaded. The typed record stores the same 7,896 IDs.

| Operation or size | Typed result |
| --- | ---: |
| Tokenization | 5.385 ms median |
| Input key hash | 0.056 ms median |
| Warm yesno hit | 0.123 ms median |
| One put and checkpoint | 23.511 ms |
| Read-only open and one get | 0.364 ms |
| Raw IDs | 31,584 B |
| Roaring payload and metadata | 36,658 B |

The warm hash plus hit is about 30 times faster than tokenization on this
host. The old writer-mode reopen and new read-only reopen are different
operations, so their times are not a direct before-and-after comparison.
Neither probe includes network transport, concurrent serving load, or full
prompt assembly. The new result is saved at
`../yesno/.agents-workspace/tmp/qwen35-kv-probe/typed-token-result.json`.

## Prefill state on a separate GPU

An attention KV entry alone cannot resume Qwen3.6 after prefill. Its 64 text
layers include 16 full-attention and 48 Gated DeltaNet ( GDN ) layers.
xinfer's GDN cache holds, per sequence and GDN layer, an FP32 convolution
state `[d_conv, kernel - 1]` and an FP32 recurrent state
`[num_v_heads, key_head_dim, value_head_dim]`. At tensor parallel size one,
Qwen3.6-27B has `d_conv = 16 * 128 * 2 + 48 * 128 = 10,240`, kernel 4,
48 value heads, and key/value dimensions 128. Its exact GDN state therefore
uses `48 * ( 10,240 * 3 + 48 * 128 * 128 ) * 4 = 156,893,184` bytes,
or 149.625 MiB, per sequence. These are calculated state bytes, not a
measured serialized artifact.

For the measured 8K prefix, the complete compact4 attention buffers were
140,104,448 bytes; add exact GDN state and the minimum full prefill bundle is
296,997,632 bytes ( 283.24 MiB ), before a manifest and token IDs. BF16
attention KV plus exact GDN state is 693,764,096 bytes ( 661.63 MiB ). The
full-state size ratio is thus 2.34 times, versus 3.83 times for attention KV
alone. Transfer and upload costs on two discrete GPUs remain unmeasured; the
GB10 has one unified-memory GPU and cannot represent that link.

At the time of the initial sizing, xinfer exposed only in-process GDN
snapshot hashes. The 2026-09-30 enhancement added a versioned portable
`GdnStateSnapshot` for the Qwen3.5/3.6 dense and MoE paths. It exports all
48 convolution and recurrent state pairs as exact FP32 bits and validates
token boundary, layer order, shapes, dtypes, tensor-parallel layout, model
fingerprint and checksum before allocating an import slot. The model
fingerprint is caller-supplied; the caller must derive it from weights,
adapters and execution settings. Attention KV and GDN must still refer to
the same exact token boundary.

Shifou now publishes a `PrefillBundle` through one yesno batch and one
checkpoint. The bundle contains exact token IDs, opaque recurrent-state
bytes split into 8 MiB pages, attention snapshots and a versioned manifest.
`CacheReader` uses `Db::open_reader` so a second process can read the last
checkpoint while the writer retains its exclusive lock. It opens one
snapshot for every bundle record, verifies each payload and the whole-state
checksum, and rejects a missing page. A handle must be reopened to see a
later checkpoint. A service API is still needed across machines; the
existing Flight surface is for yesno sets rather than binary page transport.

For exact prefix reuse, derive a rolling digest over token blocks from the
full causal input, with model weights, adapters, tokenizer, positions,
attention and numerical execution settings in the compatibility identity.
Publish full state only at selected boundaries ( for example, every 256 or
1,024 tokens ) because each Qwen3.6 GDN snapshot alone is 149.625 MiB.
Query the longest stored compatible boundary, verify its full digest and
manifest, restore its state, then prefill the remaining tokens. A GPU memory
cache can hold hot bundles; yesno is the durable shared tier. This lookup is
primarily exact-key work. yesno set operations may help admission and sparse
indices, but dense token IDs and quantized code bits are not promising set
compression targets.

Shifou now constructs dense bitmap chunks through
`BitmapContainer::from_words` and `OrdSet::from_chunks`, and reads them by
bitmap words. The old per-set-bit conversion would have staged roughly 535
million `u64` ordinals, or 4.28 GB, for a 133,693,440-byte 8K compact4 code
stream at half one bits. The direct word path removes that staging cost. It
is a conversion improvement, not additional compression or an attention
kernel improvement.

## Completed persistent prefill experiment

The opt-in scratch test at
`../yesno/.agents-workspace/tmp/qwen35-kv-probe/tests/prefill_integration.rs` uses the
local xinfer Qwen3.5/3.6 GDN API, a real verified Qwen3.6-27B-FP8 text
checkpoint and shifou's new `PrefillBundle`. One instance prefills a short
prompt, exports all 48 GDN states to a 156,893,512-byte portable snapshot
including its 328-byte envelope, and stages 16 full-attention K/V records as
lossless BF16-to-FP32 bytes. One yesno transaction publishes 37 records:
tokens, 19 GDN pages, 16 attention records and the manifest. A read-only
handle retrieves the bundle; another independent xinfer instance imports
its GDN snapshot and restores attention KV from those bytes. Four
teacher-forced continuation tokens produce a maximum absolute logit
difference of **0**. This was two instances on one GB10 GPU; no separate
physical GPU or network transfer was measured. The debug test took 52.740 s
to publish, 15.530 s to read, and 306.13 s end to end including loading and
prefill. Its attention bytes are lossless FP32 staging, not compact4.

A separate release-mode storage probe under
`../yesno/.agents-workspace/tmp/prefill-store-bench` uses a synthetic
156,893,512-byte 0x55 recurrent state, so exactly four of each eight input
bits are set, and 2,048 token IDs. The full-size arm adds 16 synthetic
attention code records of 8,756,528 bytes each, totaling 140,104,448 bytes
and matching the measured complete 8K compact4 buffer byte count. It does
not run a model. Each arm uses a new database, one writer and one
checkpoint-visible read-only handle. Timings are single cold sequential
runs on the same GB10 host.

| Arm | State and attention bytes | Records | Publish | Read | Allocated disk |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recurrent state | 156,893,512 | 21 | 4.099 s | 0.659 s | 152 MiB |
| Full-size bundle | 296,997,960 | 37 | 8.926 s | 0.980 s | 288 MiB |

The same full-size release probe was rerun with a child process opening
`CacheReader` while the parent writer remained open. The child verified all
37 records and exited successfully. In that separate cold run, publication
was 13.997 s and the parent read was 0.971 s. The difference from the
8.926 s write above illustrates run-to-run variability; neither is a
throughput distribution. This proves simultaneous cross-process access on
one host, not physical GPU separation or network transport.

The full-size logical count includes the 328-byte xinfer GDN envelope, so
it is slightly above the earlier raw-state estimate of 296,997,632 bytes.
Disk figures are allocated blocks from `du -sh`, not the shard file's sparse
apparent length. The release probe is synthetic and excludes model compute,
network transfer, CUDA upload and exact-prefix lookup. The durable bundle
path is functional; boundary selection, longest-prefix search, admission,
automatic eviction and transport across hosts remain integration work.

## Flight over-wire probe

The standalone release probe at
`../yesno/.agents-workspace/tmp/flight-prefill-probe` uses the shipped
`yesno-flight` service and `YesnoClient` across loopback TCP with the server
in a separate process. Its three logical records are 7,896 generated token
IDs encoded as 31,584 little-endian bytes, a 32,768-byte 0x55 recurrent-state
sample, and an 80-byte manifest holding lengths and SHA-256 digests. This is
a protocol sample, not xinfer's GDN snapshot or shifou's packed manifest.
Each byte is mapped to its set-bit ordinals plus a length sentinel, matching
the shifou byte-to-set representation. A Flight write transaction stages the
three keys in separate `DoPut` calls. A second read before commit finds none;
commit publishes all at one version. `DoGet` reconstructs each exact byte
buffer. The server is then stopped, yesno reopens its directory, and WAL
recovery returns all three buffers again.

| Observation | Value |
| --- | ---: |
| Logical payload | 64,432 B |
| Mutation rows including deletes and sentinels | 197,890 |
| Write transaction, one cold release run | 0.135 s |
| Three Flight reads, same run | 0.046 s |
| Earlier cross-process cold run, write / read | 0.076 / 0.084 s |

The three integer and one operation columns carry at least 25 value bytes
per mutation row, or 4,947,250 bytes of column values for this 64,432-byte
sample, before Arrow and gRPC framing. That is a derived lower bound on
column payload, not a measured network byte count. The timings are loopback
single runs; they say nothing about a remote network or serving contention.

A follow-up release probe at
`../yesno/.agents-workspace/tmp/qwen35-kv-probe/src/bin/gdn_density.rs` read the real
Qwen3.6 bundle stored by the earlier continuation test. Its portable GDN
snapshot was 156,893,512 bytes with **633,959,733 set bits**, a measured
50.5088% density. The short-prefix exact attention snapshots contributed
4,194,304 bytes and 7,014,131 set bits. Sending those actual stored buffers
through the current Flight mutation format would need **640,973,864 insert
rows**, over 38 times the transaction cap, even before sentinels, deletions
and manifest. GDN alone would require at least 15.85 GB of mutation-column
values at 25 bytes per row, versus its 156.9 MB source bytes. These are
calculations from measured payload bits and the wire schema, not transmitted
byte counts.

**The current Flight wire format cannot carry the full prefill bundle as one
atomic write.** `MAX_TRANSACTION_ROWS` is 16,777,216. At exactly half one
bits, one 8 MiB shifou state page would require 33,554,432 insert rows,
already twice the transaction cap. The 156,893,512-byte portable GDN snapshot
plus 140,104,448 bytes of compact attention data would require
1,187,991,840 insert rows, about 70.8 times that cap, before sentinels,
manifest or deletes. The client also collects every mutation into Arrow
columns before sending a staging call. The full-size failure follows from
the published row limit and the encoding; allocating and transmitting more
than a billion rows merely to receive the expected rejection would not add
evidence.

A usable Flight path needs a negotiated bulk bitmap or opaque page transfer:
small bounded binary frames on the wire, server-side word-oriented set
construction, one commit for the bundle, and snapshot-consistent page reads.
It must retain the shifou manifest's full-address and checksum checks. The
existing ordinal `DoPut` and `DoGet` are suitable for set queries and small
metadata, but not dense model-state bytes at this scale.

## Why the local full-size bundle write took seconds before b712a03

On 2026-09-30, an isolated copy of shifou and the existing release-mode
`prefill-store-bench` was instrumented under
`../yesno/.agents-workspace/tmp/{shifou-profile,prefill-store-profile}`. It used the
same 156,893,512-byte recurrent state, 140,104,448 attention bytes and 2,048
token IDs, all synthetic 0x55 payload bytes. The instrument timed collision
checks, payload preparation, batch staging, commit, visibility wait and the
explicit checkpoint. A typed yesno event sink timed any automatic checkpoint
inside commit. Every run used a new one-shard database on the same GB10 host,
release mode and `-C target-cpu=native`. These are separate cold runs, not a
paired throughput distribution.

| Run | Total publish | Preparation | Commit, including policy | Automatic checkpoint within commit | Final explicit checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default policy, run 1 | 7.168 s | 1.211 s | 5.609 s | not timed | 0.000035 s |
| Default policy, run 2 | 18.901 s | 1.246 s | 17.304 s | 1.823 s | 0.000038 s |
| Deferred automatic policy | 13.444 s | 1.239 s | 10.709 s | none | 1.151 s |

The difference between `Total publish` and the measured stage sum is
`put_prefill_bundle`'s token conversion, state-page copies, manifest creation
and full-state SHA-256, which occur before `put_packed_many`. The default
checkpoint policy triggers during `commit` because the resulting dirty
memtable is 298,310,014 bytes ( above the default 256 MiB trigger ) and the
WAL is 9,506,355,680 bytes ( above the default 1 GiB trigger ). The later
explicit checkpoint is then effectively a no-op. The deferred-policy run
raised both triggers beyond this one batch only in the scratch copy; it moved
the checkpoint out of commit and confirmed that the expensive remainder is
the WAL path. Run-to-run wall time varies substantially with host I/O.

The old WAL size was the main explanation: 296,997,960 bytes of synthetic state
and attention payload at exactly half one bits contain about 1.188 billion set
bits. `WriteBatch::store_set` stages each 8 KiB byte window as a compact
bitmap chunk, but `Planned::Whole(Op::PutChunk)::to_record` writes
`RecType::ChunkImage` as an 8-byte key followed by an 8-byte `u64` ordinal
for **every set bit**. Thus the WAL holds about 9.506 GB, or 32.01 times the
logical state and attention bytes, before checkpoint condenses it back to
roughly 288 MiB of allocated database blocks. The 32x estimate uses only the
half-dense payload bytes; the recorded WAL also covers tokens, metadata and
framing. The old WAL replay inserted each ordinal individually, so cold
recovery before checkpoint had the same dense-shape problem. `ChunkImage`
replay unions into a key, while live `PutChunk` replaces a chunk; `store_set`'s
preceding delete makes them agree. The later fix preserves that relation and
remains compatible with existing logs and replicas.

Preparation of all 37 records was about 1.2 s; batch staging, collision
checking and `wait_visible` were negligible. In the default run with the
timed automatic checkpoint, roughly 15.5 s of the 17.3 s commit was the
WAL build, append and durability path, and 1.8 s was checkpointing. This
attributes the original 8.926 s single-run result to a format expansion
whose cost fluctuates with I/O; it does not imply all deployments take eight
seconds. Skipping shifou's final checkpoint alone could not remove the
automatic policy checkpoint or the old 9.5 GB WAL write.

The values from `Db::wal_bytes()` above were sampled at the start of the
automatic checkpoint and describe retained WAL at that instant. The method
is a gauge, not a cumulative write-volume counter: a checkpoint inside
`commit` can reclaim generations and make a before/after subtraction
underflow. The compact-WAL follow-up therefore measured bytes with automatic
checkpointing deferred.

## Compact WAL follow-up in b712a03

The fix reuses the existing `RecType::ChunkPatch` record. For each `PutChunk`,
yesno chooses the smaller of the old ordinal image and a patch carrying the
container payload with an empty clear mask. That patch replays as union, just
as the old `ChunkImage` did. `store_set` still emits a leading `DeleteKey`,
which is what makes replay's union agree with the live chunk replacement.
The old image decoder remains, and the existing patch type is understood by
replicas that already support it; no WAL format version or new record type was
needed. The per-chunk choice avoids regressing sparse chunks where patch
headers cost more than a few ordinals.

The standalone `../yesno/.agents-workspace/tmp/wal-compact-measure` harness wrote one
64-chunk set per tested density, using a deferred-checkpoint policy to measure
WAL bytes across a commit and reopening without checkpoint for replay. On the
full-size synthetic prefill shape, it measured **299,893,088 WAL bytes** for
296,997,960 payload bytes, about **1.01x**. That is about 31.7 times less
than the 9,505,508,768-byte old-format prediction for the same harness shape.
The WAL commit stage took **239.9 ms**, versus the **10.71 s** deferred-policy
stage measured above; full WAL replay took **714.8 ms**. These are separate
runs on a host with variable I/O, so they establish the large direction and
size change rather than a precise steady-state speedup. The default policy
can still checkpoint during commit because the dirty set is above 256 MiB;
the new 239.9 ms figure is not a full default-policy publish latency.

`scripts/gate.sh` and `scripts/gate-pg.sh` passed for b712a03. The dense WAL
size property and old `ChunkImage` replay now have targeted tests. The
existing byte-exhaustive crash matrix passes, but no new torn-record case
specifically exercises `PutChunk` through its `ChunkPatch` body. The complete
construction, dense and sparse table, and measurement corrections are in the
2026-09-30 compact-WAL entry in `JOURNAL.md`.

## OpenZL compression of the real Qwen3.6 GDN snapshot

On 2026-10-01, the real 156,893,512-byte portable GDN state from the
Qwen3.6-27B-FP8 continuation bundle was extracted without changing the
database. Its SHA-256 is
`ee88ad5909baaba379c1b7415ca977ca2fcab0d08fe747de7639cffb1e829759`.
The current yesno size-class ladder rejects this older database, so the
extractor at `../yesno/.agents-workspace/tmp/openzl-cache-probe/old-reader` uses a
scratch copy of `yesno-core` at `b712a03` and the existing shifou reader.
The extracted input was written as
`../yesno/.agents-workspace/tmp/openzl-cache-probe/qwen36-gdn-snapshot.bin` and can be
regenerated with that extractor. The raw copy and native build outputs were
removed after measurement because the host filesystem filled; the source,
CSV results and digest remain.

The standalone OpenZL 0.1.0 probe at
`../yesno/.agents-workspace/tmp/openzl-cache-probe/src/main.rs` applies the binding's
default serial API and numeric API at element widths 1, 2, 4 and 8 bytes.
Numeric input is formed from exact little-endian words; the binding selects
numeric behavior by element width, so `u32` represents the FP32 bit patterns
without claiming float-specific transforms. Each frame is decompressed and
compared bit-for-bit to its input. The page experiment uses 18 full 8 MiB
pages and one 5,898,568-byte final page. Totals below sum all 19 independent
frames; compression and decompression times sum one release-mode run on the
GB10 AArch64 host. Timers exclude reading the source file, forming numeric
vectors, and comparing restored values. The first serial/u32 pass measured
239.6/3,078.8 ms encode, close to the later 237.7/2,984.6 ms pass shown below.

| OpenZL mode, 8 MiB pages | Total encoded bytes | Input retained | Encode | Decode |
| --- | ---: | ---: | ---: | ---: |
| Default serial | 144,348,158 | 92.00% | 237.7 ms | 106.2 ms |
| Numeric `u8` | 144,348,272 | 92.00% | 357.0 ms | 119.2 ms |
| Numeric `u16` | 139,785,419 | 89.10% | 4,324.1 ms | 116.5 ms |
| Numeric `u32` | **131,474,748** | **83.80%** | 2,984.6 ms | 72.1 ms |
| Numeric `u64` | 133,939,198 | 85.37% | 1,917.0 ms | 90.9 ms |

The `u32` page path saves 25,418,764 bytes, or 24.24 MiB and 16.20%, over
the raw state. The faster serial path saves 12,545,354 bytes, or 11.96 MiB
and 8.00%. The same source compressed as one whole-snapshot frame gave
144,334,140 bytes in 319.6 ms for serial, and 131,865,391 bytes in
2,868.4 ms for numeric `u32`. One whole numeric frame was slightly **larger**
than the sum of page-local numeric frames, and its 359.0 ms decode was slower
than the 72.1 ms sum of page-local decodes in this single run. The page-local
shape therefore does not obviously sacrifice compression ratio here.

As a control, one 8 MiB page filled with `0x55` is also 50% set bits, but
OpenZL serial encoded it in **301 bytes** and numeric `u64` in **69 bytes**.
This is why the earlier 50% synthetic payload cannot stand in for a real
compression measurement. The actual GDN state retains 83.8-92.0% under the
tested useful OpenZL modes. For a write-sensitive path, the observed 3.0 s
numeric encode cost is material beside the 16.2% space saving; the 0.24 s
serial encode is the faster option with 8.0% saving. These are CPU-side,
single-host codec measurements, not yesno write, GPU upload or network
transfer timings. They do not measure the full 8K compact4 attention buffers.

The same persisted **short-prefix** bundle also contains 4,194,304 bytes of
exact FP32 attention buffers, which were concatenated in record order as
`qwen36-attention-short.bin` ( SHA-256
`91b5d473357de88a0ab204dcd7418daf08a347c59b3cab2086f29f2749e302f1` ).
Its 20.90% set-bit density is a different shape. The experiment split the
concatenation into the 16 actual 262,144-byte attention records and gave
each an independent OpenZL frame: serial retained 1,334,496 bytes ( 31.82% )
in 29.4 ms, while numeric `u64` retained 1,033,369 bytes ( 24.64% ) in
25.1 ms. Combining GDN `u32` page frames with attention `u64` record frames
retains **132,508,117 of 161,087,816 bytes** of the actual stored state and
attention payloads, or **82.26%**. The corresponding serial path retains
145,682,654 bytes, or 90.44%. This is the short-prefix
correctness bundle, whose attention is exact FP32; it is not the synthetic
8K-sized bundle or the measured 8K compact4 attention stream. The attention
results are preserved in `result-attention-records.csv`; the one-frame
concatenation is in `result-attention-short.csv` for comparison.

The Rust binding was built with `LIBCLANG_PATH=/usr/lib/llvm-18/lib` and
`BINDGEN_EXTRA_CLANG_ARGS` pointing at the host GCC 13 and system include
directories; without those headers its native build failed to find
`stddef.h`. Reproduction tables are preserved as `result-widths.csv`,
`result-whole.csv` and `result-synthetic.csv` under the scratch probe.
