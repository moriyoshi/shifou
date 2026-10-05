# Persistent prefill bundle experiment, 2026-09-30

## Construction

The shifou `PrefillBundle` stores exact token IDs, opaque engine recurrent
state, and an ordered set of packed attention snapshots. The address identifies
the complete causal prefix and model compatibility. A caller must include
weights, adapters, positions, attention behavior and numerical execution
settings in that identity. The bundle's `state_format` identifies the engine
snapshot version. Shifou neither interprets GDN bytes nor quantizes a packed
attention snapshot again.

`put_prefill_bundle` splits recurrent state into 8 MiB records, adds token and
manifest records, then calls `put_packed_many` once. That method checks every
address collision before writing, commits one yesno batch, waits for
visibility, and checkpoints once. `CacheReader` opens yesno's checkpoint-visible
read-only mode and can coexist with the writer. `get_prefill_bundle` reads all
records under one yesno snapshot and verifies the manifest, each packed
payload, token bytes, state length and whole-state SHA-256. A missing page is
an error. `remove_prefill_bundle` deletes the manifest and owned records in
one batch. The dense conversion builds bitmap words directly from byte blocks
and copies them back by words, avoiding one staged ordinal per set bit.

The focused shifou integration test stores 8 MiB plus 17 bytes of state and an
attention snapshot, reads it through a concurrent read-only handle, reopens,
and checks an exact round trip. Other tests verify rejection before publication,
missing-page detection, and removal. A separate bitmap test checks dense,
sparse, empty and 8 KiB boundary cases against an ordinal oracle.

## Real Qwen3.6 continuation

The scratch integration test at
`../yesno/.agents-workspace/tmp/qwen35-kv-probe/tests/prefill_integration.rs`
uses the local xinfer GDN export/import enhancement and the verified 65-shard
Qwen3.6-27B-FP8 text checkpoint. It prefills one xinfer instance, serializes
its 48 GDN layers into 156,893,512 bytes including the 328-byte envelope,
and saves that plus 16 attention K/V snapshots in one shifou prefill bundle.
The attention snapshots are losslessly staged as little-endian FP32 from BF16;
they do not use compact4 in this correctness test. A read-only handle retrieves
the bundle after checkpoint. A second independent model instance restores
GDN and attention KV from those bytes and teacher-forces four continuation
tokens. The maximum absolute logit difference was **0**.

This was one GB10 GPU with two model instances, not two physical GPUs. It
establishes the process-independent byte boundary and exact continuation on
that hardware. The scratch test used a debug build and measured 52.740 s for
bundle publication and 15.530 s for reading it. These numbers include the
small attention snapshots and are not release-mode serving timings. The full
test completed in 306.13 s, including model load and prefill.

The same opt-in test was rerun on 2026-10-03 against the published xinfer
revision `78c237fdc6dd430af08e74597f386a87f95e0ebe`, using the same local
Qwen3.6-27B-FP8 checkpoint and a fresh yesno database. It again published 37
records, restored the 156,893,512-byte GDN snapshot and attention KV into a
second model instance, and obtained maximum absolute logit difference 0 over
four teacher-forced continuation tokens. The debug run took 237.29 s overall;
bundle publication took 20.910 s and reading took 16.203 s. These are single
debug observations that include encoding, verification, and database work.

## Real Nemotron attention continuation

The scratch probe at `../yesno/.agents-workspace/tmp/nemotron-probe` loads
[`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese)
at revision `3979dd16634988c34cc3bd911583c51e6a731d10` through the same
xinfer `78c237f` pin on one GB10. All four safetensors shards ( 17,776,492,488
bytes total ) matched the repository's LFS SHA-256 values. The model config
has 27 Mamba layers and four attention layers. Xinfer's Mamba convolution
and SSM tensors contain 145,539,072 logical FP32 bytes per sequence, while
BF16 attention KV adds 16,384 bytes per token. This size follows directly
from the config and
the `MambaState` tensor shapes in the pinned xinfer adapter.

The probe tokenizes a repeated Japanese system instruction with two JSON tool
definitions, prefills 512 tokens, and teacher-forces four following tokens.
It exports the four attention K/V pairs as exact BF16 bytes, writes them as
one shifou `PackedSnapshot` through yesno, reads the snapshot through a
read-only handle, checks every returned byte, and uploads new GPU tensors.
Xinfer's in-process Mamba prefix cache supplies recurrent state for this
replay. The restored path's maximum absolute logit difference from native
continuation was 0. The cache allocation has 528 token slots because the
512-token prefix plus four continuation slots round up to 33 blocks of 16;
the eight attention tensors therefore occupy 8,650,752 BF16 bytes. This is
attention-only persistence, not a durable Nemotron prefill bundle.
The scratch address binds the pinned model and xinfer revisions plus exact
prefix token IDs; a serving address must also bind adapters and numerical
execution settings, as described above.

| Item | Single debug run |
| --- | ---: |
| 512-token prefill | 10.295 s |
| Attention export to host BF16 | 0.012 s |
| BF16 attention bytes | 8,650,752 |
| Roaring payload plus metadata | 8,404,880 B |
| Publication with checkpoint | 0.744 s |
| Verified read, including read-only open | 0.444 s |
| GPU upload from retrieved BF16 | 0.000683 s |
| Four native continuation tokens | 0.344 s |
| Four restored continuation tokens | 0.343 s |

The database allocated 8.2 MiB on disk. An earlier F32 staging pass on the
same 512-token construction stored 17,301,504 attention bytes, used
16,801,662 Roaring bytes, and measured 1.670 s to publish and 0.988 s to
read. Direct BF16 avoids the unnecessary width expansion and GPU cast; the
latency differences are single-run observations, not a controlled speedup
estimate. A 20-token smoke run also matched exactly, with 1.530 s prefill
and 0.338 s for four continuation tokens.

Reproduce from the shifou directory with the verified local checkpoint and a
new database path:

```sh
RUSTFLAGS='-C target-cpu=native' cargo run \
  --manifest-path ../yesno/.agents-workspace/tmp/nemotron-probe/Cargo.toml -- \
  ../yesno/.agents-workspace/tmp/models/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
  512 ../yesno/.agents-workspace/tmp/nemotron-probe/new-attention-db
```

The preceding attention-only run used xinfer `78c237f`. Its in-process
Mamba restore ( about 12 microseconds ) does not measure restoration of
recurrent state from yesno or another GPU.

## Complete Nemotron prefill bundle

Xinfer
[`80555f8`](https://github.com/moriyoshi/xinfer/commit/80555f8c8c41b1a1a62ff90650de2e85ff2fc935)
adds a versioned portable snapshot for Nemotron-H's Mamba state. The snapshot
includes its token boundary, model fingerprint, ordered layer layout, FP32
payload, and SHA-256. Import checks those fields against the target model and
rejects an occupied sequence ID. Shifou stores the opaque snapshot alongside
the exact token IDs and BF16 attention K/V pages in `PrefillBundle`.

The scratch probe at `../yesno/.agents-workspace/tmp/nemotron-probe` ran the
same 512-token Japanese instruction and tool-definition prefix on one GB10.
It exported the complete bundle, read it from yesno, loaded a second independent
Nemotron instance, imported the Mamba snapshot and attention KV, and compared
four teacher-forced continuation steps. Maximum absolute logit difference was
0. The model fingerprint in this probe binds the checkpoint revision and BF16,
single-rank execution; a serving integration must also bind any adapters and
other numerical settings. The eight attention tensors include 528 allocated
token slots, so their byte count exceeds 512 times the per-token size.

| Item | Single debug run |
| --- | ---: |
| 512-token prefill | 10.328 s |
| Host BF16 attention export | 0.010 s |
| Portable Mamba snapshot bytes | 145,539,320 B |
| BF16 attention bytes | 8,650,752 B |
| Mamba state export and serialization | 28.645 s |
| Bundle publication | 19.112 s |
| Verified bundle read | 14.292 s |
| Mamba decode and GPU import | 23.347 s |
| BF16 attention GPU import | 0.240 s |
| Four paired continuation steps | 0.762 s |
| Maximum absolute logit difference | 0 |

The bundle occupied 21 yesno records and about 149 MiB on disk. Export/import
times include per-layer GPU transfers and FP32 byte conversion. Publication
and read include shifou encoding, verification, and yesno work. This is one
debug run, so none of these figures is a steady-state serving latency claim.
The cross-instance result establishes that the state and attention boundary
match; it does not exercise an inference server's scheduling or eviction.

Reproduce with a new database path:

```sh
RUSTFLAGS='-C target-cpu=native' cargo run \
  --manifest-path ../yesno/.agents-workspace/tmp/nemotron-probe/Cargo.toml -- \
  ../yesno/.agents-workspace/tmp/models/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
  512 ../yesno/.agents-workspace/tmp/nemotron-probe/new-full-bundle-db bundle
```

## Published xinfer remeasurement, 2026-10-04

The same 512-token Japanese instruction and tool-definition probe was rerun
against published xinfer `17499e450a174e25be333f88c654ff6743fd4465`,
using the Cargo Git dependency rather than a local xinfer checkout. Two
independent debug runs each wrote a new yesno database on one NVIDIA GB10.
The source and target models were separate instances. Each run restored
145,539,320 bytes of Mamba state and 8,650,752 bytes of BF16 attention KV,
then matched all four continuation steps exactly ( maximum absolute logit
difference 0 ). Timings exclude model loading, which took 45.581 seconds on
the first source load and 25.194-26.013 seconds on the later loads.

| Stage | Run 1 | Run 2 | Prior `80555f8` run |
| --- | ---: | ---: | ---: |
| 512-token prefill | 10.156 s | 10.078 s | 10.328 s |
| Mamba export and serialization | 0.263 s | 0.255 s | 28.645 s |
| Bundle publication | 19.304 s | 19.094 s | 19.112 s |
| Verified bundle read | 14.552 s | 14.298 s | 14.292 s |
| Mamba decode and GPU import | 0.195 s | 0.182 s | 23.347 s |
| BF16 attention GPU import | 0.243 s | 0.242 s | 0.240 s |
| Four paired continuation steps | 0.765 s | 0.742 s | 0.762 s |

The merged xinfer state-byte path accounts for the large export/import
improvement. The probe still calls the public snapshot object API, including
its validation; it does not use the newer bytes-first convenience methods.
The measured read-to-GPU path ( verified read plus both imports ) is
14.722-14.990 seconds, while the 512-token prefill is about 10.1 seconds.
These are debug-build measurements with a fresh database for each publication,
not a steady-state server benchmark. Storage publication and read remain the
largest stages in this complete-bundle path.

## Prepared BF16 host tier

The Qwen3 adapter's `prepare_packed_bf16` validates identity, format, tile
shape and payload lengths, then decodes the exact BF16 pages into retained
host slices. `restore_prepared_bf16` uploads those slices to a new GPU cache
and synchronizes once. The real `llmat/Qwen3-0.6B-NVFP4` probe in
`../yesno/.agents-workspace/tmp/practical-prompts-20261003-prepared-api-stream.json`
used xinfer `78c237f`, 56 attention tiles, and six repeated warm restores
after preparation. The read from yesno is excluded from the following times.

| Prepared prefix | Logical BF16 KV | One-time host preparation | Median GPU restore |
| --- | ---: | ---: | ---: |
| 1,307 tokens | 149,897,216 B | 47.505 ms | 3.236 ms |
| 2,035 tokens with tool definitions | 233,390,080 B | 84.910 ms | 4.648 ms |

Both restored paths matched native batched continuation logits and 48 greedy
tokens exactly. The host tier retains approximately the logical KV byte count
in RAM. It helps repeated hits after a verified read; it does not remove the
first yesno read, and it is specific to the Qwen3 BF16 attention layout.

## Release storage timings

The independent release-mode scratch probe at
`../yesno/.agents-workspace/tmp/prefill-store-bench` uses a 156,893,512-byte
0x55 recurrent-state buffer. Every source byte has four set bits, matching the
roughly 50% one-bit density of measured compact attention codes. It uses
2,048 token IDs and a fresh yesno database for each arm. One arm stores only
the recurrent state; the full-size arm also stores 16 attention code buffers
of 8,756,528 bytes each, totaling 140,104,448 attention bytes. The latter
matches the measured complete compact4 attention buffer count for an 8K
Qwen3.6 prefix, but its single-buffer record geometry is synthetic. The
recurrent state is synthetic too. Both arms run sequentially on the same GB10
host and include one checkpoint in the store time; no network transfer or
model compute is included.

| Arm | Logical state and attention bytes | Records | Publish | Read | Allocated disk |
| --- | ---: | ---: | ---: | ---: | ---: |
| Recurrent state | 156,893,512 | 21 | 4.099 s | 0.659 s | 152 MiB |
| Full-size bundle | 296,997,960 | 37 | 8.926 s | 0.980 s | 288 MiB |

A second full-size release run spawned a child process that opened
`CacheReader` while the parent writer held the directory lock. The child
verified all 37 records and exited successfully. Publication was 13.997 s
and the parent's read was 0.971 s in this cold run. That variation reinforces
that the table is a set of single observations, not a latency distribution.
The test proves simultaneous reader and writer access across processes on
one host, not transport between physical GPUs or machines.

The full-size logical count includes the 328-byte xinfer envelope relative
to the earlier 296,997,632-byte raw-state estimate. Physical allocated disk
was measured with `du -sh`; the yesno shard file is sparse and has a larger
apparent size. These are cold single runs, not throughput distributions.

## Limits

The API stores a complete state at selected token boundaries. It does not
choose boundaries, search for the longest matching prefix, admit or evict
entries automatically, transfer between machines, or upload restored state to
a GPU. The caller must export attention KV and recurrent state at the same
boundary and validate the model identity. Compact attention quality remains a
separate decision; the exact continuation experiment uses BF16 attention KV.

## Typed token cache follow-up

The typed shifou API hashes exact UTF-8 input bytes and stores `u32` IDs in
little-endian order. The caller supplies a tokenizer fingerprint covering
files and options; the probe uses SHA-256 of `tokenizer.json` and a fixed
`encode(text, false)` call. It uses the first 32,768 Unicode characters of
the pinned WikiText-2 test corpus, which are 32,796 UTF-8 bytes and produce
7,896 IDs. The scratch probe at
`../yesno/.agents-workspace/tmp/qwen35-kv-probe/src/bin/token_cache_typed.rs`
measures 21 repetitions on one GB10 host with an already loaded tokenizer.

| Item | Typed result |
| --- | ---: |
| Tokenization median | 5.385 ms |
| Input hash median | 0.056 ms |
| Warm yesno hit median | 0.123 ms |
| One put with checkpoint | 23.511 ms |
| Read-only open and one get | 0.364 ms |
| Raw ID bytes | 31,584 B |
| Roaring payload and metadata | 36,658 B |

The warm hash plus hit is about 30 times faster than tokenization in this
local setup. The read-only reopen is cheaper than the previous writer-mode
reopen and should not be attributed solely to the bitmap conversion. Results
exclude network transport, concurrent load, and full prompt assembly.

The same release probe was run with the pinned
`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese` tokenizer on 2026-10-03, using
the identical 32,796-byte WikiText input and a fresh database. It produced
7,688 IDs ( 30,752 raw bytes ). Across 21 local repetitions, tokenization
had a 6.781 ms median, input hashing 0.056 ms, and a warm yesno hit 0.121 ms.
The single write with checkpoint took 26.924 ms; read-only open plus get took
0.328 ms. The Roaring payload was 32,820 bytes, or 36,668 bytes with metadata.
This verifies that the typed token cache works with a second tokenizer; it
does not test Nemotron model execution or chat-template expansion.

## Release Nemotron cache admission experiment, 2026-10-04

The scratch harness at `../yesno/.agents-workspace/tmp/nemotron-probe` was
built with `cargo build --release --offline --locked` and
`RUSTFLAGS='-C target-cpu=native'`, against published xinfer
`17499e450a174e25be333f88c654ff6743fd4465`.
It used NVIDIA-Nemotron-Nano-9B-v2-Japanese on one GB10, a fresh yesno
directory per run, separate source and restored model instances, a Japanese
instruction/tool-definition prompt for prefixes above 20 tokens, and a short
Japanese instruction prompt for 20 and 8 tokens. The same source model's fresh
prefill was compared with a complete shifou bundle read and import into the
second model. Model loading, attention export, bundle publication, and four
paired continuation steps are excluded from the hit time. All runs produced
exact logits for the four continuation steps (maximum absolute difference 0).
The source and target model loads took about 22-44 seconds each and are
excluded from both sides.

| Prefix | Fresh prefill | Full verified read, including reader open | Mamba plus attention import | Complete cache hit | Fresh / hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.484 s | 0.547 s | 0.178 s | 0.725 s | 0.67x |
| 20 | 0.733 s | 0.550 s | 0.182 s | 0.732 s | 1.00x |
| 32 | 0.951 s | 0.548 s | 0.187 s | 0.735 s | 1.29x |
| 128 | 2.970 s | 0.568 s | 0.216 s | 0.784 s | 3.79x |
| 512, run 1 | 10.200 s | 0.559 s | 0.185 s | 0.744 s | 13.71x |
| 512, run 2 | 10.551 s | 0.566 s | 0.183 s | 0.749 s | 14.09x |

The Mamba state was 145,539,320 bytes for every prefix. BF16 attention bytes
rose from 262,144 at 8 tokens to 8,650,752 at 512; 21 records were read for
every bundle. For each run, three additional reads on the already-open reader
took 0.573-0.597 seconds with full verification and 0.082-0.109 seconds with
`StorageOnly`. The latter skips shifou's payload digests while retaining
storage checks; it is a trusted-local option, not a default for an untrusted
remote cache. These extra reads were checked against the original bundle but
were not the buffers used for GPU import. Bundle publication took 1.025-3.390
seconds across the single-run prefix sweep and 1.083-1.094 seconds in the two
512-token runs; do not interpret the single-run spread as a scaling law.

This corrects the earlier debug-build conclusion above: release-mode verified
reads take roughly 0.55-0.57 seconds, not 14 seconds. A simple model-specific
admission threshold lies near 20 tokens for this configuration, though the
single-run short-prefix timings need replication before fixing a policy. The
approximately fixed 145.5 MB recurrent state makes read and import latency
nearly independent of prefix length here. The bundle read followed its write in
the same process, so the host page cache was warm; the result is a local
steady-state hit estimate, not a cold-disk or Flight transport result. It also
excludes scheduling and cache lookup.

Reproduce with a fresh database path using the release executable after
building the scratch harness:

```sh
../yesno/.agents-workspace/tmp/qwen35-kv-probe/target/release/nemotron-probe \
  ../yesno/.agents-workspace/tmp/models/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
  512 ../yesno/.agents-workspace/tmp/nemotron-probe/new-release-db bundle
```


## Admission economics from the release Nemotron runs, 2026-10-04

A prepared prefix is worthwhile to publish for latency when expected future
hits times the per-hit saving exceeds publication time, provided the entry fits
the storage and memory budgets. For one measured tier, let `saving = fresh
prefill - cache hit`; if positive, the observed payback count is
`ceil(publication / saving)`. This excludes eviction opportunity cost, request
queueing, and concurrent work. It is a diagnostic calculation, not a production
admission policy.

| Prefix | Observed saving per hit | Bundle publication | Hits to repay publication |
| ---: | ---: | ---: | ---: |
| 8 | -0.241 s | 1.025 s | No latency payback |
| 20 | 0.001 s | 1.037 s | Indistinguishable from a tie |
| 32 | 0.216 s | 3.390 s | 16 at these single-run values |
| 128 | 2.186 s | 1.929 s | 1 |
| 512, run 1 | 9.456 s | 1.094 s | 1 |
| 512, run 2 | 9.802 s | 1.083 s | 1 |

The 32-token publication time varied substantially from the other runs and
must not become a hard threshold. The read followed a write on the same host,
so this table applies only to warm local storage. A cold or Flight hit may
shift the crossover upward. The Qwen3-0.6B comparison in the README also shows
that the weight format changes the answer: BF16 favored recomputation for its
tested long prompts, while NVFP4 favored restoration. The next useful test is
to measure the same complete path under cold local and Flight conditions, then
fit per-model, per-tier costs and validate decisions on a held-out request
trace. A later runtime policy should choose `recompute`, `restore`, or `publish`
from those measured costs rather than a global token-count cutoff.

## Release Nemotron Flight admission experiment, 2026-10-05

The scratch harness at `../yesno/.agents-workspace/tmp/nemotron-probe` was
extended to read a complete shifou `PrefillBundle` through yesno-flight on
loopback. It used the same published xinfer revision, Nemotron 9B model, GB10,
release settings, and separate source and target model instances as the local
experiment above. Each prefix got a fresh yesno directory. The writer was
closed before the Flight server opened it, so the 42 requests per bundle saw a
stable database. The Flight path fetched 21 packed records as metadata and
bitvectors, reconstructed every buffer, and checked the shifou metadata,
payload, token, and aggregate state SHA-256 digests. It imported the fetched
bundle into the second model and compared four continuation logits with fresh
prefill. Maximum absolute logit difference was zero in every run.

The table uses the first Flight read and the GPU import paired with it for the
cache-hit comparison. The local read is a full verified read that includes
`CacheReader` open. Two additional Flight reads were taken on the already
connected client for each prefix; their range shows single-run variation.
Model loading, server startup, Flight connection setup, publication, and
continuation decoding are excluded from hit latency. The page cache was warm;
the server and client ran on the same host over loopback.

| Prefix | Fresh prefill | Local verified read | Flight read, first | Other Flight reads | Mamba + attention GPU import | First Flight hit | Fresh / Flight hit |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.578 s | 0.551 s | 1.665 s | 1.836–2.016 s | 0.188 s | 1.853 s | 0.85x |
| 80 | 1.882 s | 0.553 s | 1.598 s | 1.864–1.977 s | 0.190 s | 1.788 s | 1.05x |
| 128 | 2.840 s | 0.558 s | 1.718 s | 1.655–1.662 s | 0.222 s | 1.940 s | 1.46x |
| 512 | 10.240 s | 0.568 s | 1.703 s | 1.648–1.656 s | 0.181 s | 1.884 s | 5.44x |

The Mamba state was 145,539,320 bytes at every prefix; attention ranged from
1,310,720 bytes at 64 tokens to 8,650,752 bytes at 512. At 64 tokens, the
first Flight hit lost to recomputation; at 128 it won. The 80-token run was
within variation: its first hit won by 94 ms, while its later Flight read
samples plus the same import time would lose. These observations bracket the
loopback crossover between 64 and 128 tokens, without establishing a stable
threshold. They also show why the local 20-token crossover cannot be reused
for a transport tier. At 512 tokens, publication took 1.139 seconds; the
observed first-hit saving was 8.356 seconds, so one future hit repays this
single-run publication time under the latency-only calculation above.

A read-only profile against the same 512-token database, with three complete
Flight reads after connection, split the 1.663–1.928 second total as follows:
42 `prepare_key` calls took 20 ms combined, fetch plus Arrow decoding and
buffer copy took 1.083–1.218 seconds, and SHA-256 checks took 0.506–0.638
seconds. The decoded bitvectors totaled 154,533,888 bytes per read. The
remaining approximately 50 ms covered manifest parsing, buffer assembly, and
other client work. Fetch and decode is the largest measured component; the
profile does not separate server bitmap extraction, Flight serialization,
loopback transfer, and client Arrow decoding. The digest cost is large enough
that a trusted-tier verification policy warrants separate measurement, but
skipping digests is not an acceptable default for an untrusted remote cache.
Sequential per-record Flight fetches and aggregate digest work are the next
places to experiment; query preparation itself is only about 1% of total time.

Reproduce the 512-token full run with a fresh database path:

```sh
../yesno/.agents-workspace/tmp/qwen35-kv-probe/target/release/nemotron-probe \
  ../yesno/.agents-workspace/tmp/models/NVIDIA-Nemotron-Nano-9B-v2-Japanese \
  512 ../yesno/.agents-workspace/tmp/nemotron-probe/new-flight-db bundle-flight
```

Use `flight-read-only` in place of `bundle-flight` against an already populated
bundle directory for the timing split. This is a same-host, warm-cache
transport result, not a cross-machine network or cold-storage measurement.

### Bounded parallel Flight page reads, 2026-10-05

The scratch client then fetched the independent 8 MiB recurrent-state pages with
at most four concurrent Flight requests over clones of one connected gRPC
channel. Bundle metadata, tokens, and attention remained sequential. It
reassembled pages by index before checking the aggregate digest, so the
returned bundle still passed every full checksum and the three-read equality
check. This is an experiment in the scratch harness, not a shifou runtime
change. The per-stage counters in concurrent runs sum overlapping work and
must not be added to obtain wall time.

| Prefix | Sequential Flight reads, 3 samples | Four-way page Flight reads, 3 samples |
| ---: | ---: | ---: |
| 64 | 1.665, 1.836, 2.016 s | 0.897, 1.074, 0.950 s |
| 80 | 1.599, 1.864, 1.977 s | 0.887, 0.977, 0.902 s |
| 128 | 1.718, 1.662, 1.655 s | 0.898, 0.901, 0.960 s |
| 512 | 1.703, 1.656, 1.648 s | 0.927, 0.964, 0.956 s |

The 512-token sequential read was repeated after the parallel run and yielded
1.954, 1.841, and 1.623 seconds. Thus the four-way gain is larger than the
observed run-to-run variation, although these are same-host observations and
not a throughput test under competing requests.

Fresh full inference runs checked both sides of the new admission boundary.
They used separate source and restored Nemotron instances, fresh databases,
first parallel Flight reads, and four exact continuation logits:

| Prefix | Fresh prefill | First parallel Flight read | GPU import | Complete Flight hit | Publication | Max logit difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.957 s | 0.892 s | 0.183 s | 1.075 s | 1.038 s | 0 |
| 64 | 1.695 s | 0.833 s | 0.184 s | 1.017 s | 1.040 s | 0 |

On this setup, the four-way Flight crossover is between 32 and 64 tokens.
The 64-token first hit saved 0.678 seconds, so two hits repay the observed
1.040-second publication latency under the same latency-only calculation.
This is a bracket from single fresh runs, not a hard policy threshold. The
read-only results for 80, 128, and 512 tokens show the transport improvement
but do not repeat GPU inference, so the earlier exact-logit checks remain the
accuracy evidence for those prefixes.

Reproduce the parallel run with `bundle-flight-parallel` as the final argument
to the full-run command above and a fresh directory. Use
`flight-read-only-parallel` against a populated bundle for three read-only
samples. The next runtime step is a bounded concurrent Flight bundle reader
with resource and consistency limits: avoid 18 simultaneous 8 MiB buffers,
and bind all record reads to one immutable published bundle version if writers
can update the same address during a fetch. This experiment closed the writer
first, so it did not test concurrent updates.

### Concurrency, memory, and version follow-up, 2026-10-05

The scratch Flight reader now permits `FLIGHT_PAGE_CONCURRENCY` from 1 through
16 and writes each completed page into its indexed slice of one preallocated
state buffer. It checks page lengths before copying and checks the complete
state digest afterward. Against the same populated 512-token Nemotron bundle,
with three full verified reads per setting, the sweep was:

| Concurrent state pages | Three read times | Median | Client process peak RSS |
| ---: | ---: | ---: | ---: |
| 1 | 1.884, 1.973, 1.687 s | 1.884 s | 613 MB |
| 2 | 1.038, 0.988, 1.295 s | 1.038 s | 613 MB |
| 4 | 0.869, 0.969, 0.968 s | 0.968 s | 613 MB |
| 8 | 1.004, 0.948, 0.909 s | 0.948 s | 615 MB |
| 16 | 0.975, 0.973, 0.938 s | 0.973 s | 664 MB |

Four is a reasonable bound here: eight and sixteen gave no clear latency gain
against the run-to-run spread, while sixteen raised peak client RSS. A
controlled alternating comparison at four-way concurrency measured
collect-all-pages-then-copy at 697 and 697 MB peak client RSS, versus 613 and
613 MB with direct indexed assembly. Their read times overlapped. This RSS
number is for the client process, not server plus client, and reflects the
three-read harness retaining the first bundle to compare subsequent reads.
Direct assembly removed approximately 84 MB of observed peak client RSS in
this fixture without an observed latency cost.

The reader also tested pinning every record to the exact version returned by
its first `prepare_key`. It uses public yesno-flight `Ticket::whole_key` with
`SetWire::Bitvector` for later fetches. This avoids 41 subsequent query
preparations per bundle while keeping the bare-key bitvector path. In two
alternating pinned/ordinary pairs at four-way concurrency, the 512-token
pinned reads were 0.824–0.893 and 0.851–0.871 seconds; ordinary reads were
0.898–0.980 and 0.918–0.955 seconds. Both paths decoded 154,533,888 bytes
and checked the same full bundle digests. The pinned path prepared one key,
versus 42 in the ordinary path. Read-only pinned samples at 32 tokens were
0.930, 0.809, and 0.853 seconds; at 64 they were 0.856, 0.808, and 0.837
seconds. With the separately measured GPU import costs, 32 tokens remains
near the recompute/restore boundary and 64 favors restoration. These pinned
read-only samples are not new paired GPU runs.

A separate small live-writer Flight fixture in the scratch crate verified the
consistency property directly. Two keys initially contained ordinal 1 at
version 1. After a removal and replacement advanced the database to version
3, version-1 tickets for both keys returned bit `0b10`, while current queries
for both returned `0b100`. Thus a bundle reader can use one exact version
across its keys even while writes advance. A Flight ticket is not a permanent
snapshot lease: if that version is reclaimed before a fetch, the reader must
fail and restart the entire bundle at a new version rather than quietly mix
versions. The fixture did not exercise reclamation or a full-size concurrent
writer. A production shifou reader still needs that retry and a bounded
memory budget.

Reproduce the sweep by setting `FLIGHT_PAGE_CONCURRENCY` and running
`flight-read-only-parallel` against the populated 512-token directory named
above. Set `FLIGHT_COLLECT_PAGES=1` for the older collect-and-copy assembly,
or `FLIGHT_PIN_VERSION=1` for exact-version tickets. The live-writer fixture is
`../yesno/.agents-workspace/tmp/nemotron-probe/src/bin/flight_version_probe.rs`;
it accepts a fresh scratch database path.

### Reclamation and steady-state admission follow-up, 2026-10-05

The version fixture was extended to force checkpoint reclamation. It disabled
the service's normal 30-second ticket lease, wrote two keys at version 1,
advanced both to version 3, and checkpointed. The database read floor became
3. A hand-built version-1 Flight bitvector ticket then failed with a tonic
`FailedPrecondition` status rather than returning newer data. Restarting the
whole two-key read at a newly prepared version 3 returned the new bits for
both keys. This proves the scratch client's fail-closed and whole-read-retry
shape on a small fixture; it does not exercise a 145 MB bundle or a writer
racing individual page transfers. Normal ticket leases make this failure less
likely during a short read but do not make tickets permanent leases.

The initial paired fresh-prefill runs varied enough to mislead admission at
short prefixes: one 32-token run took 0.957 seconds and another 1.257 seconds.
A separate release-mode sweep loaded Nemotron once, cleared its recurrent
state, allocated a fresh attention cache, and synchronized the GPU before and
after every prefill. It ran four rounds per prefix; the first round was
warmup, and the table uses the median of rounds 1–3. Two sweeps covered the
wider and narrower ranges. The 40-, 48-, and 64-token values below come from
the narrower sweep; 32 comes from the wider one.

| Prefix | Repeated fresh prefill median | First pinned four-way Flight read | GPU import | First Flight hit | Assessment |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0.692 s | 0.873 s | 0.180 s | 1.053 s | Recompute faster |
| 40 | 0.857 s | Not measured | Not measured | Not measured | Recompute likely faster |
| 48 | 1.015 s | 0.817 s | 0.188 s | 1.005 s | Within read variation |
| 52 | 1.090 s | 0.861 s | 0.191 s | 1.052 s | 38 ms saving in one run |
| 56 | 1.176 s | 0.924 s | 0.192 s | 1.116 s | 60 ms saving in one run |
| 64 | 1.358 s | 0.856 s | 0.184 s | 1.040 s | About 0.32 s saving |

The 64-token median here is from the narrower sweep ( 1.358 seconds ); the
wider sweep measured 1.309 seconds. Its Flight read and import are from the
pinned read-only run and the earlier exact-logit full run, rather than one
new paired pinned inference run. Full pinned replay runs at 32, 48, 52, and 56
tokens all restored the bundle into a separate model and produced four exact
continuation steps with maximum absolute logit difference zero. The 48-, 52-,
and 56-token runs published in 1.063, 1.050, and 1.050 seconds respectively.
At 52 and 56 tokens, the observed tens-of-milliseconds saving would require
many future hits to repay publication and is smaller than transport variation.
The repeated-prefill evidence supports a conservative admission point near
64 tokens for this model and same-host Flight tier, not a universal threshold.
At 64 tokens, using the 1.309-second wider-sweep prefill median, a
1.040-second pinned hit, and the earlier 1.040-second publication observation
gives about four future hits to repay publication. This calculation excludes scheduling,
contention, cold storage, cross-machine networking, and memory opportunity
cost. It supersedes the earlier two-hit estimate based on a one-off 1.695-second
fresh prefill.

Reproduce the fresh sweep with mode `prefill-sweep` or `prefill-narrow` on the
Nemotron scratch executable; the third argument is ignored in those modes.
Run full pinned replay with `FLIGHT_PIN_VERSION=1`,
`FLIGHT_PAGE_CONCURRENCY=4`, and mode `bundle-flight-parallel` against a fresh
database directory. The exact-version failure fixture is the scratch
`flight_version_probe` binary; it requires a fresh database path.

### Optional shifou Flight reader, 2026-10-05

The scratch reader's measured path now has a shifou implementation behind the
`flight` feature: `FlightBundleReader`. It accepts a connected tonic channel or
connects to an endpoint, defaults to four concurrent state-page requests, and
can bound state bytes and total logical bundle bytes. The default total logical
bundle limit is 2 GiB. It fetches the top-level packed manifest with
`prepare_key`, uses that ticket's exact database version for every subsequent
packed record, and reassembles pages directly by index. All packed metadata,
payload, token and whole-state SHA-256 checks remain enabled. A
`FailedPrecondition` during the read discards all partial pages and retries the
entire bundle once at a newly prepared version. Other errors fail closed.
Because a ticket is not a permanent lease, a second reclamation is returned
to the caller. The size limits bound logical content; Arrow and in-flight
buffers make peak process RSS higher.

The feature adds a client-only yesno-flight dependency to normal shifou
builds. The integration test uses a real yesno-flight server, publishes three
nonuniform state pages plus attention in one batch, and compares the entire
returned bundle. It also checks missing-bundle, format, memory-limit and
missing-page behavior. `cargo test --all-features --offline`, the default
`cargo test --offline`, `cargo clippy --all-targets --all-features --offline --
-D warnings`, and `cargo fmt --check` passed. The scratch Nemotron harness was
rebuilt in release mode with the feature enabled; no yesno source changed.

On the already verified 512-token Nemotron bundle, three production-reader
Flight reads took 0.877, 0.876, and 0.902 seconds. After the total-bundle
budget was added, a locked release rebuild measured 0.853, 0.857, and 0.854
seconds against the same database. A fresh 64-token run compared its Flight
bundle byte-for-byte with the local verified bundle, restored a separate
Nemotron instance, and matched four continuation steps with maximum absolute
logit difference zero. Its first production Flight read took 0.845 seconds,
Mamba plus attention import 0.182 seconds, and one-off fresh prefill 1.612
seconds. Bundle publication took 1.094 seconds. As in the prior experiments,
the server and client shared one GB10 host and warm page cache; server startup
and connection setup were excluded from read timing. This proves the shifou
API on a real model, not cross-host performance or concurrency under load.

The remaining integration work is in the serving engine: choose an admission
policy from steady-state costs, authenticate and operate a Flight endpoint,
and connect the returned host bytes to its GPU import path. A failure after
two reclaimed versions is surfaced to the caller so it can recompute rather
than using an incomplete bundle.

A deterministic integration test now covers the production retry, beyond the
small raw-ticket fixture above. A zero-lease Flight server intercepts the first
bitvector fetch after the bundle's initial versioned query, commits an
unrelated write, and checkpoints. The old version is reclaimed before the
fetch executes. `FlightBundleReader::get_prefill_bundle` retries its complete
read at the new version and returns the original published bundle exactly;
`cargo test --features flight --test flight reclaimed_first_version_restarts_the_whole_bundle`
passes. The unrelated write leaves the bundle bytes unchanged, so the test
isolates version reclamation rather than publication replacement.

### Concurrent Flight bundle readers, 2026-10-05

The release-mode Nemotron scratch probe exercised the production
`FlightBundleReader` against the verified 512-token bundle above. It started
one yesno-flight server on `127.0.0.1`, connected N independent readers,
released them together through a barrier, and completed three rounds per N.
Each read fetched and verified all packed pages, tokens, attention buffers,
and whole-state SHA-256. The bundle has 154,192,120 logical bytes per reader,
including 145,539,320 state bytes. Process setup, server startup, and client
connection setup were outside each round's clock. The server and clients ran
on the same GB10 host with a warm page cache; no GPU import or inference ran.
The table uses the median round wall time and throughput, the median of each
round's per-reader median latency, and the `/usr/bin/time` peak RSS reported
for that invocation. RSS is not an isolated client/server allocation split.

| Concurrent readers | Group wall | Per-reader median | Aggregate logical throughput | Peak process RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.963 s | 0.938 s | 160 MB/s | 587 MiB |
| 2 | 0.983 s | 0.962 s | 314 MB/s | 794 MiB |
| 4 | 1.084 s | 1.011 s | 569 MB/s | 1,131 MiB |
| 8 | 1.252 s | 1.098 s | 986 MB/s | 2,081 MiB |
| 16 | 1.723 s | 1.510 s | 1,432 MB/s | 3,798 MiB |

Throughput continues increasing through 16 readers, but the 8-to-16 step
raises median reader latency by about 0.41 seconds and process RSS by about
1.7 GiB. Four to eight simultaneous restores are a reasonable initial
same-host serving range: eight nearly doubles aggregate throughput over four
while keeping the median read near 1.1 seconds. This is a workload observation,
not a production concurrency limit. The measurements do not establish
cross-machine throughput, cold-storage behavior, tail latency under sustained
arrival, or the latency of GPU import and inference while these reads run.

To reproduce, build the scratch `nemotron-probe` in release mode and run
`FLIGHT_LOAD_CLIENTS=N nemotron-probe <model_dir> 512 <verified_db_dir>
flight-library-load` for N in 1, 2, 4, 8, and 16. The executable is under
`yesno/.agents-workspace/tmp/nemotron-probe`; the verified database for this
run is `full-bundle-db-512-flight-20261005-a` in that directory. Each command
starts its own Flight server and makes three rounds. No production yesno code
was changed for this measurement.

### Yesno peer socket on the Nemotron bundle, 2026-10-05

A scratch peer client restored the same verified 512-token Nemotron bundle
through yesno's real Unix plugin socket. A release-mode `yesnod` served the
existing database with a shared memfd arena, 64 lanes per handle, and 16 blocks
per response. The client opened one snapshot for the complete bundle, first
read the top-level manifest, then scanned all remaining packed metadata and
payload keys as lanes under that snapshot. It checked packed metadata and
payload SHA-256 values, token and whole-state hashes, page sentinels, and the
state-page layout. The resulting `PrefillBundle` was byte-for-byte equal to a
subsequent production `FlightBundleReader` restore. Three single-reader peer
reads with direct state assembly took 0.661, 0.585, and 0.593 seconds; the
Flight checks in that correctness run took 0.871, 0.909, and 0.945 seconds.

The first peer decoder staged all decoded bitvectors before constructing the
state. Its eight-reader median group time was 1.066 seconds and the invocation's
peak RSS was 3,308,300 KiB. A scratch-only direct-assembly variant writes state
page lanes into the final output allocation, keeps the sentinel outside the
payload, and verifies each page and the whole state. It passed the same exact
bundle comparison. The table shows this variant with three barrier-synchronized
rounds at each reader count. Connection and server startup were excluded from
each round; the server and clients shared a warm-page-cache GB10 host. The
four-reader entry uses a second three-round run: its first run contained a
1.240-second outlier, while the repeat was 0.697–0.736 seconds.

| Readers | Peer group wall | Peer reader median | Peer logical throughput | Peer invocation peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.616 s | 0.596 s | 250 MB/s | 478 MiB |
| 2 | 0.661 s | 0.610 s | 466 MB/s | 636 MiB |
| 4 | 0.704 s | 0.635 s | 876 MB/s | 952 MiB |
| 8 | 0.882 s | 0.705 s | 1,399 MB/s | 1,584 MiB |

A same-session refresh of the production Flight reader measured group-wall
medians of 0.967, 1.046, and 1.166 seconds at one, four, and eight readers,
respectively; its corresponding logical throughputs were 159, 589, and 1,058
MB/s, and invocation peak RSS values were 601, 1,115, and 2,162 MiB. The peer
direct-assembly group throughput was 1.32 times Flight's at eight readers, and
the reported peak RSS was about 0.56 GiB lower. These are whole-read comparisons,
not an isolated transport benchmark: the peer uses a bespoke scratch decoder
and a release `yesnod`, while Flight uses shifou's production reader and the
scratch Flight server. `/usr/bin/time` RSS is for each invocation, not an
isolated client allocation split. Neither path imported to the GPU in this
concurrency run.

The peer's default admission limit is eight connections, so this run did not
try 16 peer readers. Its batched scan used about 75 socket requests and
processed 1,054 prefix blocks per full restore. In the single-reader profile,
the two scans took roughly 0.08–0.13 seconds of a 0.59–0.66-second read; packed
validation, hashes, and output construction account for the rest of the measured
interval. The direct-assembly gain shows that client staging matters at least
as much as the socket choice for this dense bundle.

Reproduce with the release `yesnod` and the scratch `nemotron-probe` in
`yesno/.agents-workspace/tmp/nemotron-probe`: set `PEER_DIRECT_STATE=1` and run
mode `peer-read-only` for exact Flight equality, or set
`PEER_LOAD_CLIENTS=N` and run mode `peer-load` for N in 1, 2, 4, and 8. The
verified database is `full-bundle-db-512-flight-20261005-a` there. This
section records the scratch reader before its production counterpart was
implemented. No production yesno source was edited for the experiment.

### Production peer reader, 2026-10-05

Shifou now exposes synchronous `PeerBundleReader` behind the optional `peer`
feature. It connects to yesno's Unix plugin socket, uses one server-owned
snapshot for the whole bundle, and supports both the memfd arena and inline
fallback. It scans packed metadata first, checks identity, format, record
lengths and the total logical bundle budget, then scans payload lanes in
batches. State lanes go directly into the final output allocation; token,
packed-record, page and whole-state SHA-256 checks remain enabled. On any
read failure the connection is dropped, releasing its snapshot and handles;
the caller reconnects to retry. This makes a checkpoint during the read safe
without Flight's version-reclamation retry, because the server holds the
snapshot until the socket releases it.

The new real-socket integration test runs against a `yesno-plugin` session on
a Unix listener, exercises both arena and inline responses, forces multiple
lane batches with a three-lane limit, round-trips a state of 8 MiB plus 17
bytes and a multi-buffer attention record, checks an absent address and the
total-bundle budget, and rejects a missing published state page. A separate
deterministic case deletes a page and checkpoints immediately after the peer's
snapshot opens: the in-flight read returns the original complete bundle, while
the next read sees the deletion and fails. The full
shifou all-feature and default test suites, all-target all-feature Clippy
with warnings denied, and formatting check passed.

The release-mode Nemotron probe was rebuilt against this production reader.
Three 512-token reads took 0.608, 0.589 and 0.586 seconds and returned a
`PrefillBundle` byte-for-byte equal to the production Flight reader; paired
Flight reads took 0.861, 0.823 and 0.818 seconds. The following medians are
from three barrier-synchronized rounds per concurrency point, with connection
and server startup excluded. The `yesnod` peer server used 64 lanes and 16
blocks per response, shared one GB10 host with the readers, and read warm
cached data. Each reader verified 154,192,120 logical bytes.

| Readers | Group wall | Per-reader median | Logical throughput | Invocation peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.612 s | 0.589 s | 252 MB/s | 478 MiB |
| 2 | 0.634 s | 0.596 s | 486 MB/s | 635 MiB |
| 4 | 0.691 s | 0.625 s | 893 MB/s | 951 MiB |
| 8 | 0.853 s | 0.690 s | 1,447 MB/s | 1,583 MiB |

The eight-reader first round reached 1.040 seconds; the other two were 0.853
and 0.850 seconds. The median table therefore describes warm repeated reads,
not a tail guarantee. The production reader retained the scratch decoder's
speed and memory benefit. This is an exact host-byte restore; no GPU page
import or inference ran in this load sweep. The serving engine still needs a
peer-socket lifecycle and a path from returned host bytes into its GPU import.
After a checked-offset protocol hardening edit, a repeat measured 0.610 seconds
for one reader and 0.852 seconds for eight, with the same logical-byte totals;
the full peer socket integration test and all-feature lint still passed.

### Production peer reader with Nemotron GPU replay, 2026-10-05

The release `nemotron-probe` gained a `peer-library-replay` mode. It loads a
source `NVIDIA-Nemotron-Nano-9B-v2-Japanese` BF16 model, prefills the prompt,
loads a separate target model, reads a previously verified `PrefillBundle`
through shifou's production `PeerBundleReader`, imports Mamba state and the
attention buffers into target GPU memory, then compares four teacher-forced
continuation logits from the source and target. The bundle's token IDs and
prefix length are checked before import. Every run had maximum absolute logit
difference zero. This is an exact continuation check for four tokens, not a
task-level accuracy evaluation.

The model and peer server were already loaded before the timed read. The server
and client shared one GB10 host and a warm OS page cache. Each run made three
reads on one connection; the table's peer read is the third, steady read. GPU
synchronization bounds prefill and import timings. All times are seconds;
`hit` includes the third peer read, both GPU imports, and four restored decode
steps; `fresh` includes prefill and four fresh decode steps. Each row is one
process run, so small differences are not a stable admission threshold.

| Prefix | First / third peer read | Mamba / attention import | Fresh prefill | Four fresh / restored steps | Fresh / hit end to end | Logit difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0.922 / 0.581 | 0.290 / 0.003 | 0.974 | 0.341 / 0.421 | 1.315 / 1.295 | 0 |
| 48 | 0.867 / 0.585 | 0.360 / 0.005 | 1.281 | 0.384 / 0.437 | 1.666 / 1.387 | 0 |
| 64 | 0.871 / 0.566 | 0.297 / 0.003 | 1.566 | 0.416 / 0.440 | 1.982 / 1.306 | 0 |
| 512 | 0.643 / 0.597 | 0.289 / 0.004 | 10.415 | 0.334 / 0.415 | 10.749 / 1.303 | 0 |

Mamba state was 145,539,320 bytes at every prefix. Attention ranged from
786,432 bytes at 32 tokens to 8,650,752 bytes at 512. The 512-token hit was
8.25 times faster than recomputing in this paired run. At 64 tokens, the third
read plus import took 0.866 seconds before decoding. At 32 and 48 tokens,
one-off fresh-prefill times in this table were slower than the repeated
prefill medians above ( 0.692 and 1.015 seconds ). Using those medians, 32
tokens favors recomputation and 48 has only about 0.07 seconds of apparent
restore margin before decode variation. The first read plus import also loses
to the repeated 48-token prefill. A conservative admission point remains near
64 tokens for this model and same-host peer tier; measure it again under
serving contention before treating it as policy.

To reproduce, build the release scratch probe under
`yesno/.agents-workspace/tmp/nemotron-probe` and run
`nemotron-probe MODEL_DIR PREFIX DB_DIR peer-library-replay`. The 32-, 48-,
64-, and 512-token verified databases already exist in that scratch directory
under their `full-bundle-db-*` names. This mode uses the production peer reader
and imports state into a separate xinfer model instance. It excludes model
load, cache publication, peer server startup, and cross-host transport from
the hit timing.

## Prepared-prefix and session-generation workflow, 2026-10-05

Shifou now provides an exact token-chain `PrefixScope`, an atomic prepared
prefix length index, and a metadata-only longest-prefix hit. A caller can
price the hit before reading its full bundle. `SessionKey` names a private
session; `put_session_checkpoint` publishes the complete state, opaque
workflow bytes and current-generation pointer in one yesno batch. A reader
uses one snapshot for both pointer and state. `prune_session_checkpoint`
refuses to remove the current generation. `AdmissionEstimate` compares
caller-supplied first-token costs and expected future hits. The storage code
borrows existing attention snapshot buffers while preparing a batch; it does
not clone an entire KV cache merely to rewrite its address identity.

The original full-checkpoint run of the real-model probe retained at
`experiments/xinfer/src/bin/session_workflows.rs` used published xinfer
`17499e4`, two independently loaded Qwen3-0.6B model instances on one GB10,
and the WikiText-2 test text as its token source. NVFP4 quantizes weights;
the stored attention KV remains exact BF16. It published a 1,980-token
prepared prefix, found it for a request with a 32-token suffix, restored and
appended that suffix, saved the resulting session as generation 1, and then
saved generation 2 after another token. The retained probe now uses an append
checkpoint for generation 2; the full-checkpoint figures below come from the
earlier version. Both restored generations matched the
source continuation logits exactly. The prepared-host path also asserted exact
continuation logits. The workflow bytes were a small test marker; this probe
did not run real tool calls or capture a serving application's transcript.

Three sequential warm samples per arm were taken after model load and initial
CUDA work. Each durable hit includes metadata lookup, one yesno read, BF16 GPU
restore and the next model operation. `StorageOnly` was selected for this
trusted same-host probe; it retains yesno storage checks but skips shifou's
payload digests. A prepared-host hit uses xinfer's validated BF16 host slices
after a one-time verified read and preparation. Median milliseconds from
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/result-qwen-nvfp4-1980-final.json`:

| 1,980-token NVFP4 prefix | Fresh prefill | Durable yesno hit | Prepared-host hit |
| --- | ---: | ---: | ---: |
| New session plus 32-token suffix | 277.8 | 271.0 | 40.2 |
| Resume generation 2 plus one input token | 274.0 | 274.1 | 22.2 |

The prepared-host path was about 6.9 times faster for initialization and
12.4 times faster for resume in these three warm samples. First-hit host
preparation took 98.5 ms for the prepared prefix and 97.5 ms for the session
generation after its storage read. Initial trusted-local reads took 81.7 and
82.2 ms; each full bundle publication took about 1.17 seconds. Host slices
retain approximately the full BF16 KV byte count in RAM. The host tier is
therefore valuable for repeated hits while the process stays alive; a fresh
process still pays read and preparation before subsequent fast restores.

At a shorter 512-token prefix, three warm full-verified samples favored
recomputation: BF16 startup was 23.5 ms fresh versus 158.7 ms cached, and
NVFP4 startup was 62.5 ms versus 173.2 ms cached. This supports a measured
admission decision rather than a fixed token threshold. The probe excludes
model loading, server scheduling, remote transport, concurrent requests and
cold-disk behavior. Its session resume receives a new token from the corpus;
resuming generation with no new input would also require the application to
persist its pending sampled token or last logits and sampling state.

## Peer-socket prepared and session restore, 2026-10-05

The retained `experiments/xinfer/src/bin/peer_session_workflows.rs` reopens
the existing 512- and 1,980-token Qwen3-0.6B NVFP4-weight databases. A yesno
peer server and two independently loaded xinfer model instances share one GB10
host. The reader uses a server-owned snapshot and shared-memory arena. It
resolves the latest session generation, locates the longest exact prepared
prefix by metadata, then reads the full state and restores BF16 attention KV.
The fresh reference uses the same token chunk boundaries as the saved cache:
prefix, 32-token suffix, and one later token. This matters numerically: one
whole-prefill call at the session boundary differed by 0.25 in maximum logit
value from the originally chunked state, while the matched construction and
both restored paths differed by 0. The workflow payload remains a test marker.

These are individual warm runs after model load and peer-server startup. All
times are milliseconds and include GPU synchronization around model work.
The first-hit total includes metadata lookup (prepared only), full verified
peer read, GPU restore and the next model operation. The prepared-host hit
follows a one-time peer read and `prepare_packed_bf16`, then restores from
resident host slices. Its setup cost is listed separately.

| Prefix | Attention records / bytes | Fresh startup / resume | First peer startup / resume | Prepared-host startup / resume | Host setup startup / resume |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 56 / 58.7 MB | 286 / 111 | 237 / 215 | 31 / 12 | 24 / 26 |
| 1,980 | 56 / 227.1 MB | 352 / 338 | 729 / 702 | 47 / 29 | 99 / 98 |

At 1,980 tokens the first peer read alone took 551 ms for prepared state and
561 ms for session state; prepared-prefix metadata lookup took 0.5 ms. At 512
tokens the reads took 152 and 163 ms. The 1,980-token first hit loses to
recomputation on this small model, whereas a repeated hit from prepared host
pages is much faster. The 512-token first prepared hit only narrowly beats
fresh prefill in this one run, while session resume loses. `AdmissionEstimate`
should use observed tier costs rather than assuming that a cache hit helps.
The peer socket connects workers on one host; it does not provide transport
between separate machines. These runs exclude contention, cold disk and
serving-scheduler effects. The exact raw results are under
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/` in the
`result-peer-qwen-nvfp4-{512,1980}-host.json` files.

The same 1,980-token state was then read from a **separate `yesnod` process**,
rather than the probe's in-process peer thread. The source database was copied
to a fresh scratch directory, keeping its one-shard layout. `yesnod` opened
that copy as leader with its plugin channel on a Unix socket and a shared
memory arena; the probe set `PEER_SOCKET` to that socket, so it opened no
database or server itself. The temporary daemon was stopped after the run.
The exact config is retained at
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/yesnod-peer-1980.toml`,
and the raw result at `result-peer-yesnod-1980.json` beside it.

| Separate-process 1,980-token run | Fresh | First peer hit | Prepared-host repeat hit |
| --- | ---: | ---: | ---: |
| Prepared startup plus 32 tokens | 346 ms | 758 ms | 42 ms |
| Session generation 2 plus one token | 328 ms | 733 ms | 24 ms |

The first peer reads took 588 and 606 ms for 227.1 and 230.9 MB of attention
bytes. Metadata-only prefix lookup took 2.5 ms. Host preparation took 98 and
97 ms after the first read. Both paths matched source logits with maximum
absolute difference 0. This verifies the process boundary and file-descriptor
handoff, but the daemon and GPU workers still shared one host; a Unix peer
socket cannot serve a different machine.

## Append-only session checkpoints, 2026-10-05

The Qwen generation-1 and generation-2 snapshots from the 1,980-token
workflow have 56 attention records and 230,752,256 and 230,866,944 attention
bytes respectively. A scratch comparator under
`../yesno/.agents-workspace/tmp/session-kv-delta-20261005/` compared every
buffer, including its shape and byte content. All overlapping bytes were
identical. The new token appended 114,688 bytes. Splitting by 4 KiB pages
would touch 56 pages, but the actual payload remains one short tail per
record. At 64 KiB pages, changed-page payload would already be 3,325,952
bytes because every attention record crosses a page boundary.

`put_session_append_checkpoint` now stores one cumulative attention-tail
record against the latest full checkpoint, current recurrent-state pages,
complete token IDs, and a session head. The records and head share one yesno
batch. The base full checkpoint remains protected while the current head
depends on it. Intermediate append generations can be pruned without
breaking the current generation because each tail is relative to the full
base. A later full checkpoint resets the base. The append call rejects a
changed prior attention byte, nonextending token stream, changed attention
layout, or a tail over 64 MiB. Local and peer-socket readers validate the
base identity, current boundary, state and tail checksums, and reconstructed
buffer layout. Peer reads use one server-owned snapshot.

The retained `session_workflows` probe was changed only for generation-2
publication. It used Qwen3-0.6B NVFP4 weights, BF16 attention, two model
instances, a 1,980-token prepared prefix, 32-token suffix, and one more token
before generation 2. Full publication of generation 1 took 1,166 ms; append
publication of generation 2 initially took 625 ms. The prior full-checkpoint
run took approximately 1,175 ms per full publication. That first append writer
read and verified its 230.8 MB full base to prove that the candidate state
extended it. The revised writer reads the full base manifest, token record,
and 56 attention manifests, then hashes the corresponding prefix of the new
in-memory buffers against each committed record digest. It does not transfer
the stored attention payload during publication. A multi-buffer regression
proves that a changed earlier byte in the second buffer is rejected. This
preserves the logical prefix proof, but publication no longer audits the
physical base payload; full local and peer restores still validate that
payload before returning it.

On a fresh database with the same Qwen construction, generation-1 full
publication took 1,151 ms and generation-2 append publication took 442 ms,
29% less than the first append implementation and 62% less than full
publication. Reopened generation-2 continuation logits matched the source
exactly. The raw local result is
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/result-qwen-append-meta-1980.json`.
An in-process Unix peer-socket restore of that new database took 553 ms for
the state read and 675 ms through the next token, with maximum absolute logit
difference 0. Its raw result is `result-peer-qwen-append-meta-1980.json` beside
the local result. These are single warm runs; the earlier separate-process
peer run below used the pre-optimization append writer, but the persisted
checkpoint format and peer reader are the same.

The retained `peer_session_workflows` probe then read the new database through
a separate `yesnod` process and Unix peer socket with a shared-memory arena.
The daemon and GPU workers were on one GB10 host. The daemon config and raw
outputs are under
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/` as
`yesnod-peer-append-1980.toml`, `result-qwen-append-1980.json`, and
`result-peer-yesnod-append-1980.json`. Model load, server startup, and
publication were excluded from restore timings.

| Generation-2 resume, 2,013 tokens | Milliseconds |
| --- | ---: |
| Fresh recomputation plus next token | 330 |
| Verified peer read of append generation | 602 |
| BF16 GPU restore plus next token | 124 |
| First peer hit, end to end | 726 |
| Prepared-host repeat hit | 24 |

The peer restore produced maximum absolute logit difference 0. The comparable
earlier full-checkpoint peer read took 606 ms, so append publication did not
improve cold read time: the reader still needs all 230.9 MB of BF16 KV. These
single warm runs do not measure concurrency or cold storage. A prepared-host
copy remains the faster repeated-hit tier; the append format primarily lowers
write cost and storage growth across session generations.

## Repeatable end-to-end peer benchmark, 2026-10-05

`experiments/xinfer/bench_session_e2e.py` composes the retained
`session_workflows` publisher and `peer_session_workflows` reader. It requires
a local Qwen3-0.6B NVFP4 model, a corpus with enough tokens, a built `yesnod`
binary, and a new output directory. It builds the pinned release-mode probes
offline, publishes a fresh prepared prefix, full session generation 1, and
append generation 2, then launches `yesnod` as a separate process and runs
the peer reader three times by default. The server is stopped after the runs.
Each reader process loads two independent model instances and checks restored
continuation logits against a fresh state. The benchmark does not use
`FlightBundleReader`. `yesnod` still binds an unused ephemeral Flight listener
as part of its normal startup; all timed cache reads use the Unix peer socket.

Reproduction from the shifou root:

```sh
python3 experiments/xinfer/bench_session_e2e.py \
  /path/to/Qwen3-0.6B-NVFP4 /path/to/corpus.txt \
  .agents-workspace/tmp/qwen-session-e2e \
  --yesnod /path/to/yesnod --prefix-tokens 1980 --samples 3
```

The runner refuses to overwrite an existing output directory. It retains the
fresh yesno database, daemon config and log, publication JSON, each peer JSON,
and `summary.json`. The summary includes process wall times (which include
model load), per-stage timings, medians, all raw peer samples, and accuracy
checks. Serving first-hit times exclude model load, server startup, and
publication. Host repeat-hit times exclude the one-time peer read and host
preparation. The measurements are sequential on one GB10 and do not measure
cold storage, throughput, or concurrent serving.

The verified three-sample run used the local
`llmat/Qwen3-0.6B-NVFP4` weights and WikiText-2 test text. Its output is
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/e2e-bench-qwen-1980/summary.json`.
NVFP4 applies to weights; the stored attention KV is exact BF16. All three
peer samples used a separate `yesnod` process, and all prepared and session
continuations had maximum absolute logit difference 0.

| 1,980-token prepared prefix, 2,013-token session | Fresh median | First peer hit median | Prepared-host repeat median |
| --- | ---: | ---: | ---: |
| Session initialization plus 32-token suffix | 366 ms | 751 ms | 42 ms |
| Session restore plus next input token | 329 ms | 724 ms | 24 ms |

The median verified peer reads alone were 579 ms for the prepared state and
593 ms for the appended session state. GPU restoration took 110 and 109 ms.
This run published the prepared prefix in 1,219 ms, full session generation 1
in 1,212 ms, and append generation 2 in 439 ms. The first peer hit loses to
recomputation on this small model and prefix; the resident prepared-host tier
wins after its setup cost. The three reader process wall times were about
7.6-7.7 seconds each, including model load and both fresh/cache workflows.

## Cold peer read and pipeline feasibility, 2026-10-05

The three-sample E2E run above puts 579 ms of the 751 ms prepared first hit
and 593 ms of the 724 ms session first hit in the verified peer read. GPU
restore accounts for another 110 and 109 ms. At this model size, matching the
fresh 366 and 329 ms paths while leaving the other stages unchanged requires
the respective peer reads to fall below about 195 and 198 ms, a roughly
threefold read improvement. Merely hiding the 109 ms GPU restore behind a
peer read would leave the session first hit around 615 ms.

One control changed only `yesnod`'s peer `channel_max_lanes` from 16 to 64,
keeping `channel_max_blocks = 16` and the same database, model, peer probe,
and separate-process Unix socket. The arena grew from 8 to 32 MiB. With 56
attention records, this reduces payload groups from four to one and roughly
128 block-advance requests to 32. The three new peer samples are retained
beside `summary.json` as `peer-lanes64-{01,02,03}.json`, with the exact server
config in `yesnod-lanes64.toml`. Medians in milliseconds:

| Peer lanes | Prepared read | Session read | Prepared first hit | Session first hit |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 579 | 593 | 751 | 724 |
| 64 | 568 | 597 | 750 | 732 |

All six session continuation comparisons had zero maximum logit difference.
With only three sequential samples per setting, this is evidence of no large
benefit from reducing the number of requests alone, not a precise throughput
estimate. The present peer server already batches blocks and writes bitmap
words directly into the shared arena. The shifou reader still waits for each
response, copies and verifies full records, and only then calls xinfer's GPU
restore. A useful pipeline would keep at least two arena handles alive so the
server could fill one slot while a client worker decodes and verifies the
other; a single handle's next advance implicitly releases and may overwrite
its previous slot. Streaming verified records into xinfer's GPU importer could
overlap some upload cost as well. The existing blocking server processes one
request at a time on a connection, so sending more requests on that connection
without separate slots and client-side work would not itself create overlap.
Measure server container reads and encoding, client arena copying and SHA-256,
and GPU upload separately before predicting the pipeline's gain.

For a known prepared prompt or a session assigned to a worker before its next
request, a background peer read plus xinfer host preparation can move the
roughly 593 ms read and 95 ms preparation off the request path. The measured
resident-host continuation was 42 ms for initialization and 24 ms for
session restore. Such a warm tier needs an exact model/context or
session-generation key and a memory budget: this example holds about 227 MB
for the prepared prefix and 231 MB for the session state. A genuinely cold,
unpredictable session still pays to obtain its full base checkpoint.

## Shifou-side parallel BF16 restore probe, 2026-10-05

The Qwen experiment adapter's direct `restore_packed_bf16` decodes 56
independent BF16 tiles one at a time, then uploads the resulting tensors.
The prepared-host path has already decoded the tiles, so its 24 ms session
repeat hit minus roughly 17 ms for the next token suggests that much of the
109 ms direct-restore stage in the quieter E2E run was CPU preparation. This
is an inference from two paths, not an isolated decoder timer.

An opt-in `restore_packed_bf16_parallel` now uses Rayon to decode the tiles
on CPU workers while preserving their original layer order; tensor creation
and the final GPU synchronization remain sequential. The peer probe selects
it with `SHIFOU_PARALLEL_RESTORE=1`; `RAYON_NUM_THREADS=4` fixes the worker
count. The default direct restore remains serial. Three serial and three
parallel reader processes were run in alternating order against the same
16-lane `yesnod` process and stored E2E database. The exact config is
`yesnod-parallel-restore.toml` and the raw rows are
`peer-restore-{serial,parallel}-{01,02,03}.json` beside the E2E summary.

| Contended paired run, median ms | Serial decode | Four-worker decode |
| --- | ---: | ---: |
| Prepared GPU restore | 266 | 102 |
| Session GPU restore | 253 | 131 |
| Prepared peer read | 974 | 844 |
| Session peer read | 883 | 862 |
| Prepared first hit | 1,363 | 1,056 |
| Session first hit | 1,168 | 1,026 |

All six runs matched both prepared and session continuation logits with
maximum absolute difference 0. The host load average was above 40 during
this run, and even the serial peer reads were far slower than the earlier
579/593 ms E2E
baseline. The paired restore-stage reduction shows a parallel opportunity,
but these absolute times should not be treated as an unloaded performance
result. It does not change the dominant peer read. Further shifou-side work
could verify independent complete records on bounded CPU workers after each
peer batch, but first needs stage timing to separate SHA-256 from arena
copying and server work. Parallelizing 8 KiB lane copies individually would
add task overhead and may contend for memory bandwidth; the 4 MB record and
BF16 tile are the more natural units.

To isolate that CPU operation, the peer probe gained an optional
`SHIFOU_BENCH_DECODE=1` mode. After its ordinary accuracy checks, one process
decodes the same verified 230,866,944-byte session attention bundle six times
serially and six times with four Rayon workers, alternating which variant
runs first. It times validation, allocation, little-endian BF16 conversion,
and block padding; it excludes peer read, GPU tensor creation and GPU upload.
Each prepared allocation is dropped before the next trial. The raw output is
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/e2e-bench-qwen-1980/peer-decode-isolated-4w.json`.
With `RAYON_NUM_THREADS=4`, median serial decode was 305.4 ms (range
271.4-336.6 ms), and parallel decode was 89.4 ms (range 66.4-91.5 ms), a
3.42x within-process speedup. Host load averaged about 28 at collection time,
so these are contended absolute times. The normal peer restore in this run
still had zero logit difference. This isolates a useful shifou-side CPU
parallelization target; it does not show that parallel decoding can remove
the roughly 580 ms peer read observed in the earlier quiet E2E run.

## Parallel packed-record verification, 2026-10-06

The Qwen peer probe's `SHIFOU_BENCH_HASH=1` control hashes the same verified
230,866,944-byte attention bundle six times in alternating serial and
four-worker order. Its isolated median SHA-256 stage was about 403 ms serial
and 102 ms parallel; raw times are in `peer-hash-isolated-4w.json` beside the
E2E summary above. This is a second hash over already verified bytes, so it
isolates CPU opportunity without representing full peer-read latency.

The peer reader now offers `with_parallel_verification(true)`. After each
16-lane scan, it owns the packed record payloads and applies the existing
length and SHA-256 checks on Rayon workers. It collects results in manifest
order. State pages still land directly in their final allocation and are
verified serially; the socket, server snapshot, and arena reads are unchanged.
The default remains serial, and the probe selects the new path with
`SHIFOU_PARALLEL_VERIFY=1` and `RAYON_NUM_THREADS=4`.

For an end-to-end comparison, three serial and three parallel reader
processes ran in alternating order against one separate `yesnod` serving the
same 1,980-token Qwen3-0.6B NVFP4-weight database. The server had 16 peer
lanes and an 8 MiB arena; its exact config is
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/e2e-bench-qwen-1980/yesnod-parallel-verify.toml`.
The paired raw rows are `peer-verify-{serial,parallel}-{01,02,03}.json` in
that directory. Three further `peer-verify-combined-{01,02,03}.json` runs
enabled parallel BF16 restoration as well. All nine prepared and session
continuations had maximum absolute logit difference 0.

| Three-sample median, ms | Serial | Parallel verification | Verification and BF16 restore |
| --- | ---: | ---: | ---: |
| Prepared peer read | 569 | 278 | 309 |
| Session peer read | 577 | 300 | 287 |
| Prepared GPU restore | 110 | 111 | 43 |
| Session GPU restore | 109 | 109 | 45 |
| Prepared first hit | 741 | 451 | 411 |
| Session first hit | 703 | 431 | 349 |
| Fresh prepared prefill and suffix | 347 | 351 | 339 |
| Fresh session prefill and next token | 336 | 339 | 334 |

The parallel verification comparison removes about 277 ms from the session
read and 291 ms from the prepared read, consistent with the isolated hash
control. Combined session first hit is within 15 ms of fresh recomputation;
prepared first hit remains 72 ms slower. The machine's load average was about
29 during collection. These are warm, sequential, same-host measurements with
model loading and server startup excluded, not an unloaded or concurrent
serving estimate. The combined runs followed the paired runs rather than
alternating with them, so their read-stage difference from the verification
only column is not a controlled effect of BF16 restoration.

## Overlap peer scans and verification, 2026-10-06

The remaining read was split with temporary timers in the Qwen probe. In
three 16-lane runs with four-worker verification, the session data scans took
180 ms median: 78 ms inside the peer block requests and 99 ms assembling the
arena lanes into owned payloads. Packed-record verification took another
131 ms. Prepared scans took 158 ms (67 ms in requests, 89 ms in assembly),
followed by 134 ms of verification. These sums are stage observations, not
an exact decomposition of the public read timer: small metadata scans and
head resolution also run. Raw `peer-profile-{01,02,03}.stages` traces are
beside the E2E summary. The timing code was removed after the experiment.

Two small assembly experiments did not improve the full read. Appending a
contiguous bitmap lane directly, without zero-filling the same 8 KiB first,
changed median session assembly from 99 to 96 ms. Reserving the complete
known payload capacity lowered that stage to 81 ms, but request wait rose
from 78 to 91 ms and the total data scan stayed about 175-180 ms. Their raw
traces and Qwen results are `peer-bitmap-{01,02,03}.*` and
`peer-prealloc-{01,02,03}.*`. Neither variant remains in the reader.

The retained opt-in `with_pipelined_verification(true)` starts one scoped
verification worker for each completed batch. Its owned payloads are safe
to verify while the same `PeerConnection` scans the next batch. Only one
batch worker runs at a time, and Rayon spreads independent records within
that worker across four CPU threads. The reader joins results in manifest
order, retains every length and checksum check, and returns only after all
workers finish. A failed read still closes its peer connection. The probe
selects this mode with `SHIFOU_PIPELINE_VERIFY=1`, which also enables
parallel verification; `RAYON_NUM_THREADS=4` fixes the worker count.

Three non-pipelined and three pipelined reader processes ran in alternating
order against one separate 16-lane `yesnod`, the same database, model, and
four-worker parallel BF16 restore. The retained code had no stage timers.
The exact server config is `yesnod-parallel-verify.toml`, and raw output is
`peer-pipeline-clean-{sequential,pipelined}-{01,02,03}.json` beside the E2E
summary. Every prepared and session continuation had zero maximum logit
difference.

| Three-sample median, ms | Parallel verification | Pipelined verification |
| --- | ---: | ---: |
| Session peer read | 275 | 204 |
| Prepared peer read | 283 | 200 |
| Session first hit | 338 | 264 |
| Prepared first hit | 384 | 301 |
| Fresh session computation | 333 | 332 |
| Fresh prepared computation | 343 | 337 |

The 16-lane pipeline brought both first-hit workflows below recomputation
for this 1,980-token Qwen case. Eight lanes gave 203 ms session and 208 ms
prepared read medians in three subsequent `peer-pipeline-lanes8-*.json`
runs, with a 4 MiB arena; 16 lanes used 8 MiB and read in 204 and 200 ms.
The difference is too small to select eight lanes. A separate 64-lane
parallel-verification run read in 279 and 286 ms, with a 32 MiB arena;
the entire 57-record bundle fits one batch, so this pipeline has no next
batch to overlap at that setting. Its raw rows are
`peer-lanes64-{verify,combined}-{01,02,03}.json`.

These are warm, sequential, same-host measurements on one GB10 with model
load and server startup excluded. The host load average was roughly 28-30
during the stage and lane experiments. The three-run groups are useful for
identifying large differences, but they do not establish concurrent serving
throughput or cold-storage behavior. The pipeline adds one scoped worker
and may hold two owned batch payloads at once; a serving engine should
measure CPU contention and memory at its actual concurrency level.

## Concurrent readers and storage-cold restore, 2026-10-06

The model-free concurrency probe lives at
`.agents-workspace/tmp/peer-concurrency-probe/` in this repository. It uses
the same `PeerBundleReader::with_pipelined_verification(true)` path and the
same generation-2 Qwen checkpoint as the E2E run. Each reader opens its own
Unix peer connection before a barrier, reads and verifies the full session,
checks generation 2 and 56 attention records, and reports 230,866,944
attention bytes. The server is a separate `yesnod` with 16 lanes and an
8 MiB arena. There is no model load or GPU work. Six synchronized rounds
were run at each
concurrency; the first was excluded as warmup. The per-reader timer stops
after the verified data is inspected, while group wall time also includes
the release of the large result buffers. Raw JSONL rows are in
`../yesno/.agents-workspace/tmp/session-workflow-probe-20261005/e2e-bench-qwen-1980/peer-concurrency-r{4,16}-n{1,2,4}.jsonl`.

| Rayon workers shared by all readers | Simultaneous readers | Median reader ms | Median group wall ms | Aggregate GB/s from group wall |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 205 | 220 | 1.05 |
| 4 | 2 | 264 | 309 | 1.49 |
| 4 | 4 | 479 | 523 | 1.76 |
| 16 | 1 | 181 | 213 | 1.08 |
| 16 | 2 | 195 | 216 | 2.14 |
| 16 | 4 | 243 | 291 | 3.17 |

The four-worker pool becomes the dominant shared limit as readers are added:
at four readers, expanding the same-process pool to 16 workers cuts group
wall time from 523 to 291 ms. This is evidence against treating the warm
four-reader slowdown as a yesno server bottleneck. It is not proof that
yesno scales without limit: with 16 workers, four readers still take about
1.37x the one-reader group wall time, and server CPU, client memory bandwidth,
and allocation are not separately identified. Each read returns the same
checkpoint, so this measures warm shared-state contention rather than
multi-tenant key diversity.

For a cold control, the checkpointed database was copied three times to
separate inodes with reflinks disabled under the same E2E output directory:
`cold-copy-20261006/`, `cold-copy-20261006-b/`, and
`cold-copy-20261006-c/`. Each copy was fsynced and given
`POSIX_FADV_DONTNEED` before its own `yesnod` started. One client then read
the session with 16 Rayon workers. The three first-read times were 771,
836, and 781 ms (median 781 ms); each server's major-fault counter rose by
953 during that read (229 to 1,182 in the latter two runs). On the first
copy, five subsequent warm reads had a 187 ms reader median and added only
one major fault in total. The raw first-read JSONL and exact server config
are inside each copy's directory; `warm-followup.jsonl` is in the first.

The storage-faulting first read costs about 4.2x the warm read in this
single-reader control. The result does not isolate disk bandwidth from
fault handling, readahead, or server scheduling, and `DONTNEED` is a
best-effort hint, but the repeated major-fault delta shows the copies were
materially cold. A background prewarm can move this cost off a known
session's request path. For an unpredictable cold session, this is the
strongest measured reason to investigate yesno-side read-ahead or batched
query execution after upstream's planned batched-query work lands. The
current peer path already batches 16 lanes and up to 16 blocks per request;
the new upstream API should be compared against this baseline before adding
another transport mechanism. No yesno source was changed for these probes.

## Cold peer payload prewarm, 2026-10-06

A follow-up to the cold-copy control measured the same generation-2 Qwen3-0.6B
checkpoint with the model-free, fully verified `PeerBundleReader` ( 16 peer
lanes, 8 MiB arena, pipelined verification, 16 Rayon workers ). Three separate
non-reflink copies were reset with `POSIX_FADV_DONTNEED`; `mincore` confirmed
zero resident `.yno` pages before every server start. The control first reads
were 515, 553, and 535 ms ( median 535 ms ) and gave each `yesnod` 953 major
faults. These absolute times differ from the earlier 781 ms cold median as
host conditions changed, but both series were demonstrably cold.

A control read left 238.2 MiB of the 1 GiB logical file resident, including
one 220.5 MiB contiguous run at file offsets 229.938-450.438 MiB. Repeating
the experiment after prewarming only that data run with `pread` cost 255 ms
median in the background and reduced the foreground verified read to 179 ms
with six server major faults. Prewarming only the other observed ranges cost
66 ms, but the foreground read stayed at 514 ms with 946 major faults. A
whole-file `pread` cost 463 ms and left a 170 ms foreground read. Prewarming
all observed pages cost 318 ms and left a 166 ms read. The observed-page
selection is an oracle from a prior read, not a shifou facility for discovering
physical offsets on a new or rewritten database.

`POSIX_FADV_WILLNEED` and repeated Linux `readahead(2)` calls did not fill
the requested page ranges in this environment: `mincore` showed 16-32 MiB
resident, and foreground reads remained 532-571 ms. For this checkpoint,
cold payload faults dominate the delay; index and metadata faults contribute
little. A background warm-up of likely sessions can move the cost off a
request path, but shifou needs a bounded admission policy. A true targeted
prefetch plan must come from yesno's versioned physical layout rather than
shifou inferring file offsets. Batched logical queries should be remeasured
against this cold baseline when upstream work lands. The scratch harness and
all raw rows are in
`../../../yesno/.agents-workspace/tmp/cold-prewarm-20261006/`; the full
construction is in `cold-peer-page-warming-20261006.md` in this directory.

### Real Qwen continuation after selective prewarm

The existing Qwen peer E2E probe reran each cold copy with the same 16-lane
server, pipelined verification, 16 Rayon workers, and parallel BF16 GPU
restore. Each invocation reads the generation-2 session and a distinct
prepared-prefix bundle, then loads two model instances and compares native
and restored continuations. The `.yno` file had zero resident pages before
each fresh server. Foreground hit timers exclude prewarm, server startup,
model loading, and publication. All 12 runs gave zero maximum absolute
logit difference for both workflows. The raw rows and harness are
`../../../yesno/.agents-workspace/tmp/cold-prewarm-20261006/e2e-results*.json`
and `measure_e2e.py`.

| Three-copy median, ms | No prewarm | Session payload only | Prepared pages only | Both footprints |
| --- | ---: | ---: | ---: | ---: |
| Prewarm phase | 0 | 291 | 248 | 463 |
| Session hit ( peer, GPU restore, next token ) | 706 | 242 | 586 | 228 |
| Fresh session computation | 328 | 325 | 325 | 325 |
| Prepared hit ( lookup, peer, GPU restore, suffix ) | 718 | 821 | 267 | 273 |
| Fresh prepared computation | 335 | 333 | 333 | 334 |
| Server major faults across both reads | 1,901 | 954 | 956 | 0 |

An E2E read left 443.1 MiB of file pages resident. Removing the prior
session-only footprint exposed a separate 205.0 MiB prepared-page footprint.
Thus session-payload prewarm does not accelerate prepared initialization,
and prepared-page prewarm does not accelerate session restore. The both-arm
foreground times beat fresh computation in this controlled setting, but its
463 ms median prewarm is a real cost. Phase sums exclude range discovery,
server startup, model load, and overlap; they are not measured request
latency. The page masks come from prior reads and cannot safely be reused
after a physical file rewrite. A bounded background policy and a yesno-owned
logical-to-physical plan are still required for practical cold prewarm.


### Bounded prepared-host serving tier

`BoundedHostTier` now keeps engine-prepared host values within declared byte
and entry limits. `HostAdmission` accepts an entry only when expected reuse
repays its fill cost, and displaces lower expected saving per resident byte
first. The value is built only after admission has been planned; failure leaves
existing entries intact. Serving code must key each session entry by its exact
checkpoint address and invalidate prepared entries on publication or removal.
A hit does not query yesno for freshness. The limit covers declared prepared
values, not all process RSS, transient peer buffers, or GPU memory.

The optional `SHIFOU_BENCH_BOUNDED_TIER=1` arm of
`peer_session_workflows` ran a seven-request trace ( prepared, session,
prepared, prepared, session, prepared, session ) against a separate `yesnod`
Unix peer. It used the existing generation-2 Qwen3-0.6B NVFP4-weight fixture,
exact BF16 attention, full pipelined verification, 16 peer lanes and Rayon
workers, and parallel BF16 GPU restore. The policy's reuse hints were four
prepared and two session requests, seeded from the measured peer and host
latencies in each run. Each prepared host value was 233,046,016 bytes after
block padding. Three independent model processes rotated the order of 0,
256, and 512 MiB limits; the request sequence was identical in every arm.

| Declared host limit | Host hits / 7 | Median request sum | Median fill sum | Retained host bytes |
| --- | ---: | ---: | ---: | ---: |
| 0 MiB | 0 | 1,376 ms | 0 ms | 0 |
| 256 MiB | 3 prepared | 876 ms | 8.6 ms | 233,046,016 |
| 512 MiB | 3 prepared + 2 session | 524 ms | 17.4 ms | 466,092,032 |

All continuation logits matched the fresh model exactly ( maximum absolute
difference 0 ). Request sums exclude model load, server startup, initial
calibration reads, and the fill work performed after each missed request.
This is a warm peer trace: the tier cuts repeated reads, while a first cold
request still pays the cold read unless a separate background prewarm ran.
The prototype uses explicit, owner-coordinated checkpoint identities; it has
no automatic session-head change subscription or process-wide eviction. The
raw rows and repeat driver are under
`../../.agents-workspace/tmp/bounded-serving-20261006/` relative to this note.


### Proactive admission during cold peer startup

`proactive_peer_serving` plans host entries before reading them, using the
existing `BoundedHostTier` cost and byte policy. It fetches selected logical
records through a separate yesnod peer connection while one serving model
loads. The fetch worker validates and decodes exact BF16 into prepared host
values, drops each raw bundle before fetching the next, and retains only the
entries admitted under the host limit. A peer read has a separate
234,094,592-byte logical-bundle cap ( one 233,046,016-byte prepared value plus
1 MiB for bundle metadata ). This is not a cap on process RSS: model weights,
GPU allocations, and transient decode buffers are outside it. The source
model used for the accuracy oracle loads and runs *after both timed requests*,
so it cannot warm the serving GPU path or lengthen the startup overlap window.

The cold fixture is the same generation-2 Qwen3-0.6B NVFP4-weight database,
1,980-token prepared prefix and 2,013-token session, with exact BF16 attention.
Three independent database copies were reset with `POSIX_FADV_DONTNEED`;
`mincore` verified zero resident `.yno` pages before each fresh yesnod. The
server had 16 peer lanes and an 8 MiB arena. Every read used full pipelined
verification and parallel CPU decode. The two first requests ran in fixed
prepared-then-session order. No physical page offsets or prior-read page masks
were supplied to the policy. The cost hints were 718/706 ms fallback,
40/23 ms prepared-host, and 550 ms fill for prepared/session respectively.
The 256 MiB arms used 4:1 or 1:4 expected prepared-to-session reuses; the
512 MiB arm used 4:2. These are workload estimates supplied by the caller.

| Host plan | Retained bytes | Background fetch | Serving startup | First prepared request | First session request |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 0 | 0 ms | 1,734 ms | 824 ms | 666 ms |
| Prepared, 256 MiB | 233,046,016 | 589 ms | 1,776 ms | 252 ms | 614 ms |
| Session, 256 MiB | 233,046,016 | 624 ms | 1,744 ms | 844 ms | 30 ms |
| Both, 512 MiB | 466,092,032 | 1,274 ms | 1,798 ms | 252 ms | 28 ms |

Numbers are three-copy medians in milliseconds. The median sum of the two
first-request times fell from 1,490 ms without admission to 280 ms with both
entries. Across all 12 runs, maximum absolute continuation-logit difference
was zero. Server major faults remained about 1,901-1,905 over the whole run:
proactive fetch moves cold I/O before requests; it does not avoid it. CPU
preparation took 11-21 ms median in the selected arms. Background work
finished during one-model loading in 11 of 12 runs; one both-entry run waited
150 ms after model load. The shared host was busy and one no-prewarm model
load took 902 ms, so startup overlap is workload-dependent, not guaranteed.
A server that is already loaded must pay the warm-up separately. The session
head and prepared index were immutable during this owner-coordinated fixture;
production serving still needs explicit freshness checks or invalidation when
they change.

The retained source is
`experiments/xinfer/src/bin/proactive_peer_serving.rs`; the repeat harness
and final `proactive4-*.json` rows are under
`../../.agents-workspace/tmp/bounded-serving-20261006/` relative to this note.


### Background warm-up while active inference is running

The follow-up `concurrent_peer_serving` probe removed the model-load overlap.
One loaded Qwen3-0.6B model first computed a distinct 2,013-token active
session by rotating the fixture token sequence, prepared its BF16 host pages,
and warmed the GPU path. It then served 64 serial one-token host hits while
a separate peer worker read and decoded the *other* prepared/session records.
The already-hot active value ( 233,046,016 bytes ) sits outside the
speculative 0/256/512 MiB tier. Before each yesnod start, the same three
non-reflink database copies were verified to have zero resident `.yno` pages.
Peer reads used full pipelined verification, 16 lanes, and a per-connection
234,094,592-byte logical-bundle cap. This is a one-stream contention test,
not a saturated multi-request GPU server.

A ready entry is sent to the serving thread as soon as its own peer read and
CPU decode finish. The serving thread polls between active requests. The two
follow-up requests never wait for the background worker: a missing entry
falls back to a verified peer read immediately. This can duplicate an in-flight
read, a cost that a production scheduler would need to cancel or coalesce.

| Plan | Active request p50 | Active request p95 | 64 active requests | Prepared ready | Session ready | Follow-up prepared | Follow-up session |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 13.8 ms | 18.6 ms | 1,007 ms | - | - | 678 ms | 729 ms |
| Prepared, 256 MiB | 13.9 ms | 16.5 ms | 913 ms | 716 ms | - | 36 ms | 841 ms |
| Session, 256 MiB | 13.8 ms | 21.1 ms | 964 ms | - | 663 ms | 677 ms | 24 ms |
| Both serial, 512 MiB | 13.9 ms | 14.7 ms | 908 ms | 623 ms | 1,217 ms | 33 ms | 383 ms |

These are three-copy medians with arm order rotated across copies. With both
entries scheduled serially, the prepared value was ready for all three
follow-up requests; the session value was still pending after the 64 active
requests in all three, so the session request used the peer path. The median
64-request loop plus two follow-ups was 2,496 ms without warming and
1,314 ms with serial both-entry warming. Active p50 differences varied from
-4.0 to +4.9 ms across paired copies; the small sample and busy shared host
do not establish a stable inference penalty or a zero-cost background path.
All active and follow-up continuation logits matched exactly.

A second arm opened **two** independent peer connections for the 512 MiB
plan, one per entry. In a fresh paired three-copy comparison, serial warming
made the prepared value ready first in all three runs, while parallel warming
made the session value ready for all three follow-up requests but the prepared
value for only one. Parallel median ready times were 1,110 ms prepared and
1,109 ms session; the two reads contended for server, CPU, and page-cache
resources. Its median 64-request loop plus follow-ups was 1,385 ms against
1,293 ms for serial in that paired series. The per-copy parallel-minus-serial
differences were -16, +131, and -74 ms, so there is no repeatable win from
two readers on this frame. Per-connection bundle limits also do not cap their
combined in-flight allocation.

The probe and repeat driver are
`experiments/xinfer/src/bin/concurrent_peer_serving.rs` and
`../../.agents-workspace/tmp/bounded-serving-20261006/run_contention.py`;
raw rows use the `contention2-*.json` and `contention3-*.json` prefixes in that
scratch directory. Production work still needs a request-aware scheduler,
in-flight read coalescing or cancellation, and freshness invalidation on
prepared-index or session-head publication.
