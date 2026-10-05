# Page residency planning with yesno sets, 2026-10-02

## Question and construction

Can yesno set arithmetic efficiently merge attention-page requests and subtract
pages already present in GPU RAM, CPU RAM, or an in-flight read? The planner
computes `needed = union(requests)`, `upload = needed - gpu`, and
`new_remote_reads = upload - cpu - in_flight`. A separate dense-word planner
computes the same result. The probe compares their complete output sets and
counts before timing; no model logits or serving path are timed.

The request masks come from the retained Qwen3-0.6B, 8,192-token,
WikiText-2-offset-16,384 BF16-centroid page-selection fixture. It has 28
layers, 16 query heads, 8 KV heads, and 513 candidate 16-token pages per KV
head. The probe groups four adjacent pages into one 8 KiB fetch unit and
unions the two query heads sharing each KV head. The one-eighth and one-quarter
budgets use their recorded request sets. The `all` budget requests every page
and models exact attention-cache restoration, without sparse-attention quality
assumptions. IDs are unique across layers and KV heads. Each head receives
three local u64 mask words, covering 129 used fetch-unit IDs; this lets the
dense baseline OR only those three words per request. GPU residency uses
`id * 2654435761 % 100 < resident_pct`; CPU residency uses
`id * 1140071481 % 2 == 0`; every seventh ID has an in-flight read. These
are deterministic synthetic placements, not measured xinfer residency.

Release build, pinned to Cortex-X925 CPU 16. The result is the median of nine
batches; each batch performs 5,000 iterations for one layer or 300 for all
layers. Two timed paths perform union and subtraction, then either count the new
remote reads or enumerate their page IDs. The enumeration path allocates its
result vector. Both reuse input sets and word masks; they exclude manifest lookup, page reads, transfer,
GPU synchronization, and set construction. The source and raw CSV live at
`../yesno/.agents-workspace/tmp/residency-planner-probe/` relative to the
shifou repository. Reproduce from yesno's root with:

```sh
cargo run --release --offline --manifest-path .agents-workspace/tmp/residency-planner-probe/Cargo.toml -- .agents-workspace/tmp/kv-page-probe-20260930/result-8192-16384-masks-both.json
taskset -c 16 .agents-workspace/tmp/residency-planner-probe/target/release/residency-planner-probe .agents-workspace/tmp/kv-page-probe-20260930/result-8192-16384-masks-both.json
```

## Measured result

| Layers | Request | GPU resident | Needed units | GPU uploads | New remote reads | yesno count | Dense count | yesno ID list | Dense ID list |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1/8 | 0% | 299 | 299 | 118 | 3.990 us | 0.068 us | 4.558 us | 0.253 us |
| 1 | 1/8 | 90% | 299 | 33 | 16 | 5.140 us | 0.065 us | 5.409 us | 0.104 us |
| 1 | all | 50% | 1,032 | 514 | 220 | 8.666 us | 0.069 us | 9.828 us | 0.385 us |
| 28 | 1/8 | 50% | 8,899 | 4,424 | 1,862 | 21.300 us | 1.295 us | 26.320 us | 2.236 us |
| 28 | 1/4 | 50% | 14,845 | 7,383 | 3,157 | 34.037 us | 1.306 us | 42.327 us | 2.944 us |
| 28 | all | 0% | 28,896 | 28,896 | 12,288 | 97.302 us | 1.300 us | 128.488 us | 8.643 us |
| 28 | all | 50% | 28,896 | 14,443 | 6,143 | 97.293 us | 1.300 us | 113.763 us | 4.751 us |
| 28 | all | 90% | 28,896 | 2,884 | 1,229 | 101.804 us | 1.301 us | 105.175 us | 1.898 us |

For the 28-layer cases at 50% GPU residency, dense words are 16-75 times
faster when counting and 12-24 times faster when returning the page IDs.
Both planner times are small beside an actual page transfer.
The practical benefit comes from skipping existing bytes, not from yesno's
set kernel outperforming dense words at this cardinality. At 50% GPU
residency, the exact all-page case needs 14,443 uploads rather than 28,896;
that is about 112.8 MiB rather than 225.8 MiB of 8 KiB units. The synthetic
CPU and in-flight masks leave 6,143 new remote reads. Real overlap and
transfer savings require a xinfer trace.

## Exact partial reads from a fresh yesno page store

A second standalone probe under
`../yesno/.agents-workspace/tmp/residency-read-probe/` measures the read path.
The earlier real Qwen page-store database could not be reopened by the current
yesno build because its recorded size-class ladder differs from the current
one. The probe therefore created a fresh database with the current build:
1,024 bitmap chunks, 8,192 bytes each, filled by SplitMix64-derived words.
This gives a roughly half-dense 8 MiB payload, matching the size and near-half
bit density of the earlier real compact4 layer, while the actual bytes are
synthetic. It commits, checkpoints, closes, and reopens the database. The
probe derives expected per-chunk SHA-256 digests before timing.

Each timed arm acquires a snapshot. The full ordinal arm models shifou's
`Snapshot::load` plus bit-by-bit reconstruction and a whole-payload SHA-256.
The dense full and selected arms use one `key_stream`, copy bitmap words, and
verify a SHA-256 for every returned 8 KiB chunk. The selected arm seeks to
ascending missing IDs. For every residency scenario, the probe checks every
returned chunk byte-for-byte against the full payload before timing. Nine
rotated warm rounds ran pinned to Cortex-X925 CPU 16; medians follow.

| Read arm | Chunks read | Bytes read | Median |
| --- | ---: | ---: | ---: |
| Current ordinal full read | 1,024 | 8,388,608 | 81.898 ms |
| Dense full read, per-chunk SHA-256 | 1,024 | 8,388,608 | 15.726 ms |
| Selected, 0% GPU resident | 1,024 | 8,388,608 | 15.724 ms |
| Selected, 50% GPU resident | 511 | 4,186,112 | 7.875 ms |
| Selected, 90% GPU resident | 102 | 835,584 | 1.590 ms |

With matching per-chunk integrity, the selected reader was about 2.0 times
faster at 50% residency and 9.9 times faster at 90% than the dense full read.
The ordinal full path also pays a separate 5.2 times reconstruction penalty
relative to the verified dense full path. These are warm local storage reads;
GPU transfer, manifest lookup, planning, cache synchronization, and actual
xinfer residency are excluded. A production selected-page format still needs
independent page digests and a manifest. Read savings are exact because every
missing chunk is fetched and validated; no attention pages are omitted for
approximation.

Reproduce from yesno's root after building the probe:

```sh
cargo build --release --offline --manifest-path .agents-workspace/tmp/residency-read-probe/Cargo.toml
taskset -c 16 .agents-workspace/tmp/residency-read-probe/target/release/residency-read-probe .agents-workspace/tmp/residency-read-probe/db-synthetic-current
```

The probe refuses to replace an existing database. Its source and raw CSV are
retained alongside that database.

## Fit to the present storage format

The current prefill API reads all recurrent-state pages, checks a checksum of
the concatenated state, then loads every attention snapshot. State pages are
8 MiB and can be reused at that granularity, but an 8 MiB state object is
only one page. Each packed attention record also loads, reconstructs, and
checks its whole payload. Fine-grained fetch planning cannot reduce those
reads until shifou exposes a page manifest, independent page checksums, and a
selected-page read or restore API. A compatible page identity must include the
model, prefix, layer, KV head, codec/layout version, and page number. Residency
must be marked only after a completed transfer, and an in-flight read must be
awaited rather than counted as present.

For local head-scoped masks, use dense words in the runtime planner. yesno
remains useful to persist and query page membership or larger sparse global
indices, but this experiment finds no arithmetic speed reason to put an
`OrdSet` in the hot planner. The next measurement is full cache-hit latency and bytes transferred at
measured xinfer GPU residency, using a selected-page format and manifest. The sparse attention rows remain research cases because
the prior continuation-quality experiment did not establish safe selection.

## Sparse exception-mask applicability

The affine codec has an exception-position mask, but its default policy uses
32-value groups and an outlier fraction of 0.02. The encoder computes
`floor(group_length * outlier_fraction)`, so every default group has trim zero
and emits no trimmed exceptions. The Qwen affine experiments explicitly use
64-value groups, allowing one value from each end to be excluded before a
candidate is evaluated. Even then, the exception mask, exact words, and code
planes occupy the same group payload. An `OrdSet` intersection could find
requested exception positions after loading that payload, but it cannot avoid
the payload read or current full-tile decode. The compact 1/2/4-bit codec used
in the page experiment has no exception mask at all. A useful read-savings
experiment first requires separately addressable corrections and measured
exception density from the target codec; a standalone mask-intersection
microbenchmark would not establish an application benefit today.

## Fragmented pages over yesno-flight

The next probe uses the existing yesno-flight protocol over loopback. It
creates a fresh database with one 8 MiB payload key holding 1,024 bitmap
chunks of `0x55` bytes, then stores seven full-bit page masks as separate keys. The masks use
the same deterministic residency rule as the local experiments. Server and
client run on Cortex-X925 CPUs 16-17 with `TCP_NODELAY=true` on both sockets.
A full fetch requests one Bitvector window. Exact ranges use one Bitvector
`DoGet` per consecutive missing-page run. Coalesced ranges allow some resident
pages to be fetched again to reduce the request count. A stored-mask fetch
asks for `And(Key(payload), Key(missing_mask))` on the Containers wire. An
in-band fetch asks for `And(Key(payload), Or(Range(...)))`, with one range for
each consecutive missing-page run, and needs no mask-key write. Both filtered
paths return only requested chunks. The probe validates each returned prefix
and cardinality; one untimed filtered read per mask checks every bitmap word.
Timed reads decode every Arrow batch. Byte totals below count decoded payload
and omit Arrow and gRPC framing.

Fragmentation matters. At 50% residency, 511 missing chunks form 399 runs;
at 90%, 102 missing chunks form 102 runs. The first three independent sweeps
found about 19.6-20.2 ms for the 399 exact-range requests and 4.7-5.0 ms for
the 102 exact-range requests. A full 8 MiB Flight fetch was roughly 2.5-4.7
ms across runs. Coalescing gaps reduced the 50% case to 113 requests and
6.23 MiB of decoded payload, at about 7.3 ms. The 90% case became 51
requests and 2.39 MiB, at about 3.1 ms. A single filtered Containers request
with a pre-stored mask took about 0.65-0.78 ms at 90% in paired runs, but this
excludes publishing and maintaining that mask. It is an upper bound on what
runtime set filtering can achieve with an already available mask.

The in-band expression avoids a mask write. Its size and one-time ticket mint
cost scale with the number of missing runs: 399 ranges produce a 6,804-byte
command and 6,852-byte ticket, with about 2.4 ms to mint; 102 ranges produce
1,755- and 1,803-byte messages with about 0.3 ms to mint. The paired
measurement below times a new `GetFlightInfo` ticket and the filtered `DoGet`
on every iteration. It alternates that path with a full fetch for 30 requests
per arm. These are medians from the final run; the preceding run gave the same
crossover at 80%, 85%, and 90% residency.

| GPU resident | Missing chunks | Full Flight | In-band mint plus filtered read |
| ---: | ---: | ---: | ---: |
| 50% | 511 | 2.886 ms | 7.501 ms |
| 70% | 307 | 2.806 ms | 4.207 ms |
| 80% | 205 | 2.850 ms | 2.485 ms |
| 85% | 153 | 2.599 ms | 1.686 ms |
| 90% | 102 | 2.718 ms | 1.126 ms |

Thus an existing yesno set intersection can reduce wire work when only a
small, fragmented part of an entry is missing. With this workload, minting an
in-band expression starts to pay around 80% residency. This is a measured
crossover for this payload and loopback protocol, not a cache policy constant.
At 50%, the full fetch wins despite halving payload bytes. A compact
128-byte chunk mask carried directly in a ticket might reduce the expression
cost, but that protocol and its integrity rules have not been implemented or
measured. One run also had isolated 200 ms-class Flight tails, so the median
benefit does not establish tail behavior.

The probe source is
`../yesno/.agents-workspace/tmp/flight-bitvector-bench/src/bin/page_filter.rs`.
The raw independent sweeps and paired runs are
`../yesno/.agents-workspace/tmp/flight-bitvector-bench/page-filter-*.csv` and
`page-filter-*.log` in the same directory. Reproduce from yesno's root with:

```sh
cargo build --release --offline --manifest-path .agents-workspace/tmp/flight-bitvector-bench/Cargo.toml --bin page_filter
taskset -c 16-17 .agents-workspace/tmp/flight-bitvector-bench/target/release/page_filter
```

This still uses a synthetic dense byte pattern and a prebuilt yesno database.
It excludes GPU upload, xinfer cache restoration, a real request trace, and a
production page manifest. A runtime choice between full and filtered fetches
needs measured residency and must keep the cache identity and snapshot
version fixed across the requested chunks.

## Why the 50% in-band request takes about 7.5 ms

A follow-up corrected the Flight probe's byte counter. It had re-encoded each
returned bitmap merely to count bytes, allocating and copying another 8 KiB
per selected chunk on the client. It now counts the known bitmap size. The
in-band 50% paired median remained 7.445 ms against a 3.288 ms full fetch in
the corrected run. The earlier observation therefore survives that benchmark
error; the stored-mask path improved and its earlier timings should not be
used to attribute server cost.

Source inspection gives the request path. `GetFlightInfo` computes the exact
cardinality of a new expression before issuing its ticket. `DoGet` then lowers
and materializes the same expression into an `OrdSet`, and the Containers arm
encodes the returned chunks. Its batch limit is 512 containers, so this
50% mask's 511 chunks become one roughly 4 MiB Arrow batch. The full Bitvector
path can lend contiguous bitmap bytes and emits at most 64 chunks per batch.

A separate CPU-16/17 phase probe measured the 399-range expression locally
across 21 warm repetitions: exact cardinality 2.07-2.12 ms; materialized
result 2.18-2.20 ms; container batch encoding 1.33-1.34 ms. The client-side
`GetFlightInfo` mint took 2.6-2.8 ms, consistent with cardinality dominating
that RPC. The decoded first batch of the `DoGet` was ready about 7.2-7.4 ms
after that call began, while converting its container rows on the client took
about 0.19-0.20 ms. The local stages and network phases overlap and should
not be added as a wall-clock identity, but they identify repeated expression
work and delayed emission of a large batch as the main costs. The 4 MiB wire
payload alone is not the explanation: the full path moves 8 MiB in about
2.5-3.3 ms in the paired samples.

The probe also opened the same yesno expression as a lazy chunk stream and
flushed every 64 containers, without changing production Flight code. With
the 399-range mask, the first local batch was ready in 0.63-0.64 ms and the
full local encode walk took 2.44-2.46 ms. The current local materialize-then-
encode sequence took about 3.5 ms. This is a concrete candidate for reducing
first-batch delay, but its over-wire gain and error behavior are unmeasured.
A reusable snapshot ticket could remove the extra `GetFlightInfo` cardinality
pass for a new in-band expression; the ticket's version must still be the
version of the resident pages. A compact page-mask ticket and direct
selected-chunk stream may be better than a 399-term range union, but that is
also unimplemented.

Phase code and three raw logs live under
`../yesno/.agents-workspace/tmp/flight-bitvector-bench/` as
`src/bin/page_filter.rs`, `page-filter-phases-card-1.log`, and
`page-filter-lazy-{1,2}.log`. Set `PAGE_FILTER_PHASES=1` when running the
probe to reproduce the decomposition.

## Over-wire lazy expression and batch-size experiment

On 2026-10-02, an isolated copy of `yesno-flight` in
`../yesno/.agents-workspace/tmp/flight-bitvector-bench/flight-variant/`
was used to test two independent changes to the Containers response: open a
filtered expression as a lazy chunk stream rather than materializing its full
`OrdSet`, and flush every 64 containers rather than every 512. The production
Flight crate was not edited. `PAGE_FILTER_VARIANT=1` selects the copy;
`YESNO_FLIGHT_LAZY_STREAM=1` and `YESNO_FLIGHT_BATCH64=1` enable the changes.
The original server, each change alone, and both changes were run against the
same prebuilt 8 MiB key on CPU 16-17 over loopback Flight. The client checked
returned chunk identities and bitmap contents. Each paired result alternated
30 full-key reads with 30 new `GetFlightInfo` plus filtered `DoGet` reads.

| Server path | 50% resident, mint plus read median | 90% resident, mint plus read median |
| --- | ---: | ---: |
| Original, 512-container batches | 7.163 ms | 1.255 ms |
| 64-container batches only | 7.126 ms | 1.128 ms |
| Lazy expression only | 7.196 ms | 1.129 ms |
| Lazy plus 64-container batches | 6.609 ms | 1.072 ms |

A reversed-order repeat put the original at 7.303 ms and the combined variant
at 6.601 ms at 50% residency; at 90% it put them at 1.061 and 1.000 ms.
Thus the combined variant saves about 0.55-0.70 ms ( 8-10% ) at 50% in the
actual mint-plus-read workload. Each change alone showed little stable benefit
at 50%; the interaction matters. It does not change the decision there: a
full-key Bitvector fetch took about 2.9 ms. The existing crossover around
80% residency remains. At 90%, the request is already short enough that the
absolute improvement is small.

The first-batch phase probe showed why the idea looked stronger locally. At
50% residency, the direct-ticket first batch fell from 7.363 ms in one
original run to 1.357 ms with both changes, and end-to-end direct-ticket time
fell from 8.452 to 3.547 ms. Those phase samples were not stable enough to
predict the paired mint-plus-read result: a later seven-repetition direct
`DoGet` comparison put the original at 8.490 ms and combined at 5.134 ms,
while the 30-pair mint-plus-read comparison above found a much smaller overall
saving. A fresh `GetFlightInfo` still computes exact cardinality, about
2.6 ms for the 399-range 50% request, and remains a large fixed cost. The
combined variant also had a 211 ms p95 tail in one reversed-order run, so this
sample does not establish a tail-latency improvement.

Raw logs are `phase-{original,batch64,lazy,both}.log` and
`paired-{original,batch64,lazy,both,both2,original2}.log` beside the probe.
Build with `cargo build --offline --release --bin page_filter` from the probe
crate, then run its binary with `taskset -c 16,17` and the environment switches
above. The dataset uses a synthetic half-dense bitmap pattern; this remains a
loopback and CPU-only result, without GPU upload or an application request
trace. The main next opportunity is avoiding repeated cardinality work for
new in-band range tickets, subject to snapshot-version and ticket-integrity
rules; the present lazy/batch change alone is a modest end-to-end improvement.

## Isolating the yesno expression cost

A follow-up core-only profile separated the container kernel from the 399-range
stream used above. At 50% residency, lowering took 0.010 ms, planning 0.031
ms, and counting the planned intersection stream 1.84-1.87 ms. The 511 direct
bitmap x run `and_cardinality` calls took only 0.066 ms. Seeking the 399-range
union once per selected chunk cost 0.92-0.93 ms even without intersection.
The same 511 chunks selected by a stored bitmap mask counted in 0.22-0.24 ms.
Thus the slowness can be attributed to yesno's current stream composition for
many disjoint ranges, especially repeated seeks through a left-deep `Concat`
chain, rather than to its bitmap arithmetic kernel. This is a CPU-local
attribution; Flight still adds ticket minting, serialization, and transport.
The exact profiler and raw results are under
`../yesno/.agents-workspace/tmp/flight-bitvector-bench/` as
`src/bin/kernel_profile.rs` and `kernel-profile-{3,4}.log`.

A later control repeated the cardinality kernel on all 511 distinct persisted
bitmap chunks instead of reusing one chunk; it took 0.083 ms
( `kernel-profile-5.log` ). The attribution is unchanged.

## Flight remeasurement after yesno ConcatAll

Commit `2dc4233` replaced yesno-core's boundless binary `Concat` chain with
`ConcatAll`, which skips disjoint parts by their prefix upper bounds. The
original yesno-flight server was rebuilt against that commit, with no lazy
Flight variant enabled. The same 8 MiB checkpointed payload and 50-90%
residency masks were used on CPU 16-17 over loopback. Each row below is a
paired 30-request median for a fresh `GetFlightInfo` plus filtered `DoGet`;
three post-change runs are shown as ranges. The old values are the two paired
runs before the core change.

| GPU resident | Old in-band mint plus read | With ConcatAll | Post-change full fetch |
| ---: | ---: | ---: | ---: |
| 50% | 7.16-7.30 ms | 3.44-3.83 ms | 2.54-2.98 ms |
| 70% | 4.07-4.16 ms | 2.06-2.20 ms | 2.41-2.65 ms |
| 80% | 2.22-2.48 ms | 1.45-1.52 ms | 2.36-2.72 ms |
| 85% | 1.57-1.68 ms | 1.13-1.24 ms | 2.36-2.73 ms |
| 90% | 1.06-1.26 ms | 0.82-0.91 ms | 2.48-2.66 ms |

This moves the measured crossover: the filtered request is still slower at
50% but already faster at 70%. It removes about 3.5-3.9 ms from the 50%
in-band request. A phase run measured local expression cardinality at 0.391 ms
( previously 2.07 ms ) and collection at 0.485 ms ( previously 2.13 ms );
`GetFlightInfo` mint fell from about 2.6 ms to 1.09 ms in that phase run.
Some paired runs still had isolated 200 ms-class tails, so the medians do not
establish tail behavior. The raw new logs are
`../yesno/.agents-workspace/tmp/flight-bitvector-bench/phase-after.log` and
`paired-after{,-2,-3}.log`.


## Real Qwen3-0.6B KV payload over Flight, 2026-10-03

The synthetic `0x55` payload was replaced with data produced by an actual
Qwen3-0.6B prefill on the NVIDIA GB10. The local model revision was
`c1899de289a04d12100db370d81485cdf75e47ca`. The input was 8,192
WikiText-2 tokens beginning at corpus offset 16,384. The probe exported
layer 0 K/V, encoded it with shifou compact4, and rearranged the code bytes
into 1,024 page-major 8 KiB bitmap chunks ( four adjacent 16-token K/V pages
per chunk ). It committed and checkpointed one 8 MiB yesno key, closed it,
then reopened it with the current yesno build. The actual code bits are
51.1845% one; portable Roaring occupies 8,396,820 bytes versus 8,388,608
raw bytes. Model prefill took 969 ms, compact4 encoding 213 ms, rearrangement
43 ms, and commit plus checkpoint 45 ms in this one construction run. Those
creation times are not part of cache-hit timing.

The over-wire probe served this reopened key through the original production
yesno-flight service over loopback, with client/server pinned to CPU 16-17 and
`TCP_NODELAY` enabled. Each case alternated 30 full Bitvector reads with 30
fresh `GetFlightInfo` plus filtered Containers reads. The filtered ticket was
`And(Key(payload), Or(Range(...)))`; each half-open range covered complete
8 KiB chunk ordinals. One untimed read in every case compared every returned
bitmap word with the full-read bytes and checked its prefix. `full` below is
the paired full-read median. `filtered` includes ticket mint and `DoGet`.
These are ranges of medians from two saved reverse-order runs; timings are
wall-clock milliseconds, and payload excludes Arrow/gRPC framing.

| Request | Chunks fetched | Runs in mask | Payload | Full | Filtered |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact, 50% resident | 511 | 399 | 4.00 MiB | 2.73-3.00 | 3.46-3.64 |
| Exact, 70% resident | 307 | 307 | 2.40 MiB | 2.63-2.85 | 2.06-2.34 |
| Exact, 90% resident | 102 | 102 | 0.80 MiB | 2.45-2.80 | 0.80-1.03 |
| Qwen centroid 1/8, 0% resident | 291 | 80 | 2.27 MiB | 2.64-2.72 | 1.75-1.78 |
| Qwen centroid 1/4, 0% resident | 493 | 84 | 3.85 MiB | 2.82-3.46 | 6.33-6.77 |
| Qwen centroid 1/8, 50% resident | 144 | 121 | 1.13 MiB | 2.43-2.71 | 1.00-1.10 |
| Qwen centroid 1/4, 50% resident | 240 | 197 | 1.88 MiB | 2.59-2.67 | 1.58-1.69 |

The larger 1/4, 0%-resident case varied substantially: two earlier
reverse-order runs put its filtered median at 2.92-2.94 ms. The 6.33-6.77 ms
saved runs therefore do not support a stable win for this case. Exact 50%
reads were also 6.79-7.22 ms when this case ran first in three processes;
when moved last, their medians fell to 3.43-3.97 ms across four runs. A local
stage probe showed the first 511-chunk expression collection took about
2.0 ms regardless of whether its ranges were half-open or ended one ordinal
early; the same collection later took about 0.41 ms. This implicates first
large-result materialization or memory warm-up, but the probe does not isolate
which component owns that cost. It is a real cold-path consideration, not
proof that Qwen bitmap bits make yesno's set kernel slower. Across the
warmed exact cases, the crossover remains between 50% and 70% resident.

The 1/8 and 1/4 masks are real BF16-centroid selections recorded for this
prefix, but GPU residency placements still use the deterministic synthetic
rule, `id * 2654435761 % 100 < resident_pct`. No xinfer page restoration,
GPU transfer, continuation logits, or serving latency was timed. The sparse
selection rows do not establish acceptable model quality. The exact rows
fetch every missing page and only test cache retrieval. The local model probe
also measured a warm verified selected read of 4.43 ms for 291 chunks and
7.48 ms for 493 chunks; those paths include one SHA-256 per chunk and are not
directly comparable to the Flight timings, which do not hash payloads.

Source, reopened database, model-run log, JSON result, and Flight raw logs are
under `../yesno/.agents-workspace/tmp/kv-page-probe-20260930/` and
`../yesno/.agents-workspace/tmp/flight-bitvector-bench/`, relative to shifou.
The key files are `result-page-store-real-20261003.json`,
`run-page-store-real-20261003.log`, `src/bin/real_qwen_flight.rs`,
`real-qwen-flight-{1,2,3}.log`, and
`real-qwen-flight-reverse-{1,2}.log`. Reproduce the Flight part from yesno's
root after building the scratch crate:

```sh
cargo build --offline --release --manifest-path .agents-workspace/tmp/flight-bitvector-bench/Cargo.toml --bin real_qwen_flight
cd .agents-workspace/tmp/flight-bitvector-bench
REVERSE_CASES=1 taskset -c 16,17 target/release/real_qwen_flight
```

This confirms that the set-expression improvement helps on actual compact4
model bytes and model-derived request masks, with the same limits as the
synthetic experiment. The next application step is a versioned page manifest
and a measured xinfer GPU-residency trace before treating these numbers as a
serving-cache speedup.


## GPU page restoration and inference after a persistent cache hit, 2026-10-03

The preceding Flight measurement stopped at decoded CPU bytes. This follow-up
ran Qwen3-0.6B through GPU KV restoration and resumed autoregressive inference
on the NVIDIA GB10. The pinned model revision was
`c1899de289a04d12100db370d81485cdf75e47ca`, using xinfer
`b88c15334fb607ff52cdc3fd875c3da796dcc020` and the existing Qwen3
scratch adapter. The two WikiText-2 prefixes were 2,048 tokens at corpus
offset 4,096 and 8,192 tokens at offset 16,384. All 28 layers and both K/V
tensors per layer participated; no missing layer was left in the original
prefill cache. Greedy generation used eight tokens after the prefix.
Teacher-forced accuracy used the next 64 held-out corpus tokens and compared
logits with a native BF16 prefill on the same prefix.

### Lossless BF16 pages: actual GPU page writes

For the exact page experiment, each K or V tensor was exported as its BF16
bytes, split into 16-token pages ( 32 KiB per K or V page ), and persisted in
yesno as four 8 KiB bitmap chunks per page. The complete 28-layer cache was
224 MiB at 2K and 896 MiB at 8K. One yesno batch committed all 56 keys;
the database was checkpointed, closed, and reopened before reads. A fixture
held one SHA-256 digest per 8 KiB chunk. The timed read used an ascending
key stream with seeks to each missing page, copied its bytes, and checked
every chunk digest. The timed GPU stage created BF16 page tensors, wrote them
in place with Candle `slice_set`, and synchronized the device. A full BF16
readback of every layer after generation confirmed the restored prefix bits
were identical to the original prefill cache.

Synthetic resident pages were placed on the GPU before timing, using
`(tile * pages_per_tile + page) * 2654435761 % 100 < resident_pct`.
These are controlled placements, not xinfer's measured eviction history.
The wall clock starts before the yesno read and stops after the eighth greedy
token. It includes per-page SHA-256, read allocation, CPU-to-GPU transfer,
page writes, GPU synchronization, and inference. It excludes creating the
blank GPU cache, prepopulating resident pages, the later bitwise oracle
readback, model load, prefix prefill, tokenizer work, and any network hop.
The continuous wall time agreed with the sum of its separately timed stages
within display rounding. The following table is the final continuous run;
prior independent stage-sum runs gave 2K totals of 672-677, 383-386, and
158-164 ms, and 8K totals of 2,197-2,333, 1,157-1,251, and 346-417 ms
at 0%, 50%, and 90% residency respectively.

| Prefix | GPU resident | Missing K/V pages | yesno read plus SHA | GPU page restore | Eight-token decode | Cache read to eighth token |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 0% | 7,168 | 537 ms | 40 ms | 101 ms | 678 ms |
| 2K | 50% | 3,584 | 263 ms | 21 ms | 99 ms | 383 ms |
| 2K | 90% | 716 | 52 ms | 4 ms | 101 ms | 157 ms |
| 8K | 0% | 28,672 | 1,905 ms | 160 ms | 137 ms | 2,203 ms |
| 8K | 50% | 14,335 | 941 ms | 80 ms | 138 ms | 1,160 ms |
| 8K | 90% | 2,866 | 191 ms | 16 ms | 138 ms | 345 ms |

The same run's native prefill plus eight-token greedy decode took about
429 ms at 2K ( 323 + 106 ms ) and 951 ms at 8K ( 808 + 143 ms ). Thus
restoring this uncompressed cache beat recomputing the prefix at 50% and
90% resident for 2K, but only at 90% for 8K. This crossover is specific to
the small model, GPU, local read path, and synthetic residency. Read plus
checksum dominates the exact page path; these timings do not isolate the
yesno bitmap kernel from stream, copying, hashing, or allocation.

The lossless path matched the native BF16 cache bit for bit. On both prefixes,
all eight generated IDs and all 64 teacher-forced top choices matched the
native path, with zero observed logit difference and KL divergence.

### Current shifou compact4: full-cache restore and quality

A second arm used the actual shifou compact4 format for all 56 K/V tiles.
`Cache::put_packed_many` published the snapshots in one yesno transaction;
a reopened `CacheReader::get_packed` read and verified them. Each address's
prefix fingerprint was the SHA-256 of the complete token-ID prefix. The
timed cache hit then decoded every packed tile on the CPU, restored a complete
BF16 GPU cache, and generated eight tokens. This is shifou's current
full-cache restore path; it does not use the selective BF16 page writer above.
The oracle comparison with the original packed snapshots happened after the
timed generation. Packed buffers occupied 62.30 MiB at 2K and 240.80 MiB at
8K, versus 224 and 896 MiB for the exact BF16 cache. The table uses the
final prefix-correct run. Two preceding continuous runs put the 2K total at
1,125-1,154 ms; three put the 8K total at 5,566-6,243 ms.

| Prefix | shifou read and verify | CPU compact4 decode | Full GPU restore | Eight-token decode | Cache read to eighth token |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 138 ms | 845 ms | 77 ms | 101 ms | 1,161 ms |
| 8K | 527 ms | 4,301 ms | 1,033 ms | 139 ms | 6,000 ms |

CPU compact4 decode is the largest cost despite its smaller stored payload.
At 8K, full-tile GPU restoration is also much slower than the 16-token BF16
page writer, but the paths use different host representations and GPU upload
primitives, so this is not a bandwidth-only comparison. The 8K compact4
cache hit was much slower than recomputing this 0.6B model's prefix locally.
That finding applies to the current CPU decoder and full-tile restore path;
it does not establish that quantized KV caching is intrinsically slower.

| Prefix | Path | Mean next-token NLL, 64 tokens | Top choice versus BF16 | Mean KL versus BF16 | Greedy eight-token sequence |
| ---: | --- | ---: | ---: | ---: | --- |
| 2K | Native BF16 | 2.2329 | 64/64 | 0 | Reference |
| 2K | Exact BF16 pages | 2.2329 | 64/64 | 0 | Identical |
| 2K | Compact4 | 2.2282 | 61/64 | 0.0091 | Identical |
| 8K | Native BF16 | 3.1063 | 64/64 | 0 | Reference |
| 8K | Exact BF16 pages | 3.1063 | 64/64 | 0 | Identical |
| 8K | Compact4 | 3.1678 | 59/64 | 0.0402 | Identical |

The lower 2K compact4 NLL on this particular 64-token continuation does
not establish a quality gain. The sample is too small for a broad accuracy
claim, and these metrics test one text prefix per length. The eight-token
greedy match also does not imply longer generations will match. It does show
that storage and restoration introduced no additional error beyond compact4:
the observed compact4 teacher-forced metrics agree with the earlier
in-memory compact4 fixture for the first eight scored tokens.

The retained scratch source is
`../yesno/.agents-workspace/tmp/kv-page-probe-20260930/src/{restore_e2e,compact_e2e}.rs`.
Machine-readable results and raw logs are in the same yesno scratch
directory: `result-restore-e2e-{2048-4096,8192-16384}-wall.json` and
`result-compact-e2e-{2048-4096,8192-16384}-final.json`, with matching
`run-*.log` files. The earlier repeated `-r2`/`-r3` results are retained.
Build from the scratch crate with the shared shifou target directory, then
run either mode using this command shape:

```sh
CUDA_COMPUTE_CAP=121 CARGO_TARGET_DIR=../shifou/.agents-workspace/tmp/target cargo build --offline --release --manifest-path .agents-workspace/tmp/kv-page-probe-20260930/Cargo.toml
SCORE_STEPS=64 CUDA_COMPUTE_CAP=121 ../shifou/.agents-workspace/tmp/target/release/kv-page-probe MODEL_DIR CORPUS_PATH OUTPUT_JSON PREFIX_TOKENS CORPUS_OFFSET restore-e2e NEW_DB_DIR
SCORE_STEPS=64 CUDA_COMPUTE_CAP=121 ../shifou/.agents-workspace/tmp/target/release/kv-page-probe MODEL_DIR CORPUS_PATH OUTPUT_JSON PREFIX_TOKENS CORPUS_OFFSET compact-e2e NEW_DB_DIR
```

Every output and database path must be new; the probe refuses to replace an
existing result. No production yesno, shifou, or xinfer code was changed for
this measurement. An over-wire Flight-to-GPU cache-hit path, actual xinfer
GPU-residency trace, and broader model-quality evaluation remain unmeasured.


## Qwen cache restoration over yesno-flight, 2026-10-03

The full-model GPU experiment above was repeated with the original
`yesno-flight` server and an Arrow Flight client over TCP loopback, with
`TCP_NODELAY` enabled. Server and client ran in one process; the wire path was
real gRPC/Arrow rather than a direct database call. The database was the
already checkpointed, reopened 2K or 8K Qwen3-0.6B cache. The same 28-layer
model, WikiText-2 prefixes, eight greedy tokens, 64 teacher-forced score
tokens, and deterministic synthetic GPU residency rule were used. Every
transported bitmap chunk was checked for the expected key and prefix and
against the model-derived SHA-256. After generation, the entire GPU prefix
was read back and compared bit for bit with the native BF16 cache.

For the **lossless BF16 page fixture**, each K or V tile uses one Flight
request. A full missing tile is sent on the Bitvector wire; a partly resident
tile uses a fresh in-band `And(Key, Or(Range(...)))` ticket on the Containers
wire. Both routes fetch only whole 16-token pages. The timed path includes
`GetFlightInfo`, `DoGet`, Arrow decoding, per-chunk SHA-256, GPU page writes
and synchronization, and eight-token generation. One mode issues the 56 tile
requests sequentially; the other allows eight in flight. These are ranges
from two independent runs per mode. The local numbers are the continuous
single-run result above for context.

| Prefix | GPU resident | Local cache hit | Flight, sequential | Flight, 8 concurrent |
| ---: | ---: | ---: | ---: | ---: |
| 2K | 0% | 678 ms | 746-753 ms | 705-906 ms |
| 2K | 50% | 383 ms | 530-537 ms | 380-399 ms |
| 2K | 90% | 157 ms | 237-248 ms | 168-171 ms |
| 8K | 0% | 2,203 ms | 2,404-2,407 ms | 2,315-2,555 ms |
| 8K | 50% | 1,160 ms | 1,526-1,584 ms | 1,219-1,289 ms |
| 8K | 90% | 345 ms | 500-517 ms | 386-395 ms |

All 56 requests were needed in these placements: even at 90% residency,
each tile had at least one missing page. Eight-way concurrency reduced the
partial-request transport wall time. At 8K and 90%, transport itself fell
from 345-363 ms sequential to 232-239 ms at concurrency eight; the rest of
the cache hit was GPU restoration and model decoding. Full Bitvector
transfer did not show a stable eight-way benefit; 0%-resident time varied
more under concurrency. The sums of per-request `GetFlightInfo` and `DoGet`
measurements overlap when requests are concurrent and must not be read as
wall time. The actual cache-hit timer starts before the first Flight call
and stops at the eighth generated token.

For the **actual persisted shifou compact4 cache**, the scratch client fetched
the 56 `SHIFOU02` manifests and 56 packed payloads through 112 Bitvector
Flight requests. It rebuilt each `PackedSnapshot`, checked address, manifest
hash, payload hash and buffer geometry, then used shifou's `decode_compact`
and the model's full-cache GPU restore. A post-timing comparison with
`CacheReader::get_packed` established byte-for-byte identity for every tile.
The table gives the sequential sample and the range from two eight-way runs.

| Prefix | Local compact4 cache hit | Flight, sequential | Flight, 8 concurrent | Eight-way Flight read | CPU decode | Full GPU restore |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 1,161 ms | 1,405 ms | 1,275-1,288 ms | 166-169 ms | 933-946 ms | 72-76 ms |
| 8K | 6,000 ms | 6,426 ms | 6,411-6,417 ms | 786-809 ms | 4,363-4,376 ms | 1,107-1,111 ms |

The Flight compact4 path preserved the accuracy of the local compact4 path:
all eight greedy tokens matched native BF16 at both prefixes; the
teacher-forced top choice matched on 61/64 tokens at 2K and 59/64 at 8K,
with mean KL 0.0091 and 0.0402 respectively. The BF16 page path matched
all 64 top choices and every measured logit exactly. Thus neither Flight
transport nor GPU restoration introduced additional model error in these
fixtures. At 8K, compact4's CPU decode remains the dominant latency; moving
the smaller packed payload over Flight does not change that conclusion.

Scratch source is
`../yesno/.agents-workspace/tmp/kv-page-probe-20260930/src/{flight_e2e,compact_flight_e2e}.rs`.
Machine-readable outputs and raw logs there use
`result-flight-e2e-{2048-4096,8192-16384}{,-c8,-c1-r2,-c8-r2}.json`
and `result-compact-flight-e2e-{2048-4096,8192-16384}-c{1,8}{,-r2}.json`
where the matching file exists. From yesno's root, build the scratch crate as
above and run its binary with `flight-e2e DB_DIR` for the BF16 page store or
`compact-flight-e2e DB_DIR` for the shifou packed store. Set
`FLIGHT_CONCURRENCY=1` or `8` and `SCORE_STEPS=64` in the environment.
The local server and client ran on the same host, with no external network
latency, production request scheduler, or measured xinfer eviction trace.
The packed Flight client is a research decoder of the current shifou manifest;
it has not been installed as a production shifou transport adapter.
